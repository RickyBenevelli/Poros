"""Gaussian-splatting detector (residual-driven matching pursuit, torch-free).

Inspired by 3D Gaussian Splatting: instead of finding blob centres at a
characteristic scale (the LoG detector) or thresholding a response (bottom-hat),
this detector *explains the darkness* with a population of 2D anisotropic
Gaussians. Cells emerge from coverage - and an irregular cavity is formed by
**several overlapping Gaussians merged together**.

The optimise -> densify -> prune loop of 3DGS is reproduced without autograd:

* **target**   - the bottom-hat response masked to the eroded slice interior.
* **densify**  - repeatedly seed a Gaussian at the darkest still-unexplained
  pixel, fit its mean/covariance/amplitude from intensity-weighted moments of the
  local dark basin (closed form), and subtract its splat from the residual.
* **prune**    - reject Gaussians that are too shallow (crumb shadow) or whose
  footprint falls outside the plausible cell-area band.
* **refine**   - optional coordinate-descent sweeps re-fitting each Gaussian
  against the residual with its own splat added back.
* **group**    - rasterise every surviving Gaussian's iso-contour footprint and
  union them; overlapping footprints merge into one hole, then the union is run
  through the shared :func:`~poros.detectors._common.filter_components`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np
import numpy.typing as npt

from .._types import (
    DetectionContext,
    DetectionResult,
    DetectorName,
    FloatImage,
    Mask,
)
from ._base import BaseDetector
from ._common import erode_interior, filter_components

_COV_REG = 0.5
"""Covariance ridge (px^2) added to keep the 2x2 matrix invertible."""


@dataclass
class GaussianSplatConfig:
    """Tuning for :class:`GaussianSplatDetector`.

    ``fit_window_frac`` is tied to cell scale (not the morphology kernel) so
    Gaussians stay cell-sized and don't bridge across the network.
    ``max_sigma_frac`` caps each semi-axis as a fraction of the slice diameter;
    broader fits are clamped then pruned. ``min_circularity=0`` so elongated
    cells — the motivation for this detector — are not discarded.
    """

    max_gaussians: int = 8000
    residual_stop_frac: float = 0.05
    min_amplitude_frac: float = 0.25
    iso_sigma: float = 1.2
    refine_iters: int = 1
    min_circularity: float = 0.0
    max_area_frac: float = 0.30
    fit_window_frac: float = 0.03
    max_sigma_frac: float = 0.04
    downsample: float = 1.0


@dataclass
class Gaussian2D:
    """A single 2D anisotropic Gaussian splat.

    ``cov`` is a 2×2 covariance matrix (px²); eigenvectors give orientation,
    eigenvalues the squared semi-axes.
    """

    cx: float
    cy: float
    cov: npt.NDArray[np.float64]
    amplitude: float


class GaussianSplatDetector(BaseDetector):
    """Detect cells by covering the darkness map with 2D Gaussians.

    See the module docstring for the optimise/densify/prune/group loop. Stats:
    ``n_gaussians``, ``residual_explained`` (fraction of initial darkness energy
    removed), plus ``raw_components`` / ``kept_components``.
    """

    name: ClassVar[DetectorName] = DetectorName.GAUSSIAN

    def __init__(self, config: GaussianSplatConfig | None = None) -> None:
        self._cfg = config or GaussianSplatConfig()

    def detect(self, ctx: DetectionContext) -> DetectionResult:
        cfg = self._cfg
        inner = erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)
        target = _bottomhat_target(ctx, inner)

        downsampled = cfg.downsample > 1.0
        sf = 1.0 / cfg.downsample if downsampled else 1.0
        work = (
            cv2.resize(target, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
            if downsampled
            else target
        )

        win = max(5, int(round(ctx.scale.diameter * cfg.fit_window_frac * sf)))
        max_sigma = ctx.scale.diameter * cfg.max_sigma_frac * sf
        min_area_w = max(1.0, ctx.scale.min_area * sf * sf)
        max_area_w = cfg.max_area_frac * ctx.scale.slice_area * sf * sf

        gaussians, explained = _splat_pursuit(
            work, cfg, win=win, max_sigma=max_sigma,
            min_area=min_area_w, max_area=max_area_w,
        )

        footprint = _rasterize_footprints(gaussians, work.shape, cfg.iso_sigma)
        if downsampled:
            footprint = cv2.resize(
                footprint, (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        footprint = cv2.bitwise_and(footprint, inner)

        accepted, holes, stats = filter_components(
            footprint, target, ctx.scale,
            min_circularity=cfg.min_circularity, max_area_frac=cfg.max_area_frac,
        )
        result_stats: dict[str, int | float | str] = {
            "n_gaussians": len(gaussians),
            "residual_explained": round(explained, 3),
            **stats,
        }
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )


# ----------------------------------------------------------------------------- #
# Matching-pursuit core                                                         #
# ----------------------------------------------------------------------------- #


def _splat_pursuit(
    work: FloatImage,
    cfg: GaussianSplatConfig,
    *,
    win: int,
    max_sigma: float,
    min_area: float,
    max_area: float,
) -> tuple[list[Gaussian2D], float]:
    """Greedily seed/fit/subtract Gaussians until the darkness is explained.

    Returns the surviving Gaussians and the fraction of the initial darkness
    energy removed from the residual.
    """
    residual = work.astype(np.float32, copy=True)
    total_energy = float(residual.sum())
    if total_energy <= 0.0:
        return [], 0.0

    amp_floor = cfg.min_amplitude_frac * float(residual.max())
    gaussians: list[Gaussian2D] = []

    for _ in range(cfg.max_gaussians):
        _, peak, _, (mx, my) = cv2.minMaxLoc(residual)
        if peak <= amp_floor:
            break
        if residual.sum() <= cfg.residual_stop_frac * total_energy:
            break

        g = _fit_local_gaussian(residual, int(my), int(mx), win)
        if g is None:
            _suppress(residual, int(mx), int(my), win // 2)
            continue
        _clamp_cov(g, max_sigma)

        area = float(np.pi * cfg.iso_sigma * cfg.iso_sigma * np.sqrt(_det(g.cov)))
        if g.amplitude < amp_floor or area < min_area or area > max_area:
            _suppress(residual, int(mx), int(my), win // 2)
            continue

        _subtract_splat(residual, g, radius_sigma=3.0)
        gaussians.append(g)

    for _ in range(cfg.refine_iters):
        _refine(residual, gaussians, win)

    explained = 1.0 - float(residual.sum()) / total_energy
    return gaussians, explained


def _fit_local_gaussian(
    residual: FloatImage, y: int, x: int, win: int, peak_frac: float = 0.5
) -> Gaussian2D | None:
    """Fit one Gaussian to the dark basin around ``(y, x)`` by weighted moments.

    The window is restricted to the connected component (above ``peak_frac`` of
    the seed value) that contains the seed, so neighbouring cells do not pull the
    covariance outward.
    """
    h, w = residual.shape[:2]
    peak = float(residual[y, x])
    if peak <= 0.0:
        return None

    y0, y1 = max(0, y - win), min(h, y + win + 1)
    x0, x1 = max(0, x - win), min(w, x + win + 1)
    patch = residual[y0:y1, x0:x1]

    basin = (patch >= peak_frac * peak).astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(basin)
    if n_lab <= 1:
        return None
    seed_lab = int(labels[y - y0, x - x0])
    if seed_lab == 0:
        return None

    weights = np.where(labels == seed_lab, patch, 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return None

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    cx = float((xs * weights).sum() / total)
    cy = float((ys * weights).sum() / total)
    dx, dy = xs - cx, ys - cy
    sxx = float((weights * dx * dx).sum() / total) + _COV_REG
    syy = float((weights * dy * dy).sum() / total) + _COV_REG
    sxy = float((weights * dx * dy).sum() / total)

    cov = np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)
    return Gaussian2D(cx=cx, cy=cy, cov=cov, amplitude=peak)


def _refine(residual: FloatImage, gaussians: list[Gaussian2D], win: int) -> None:
    """One coordinate-descent sweep: re-fit each Gaussian against residual + self."""
    for i, g in enumerate(gaussians):
        _add_splat(residual, g, radius_sigma=3.0)
        refit = _fit_local_gaussian(residual, int(round(g.cy)), int(round(g.cx)), win)
        new = refit if refit is not None else g
        _subtract_splat(residual, new, radius_sigma=3.0)
        gaussians[i] = new


# ----------------------------------------------------------------------------- #
# Splat evaluation / rasterisation                                              #
# ----------------------------------------------------------------------------- #


def _splat_bbox(
    g: Gaussian2D, shape: tuple[int, ...], radius_sigma: float
) -> tuple[int, int, int, int, npt.NDArray[np.float64]] | None:
    h, w = shape[:2]
    eig = np.linalg.eigvalsh(g.cov)
    s_max = float(np.sqrt(max(float(eig.max()), 1e-6)))
    r = int(np.ceil(radius_sigma * s_max))
    if r <= 0:
        return None
    cx, cy = int(round(g.cx)), int(round(g.cy))
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return None

    inv = np.linalg.inv(g.cov)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dx, dy = xs - g.cx, ys - g.cy
    mahal = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return y0, y1, x0, x1, mahal


def _subtract_splat(residual: FloatImage, g: Gaussian2D, *, radius_sigma: float) -> None:
    box = _splat_bbox(g, residual.shape, radius_sigma)
    if box is None:
        return
    y0, y1, x0, x1, mahal = box
    vals = (g.amplitude * np.exp(-0.5 * mahal)).astype(np.float32)
    region = residual[y0:y1, x0:x1]
    np.clip(region - vals, 0.0, None, out=region)


def _add_splat(residual: FloatImage, g: Gaussian2D, *, radius_sigma: float) -> None:
    box = _splat_bbox(g, residual.shape, radius_sigma)
    if box is None:
        return
    y0, y1, x0, x1, mahal = box
    residual[y0:y1, x0:x1] += (g.amplitude * np.exp(-0.5 * mahal)).astype(np.float32)


def _rasterize_footprints(
    gaussians: list[Gaussian2D], shape: tuple[int, ...], iso_sigma: float
) -> Mask:
    mask: Mask = np.zeros(shape[:2], dtype=np.uint8)
    iso_sq = iso_sigma * iso_sigma
    for g in gaussians:
        box = _splat_bbox(g, shape, iso_sigma)
        if box is None:
            continue
        y0, y1, x0, x1, mahal = box
        sub = mask[y0:y1, x0:x1]
        sub[mahal <= iso_sq] = 255
    return mask


# ----------------------------------------------------------------------------- #
# Small helpers                                                                 #
# ----------------------------------------------------------------------------- #


def _bottomhat_target(ctx: DetectionContext, inner: Mask) -> FloatImage:
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ctx.scale.kernel_size, ctx.scale.kernel_size)
    )
    blackhat = cv2.morphologyEx(ctx.gray, cv2.MORPH_BLACKHAT, se)
    masked = cv2.bitwise_and(blackhat, blackhat, mask=inner)
    return masked.astype(np.float32)


def _suppress(residual: FloatImage, x: int, y: int, r: int) -> None:
    """Zero a small disk so a failed/pruned seed cannot be re-picked forever."""
    r = max(1, r)
    h, w = residual.shape[:2]
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    residual[y0:y1, x0:x1] = 0.0


def _clamp_cov(g: Gaussian2D, max_sigma: float) -> None:
    """Clamp a Gaussian's covariance so no semi-axis exceeds ``max_sigma``.

    Keeps a single splat from growing large enough to bridge neighbouring cells;
    orientation (the eigenvectors) is preserved, only the eigenvalues are capped.
    """
    if max_sigma <= 0.0:
        return
    vals, vecs = np.linalg.eigh(g.cov)
    capped = np.clip(vals, _COV_REG, max_sigma * max_sigma)
    g.cov = (vecs @ np.diag(capped) @ vecs.T).astype(np.float64)


def _det(cov: npt.NDArray[np.float64]) -> float:
    """2×2 determinant, floored at a tiny positive value."""
    return max(float(cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]), 1e-6)

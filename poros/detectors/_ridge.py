"""Ridge / reticulum detector: watershed flooded along the cell-wall network."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Literal

import cv2
import numpy as np
from skimage.feature import (
    peak_local_max,
    structure_tensor,
    structure_tensor_eigenvalues,
)
from skimage.filters import frangi, meijering, sato
from skimage.segmentation import watershed

from .._types import (
    DetectionContext,
    DetectionResult,
    DetectorName,
    FloatImage,
    GrayImage,
)
from ._base import BaseDetector
from ._common import erode_interior, filter_components

RidgeFilter = Literal["frangi", "sato", "meijering", "structure_tensor"]
"""Which ridge/vesselness operator highlights the cell walls (the reticulum)."""


def _ridge_response(
    gray: GrayImage, kind: RidgeFilter, sigmas: tuple[float, ...]
) -> FloatImage:
    """Return a [0, 1] vesselness response that is high on bright tubular walls.

    The crumb walls form a bright network that is structurally identical to the
    tubular patterns targeted by blood-vessel filters.
    """
    img = gray.astype(np.float64) / 255.0
    sigma_list = list(sigmas)

    if kind == "frangi":
        return frangi(img, sigmas=sigma_list, black_ridges=False)
    if kind == "sato":
        return sato(img, sigmas=sigma_list, black_ridges=False)
    if kind == "meijering":
        return meijering(img, sigmas=sigma_list, black_ridges=False)

    # structure_tensor: largest eigenvalue magnitude marks oriented structure.
    a_rr, a_rc, a_cc = structure_tensor(img, sigma=sigma_list[0], order="rc")
    larger, _ = structure_tensor_eigenvalues((a_rr, a_rc, a_cc))
    max_val = float(larger.max())
    return larger / max_val if max_val > 0 else larger


def _as_u8(image: np.ndarray) -> np.ndarray:
    """Convert a debug image to uint8 so it can be written as a PNG."""
    if image.dtype == np.uint8:
        return image
    finite = np.nan_to_num(image.astype(np.float64, copy=False))
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(finite.shape, dtype=np.uint8)
    return ((finite - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _save_step_images(
    out_dir: str, steps: list[tuple[str, np.ndarray]], cfg: RidgeConfig
) -> None:
    sigmas = "-".join(str(sigma) for sigma in cfg.sigmas)
    out_dir = (
        f"{out_dir}/{cfg.ridge_filter}_{sigmas}_{cfg.seed_percentile}_"
        f"{cfg.wall_cut_percentile}_{cfg.max_area_frac}_{cfg.min_circularity}"
    )
    os.makedirs(out_dir, exist_ok=True)
    for index, (name, image) in enumerate(steps, start=1):
        path = os.path.join(out_dir, f"{index:02d}_{name}.png")
        cv2.imwrite(path, _as_u8(image))


@dataclass
class RidgeConfig:
    """Tuning for :class:`RidgeDetector`.

    ``seed_percentile``: seeds are placed where the ridge response is below this
    percentile (flat cell interiors, away from walls).
    ``wall_cut_percentile``: pixels above this percentile are treated as wall and
    excluded from cells so adjacent cells stay physically separate.
    """

    ridge_filter: RidgeFilter = "meijering"
    sigmas: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    seed_percentile: float = 35.0
    wall_cut_percentile: float = 55.0
    min_distance: int | None = None
    max_area_frac: float = 0.40
    min_circularity: float = 0.0
    save_steps_dir: str | None = "ridge_steps"
    """Optional directory for intermediate debug PNGs.

    The same behavior can be enabled from the CLI with
    ``POROS_RIDGE_STEPS_DIR=/path/to/dir``.
    """


class RidgeDetector(BaseDetector):
    """Segment cells by flooding a watershed along the cell-wall network.

    The vesselness/structure-tensor response is high on the bright walls (the
    reticolo) and low inside cells. Rather than thresholding the walls (which
    breaks where the network has gaps), the response is used as the watershed
    *elevation*: seeds are placed in the low-response cell interiors and the
    flood rises until it meets the wall ridges, so cell boundaries follow the
    reticolo even across small gaps. Regions are then filtered by area,
    circularity, and cell darkness.

    Stats: ``ridge_filter``, ``n_seeds``, plus ``raw_components`` /
    ``kept_components``.
    """

    name: ClassVar[DetectorName] = DetectorName.RIDGE

    def __init__(self, config: RidgeConfig | None = None) -> None:
        self._cfg = config or RidgeConfig()

    def detect(self, ctx: DetectionContext) -> DetectionResult:
        cfg = self._cfg
        # Erode the interior mask to avoid seeds too close to the walls
        inner = erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)
        # Compute the raw ridge/vesselness response from the selected filter
        response = _ridge_response(ctx.gray, cfg.ridge_filter, cfg.sigmas)
        # normalize to [0, 255] for percentile thresholding and watershed input
        resp_u8 = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        # high values in resp_u8 correspond to walls
        # inner > 0 restricts the calculation to pixels inside the eroded slice mask
        # takes the seed_percentile as the threshold to identify the holes
        seed_level = float(np.percentile(resp_u8[inner > 0], cfg.seed_percentile))
        # create a binary mask of the interiors where the ridge response is below the seed_level
        interiors = cv2.bitwise_and(
            (resp_u8 <= seed_level).astype(np.uint8) * 255, inner
        )
        dist = cv2.distanceTransform(interiors, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        min_distance = (
            cfg.min_distance
            if cfg.min_distance is not None
            else max(3, int(ctx.scale.diameter * 0.0))  # !
        )

        coords = peak_local_max(
            dist, min_distance=min_distance, labels=interiors > 0, exclude_border=False
        )

        markers = np.zeros(ctx.gray.shape, dtype=np.int32)
        for idx, (y, x) in enumerate(coords, start=1):
            markers[y, x] = idx

        # Ridge response as watershed elevation: walls (high response) become
        # the watershed lines separating neighbouring cells.
        labels = watershed(resp_u8, markers, mask=inner > 0, watershed_line=True)
        # wall_cut_percentile is used to identify pixels that are likely part of the walls
        wall_cut = float(np.percentile(resp_u8[inner > 0], cfg.wall_cut_percentile))
        # a pixel is part of a hole if it is labeled as a cell (labels > 0)
        # but has a ridge response above the wall_cut threshold
        candidate: GrayImage = ((labels > 0) & (resp_u8 <= wall_cut)).astype(
            np.uint8
        ) * 255

        darkness: GrayImage = (255 - ctx.gray).astype(np.uint8)
        accepted, holes, stats = filter_components(
            candidate,
            darkness,
            ctx.scale,
            min_circularity=cfg.min_circularity,
            max_area_frac=cfg.max_area_frac,
        )
        result_stats: dict[str, int | float | str] = {
            "ridge_filter": cfg.ridge_filter,
            "n_seeds": int(coords.shape[0]),
            **stats,
        }
        save_steps_dir = os.environ.get("POROS_RIDGE_STEPS_DIR") or cfg.save_steps_dir
        if save_steps_dir:
            _save_step_images(
                save_steps_dir,
                [
                    ("gray", ctx.gray),
                    ("inner_mask", inner),
                    ("ridge_response", response),
                    ("ridge_response_u8", resp_u8),
                    ("interiors", interiors),
                    ("distance_transform", dist),
                    ("markers", markers),
                    ("watershed_labels", labels),
                    ("candidate_wall_cut", candidate),
                    ("darkness_score", darkness),
                    ("accepted_mask", accepted),
                ],
                cfg,
            )
            result_stats["steps_dir"] = save_steps_dir
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )

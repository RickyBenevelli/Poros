"""Laplacian-of-Gaussian blob detector.

Two modes share one detector:

* ``"blob"`` (default) - OpenCV ``SimpleBlobDetector``, which thresholds the
  image at many levels and keeps dark regions that stay stable across them,
  then verifies each with a local-contrast check. Robust and selective.
* ``"oriented"`` - a bank of anisotropic LoG kernels at several scales x angles
  (max response). Targets elongated cells, but is noisier; kept for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import cv2
import numpy as np

from .._types import (
    DetectionContext,
    DetectionResult,
    DetectorName,
    FloatImage,
    GrayImage,
    Hole,
    Mask,
)
from ._base import BaseDetector
from ._common import erode_interior, filter_components, local_contrast, split_touching_blobs

LogMode = Literal["blob", "oriented"]
"""``blob`` = SimpleBlobDetector; ``oriented`` = anisotropic LoG filter bank."""


def _local_ellipse(
    gray: GrayImage, cx: int, cy: int, r: int
) -> tuple[float, float, float, float, float] | None:
    """Fit an oriented ellipse to the dark blob around ``(cx, cy)``.

    Segments the dark region in a local window and fits an ellipse to it, so
    elongated/tilted cells are captured as ovals rather than fixed circles.
    Returns ``(ecx, ecy, major, minor, angle_deg)`` in full-image coordinates,
    or ``None`` if no fittable region exists.
    """
    h, w = gray.shape
    pad = max(5, 2 * r)
    y0, y1 = max(0, cy - pad), min(h, cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(w, cx + pad + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size < 25:
        return None

    thr, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (patch < thr).astype(np.uint8)
    ly, lx = cy - y0, cx - x0
    _, comp_labels = cv2.connectedComponents(dark)
    center_label = int(comp_labels[ly, lx]) if dark[ly, lx] else 0
    if center_label == 0:
        return None

    component = (comp_labels == center_label).astype(np.uint8)
    cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)  # ty: ignore[no-matching-overload]
    if len(cnt) < 5:
        return None
    (ecx, ecy), (major, minor), angle = cv2.fitEllipse(cnt)
    return ecx + x0, ecy + y0, major, minor, angle


def _anisotropic_log_kernel(
    sigma_par: float, sigma_perp: float, theta: float
) -> FloatImage:
    """Build a Laplacian-of-anisotropic-Gaussian kernel oriented at ``theta``."""
    radius = int(np.ceil(3.0 * max(sigma_par, sigma_perp)))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_r = xx * cos_t + yy * sin_t
    y_r = -xx * sin_t + yy * cos_t
    sx2, sy2 = sigma_par**2, sigma_perp**2
    gauss = np.exp(-(x_r**2 / (2.0 * sx2) + y_r**2 / (2.0 * sy2)))
    log = ((x_r**2 - sx2) / sx2**2 + (y_r**2 - sy2) / sy2**2) * gauss
    return log - log.mean()


@dataclass
class LogConfig:
    """Tuning for :class:`LogBlobDetector`.

    Blob mode params: ``fit_ellipse``, ``min_contrast``, ``min/max_threshold``,
    ``threshold_step``, ``min_convexity``.
    Oriented mode params: ``sigmas``, ``anisotropy``, ``n_orientations``,
    ``threshold_scale``.
    """

    mode: LogMode = "blob"
    fit_ellipse: bool = True
    min_contrast: float = 8.0
    min_threshold: float = 10.0
    max_threshold: float = 220.0
    threshold_step: float = 10.0
    min_convexity: float = 0.5
    sigmas: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    anisotropy: float = 2.0
    n_orientations: int = 6
    threshold_scale: float = 1.0
    max_area_frac: float = 0.30
    min_circularity: float = 0.20


class LogBlobDetector(BaseDetector):
    """Multi-scale dark-blob detector (SimpleBlobDetector or oriented LoG bank).

    Stats (blob mode): ``mode``, ``raw_keypoints``, ``rejected_boundary``,
    ``rejected_contrast``, ``kept_components``. Stats (oriented mode): ``mode``,
    ``bank_size``, ``threshold_used``, ``raw_components`` / ``kept_components``.
    """

    name: ClassVar[DetectorName] = DetectorName.LOG

    def __init__(self, config: LogConfig | None = None) -> None:
        self._cfg = config or LogConfig()

    def detect(self, ctx: DetectionContext) -> DetectionResult:
        if self._cfg.mode == "oriented":
            return self._detect_oriented(ctx)
        return self._detect_blob(ctx)

    # --- blob mode (SimpleBlobDetector) ------------------------------------- #

    def _detect_blob(self, ctx: DetectionContext) -> DetectionResult:
        """SimpleBlobDetector with a per-keypoint local-contrast verification."""
        cfg = self._cfg
        gray = ctx.gray
        inner = erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)
        max_area = cfg.max_area_frac * ctx.scale.slice_area

        # cv2 stubs omit SimpleBlobDetector_* though they exist at runtime.
        params = cv2.SimpleBlobDetector_Params()  # ty: ignore[unresolved-attribute]
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = float(ctx.scale.min_area)
        params.maxArea = float(max_area)
        params.filterByCircularity = True
        params.minCircularity = float(cfg.min_circularity)
        params.filterByConvexity = True
        params.minConvexity = float(cfg.min_convexity)
        params.filterByInertia = False
        params.minThreshold = float(cfg.min_threshold)
        params.maxThreshold = float(cfg.max_threshold)
        params.thresholdStep = float(cfg.threshold_step)

        detector = cv2.SimpleBlobDetector_create(params)  # ty: ignore[unresolved-attribute]
        keypoints = detector.detect(gray)

        accepted: Mask = np.zeros(gray.shape, dtype=np.uint8)
        holes: list[Hole] = []
        rejected_boundary = 0
        rejected_contrast = 0
        for kp in keypoints:
            ix, iy = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if not (0 <= iy < inner.shape[0] and 0 <= ix < inner.shape[1]) or inner[iy, ix] == 0:
                rejected_boundary += 1
                continue
            r = max(1, int(round(kp.size / 2.0)))
            contrast = local_contrast(gray, ix, iy, r)
            if contrast < cfg.min_contrast:
                rejected_contrast += 1
                continue

            ellipse = _local_ellipse(gray, ix, iy, r) if cfg.fit_ellipse else None
            if ellipse is not None:
                ecx, ecy, major, minor, angle = ellipse
                cv2.ellipse(
                    accepted, (round(ecx), round(ecy)),
                    (round(major / 2), round(minor / 2)), angle, 0, 360, 255, -1,
                )
                hole = Hole(
                    id=len(holes) + 1, cx=ecx, cy=ecy,
                    area=int(np.pi * major * minor / 4.0),
                    circularity=round(minor / major, 3) if major > 0 else 1.0,
                    mean_depth=round(contrast, 1),
                    bbox_x=round(ecx - major / 2), bbox_y=round(ecy - minor / 2),
                    bbox_w=round(major), bbox_h=round(minor), angle=round(angle, 1),
                )
            else:
                cv2.circle(accepted, (ix, iy), r, 255, -1)
                hole = Hole(
                    id=len(holes) + 1, cx=float(kp.pt[0]), cy=float(kp.pt[1]),
                    area=int(np.pi * r * r), circularity=1.0,
                    mean_depth=round(contrast, 1),
                    bbox_x=ix - r, bbox_y=iy - r, bbox_w=2 * r, bbox_h=2 * r,
                )
            holes.append(hole)

        stats: dict[str, int | float | str] = {
            "mode": "blob",
            "raw_keypoints": len(keypoints),
            "rejected_boundary": rejected_boundary,
            "rejected_contrast": rejected_contrast,
            "kept_components": len(holes),
        }
        return DetectionResult(name=self.name, mask=accepted, holes=holes, stats=stats)

    # --- oriented mode (anisotropic LoG bank) ------------------------------- #

    def _oriented_response(self, gray: FloatImage) -> FloatImage:
        """Max LoG response across the scale x orientation bank (dark blobs)."""
        cfg = self._cfg
        inverted = 255.0 - gray
        thetas = np.linspace(0.0, np.pi, cfg.n_orientations, endpoint=False)
        response = np.zeros_like(gray, dtype=np.float64)
        for sigma in cfg.sigmas:
            sigma_par = sigma * cfg.anisotropy
            for theta in thetas:
                kernel = _anisotropic_log_kernel(sigma_par, sigma, float(theta))
                filtered = cv2.filter2D(inverted, cv2.CV_64F, kernel) * sigma
                np.maximum(response, filtered, out=response)
        return response

    def _detect_oriented(self, ctx: DetectionContext) -> DetectionResult:
        cfg = self._cfg
        response = self._oriented_response(ctx.gray.astype(np.float64))
        response[ctx.slice_mask == 0] = 0.0

        resp_u8 = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # ty: ignore[no-matching-overload]
        vals = resp_u8[ctx.slice_mask > 0]
        otsu_t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_val = int(min(254, otsu_t * cfg.threshold_scale))
        _, raw = cv2.threshold(resp_u8, thr_val, 255, cv2.THRESH_BINARY)

        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_k)
        cleaned = cv2.bitwise_and(
            cleaned, erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)
        )
        candidate = split_touching_blobs(cleaned)
        accepted, holes, stats = filter_components(
            candidate, resp_u8, ctx.scale,
            min_circularity=cfg.min_circularity, max_area_frac=cfg.max_area_frac,
        )
        result_stats: dict[str, int | float | str] = {
            "mode": "oriented",
            "bank_size": len(cfg.sigmas) * cfg.n_orientations,
            "threshold_used": int(thr_val),
            **stats,
        }
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )

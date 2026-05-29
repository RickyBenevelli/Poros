"""Adaptive-threshold detector (Sauvola / Niblack / local mean)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import cv2
import numpy as np
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola

from .._types import DetectionContext, DetectionResult, DetectorName, GrayImage
from ._base import BaseDetector
from ._common import erode_interior, filter_components, split_touching_blobs
from .._geometry import odd

AdaptiveMethod = Literal["sauvola", "niblack", "local"]
"""Which local thresholding rule to use for the dark-cell binarization."""


def _local_threshold(gray: GrayImage, method: AdaptiveMethod, window: int, k: float) -> np.ndarray:
    """Return the per-pixel threshold surface for the chosen method.

    The uint8 image is passed directly so skimage infers the correct dynamic
    range ``r`` from the dtype; casting to float first makes Sauvola's auto
    ``r`` blow up and the threshold surface degenerate.
    """
    if method == "sauvola":
        return threshold_sauvola(gray, window_size=window, k=k)
    if method == "niblack":
        return threshold_niblack(gray, window_size=window, k=k)
    return threshold_local(gray, block_size=window, method="gaussian")


@dataclass
class AdaptiveConfig:
    """Tuning for :class:`AdaptiveThresholdDetector`.

    ``window_size=None`` derives a window ~5% of the slice diameter. ``k`` is
    the Sauvola/Niblack sensitivity (ignored by the ``local`` method).
    """

    method: AdaptiveMethod = "sauvola"
    window_size: int | None = None
    k: float = 0.2
    max_area_frac: float = 0.30
    min_circularity: float = 0.20
    split_touching: bool = True


class AdaptiveThresholdDetector(BaseDetector):
    """Detect cells via adaptive (local) thresholding of the gray image.

    A per-pixel threshold surface handles uneven illumination; pixels darker
    than their local threshold become cell candidates. Stats: ``method``,
    ``window_size``, ``raw_components``, ``kept_components``.
    """

    name: ClassVar[DetectorName] = DetectorName.ADAPTIVE

    def __init__(self, config: AdaptiveConfig | None = None) -> None:
        self._cfg = config or AdaptiveConfig()

    def detect(self, ctx: DetectionContext) -> DetectionResult:
        cfg = self._cfg
        # Window tracks cell scale (~5% of diameter), not the morphology kernel.
        window = cfg.window_size if cfg.window_size is not None else odd(
            max(15, int(ctx.scale.diameter * 0.05))
        )

        thr_surface = _local_threshold(ctx.gray, cfg.method, window, cfg.k)
        raw = ((ctx.gray.astype(np.float64) < thr_surface).astype(np.uint8)) * 255

        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_k)
        cleaned = cv2.bitwise_and(
            cleaned, erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)
        )

        candidate = split_touching_blobs(cleaned) if cfg.split_touching else cleaned
        darkness: GrayImage = (255 - ctx.gray).astype(np.uint8)
        accepted, holes, stats = filter_components(
            candidate,
            darkness,
            ctx.scale,
            min_circularity=cfg.min_circularity,
            max_area_frac=cfg.max_area_frac,
        )

        result_stats: dict[str, int | float | str] = {
            "method": cfg.method,
            "window_size": int(window),
            **stats,
        }
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )

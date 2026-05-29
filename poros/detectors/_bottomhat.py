"""Morphological bottom-hat detector"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import cv2

from .._types import DetectionContext, DetectionResult, DetectorName
from ._base import BaseDetector
from ._common import erode_interior, filter_components, split_touching_blobs


@dataclass
class BottomHatConfig:
    """Tuning for :class:`BottomHatDetector`.

    ``threshold_value=None`` uses Otsu scaled by ``threshold_scale`` (>1
    suppresses shallow texture, <1 boosts recall). ``min_depth_frac`` rejects
    blobs whose mean bottom-hat response is below that fraction of the threshold.
    """

    max_area_frac: float = 0.30
    min_circularity: float = 0.20
    threshold_value: int | None = None
    threshold_scale: float = 1.0
    min_depth_frac: float = 0.4
    split_touching: bool = True


class BottomHatDetector(BaseDetector):
    """Detect cells via the morphological bottom-hat (black-hat) transform.

    ``blackhat(I) = closing(I) - I``. Closing fills dark structures smaller than
    the structuring element, so subtracting the original isolates exactly those
    dark structures (the cells). The residual is Otsu-thresholded on slice
    pixels, cleaned, optionally watershed-split, then filtered by area,
    circularity, and per-blob depth.

    Stats: ``threshold_used``, ``depth_floor``, plus ``raw_components`` /
    ``kept_components``.
    """

    name: ClassVar[DetectorName] = DetectorName.BOTTOMHAT

    def __init__(self, config: BottomHatConfig | None = None) -> None:
        self._cfg = config or BottomHatConfig()

    def detect(self, ctx: DetectionContext) -> DetectionResult:
        cfg = self._cfg
        gray = ctx.gray
        slice_mask = ctx.slice_mask

        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ctx.scale.kernel_size, ctx.scale.kernel_size)
        )
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        blackhat_in = cv2.bitwise_and(blackhat, blackhat, mask=slice_mask)

        if cfg.threshold_value is None:
            vals = blackhat[slice_mask > 0]
            otsu_t, _ = cv2.threshold(
                vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thr_val = int(min(254, otsu_t * cfg.threshold_scale))
        else:
            thr_val = cfg.threshold_value
        _, raw = cv2.threshold(blackhat_in, thr_val, 255, cv2.THRESH_BINARY)

        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_k)
        cleaned = cv2.bitwise_and(
            cleaned, erode_interior(slice_mask, ctx.scale.boundary_erosion)
        )

        candidate = split_touching_blobs(cleaned) if cfg.split_touching else cleaned

        depth_floor = thr_val * cfg.min_depth_frac
        accepted, holes, stats = filter_components(
            candidate,
            blackhat_in,
            ctx.scale,
            min_circularity=cfg.min_circularity,
            max_area_frac=cfg.max_area_frac,
            min_score=depth_floor,
        )

        result_stats: dict[str, int | float | str] = {
            "threshold_used": int(thr_val),
            "depth_floor": round(float(depth_floor), 1),
            **stats,
        }
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )

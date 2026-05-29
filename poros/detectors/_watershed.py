"""Marker-controlled watershed detector (cell clustering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from skimage.segmentation import watershed

from .._types import DetectionContext, DetectionResult, DetectorName, GrayImage, Mask
from .._geometry import odd
from ._base import BaseDetector
from ._common import erode_interior, filter_components

ForegroundMethod = Literal["local", "otsu"]
"""How the dark-cell foreground is thresholded before watershed."""


@dataclass
class WatershedConfig:
    """Tuning for :class:`WatershedDetector`.

    ``foreground="local"`` adapts to brightness gradients so bright zones still
    get foreground; ``"otsu"`` blanks them out. ``foreground_close`` solidifies
    large cavities but off by default because closing merges the cell network
    and collapses per-cell separation. ``min_circularity`` is intentionally low
    because large watershed basins are not round.
    """

    foreground: ForegroundMethod = "local"
    foreground_offset: float = 6.0
    foreground_close: int = 0
    marker_blur: float = 1.0
    min_distance: int | None = None
    max_area_frac: float = 0.30
    min_circularity: float = 0.05


class WatershedDetector(BaseDetector):
    """Cluster the dark crumb into individual cells via marker-controlled watershed.

    Stats: ``n_markers``, ``min_distance``, ``raw_components``,
    ``kept_components``.
    """

    name: ClassVar[DetectorName] = DetectorName.WATERSHED

    def __init__(self, config: WatershedConfig | None = None) -> None:
        self._cfg = config or WatershedConfig()

    def detect(self, ctx: DetectionContext) -> DetectionResult:
        cfg = self._cfg
        inner = erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)

        # Local thresholding keeps bright zones populated; global Otsu blanks them.
        if cfg.foreground == "local":
            block = odd(max(15, int(ctx.scale.diameter * 0.05)))
            surface = threshold_local(
                ctx.gray, block_size=block, method="gaussian", offset=cfg.foreground_offset
            )
            foreground = (ctx.gray.astype(np.float64) < surface).astype(np.uint8) * 255
        else:
            vals = ctx.gray[inner > 0]
            otsu_t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            foreground = ((ctx.gray < otsu_t).astype(np.uint8)) * 255
        foreground = cv2.bitwise_and(foreground, inner)
        if cfg.foreground_close > 0:
            c = cfg.foreground_close * 2 + 1
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)),
            )

        dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        if cfg.marker_blur > 0:
            dist = cv2.GaussianBlur(dist, (0, 0), cfg.marker_blur)

        min_distance = (
            cfg.min_distance
            if cfg.min_distance is not None
            else max(3, int(ctx.scale.diameter * 0.01))
        )
        coords = peak_local_max(
            dist, min_distance=min_distance, labels=foreground > 0, exclude_border=False
        )
        markers = np.zeros(ctx.gray.shape, dtype=np.int32)
        for idx, (y, x) in enumerate(coords, start=1):
            markers[y, x] = idx

        # watershed_line=True leaves 0-valued ridges between regions so touching
        # cells stay separated when re-labeled by filter_components.
        labels = watershed(-dist, markers, mask=foreground > 0, watershed_line=True)
        candidate: Mask = (labels > 0).astype(np.uint8) * 255

        darkness: GrayImage = (255 - ctx.gray).astype(np.uint8)
        accepted, holes, stats = filter_components(
            candidate, darkness, ctx.scale,
            min_circularity=cfg.min_circularity, max_area_frac=cfg.max_area_frac,
        )
        result_stats: dict[str, int | float | str] = {
            "n_markers": int(coords.shape[0]),
            "min_distance": int(min_distance),
            **stats,
        }
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )

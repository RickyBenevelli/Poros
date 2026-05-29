"""No-GT metrics: descriptive (count, porosity, mean area) and validity proxies (contrast, edge alignment)."""

from __future__ import annotations

import cv2
import numpy as np

from .._types import DetectionContext, DetectionResult
from ._base import BaseMetric


class CountMetric(BaseMetric):
    """Number of detected cells. Descriptive only: a fragmenting detector can
    inflate this without detecting anything real."""

    def score(self, result: DetectionResult, ctx: DetectionContext) -> float:
        return float(len(result.holes))


class PorosityMetric(BaseMetric):
    """Cell-area fraction of the slice (crumb porosity).

    Comparable across methods: wildly different porosities flag a detector that
    is over- or under-segmenting.
    """

    def score(self, result: DetectionResult, ctx: DetectionContext) -> float:
        slice_area = float(np.count_nonzero(ctx.slice_mask))
        return float(np.count_nonzero(result.mask)) / slice_area if slice_area else 0.0


class MeanCellAreaMetric(BaseMetric):
    """Mean detected cell area in pixels. Descriptive."""

    def score(self, result: DetectionResult, ctx: DetectionContext) -> float:
        if not result.holes:
            return 0.0
        return float(np.mean([h.area for h in result.holes]))


class MeanContrastMetric(BaseMetric):
    """Mean darkness of detected cells vs. their surround (precision proxy).

    Compares mean gray inside cells with the mean of a thin ring just outside.
    Real pores are clearly darker than the surrounding crumb; a detector firing
    on flat texture scores near zero.
    """

    def score(self, result: DetectionResult, ctx: DetectionContext) -> float:
        cells = result.mask > 0
        if not cells.any():
            return 0.0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ring = (cv2.dilate(result.mask, k) > 0) & ~cells & (ctx.slice_mask > 0)
        if not ring.any():
            return 0.0
        gray = ctx.gray.astype(np.float64)
        return float(gray[ring].mean() - gray[cells].mean())


class EdgeAlignmentMetric(BaseMetric):
    """How well cell boundaries sit on real image edges (validity proxy).

    Mean gradient magnitude along detected contours normalised by the mean
    gradient over the slice interior. Values > 1 mean boundaries follow genuine
    intensity edges rather than cutting through flat crumb.
    """

    def score(self, result: DetectionResult, ctx: DetectionContext) -> float:
        gray = ctx.gray.astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        contour = (cv2.morphologyEx(result.mask, cv2.MORPH_GRADIENT, k) > 0)
        inside = ctx.slice_mask > 0
        if not contour.any() or not inside.any():
            return 0.0
        ref = float(grad[inside].mean())
        return float(grad[contour].mean()) / ref if ref > 0 else 0.0

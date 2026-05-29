"""Otsu-based slice segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from .._types import BGRImage, Mask
from ._base import BaseSegmenter


class OtsuSliceSegmenter(BaseSegmenter):
    """Otsu threshold → largest component → flood-fill internal holes.

    The flood-fill is essential: bread cells are dark, so Otsu marks them as
    background. Without it, downstream detectors would see "holes in the mask"
    where the real cells are and skip them entirely.
    """

    def segment(self, bgr: BGRImage) -> Mask:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2 stub types the 2nd positional (connectivity) as a labels array.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)  # ty: ignore[no-matching-overload]
        if n_labels <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        slice_mask = (labels == largest).astype(np.uint8) * 255

        h, w = slice_mask.shape
        flood = slice_mask.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        holes_inside = cv2.bitwise_not(flood)
        return cv2.bitwise_or(slice_mask, holes_inside)

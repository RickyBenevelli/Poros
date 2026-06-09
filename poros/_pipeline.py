"""Build the shared DetectionContext (segment -> scale)."""

from __future__ import annotations

import cv2

from ._geometry import compute_scale
from ._types import BGRImage, DetectionContext
from .segmentation import BaseSegmenter, OtsuSliceSegmenter


def build_context(
    bgr: BGRImage,
    *,
    image_name: str | None = None,
    segmenter: BaseSegmenter | None = None,
    kernel_size: int | None = None,
    min_area: int | None = None,
    boundary_erosion: int | None = None,
) -> DetectionContext:
    """Segment the slice, convert to grayscale, and derive scale parameters.

    Computed once so all detectors share an identical view of the image.
    """
    segmenter = segmenter or OtsuSliceSegmenter()
    slice_mask = segmenter.segment(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = compute_scale(
        slice_mask,
        kernel_size=kernel_size,
        min_area=min_area,
        boundary_erosion=boundary_erosion,
    )
    return DetectionContext(
        bgr=bgr,
        gray=gray,
        slice_mask=slice_mask,
        scale=scale,
        image_name=image_name or "",
    )

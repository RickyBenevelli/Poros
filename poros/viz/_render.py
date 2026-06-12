"""Overlay rendering and multi-detector comparison grids."""

from __future__ import annotations

import cv2
import numpy as np

from .._types import BGRImage, DetectionResult, Mask

_SLICE_COLOR = (255, 255, 0)
_HOLE_COLOR = (0, 0, 255)
_CENTROID_COLOR = (0, 255, 255)
_HEADER_H = 34


def render(
    bgr: BGRImage,
    slice_mask: Mask,
    result: DetectionResult,
    *,
    draw_banner: bool = True,
) -> BGRImage:
    """Draw slice outline, cell contours, and centroids over the image.

    ``draw_banner=False`` skips the corner label — used by the grid, which
    adds its own un-clippable header strip.
    """
    vis: BGRImage = bgr.copy()
    slice_cnts, _ = cv2.findContours(
        slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, slice_cnts, -1, _SLICE_COLOR, 1)
    hole_cnts, _ = cv2.findContours(
        result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, hole_cnts, -1, _HOLE_COLOR, 1)
    for h in result.holes:
        cv2.circle(vis, (round(h.cx), round(h.cy)), 1, _CENTROID_COLOR, -1)

    if draw_banner:
        banner = f"{result.name}: {len(result.holes)} cells"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.rectangle(vis, (15, 15), (35 + tw, 35 + th + 10), (0, 0, 0), -1)
        cv2.putText(
            vis,
            banner,
            (25, 25 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return vis


def compare_grid(
    bgr: BGRImage, slice_mask: Mask, results: list[DetectionResult], cols: int = 3
) -> BGRImage:
    """Tile one labeled overlay per detector into a single comparison image.

    Each panel is cropped to the slice bounding box so the bread fills the tile.
    """
    x0, y0, x1, y1 = _slice_bbox(slice_mask, margin=0.02)
    panels = [
        (
            f"{r.name}: {len(r.holes)} cells",
            render(bgr, slice_mask, r, draw_banner=False)[y0:y1, x0:x1],
        )
        for r in results
    ]
    return _tile(panels, cols=cols)


def _slice_bbox(slice_mask: Mask, margin: float) -> tuple[int, int, int, int]:
    h, w = slice_mask.shape[:2]
    ys, xs = np.nonzero(slice_mask > 0)
    if ys.size == 0:
        return 0, 0, w, h
    pad = int(margin * max(h, w))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    y1 = min(h, int(ys.max()) + pad)
    return x0, y0, x1, y1


def _tile(panels: list[tuple[str, BGRImage]], cols: int) -> BGRImage:
    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    h, w = panels[0][1].shape[:2]
    pw = 480
    ph = int(480 * h / w)
    rows = (len(panels) + cols - 1) // cols
    cell_h = ph + _HEADER_H
    grid: BGRImage = np.zeros((cell_h * rows, pw * cols, 3), dtype=np.uint8)
    for i, (label, panel) in enumerate(panels):
        r, c = divmod(i, cols)
        y0, x0 = r * cell_h, c * pw
        cv2.rectangle(grid, (x0, y0), (x0 + pw, y0 + _HEADER_H), (40, 40, 40), -1)
        cv2.putText(
            grid,
            label,
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        grid[y0 + _HEADER_H : y0 + cell_h, x0 : x0 + pw] = cv2.resize(panel, (pw, ph))
    return grid

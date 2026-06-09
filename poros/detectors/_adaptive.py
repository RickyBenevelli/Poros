"""Adaptive-threshold detector (Sauvola / Niblack / local mean)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Literal

import cv2
import numpy as np
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola

from .._geometry import odd
from .._types import DetectionContext, DetectionResult, DetectorName, GrayImage
from ._base import BaseDetector
from ._common import erode_interior, filter_components, split_touching_blobs

AdaptiveMethod = Literal["sauvola", "niblack", "local"]
"""Which local thresholding rule to use for the dark-cell binarization."""


def _local_threshold(
    gray: GrayImage, method: AdaptiveMethod, window: int, k: float
) -> np.ndarray:
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
    out_dir: str, steps: list[tuple[str, np.ndarray]], cfg: AdaptiveConfig
) -> None:
    out_dir = f"{out_dir}/{cfg.window_size}_{cfg.k}_{cfg.max_area_frac}_{cfg.min_circularity}_{cfg.split_touching}"
    os.makedirs(
        out_dir,
        exist_ok=True,
    )
    for index, (name, image) in enumerate(steps, start=1):
        path = os.path.join(out_dir, f"{index:02d}_{name}.png")
        cv2.imwrite(path, _as_u8(image))


@dataclass
class AdaptiveConfig:
    """Tuning for :class:`AdaptiveThresholdDetector`.

    ``window_size=None`` derives a window ~5% of the slice diameter. ``k`` is
    the Sauvola/Niblack sensitivity (ignored by the ``local`` method).
    """

    method: AdaptiveMethod = "sauvola"
    window_size: int | None = None
    k: float = 0.1
    max_area_frac: float = 0.30
    min_circularity: float = 0.0
    split_touching: bool = True
    save_steps_dir: str | None = "adaptive_steps"
    """Optional directory for intermediate debug PNGs.

    The same behavior can be enabled from the CLI with
    ``POROS_ADAPTIVE_STEPS_DIR=/path/to/dir``.
    """


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
        # window_size is the size in pixels of the local neighborhood to compute
        # each pixel's adaptive threshold; e.g. 31 -> 31x31 neighborhood

        window = (
            cfg.window_size
            if cfg.window_size is not None
            else odd(max(15, int(ctx.scale.diameter * 0.05)))
        )

        thr_surface = _local_threshold(ctx.gray, cfg.method, window, cfg.k)
        raw = ((ctx.gray.astype(np.float64) < thr_surface).astype(np.uint8)) * 255

        # morphological opening: erosion followed by dilation
        # removes small bright artifacts and smooths the edges of dark blobs
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_k)
        # Erode the slice interior mask to exclude detections near the boundary
        # then keep only candidates that are fully inside the eroded interior
        interior = erode_interior(ctx.slice_mask, ctx.scale.boundary_erosion)
        cleaned = cv2.bitwise_and(opened, interior)
        # the detector separate connected blobs that may represent multiple touching cells.
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
        save_steps_dir = f"{cfg.save_steps_dir}/" or os.environ.get(
            "POROS_ADAPTIVE_STEPS_DIR"
        )
        if save_steps_dir:
            _save_step_images(
                save_steps_dir,
                [
                    ("gray", ctx.gray),
                    ("threshold_surface", thr_surface),
                    ("raw_dark_pixels", raw),
                    ("opened", opened),
                    ("interior_mask", interior),
                    ("cleaned_inside_slice", cleaned),
                    ("candidate_split", candidate),
                    ("darkness_score", darkness),
                    ("accepted_mask", accepted),
                ],
                cfg,
            )
            result_stats["steps_dir"] = save_steps_dir
        return DetectionResult(
            name=self.name, mask=accepted, holes=holes, stats=result_stats
        )

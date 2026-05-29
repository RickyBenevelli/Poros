"""Majority-voting and centroid-merge fusion strategies."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .._types import DetectionContext, DetectionResult, FusionName, Hole, Mask
from ..detectors._common import filter_components
from ._base import BaseFusion


class MajorityVotingFusion(BaseFusion):
    """Keep pixels accepted by at least ``min_votes`` detectors.

    Each detector's mask casts one vote per pixel; vote count also serves as
    the per-cell confidence score. ``min_votes=None`` uses a strict majority.
    """

    def __init__(
        self,
        min_votes: int | None = None,
        min_circularity: float = 0.15,
        max_area_frac: float = 0.30,
    ) -> None:
        self._min_votes = min_votes
        self._min_circularity = min_circularity
        self._max_area_frac = max_area_frac

    def fuse(
        self, results: Sequence[DetectionResult], ctx: DetectionContext
    ) -> DetectionResult:
        if not results:
            empty: Mask = np.zeros(ctx.gray.shape, dtype=np.uint8)
            return DetectionResult(name=FusionName.VOTING, mask=empty)

        votes = np.zeros(ctx.gray.shape, dtype=np.uint16)
        for r in results:
            votes += (r.mask > 0).astype(np.uint16)

        n = len(results)
        min_votes = self._min_votes if self._min_votes is not None else n // 2 + 1
        consensus: Mask = ((votes >= min_votes).astype(np.uint8)) * 255

        accepted, holes, stats = filter_components(
            consensus,
            votes.astype(np.float64),
            ctx.scale,
            min_circularity=self._min_circularity,
            max_area_frac=self._max_area_frac,
        )

        result_stats: dict[str, int | float | str] = {
            "n_detectors": n,
            "min_votes": int(min_votes),
            **stats,
        }
        return DetectionResult(
            name=FusionName.VOTING, mask=accepted, holes=holes, stats=result_stats
        )


class CentroidMergeFusion(BaseFusion):
    """Union of all detectors' masks with nearby centroids collapsed into one.

    Cells whose centroids fall within ``merge_radius`` are deduplicated, keeping
    the largest. ``merge_radius=None`` derives a scale-aware default (~1.5% of
    the slice diameter).
    """

    def __init__(self, merge_radius: float | None = None) -> None:
        self._merge_radius = merge_radius

    def fuse(
        self, results: Sequence[DetectionResult], ctx: DetectionContext
    ) -> DetectionResult:
        union: Mask = np.zeros(ctx.gray.shape, dtype=np.uint8)
        for r in results:
            union = cv2.bitwise_or(union, r.mask)

        radius = (
            self._merge_radius
            if self._merge_radius is not None
            else max(3.0, ctx.scale.diameter * 0.015)
        )
        pooled = [h for r in results for h in r.holes]
        # Greedy merge: keep larger blobs first, drop any later blob whose
        # centroid is within `radius` of an already-kept one.
        pooled.sort(key=lambda h: h.area, reverse=True)
        kept: list[Hole] = []
        for hole in pooled:
            if any(
                (hole.cx - k.cx) ** 2 + (hole.cy - k.cy) ** 2 <= radius**2 for k in kept
            ):
                continue
            kept.append(hole)
        merged = [
            Hole(
                id=i + 1,
                cx=h.cx,
                cy=h.cy,
                area=h.area,
                circularity=h.circularity,
                mean_depth=h.mean_depth,
                bbox_x=h.bbox_x,
                bbox_y=h.bbox_y,
                bbox_w=h.bbox_w,
                bbox_h=h.bbox_h,
            )
            for i, h in enumerate(kept)
        ]

        result_stats: dict[str, int | float | str] = {
            "n_detectors": len(results),
            "pooled_holes": len(pooled),
            "merge_radius": round(float(radius), 1),
            "kept_components": len(merged),
        }
        return DetectionResult(
            name=FusionName.CENTROID, mask=union, holes=merged, stats=result_stats
        )

"""Inter-method agreement and a no-GT metrics table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._types import DetectionContext, DetectionResult
from ._metrics import (
    CountMetric,
    EdgeAlignmentMetric,
    MeanCellAreaMetric,
    MeanContrastMetric,
    PorosityMetric,
)


def centroid_agreement(
    results: Sequence[DetectionResult], ctx: DetectionContext, radius: float | None = None
) -> dict[str, float]:
    """Fraction of each method's centroids corroborated by at least one other method.

    Agreement across independent methods is a proxy for correctness (analogous to
    inter-annotator agreement): a detection placed at the same spot by several
    methods is far more likely to correspond to a real cell.
    """
    r = radius if radius is not None else max(3.0, ctx.scale.diameter * 0.015)
    centroids = {
        res.name: np.array([[h.cx, h.cy] for h in res.holes], dtype=np.float64)
        for res in results
    }
    out: dict[str, float] = {}
    for res in results:
        mine = centroids[res.name]
        others = [c for name, c in centroids.items() if name != res.name and len(c)]
        if len(mine) == 0 or not others:
            out[res.name] = 0.0
            continue
        pool = np.vstack(others)
        matched = sum(
            1 for p in mine if ((pool - p) ** 2).sum(axis=1).min() <= r * r
        )
        out[res.name] = matched / len(mine)
    return out


def metrics_table(
    results: Sequence[DetectionResult], ctx: DetectionContext
) -> list[dict[str, float | str]]:
    """Return one row per method: count, porosity, mean_area, contrast,
    edge_align, and centroid agreement."""
    metrics = {
        "count": CountMetric(),
        "porosity": PorosityMetric(),
        "mean_area": MeanCellAreaMetric(),
        "contrast": MeanContrastMetric(),
        "edge_align": EdgeAlignmentMetric(),
    }
    agreement = centroid_agreement(results, ctx)
    rows: list[dict[str, float | str]] = []
    for res in results:
        row: dict[str, float | str] = {"method": res.name}
        for col, metric in metrics.items():
            row[col] = round(metric.score(res, ctx), 3)
        row["agreement"] = round(agreement[res.name], 3)
        rows.append(row)
    return rows

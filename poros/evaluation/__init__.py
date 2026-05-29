"""Evaluation — score detection quality without ground truth.

├── _base.py     BaseMetric(ABC)
├── _metrics.py  CountMetric, PorosityMetric, MeanCellAreaMetric,
│                MeanContrastMetric, EdgeAlignmentMetric
└── _report.py   centroid_agreement + metrics_table
"""

from __future__ import annotations

from ._base import BaseMetric
from ._metrics import (
    CountMetric,
    EdgeAlignmentMetric,
    MeanCellAreaMetric,
    MeanContrastMetric,
    PorosityMetric,
)
from ._report import centroid_agreement, metrics_table

__all__ = [
    "BaseMetric",
    "CountMetric",
    "EdgeAlignmentMetric",
    "MeanCellAreaMetric",
    "MeanContrastMetric",
    "PorosityMetric",
    "centroid_agreement",
    "metrics_table",
]

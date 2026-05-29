"""Metric contract.

No-GT metrics fall in two families:
- **descriptive** (count, porosity, mean size): summarise what was detected.
- **validity proxies** (contrast, edge alignment, agreement): estimate whether
  detections correspond to real cells without needing ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .._types import DetectionContext, DetectionResult


class BaseMetric(ABC):
    @abstractmethod
    def score(self, result: DetectionResult, ctx: DetectionContext) -> float:
        """Return a scalar score for ``result``."""
        ...

"""Detector contract + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .._types import DetectionContext, DetectionResult, DetectorName


class BaseDetector(ABC):
    """Base class for all cell detectors.

    Subclasses declare ``name`` (their registry key) and implement ``detect``.
    """

    name: ClassVar[DetectorName]

    @abstractmethod
    def detect(self, ctx: DetectionContext) -> DetectionResult: ...


def make_detector(name: DetectorName) -> BaseDetector:
    """Instantiate the detector registered under ``name`` with its defaults."""
    return _registry()[name]()


def available_detectors() -> list[DetectorName]:
    """Return every registered detector name (stable order)."""
    return list(_registry().keys())


def _registry() -> dict[DetectorName, type[BaseDetector]]:
    from ._adaptive import AdaptiveThresholdDetector
    from ._bottomhat import BottomHatDetector
    from ._gaussian import GaussianSplatDetector
    from ._log import LogBlobDetector
    from ._ridge import RidgeDetector
    from ._watershed import WatershedDetector

    return {
        DetectorName.BOTTOMHAT: BottomHatDetector,
        DetectorName.LOG: LogBlobDetector,
        DetectorName.RIDGE: RidgeDetector,
        DetectorName.ADAPTIVE: AdaptiveThresholdDetector,
        DetectorName.WATERSHED: WatershedDetector,
        DetectorName.GAUSSIAN: GaussianSplatDetector,
    }

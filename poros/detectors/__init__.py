"""Detectors — interchangeable cell-detection strategies.

├── _base.py       BaseDetector(ABC) + registry
├── _common.py     erode_interior, split_touching_blobs, filter_components
├── _bottomhat.py  BottomHatDetector    (morphological black-hat)
├── _log.py        LogBlobDetector      (multi-scale LoG / SimpleBlobDetector)
├── _ridge.py      RidgeDetector        (vesselness walls → watershed cells)
├── _adaptive.py   AdaptiveThresholdDetector  (Sauvola / Niblack / local)
├── _watershed.py  WatershedDetector    (marker-controlled flooding)
└── _gaussian.py   GaussianSplatDetector      (matching-pursuit 2D Gaussians)
"""

from __future__ import annotations

from ._adaptive import AdaptiveConfig, AdaptiveThresholdDetector
from ._base import BaseDetector, available_detectors, make_detector
from ._bottomhat import BottomHatConfig, BottomHatDetector
from ._gaussian import GaussianSplatConfig, GaussianSplatDetector
from ._log import LogBlobDetector, LogConfig
from ._ridge import RidgeConfig, RidgeDetector
from ._watershed import WatershedConfig, WatershedDetector

__all__ = [
    "AdaptiveConfig",
    "AdaptiveThresholdDetector",
    "BaseDetector",
    "BottomHatConfig",
    "BottomHatDetector",
    "GaussianSplatConfig",
    "GaussianSplatDetector",
    "LogBlobDetector",
    "LogConfig",
    "RidgeConfig",
    "RidgeDetector",
    "WatershedConfig",
    "WatershedDetector",
    "available_detectors",
    "make_detector",
]

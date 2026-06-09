"""Core data types shared across the whole package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

import numpy.typing as npt

# Dtype is left open (Any) so these stay compatible with cv2's loosely-typed
# MatLike returns without scattering casts at every cv2 call site.
type BGRImage = npt.NDArray[Any]
"""Colour image, shape (H, W, 3), uint8, BGR channel order."""

type GrayImage = npt.NDArray[Any]
"""Single-channel image, shape (H, W), uint8."""

type Mask = npt.NDArray[Any]
"""Binary mask, shape (H, W), uint8, values in {0, 255}."""

type FloatImage = npt.NDArray[Any]
"""Floating-point response map, shape (H, W)."""


class DetectorName(StrEnum):
    BOTTOMHAT = auto()
    LOG = auto()
    RIDGE = auto()
    ADAPTIVE = auto()
    WATERSHED = auto()
    GAUSSIAN = auto()


class FusionName(StrEnum):
    VOTING = auto()
    CENTROID = auto()


@dataclass
class Hole:
    """A single detected cell (pore / cavity).

    ``circularity`` = 4π·area / perimeter² in [0, 1].
    ``mean_depth`` is a per-detector confidence proxy (bottom-hat response,
    local contrast, or vesselness — depending on which detector found the cell).
    ``angle`` is the fitted-ellipse orientation in degrees (0 when not fitted).
    """

    id: int
    cx: float
    cy: float
    area: int
    circularity: float
    mean_depth: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    angle: float = 0.0


@dataclass
class ScaleParams:
    """Resolution-aware parameters derived from the segmented slice.

    Computed once by :func:`poros._geometry.compute_scale` and shared by
    every detector so they all adapt to image resolution without re-deriving the
    slice geometry.
    """

    slice_area: int
    diameter: float
    kernel_size: int
    min_area: int
    boundary_erosion: int


@dataclass
class DetectionContext:
    """Everything a detector needs, computed once and shared by all detectors.

    ``gray`` is the grayscale conversion of ``bgr`` — detectors must not use
    the raw BGR image for analysis.
    ``image_name`` is optional source metadata for diagnostics and debug output.
    """

    bgr: BGRImage
    gray: GrayImage
    slice_mask: Mask
    scale: ScaleParams
    image_name: str = ""


@dataclass
class DetectionResult:
    """Uniform output of every detector and fusion strategy.

    The common shape makes detectors directly comparable in montages and
    interchangeable as input to the fusion strategies.
    """

    name: str
    mask: Mask
    holes: list[Hole] = field(default_factory=list)
    stats: dict[str, int | float | str] = field(default_factory=dict)

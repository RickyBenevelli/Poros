"""Scale estimation helpers — kernel size, min area, and crust margin all
auto-scale from the slice diameter so the pipeline adapts to different
image resolutions without manual tuning."""

from __future__ import annotations

import numpy as np

from ._types import Mask, ScaleParams


def odd(n: int) -> int:
    """Return ``n`` rounded up to the nearest odd integer (>= 1)."""
    n = max(1, n)
    return n if n % 2 == 1 else n + 1


def slice_diameter(slice_mask: Mask) -> float:
    """Return the equivalent circular diameter of the slice, in pixels."""
    area = float(np.count_nonzero(slice_mask))
    return 2.0 * np.sqrt(area / np.pi)


def compute_scale(
    slice_mask: Mask,
    *,
    kernel_size: int | None = None,
    min_area: int | None = None,
    boundary_erosion: int | None = None,
) -> ScaleParams:
    """Derive resolution-aware parameters from the slice mask.

    Any explicit override is used as-is; unset values are derived from the
    slice geometry.
    """
    slice_area = int(np.count_nonzero(slice_mask))
    diameter = slice_diameter(slice_mask)

    ks = kernel_size if kernel_size is not None else odd(int(diameter / 7))
    ma = min_area if min_area is not None else max(8, int(slice_area * 0.00003))
    be = (
        boundary_erosion
        if boundary_erosion is not None
        else max(8, int(diameter * 0.025))
    )
    return ScaleParams(
        slice_area=slice_area,
        diameter=diameter,
        kernel_size=odd(ks),
        min_area=ma,
        boundary_erosion=be,
    )

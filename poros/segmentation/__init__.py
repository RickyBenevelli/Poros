"""Segmentation — isolate the bread slice from the background.

├── _base.py   BaseSegmenter(ABC)
└── _otsu.py   OtsuSliceSegmenter  (Otsu + largest component + hole fill)
"""

from __future__ import annotations

from ._base import BaseSegmenter
from ._otsu import OtsuSliceSegmenter

__all__ = ["BaseSegmenter", "OtsuSliceSegmenter"]

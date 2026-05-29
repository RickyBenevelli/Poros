"""Segmenter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .._types import BGRImage, Mask


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, bgr: BGRImage) -> Mask:
        """Return a binary mask (values {0, 255}) isolating the slice."""
        ...

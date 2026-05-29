"""Fusion contract + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .._types import DetectionContext, DetectionResult, FusionName


class BaseFusion(ABC):
    @abstractmethod
    def fuse(
        self, results: Sequence[DetectionResult], ctx: DetectionContext
    ) -> DetectionResult:
        """Combine ``results`` into a single consensus :class:`DetectionResult`."""
        ...


def make_fusion(name: FusionName) -> BaseFusion:
    """Instantiate the fusion strategy registered under ``name``."""
    from ._voting import CentroidMergeFusion, MajorityVotingFusion

    registry: dict[FusionName, type[BaseFusion]] = {
        FusionName.VOTING: MajorityVotingFusion,
        FusionName.CENTROID: CentroidMergeFusion,
    }
    return registry[name]()

"""Fusion — combine multiple detector results into one consensus.

├── _base.py    BaseFusion(ABC) + make_fusion
└── _voting.py  MajorityVotingFusion  (pixel votes) + CentroidMergeFusion  (centroid dedup)
"""

from __future__ import annotations

from ._base import BaseFusion, make_fusion
from ._voting import CentroidMergeFusion, MajorityVotingFusion

__all__ = [
    "BaseFusion",
    "CentroidMergeFusion",
    "MajorityVotingFusion",
    "make_fusion",
]

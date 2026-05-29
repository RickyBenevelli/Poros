"""Visualization — overlays, comparison grids, and the interactive viewer.

├── _render.py  render (single overlay) + compare_grid (labeled detector grid)
└── _viewer.py  export_gallery / export_viewer (self-contained HTML gallery)
"""

from __future__ import annotations

from ._render import compare_grid, render
from ._viewer import GalleryItem, export_gallery, export_viewer

__all__ = ["GalleryItem", "compare_grid", "export_gallery", "export_viewer", "render"]

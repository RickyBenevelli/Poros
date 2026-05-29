"""Build a combined interactive gallery over many images.

Scans a root folder of category subfolders (e.g. ``data/RGB2/<category>/``),
runs every detector on the first image of each, and writes one self-contained
HTML gallery (see :func:`poros.viz.export_gallery`).

    uv run python -m poros.gallery                       # data/RGB2 -> results/gallery
    uv run python -m poros.gallery <root> <out_dir>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

import cv2

from . import available_detectors, build_context, make_detector
from .viz import GalleryItem, export_gallery

log = logging.getLogger("poros.gallery")

_IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _first_image(folder: str) -> str | None:
    """Return the first image file in ``folder`` (alphabetical), or None."""
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(_IMAGE_EXTS))
    return os.path.join(folder, names[0]) if names else None


def run(argv: Sequence[str] | None = None) -> int:
    """Build the gallery from a root of category subfolders."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="data/RGB2", help="root with category subfolders")
    p.add_argument("out_dir", nargs="?", default="results/gallery", help="output gallery folder")
    args = p.parse_args(argv)

    subdirs = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )

    items: list[GalleryItem] = []
    for name in subdirs:
        image_path = _first_image(os.path.join(args.root, name))
        if image_path is None:
            continue
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("could not read %s", image_path)
            continue
        ctx = build_context(bgr)
        results = [make_detector(d).detect(ctx) for d in available_detectors()]
        log.info("%s: %s -> %s", name, os.path.basename(image_path),
                 {r.name: len(r.holes) for r in results})
        items.append((name, bgr, ctx.slice_mask, results))

    if not items:
        log.error("no images found under %s", args.root)
        return 1

    index = export_gallery(items, args.out_dir, title="bread cells")
    log.info("wrote %s", index)
    return 0


if __name__ == "__main__":
    sys.exit(run())

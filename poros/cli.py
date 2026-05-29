from __future__ import annotations

import argparse
import csv
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict

import cv2

from . import (
    BGRImage,
    DetectionContext,
    DetectionResult,
    DetectorName,
    FusionName,
    Hole,
    available_detectors,
    build_context,
    make_detector,
    make_fusion,
)
from .evaluation import metrics_table
from .viz import compare_grid, export_viewer, render

log = logging.getLogger("poros")

_METHOD_ALL = "all"
_METHOD_FUSION = "fusion"
_WROTE = "wrote %s"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    detector_names = [d.value for d in available_detectors()]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", help="input image (.bmp / .png / .jpg / ...)")
    p.add_argument("-o", "--output", default=None, help="output image path/prefix")
    p.add_argument("--csv", default=None, help="write per-cell stats to this CSV file")
    p.add_argument(
        "--method",
        default=DetectorName.BOTTOMHAT.value,
        choices=[*detector_names, _METHOD_ALL, _METHOD_FUSION],
        help="detector to run, 'all' for every detector, or 'fusion' for a consensus",
    )
    p.add_argument(
        "--fusion",
        default=FusionName.VOTING.value,
        choices=[f.value for f in FusionName],
        help="fusion strategy used by --method fusion",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="also save a grid with every detector's overlay side by side",
    )
    p.add_argument(
        "--viewer",
        action="store_true",
        help="export an interactive HTML viewer (synced pan/zoom, overlay toggles) "
        "with every detector under <output>_viewer/",
    )
    p.add_argument(
        "--metrics",
        action="store_true",
        help="print a no-ground-truth metrics table comparing every detector",
    )
    p.add_argument("--kernel-size", type=int, default=None, help="morphology kernel (odd)")
    p.add_argument("--min-area", type=int, default=None, help="minimum cell area (px)")
    p.add_argument(
        "--boundary-erosion", type=int, default=None, help="crust margin to ignore (px)"
    )
    return p.parse_args(argv)


def _run_detectors(
    ctx: DetectionContext, names: Sequence[DetectorName]
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for name in names:
        result = make_detector(name).detect(ctx)
        log.info("%s: %s", name.value, result.stats)
        results.append(result)
    return results


def _write_csv(path: str, holes: list[Hole]) -> None:
    fieldnames = list(asdict(Hole(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in holes:
            writer.writerow(asdict(h))
    log.info(_WROTE, path)


def _save_image(path: str, image: BGRImage) -> None:
    cv2.imwrite(path, image)
    log.info(_WROTE, path)


def _select_primary(
    args: argparse.Namespace, results: list[DetectionResult], ctx: DetectionContext
) -> DetectionResult | None:
    if args.method == _METHOD_FUSION:
        fused = make_fusion(FusionName(args.fusion)).fuse(results, ctx)
        log.info("fusion(%s): %s", args.fusion, fused.stats)
        return fused
    if args.method == _METHOD_ALL:
        return None
    return next(r for r in results if r.name == args.method)


def _emit_outputs(
    args: argparse.Namespace,
    ctx: DetectionContext,
    base: str,
    results: list[DetectionResult],
    primary: DetectionResult | None,
) -> None:
    bgr = ctx.bgr
    if primary is not None:
        out_path = args.output if args.output else f"{base}_{primary.name}.png"
        _save_image(out_path, render(bgr, ctx.slice_mask, primary))
        if args.csv:
            _write_csv(args.csv, primary.holes)

    is_fusion = args.method == _METHOD_FUSION and primary is not None
    # `primary` is one of `results` except for fusion (a new consensus entry).
    full_set = [*results, primary] if is_fusion else results

    if args.compare or args.method == _METHOD_ALL:
        _save_image(f"{base}_compare.png", compare_grid(bgr, ctx.slice_mask, full_set))

    if args.viewer:
        viewer_dir = f"{base}_viewer"
        html = export_viewer(
            bgr, ctx.slice_mask, full_set, viewer_dir, title=os.path.basename(base)
        )
        log.info("wrote %s", html)


def _print_metrics(results: list[DetectionResult], ctx: DetectionContext) -> None:
    rows = metrics_table(results, ctx)
    cols = ["method", "count", "porosity", "mean_area", "contrast", "edge_align", "agreement"]
    widths = {c: max(len(c), *(len(f"{row[c]}") for row in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    log.info("no-GT metrics (contrast/edge_align/agreement = validity proxies):")
    log.info(header)
    for row in rows:
        log.info("  ".join(f"{row[c]}".ljust(widths[c]) for c in cols))


def run(argv: Sequence[str] | None = None) -> int:
    """Entry point: load image, run detection/fusion, write outputs."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        log.error("could not read %s", args.input)
        return 1

    base = os.path.splitext(args.output or os.path.splitext(args.input)[0] + "_holes")[0]

    log.info("loaded %s %dx%d", args.input, bgr.shape[1], bgr.shape[0])
    ctx = build_context(
        bgr,
        kernel_size=args.kernel_size,
        min_area=args.min_area,
        boundary_erosion=args.boundary_erosion,
    )
    log.info(
        "slice area %d px | diameter %.0f px | kernel %d | min_area %d | erosion %d",
        ctx.scale.slice_area, ctx.scale.diameter, ctx.scale.kernel_size,
        ctx.scale.min_area, ctx.scale.boundary_erosion,
    )

    run_all = (
        args.method in (_METHOD_ALL, _METHOD_FUSION)
        or args.compare or args.viewer or args.metrics
    )
    names = available_detectors() if run_all else [DetectorName(args.method)]
    results = _run_detectors(ctx, names)

    primary = _select_primary(args, results, ctx)
    _emit_outputs(args, ctx, base, results, primary)
    if args.metrics:
        _print_metrics(results, ctx)
    return 0

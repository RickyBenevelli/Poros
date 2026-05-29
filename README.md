# Poros

Bread-crumb porosity analysis from a single cross-section image. Six classical CV detectors, same interface, fuseable into a consensus. No training data required.

## Install

```bash
uv sync
```

Python >= 3.13. Deps: OpenCV, NumPy, scikit-image, SciPy.

## Usage

```bash
uv run python main.py slice.bmp
uv run python main.py slice.bmp --method ridge
uv run python main.py slice.bmp --method all --compare
uv run python main.py slice.bmp --method fusion --fusion voting
uv run python main.py slice.bmp --metrics
uv run python main.py slice.bmp --viewer
uv run python main.py slice.bmp --method ridge --csv cells.csv
```

## Detectors

| Method        | Strategy                                                   |
| ------------- | ---------------------------------------------------------- |
| `bottomhat` | Morphological black-hat + Otsu threshold + watershed split |
| `log`       | Multi-scale dark-blob detection, ellipse fit per blob      |
| `ridge`     | Frangi/Sato/Meijering vesselness on the wall network       |
| `adaptive`  | Sauvola / Niblack / local-mean adaptive thresholding       |
| `watershed` | Marker-controlled watershed with adaptive foreground       |
| `gaussian`  | Gaussian-splat / DoG blob detector                         |

## Metrics

`--metrics` prints per-detector stats without ground truth. `count` and `porosity` describe what was found; `contrast`, `edge_align`, and `agreement` are the validity proxies to compare methods on. `IoUMetric` and `F1Metric` are ready for supervised evaluation when GT masks are available.

## Gallery

```bash
uv run python -m poros.gallery              # data/RGB2/ -> results/gallery/index.html
uv run python -m poros.gallery <root> <out>
```

Self-contained HTML viewer: synchronized pan/zoom, per-method show/hide, overlay styles, opacity. `main.py --viewer` does the same for a single image.

## Structure

```
poros/
├── _types.py        # Hole, DetectionContext, DetectionResult, enums
├── _geometry.py     # ScaleParams, compute_scale
├── _pipeline.py     # build_context
├── cli.py
├── gallery.py
├── segmentation/
├── detectors/
├── fusion/
├── evaluation/
└── viz/
```

## API

```python
import cv2
from poros import build_context, make_detector, DetectorName

bgr = cv2.imread("slice.bmp")
ctx = build_context(bgr)
result = make_detector(DetectorName.BOTTOMHAT).detect(ctx)
print(len(result.holes), result.stats["porosity"])
```

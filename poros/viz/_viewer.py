"""Export a self-contained interactive HTML gallery for qualitative analysis"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .._types import BGRImage, DetectionResult, Mask
from ._render import _slice_bbox

# (B, G, R, A) — OpenCV channel order; cv2.imwrite emits correct RGBA PNGs.
_SLICE_COLOR = (255, 255, 0, 255)
_CONTOUR_COLOR = (0, 0, 255, 255)
_CENTROID_COLOR = (0, 255, 255, 255)
_FILL_COLOR = (60, 220, 40, 120)

GalleryItem = tuple[str, BGRImage, Mask, list[DetectionResult]]
"""``(category_name, bgr, slice_mask, results)`` for one image."""


def _contour_overlay(slice_crop: Mask, mask_crop: Mask, holes_xy: list[tuple[int, int]]) -> np.ndarray:
    h, w = slice_crop.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    slice_cnts, _ = cv2.findContours(slice_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, slice_cnts, -1, _SLICE_COLOR, 2)
    hole_cnts, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, hole_cnts, -1, _CONTOUR_COLOR, 2)
    for x, y in holes_xy:
        cv2.circle(overlay, (x, y), 2, _CENTROID_COLOR, -1)
    return overlay


def _fill_overlay(mask_crop: Mask) -> np.ndarray:
    h, w = mask_crop.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask_crop > 0] = _FILL_COLOR
    return overlay


def export_gallery(items: list[GalleryItem], out_dir: str, *, title: str = "bread cells") -> str:
    """Write per-category originals + overlays and a combined ``index.html``.

    Returns the path to the generated ``index.html``.
    """
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    categories: list[dict[str, object]] = []
    method_names: list[str] = []
    for name, bgr, slice_mask, results in items:
        x0, y0, x1, y1 = _slice_bbox(slice_mask, margin=0.02)
        slug = "".join(c if c.isalnum() else "_" for c in name)
        cv2.imwrite(os.path.join(data_dir, f"{slug}__original.png"), bgr[y0:y1, x0:x1])
        slice_crop = slice_mask[y0:y1, x0:x1]

        methods: list[dict[str, object]] = []
        for r in results:
            mask_crop = r.mask[y0:y1, x0:x1]
            holes_xy = [(round(h.cx) - x0, round(h.cy) - y0) for h in r.holes]
            cv2.imwrite(
                os.path.join(data_dir, f"{slug}__{r.name}__contour.png"),
                _contour_overlay(slice_crop, mask_crop, holes_xy),
            )
            cv2.imwrite(
                os.path.join(data_dir, f"{slug}__{r.name}__fill.png"), _fill_overlay(mask_crop)
            )
            methods.append({
                "name": r.name,
                "count": len(r.holes),
                "contour": f"data/{slug}__{r.name}__contour.png",
                "fill": f"data/{slug}__{r.name}__fill.png",
            })
            if r.name not in method_names:
                method_names.append(r.name)
        categories.append({
            "name": name,
            "original": f"data/{slug}__original.png",
            "methods": methods,
        })

    html = (
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__DATA__", json.dumps(categories))
        .replace("__METHODS__", json.dumps(method_names))
    )
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(html)
    return index_path


def export_viewer(
    bgr: BGRImage, slice_mask: Mask, results: list[DetectionResult], out_dir: str,
    *, title: str = "bread cells",
) -> str:
    """Single-image gallery (one category). See :func:`export_gallery`."""
    return export_gallery([(title, bgr, slice_mask, results)], out_dir, title=title)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; --bg:#0e0f12; --panel:#16181d; --line:#262a31;
          --fg:#e7e9ee; --mut:#8b92a0; --accent:#5b8cff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { position:sticky; top:0; z-index:10; background:rgba(14,15,18,.92);
           backdrop-filter:blur(6px); border-bottom:1px solid var(--line); padding:10px 16px;
           display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:13px; font-weight:600; letter-spacing:.3px; margin:0; color:var(--mut); text-transform:uppercase; }
  .group { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .lbl { font-size:12px; color:var(--mut); }
  .chip { font-size:12px; padding:5px 11px; border:1px solid var(--line); border-radius:999px;
          background:var(--panel); color:var(--fg); cursor:pointer; user-select:none; transition:.12s; }
  .chip:hover { border-color:#3a4150; }
  .chip.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  .chip.off { opacity:.4; text-decoration:line-through; }
  .seg { display:flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .seg button { background:var(--panel); color:var(--fg); border:0; padding:5px 12px; font-size:12px; cursor:pointer; }
  .seg button.on { background:var(--accent); color:#fff; }
  input[type=range] { accent-color:var(--accent); }
  .panels { display:grid; gap:8px; padding:8px;
            grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); }
  .panel { background:#000; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .panel.hidden { display:none; }
  .phead { display:flex; justify-content:space-between; align-items:center;
           padding:7px 11px; background:var(--panel); border-bottom:1px solid var(--line); }
  .pname { font-size:13px; font-weight:600; }
  .pcount { font-size:12px; color:var(--accent); }
  .viewport { position:relative; overflow:hidden; height:74vh; cursor:grab; touch-action:none; background:#000; }
  .viewport.grab { cursor:grabbing; }
  .content { position:absolute; top:0; left:0; transform-origin:0 0; will-change:transform; }
  .content img { position:absolute; top:0; left:0; display:block; image-rendering:pixelated; }
  .content img.ov.off { display:none; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="group" id="cats"></div>
  <div class="group"><span class="lbl">overlay</span>
    <div class="seg" id="ovToggle"><button data-v="1" class="on">on</button><button data-v="0">off</button></div>
    <div class="seg" id="style"><button data-v="contour" class="on">contorno</button><button data-v="fill">pieno</button></div>
    <input type="range" id="opacity" min="0" max="100" value="100" title="opacità">
    <button class="chip" id="reset">reset zoom</button>
  </div>
  <div class="group" id="methodChips"></div>
</header>
<div class="panels" id="panels"></div>
<script>
const CATEGORIES = __DATA__;
const METHODS = __METHODS__;
const LS = "poros.viewer";
const saved = JSON.parse(localStorage.getItem(LS) || "{}");

const state = {
  cat: 0,
  overlay: saved.overlay !== false,
  style: saved.style || "contour",
  opacity: saved.opacity ?? 100,
  hidden: new Set(saved.hidden || []),
  scale: 1, tx: 0, ty: 0, natW: 0, natH: 0,
};
const contents = [];

function persist() {
  localStorage.setItem(LS, JSON.stringify({
    overlay: state.overlay, style: state.style, opacity: state.opacity,
    hidden: [...state.hidden],
  }));
}

function applyTransform() {
  const t = `translate(${state.tx}px,${state.ty}px) scale(${state.scale})`;
  for (const c of contents) c.style.transform = t;
}
function fit() {
  const vp = document.querySelector(".viewport");
  if (!vp || !state.natW) return;
  const s = Math.min(vp.clientWidth / state.natW, vp.clientHeight / state.natH);
  state.scale = s;
  state.tx = (vp.clientWidth - state.natW * s) / 2;
  state.ty = (vp.clientHeight - state.natH * s) / 2;
  applyTransform();
}
function zoomAt(cx, cy, f) {
  const ns = Math.max(0.05, Math.min(40, state.scale * f));
  state.tx = cx - (cx - state.tx) * (ns / state.scale);
  state.ty = cy - (cy - state.ty) * (ns / state.scale);
  state.scale = ns; applyTransform();
}

function bindViewport(vp) {
  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }, { passive: false });
  let drag = false, lx = 0, ly = 0;
  vp.addEventListener("pointerdown", (e) => {
    drag = true; lx = e.clientX; ly = e.clientY; vp.classList.add("grab");
    vp.setPointerCapture(e.pointerId);
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    state.tx += e.clientX - lx; state.ty += e.clientY - ly;
    lx = e.clientX; ly = e.clientY; applyTransform();
  });
  const end = () => { drag = false; vp.classList.remove("grab"); };
  vp.addEventListener("pointerup", end);
  vp.addEventListener("pointercancel", end);
  vp.addEventListener("pointerleave", end);
}

function renderCategory() {
  const cat = CATEGORIES[state.cat];
  const panels = document.getElementById("panels");
  panels.innerHTML = "";
  contents.length = 0;
  let firstImg = null;
  for (const m of cat.methods) {
    const panel = document.createElement("div");
    panel.className = "panel" + (state.hidden.has(m.name) ? " hidden" : "");
    panel.innerHTML =
      `<div class="phead"><span class="pname">${m.name}</span>` +
      `<span class="pcount">${m.count} buchi</span></div>`;
    const vp = document.createElement("div"); vp.className = "viewport";
    const content = document.createElement("div"); content.className = "content";
    const base = document.createElement("img"); base.src = cat.original;
    const ov = document.createElement("img"); ov.className = "ov";
    ov.src = state.style === "fill" ? m.fill : m.contour;
    ov.style.opacity = state.opacity / 100;
    if (!state.overlay) ov.classList.add("off");
    content.append(base, ov); vp.append(content); panel.append(vp);
    panels.append(panel);
    contents.push(content);
    bindViewport(vp);
    if (!firstImg) { firstImg = base; }
  }
  if (firstImg) {
    firstImg.addEventListener("load", () => {
      state.natW = firstImg.naturalWidth; state.natH = firstImg.naturalHeight; fit();
    });
    if (firstImg.complete && firstImg.naturalWidth) {
      state.natW = firstImg.naturalWidth; state.natH = firstImg.naturalHeight; fit();
    }
  }
}

// --- header controls ---
const catsEl = document.getElementById("cats");
CATEGORIES.forEach((c, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " on" : "");
  b.textContent = c.name;
  b.onclick = () => {
    state.cat = i;
    [...catsEl.children].forEach((x, j) => x.classList.toggle("on", j === i));
    renderCategory();
  };
  catsEl.append(b);
});

const chipsEl = document.getElementById("methodChips");
METHODS.forEach((name) => {
  const b = document.createElement("button");
  b.className = "chip " + (state.hidden.has(name) ? "off" : "on");
  b.textContent = name;
  b.onclick = () => {
    if (state.hidden.has(name)) state.hidden.delete(name); else state.hidden.add(name);
    b.classList.toggle("on"); b.classList.toggle("off");
    persist(); renderCategory();
  };
  chipsEl.append(b);
});

function segSetup(id, key, apply) {
  const seg = document.getElementById(id);
  seg.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      seg.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
      btn.classList.add("on");
      state[key] = btn.dataset.v === "1" ? true : (btn.dataset.v === "0" ? false : btn.dataset.v);
      persist(); apply();
    };
    const v = btn.dataset.v === "1" ? true : (btn.dataset.v === "0" ? false : btn.dataset.v);
    btn.classList.toggle("on", state[key] === v);
  });
}
segSetup("ovToggle", "overlay", () =>
  document.querySelectorAll(".content img.ov").forEach((o) => o.classList.toggle("off", !state.overlay)));
segSetup("style", "style", renderCategory);

const op = document.getElementById("opacity");
op.value = state.opacity;
op.addEventListener("input", (e) => {
  state.opacity = +e.target.value; persist();
  document.querySelectorAll(".content img.ov").forEach((o) => o.style.opacity = state.opacity / 100);
});
document.getElementById("reset").onclick = fit;
window.addEventListener("resize", fit);

renderCategory();
</script>
</body>
</html>
"""

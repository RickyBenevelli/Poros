"""Shared building blocks: erode_interior, split_touching_blobs,
filter_components, and local_contrast."""

from __future__ import annotations

import cv2
import numpy as np

from .._types import FloatImage, GrayImage, Hole, Mask, ScaleParams


def local_contrast(gray: GrayImage, cx: int, cy: int, r: int) -> float:
    """Return mean(annulus) - mean(disk) — how much darker the blob is than its surround.

    Disk spans radius ``r``; annulus spans ``[r, 2r]``. Real pores are clearly
    darker than the crumb around them; texture blobs are not.
    """
    h, w = gray.shape
    r_out = max(r + 1, 2 * r)
    y0, y1 = max(0, cy - r_out), min(h, cy + r_out + 1)
    x0, x1 = max(0, cx - r_out), min(w, cx + r_out + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    ys = np.arange(y0, y1) - cy
    xs = np.arange(x0, x1) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist2 = yy**2 + xx**2
    disk = patch[dist2 <= r**2]
    annulus = patch[(dist2 > r**2) & (dist2 <= r_out**2)]
    mean_in = float(disk.mean()) if disk.size else 128.0
    mean_ann = float(annulus.mean()) if annulus.size else 128.0
    return mean_ann - mean_in


def erode_interior(slice_mask: Mask, boundary_erosion: int) -> Mask:
    e_size = boundary_erosion * 2 + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_size, e_size))
    return cv2.erode(slice_mask, se)


def split_touching_blobs(
    mask: Mask, min_peak_dist: float = 2.0, peak_neighborhood: int = 9
) -> Mask:
    """Split touching blobs using a distance transform + watershed.

    Peaks are local maxima within a ``peak_neighborhood``-pixel window (the
    ``cv2.dilate`` trick), so both large cavities and tiny pores get a seed —
    a global threshold would miss the small ones.
    """
    if cv2.countNonZero(mask) == 0:
        return mask.copy()

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask.copy()

    nb = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (peak_neighborhood, peak_neighborhood)
    )
    dist_dil = cv2.dilate(dist, nb)
    peaks = ((dist == dist_dil) & (dist > min_peak_dist)).astype(np.uint8) * 255

    n_peaks, peak_labels = cv2.connectedComponents(peaks)
    if n_peaks <= 2:
        return mask.copy()

    # 0 = unknown (watershed fills), 1 = background, 2..N+1 = peak seeds.
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[peak_labels > 0] = peak_labels[peak_labels > 0] + 1
    markers[mask == 0] = 1

    # cv2 stub requires a non-None dst; passing None is the documented API.
    dist_8u = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # ty: ignore[no-matching-overload]
    img3 = cv2.cvtColor(255 - dist_8u, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)

    out: Mask = np.zeros_like(mask)
    out[markers >= 2] = 255
    return out


def filter_components(
    candidate: Mask,
    score_map: FloatImage | GrayImage,
    scale: ScaleParams,
    *,
    min_circularity: float,
    max_area_frac: float,
    min_score: float = 0.0,
) -> tuple[Mask, list[Hole], dict[str, int]]:
    """Filter connected components by area, circularity, and per-blob score.

    Common back-end for all region-based detectors so accepted cells are
    produced consistently regardless of how the candidate mask was generated.
    Returns ``(accepted_mask, holes, stats)``; stats reports
    ``raw_components`` and ``kept_components``.
    """
    # cv2 stub types the 2nd positional (connectivity) as a labels array.
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(candidate, 8)  # ty: ignore[no-matching-overload]
    max_area = max_area_frac * scale.slice_area
    score = score_map.astype(np.float64, copy=False)

    accepted: Mask = np.zeros_like(candidate)
    holes: list[Hole] = []
    for i in range(1, n_lab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < scale.min_area or area > max_area:
            continue

        component = labels == i
        mean_score = float(score[component].mean()) if area else 0.0
        if mean_score < min_score:
            continue

        comp_u8 = component.astype(np.uint8)
        cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)  # ty: ignore[no-matching-overload]
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue

        cx, cy = cents[i]
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        holes.append(
            Hole(
                id=len(holes) + 1,
                cx=float(cx),
                cy=float(cy),
                area=area,
                circularity=round(float(circularity), 3),
                mean_depth=round(mean_score, 1),
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
            )
        )
        accepted[component] = 255

    return accepted, holes, {"raw_components": n_lab - 1, "kept_components": len(holes)}

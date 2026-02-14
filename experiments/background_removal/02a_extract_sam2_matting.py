#!/usr/bin/env python3
"""
02a_extract_sam2_matting.py  —  Extract objects via SAM2 auto + alpha matting

Production pipeline:
  1. SAM2 automatic mask generation (finds all distinct regions)
  2. Three-stage filtering: area size, color uniformity, IoU dedup
  3. Trimap generation from binary masks (erode/dilate)
  4. PyMatting closed-form alpha estimation (soft, continuous edges)
  5. Foreground color estimation (eliminates compositing halos)
  6. Final RGBA PNG with items in exact original positions

Usage:
  python 02a_extract_sam2_matting.py --batch --input-dir ./cutlery --output-dir ./outputs/cutlery_A --debug
  python 02a_extract_sam2_matting.py --image scene.jpg --output extracted.png --debug

Debug outputs (--debug):
  *_1_masks.png      -> all SAM2 regions (green=kept, red=rejected)
  *_2_trimap.png     -> combined trimap (white=fg, black=bg, gray=unknown)
  *_3_alpha.png      -> final alpha matte from PyMatting
  *_4_overlay.png    -> alpha overlay on original image
  *_5_rgba_full.png  -> full-size RGBA (uncropped)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAM2_HF_ID = "facebook/sam2-hiera-large"

# SAM2 auto filtering
DEFAULT_MAX_AREA_PCT = 8.0
DEFAULT_MIN_AREA_PX = 100
DEFAULT_MIN_COLOR_STD = 20.0
DEFAULT_IOU_DEDUP = 0.5
DEFAULT_POINTS_PER_SIDE = 64

# Alpha matting
DEFAULT_ERODE_PX = 3
DEFAULT_DILATE_PX = 7

# Output
DEFAULT_PADDING = 20


# ══════════════════════════════════════════════════════════════════
# SAM2 Automatic Mask Generator
# ══════════════════════════════════════════════════════════════════
class SAM2AutoExtractor:
    def __init__(self, device: str = "cuda", points_per_side: int = DEFAULT_POINTS_PER_SIDE):
        self.device = device
        log.info("Loading SAM2 for automatic segmentation...")
        from sam2.build_sam import build_sam2_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        model = build_sam2_hf(SAM2_HF_ID, device=device)
        self.generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.70,
            box_nms_thresh=0.7,
            min_mask_region_area=50,
        )
        log.info("SAM2 auto-mask generator ready (grid=%dx%d)", points_per_side, points_per_side)

    def generate(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        return self.generator.generate(image_np)


# ══════════════════════════════════════════════════════════════════
# Mask Filtering
# ══════════════════════════════════════════════════════════════════
def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def deduplicate_masks(masks: List[Dict], iou_threshold: float) -> List[Dict]:
    ranked = sorted(masks, key=lambda m: m["stability_score"], reverse=True)
    kept: List[Dict] = []
    for m in ranked:
        dup = any(compute_iou(m["segmentation"], e["segmentation"]) > iou_threshold for e in kept)
        if not dup:
            kept.append(m)
    return kept


def filter_masks(
    all_masks: List[Dict],
    image_np: np.ndarray,
    total_px: int,
    max_area_pct: float,
    min_area_px: int,
    min_color_std: float,
    iou_dedup: float,
) -> Tuple[List[Dict], List[int], Dict[str, int]]:
    """Three-stage filtering: area, color uniformity, dedup. Returns (kept_masks, kept_indices, stats)."""
    stats = {"rejected_large": 0, "rejected_small": 0, "rejected_flat": 0}
    kept_indices: List[int] = []

    for i, m in enumerate(all_masks):
        area_pct = m["area"] / total_px * 100
        if area_pct > max_area_pct:
            log.info("  [x] region %2d: %6d px (%5.1f%%) — too large", i, m["area"], area_pct)
            stats["rejected_large"] += 1
            continue
        if m["area"] < min_area_px:
            stats["rejected_small"] += 1
            continue
        pixels = image_np[m["segmentation"]]
        cstd = float(np.std(pixels.astype(np.float32), axis=0).mean())
        if cstd < min_color_std:
            log.info("  [x] region %2d: %6d px (%5.1f%%)  color_std=%.1f — flat surface", i, m["area"], area_pct, cstd)
            stats["rejected_flat"] += 1
            continue
        log.info("  [ok] region %2d: %6d px (%5.1f%%)  stab=%.2f  iou=%.2f  clr=%.1f",
                 i, m["area"], area_pct, m["stability_score"], m["predicted_iou"], cstd)
        kept_indices.append(i)

    kept = [all_masks[i] for i in kept_indices]
    log.info("After size+color: %d kept  (%d large, %d small, %d flat rejected of %d)",
             len(kept), stats["rejected_large"], stats["rejected_small"], stats["rejected_flat"], len(all_masks))

    if len(kept) > 1:
        before = len(kept)
        kept = deduplicate_masks(kept, iou_dedup)
        if len(kept) < before:
            log.info("  Dedup: %d -> %d masks", before, len(kept))

    # Rebuild indices for viz
    kept_seg_ids = {id(m["segmentation"]) for m in kept}
    kept_indices = [i for i, m in enumerate(all_masks) if id(m["segmentation"]) in kept_seg_ids]
    return kept, kept_indices, stats


# ══════════════════════════════════════════════════════════════════
# Alpha Matting (the production upgrade)
# ══════════════════════════════════════════════════════════════════
def binary_mask_to_trimap(mask_bool: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
    """Convert binary mask to trimap: 1.0=definite fg, 0.0=definite bg, 0.5=unknown."""
    mask_u8 = mask_bool.astype(np.uint8)
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    fg = cv2.erode(mask_u8, k_e, iterations=1)
    outer = cv2.dilate(mask_u8, k_d, iterations=1)
    trimap = np.full(mask_u8.shape, 0.5, dtype=np.float64)
    trimap[fg == 1] = 1.0
    trimap[outer == 0] = 0.0
    return trimap


def matting_alpha(image_f64: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    """Run closed-form alpha matting. Falls back to trimap threshold if pymatting fails."""
    try:
        from pymatting import estimate_alpha_cf
        alpha = estimate_alpha_cf(image_f64, trimap)
        return np.clip(alpha, 0.0, 1.0)
    except Exception as e:
        log.warning("PyMatting failed (%s), falling back to trimap threshold", e)
        alpha = np.zeros_like(trimap)
        alpha[trimap > 0.8] = 1.0
        alpha[(trimap >= 0.2) & (trimap <= 0.8)] = 0.5
        return alpha


def estimate_foreground(image_f64: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Estimate true foreground color to eliminate halos on compositing."""
    try:
        from pymatting import estimate_foreground_ml
        fg = estimate_foreground_ml(image_f64, alpha)
        return np.clip(fg, 0.0, 1.0)
    except Exception as e:
        log.warning("Foreground estimation failed (%s), using raw image", e)
        return image_f64


# ══════════════════════════════════════════════════════════════════
# Post-processing
# ══════════════════════════════════════════════════════════════════
def crop_content(rgba: np.ndarray, pad: int = DEFAULT_PADDING) -> np.ndarray:
    a = rgba[:, :, 3]
    rows = np.any(a > 0, axis=1)
    cols = np.any(a > 0, axis=0)
    if not rows.any():
        return rgba
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = rgba.shape[:2]
    return rgba[max(0, rmin - pad):min(h, rmax + pad + 1),
                max(0, cmin - pad):min(w, cmax + pad + 1)]


# ══════════════════════════════════════════════════════════════════
# Debug Visualizations
# ══════════════════════════════════════════════════════════════════
def viz_all_masks(image_bgr: np.ndarray, all_masks: List[Dict],
                  kept_indices: Set[int], output_path: str):
    overlay = image_bgr.copy()
    for i, m in enumerate(all_masks):
        if i in kept_indices:
            continue
        mask = m["segmentation"]
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.5
                         + np.array([0, 0, 128], dtype=np.float32)).clip(0, 255).astype(np.uint8)
    for i in kept_indices:
        mask = all_masks[i]["segmentation"]
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.5
                         + np.array([0, 128, 0], dtype=np.float32)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, overlay)
    log.info("  [debug] all masks -> %s", output_path)


def viz_trimap(trimap: np.ndarray, output_path: str):
    vis = (trimap * 255).astype(np.uint8)
    cv2.imwrite(output_path, vis)
    log.info("  [debug] trimap -> %s", output_path)


def viz_alpha(alpha: np.ndarray, output_path: str):
    vis = (alpha * 255).astype(np.uint8)
    cv2.imwrite(output_path, vis)
    log.info("  [debug] alpha matte -> %s", output_path)


def viz_overlay(image_bgr: np.ndarray, alpha: np.ndarray, output_path: str):
    ann = image_bgr.copy()
    mask = alpha < 0.05
    ann[mask] = (ann[mask] * 0.15).astype(np.uint8)
    cv2.imwrite(output_path, ann)
    log.info("  [debug] overlay -> %s", output_path)


# ══════════════════════════════════════════════════════════════════
# Core Extraction
# ══════════════════════════════════════════════════════════════════
def extract(
    image_path: str,
    output_path: str,
    extractor: SAM2AutoExtractor,
    max_area_pct: float = DEFAULT_MAX_AREA_PCT,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    min_color_std: float = DEFAULT_MIN_COLOR_STD,
    iou_dedup: float = DEFAULT_IOU_DEDUP,
    erode_px: int = DEFAULT_ERODE_PX,
    dilate_px: int = DEFAULT_DILATE_PX,
    padding: int = DEFAULT_PADDING,
    debug: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    stem = Path(output_path).stem
    odir = Path(output_path).parent
    os.makedirs(str(odir), exist_ok=True)

    # Load
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_f64 = image_np.astype(np.float64) / 255.0
    h, w = image_np.shape[:2]
    total_px = h * w
    log.info("=== %s (%dx%d) ===", Path(image_path).name, w, h)

    # 1. SAM2 auto segmentation
    t_seg = time.time()
    all_masks = extractor.generate(image_np)
    seg_time = time.time() - t_seg
    all_masks.sort(key=lambda m: m["area"], reverse=True)
    log.info("SAM2: %d regions in %.1fs", len(all_masks), seg_time)

    # 2. Filter
    kept_masks, kept_indices, stats = filter_masks(
        all_masks, image_np, total_px,
        max_area_pct, min_area_px, min_color_std, iou_dedup,
    )
    log.info("Kept: %d object masks", len(kept_masks))

    if not kept_masks:
        log.warning("No objects found in %s", image_path)
        return {"input": image_path, "output": None, "status": "no_objects",
                "regions": len(all_masks), "kept": 0, "elapsed": round(time.time() - t0, 2)}

    if debug:
        viz_all_masks(image_bgr, all_masks, set(kept_indices),
                      str(odir / f"{stem}_1_masks.png"))

    # 3. Combine binary masks -> trimap
    combined_binary = np.zeros((h, w), dtype=bool)
    for m in kept_masks:
        combined_binary |= m["segmentation"]

    trimap = binary_mask_to_trimap(combined_binary, erode_px, dilate_px)
    if debug:
        viz_trimap(trimap, str(odir / f"{stem}_2_trimap.png"))

    # 4. Alpha matting
    t_mat = time.time()
    alpha = matting_alpha(image_f64, trimap)
    mat_time = time.time() - t_mat
    log.info("Alpha matting: %.2fs", mat_time)

    if debug:
        viz_alpha(alpha, str(odir / f"{stem}_3_alpha.png"))
        viz_overlay(image_bgr, alpha, str(odir / f"{stem}_4_overlay.png"))

    # 5. Foreground estimation (anti-halo)
    foreground = estimate_foreground(image_f64, alpha)

    # 6. Build RGBA
    fg_u8 = (foreground * 255).clip(0, 255).astype(np.uint8)
    alpha_u8 = (alpha * 255).clip(0, 255).astype(np.uint8)
    rgba = np.dstack([fg_u8, alpha_u8])

    if debug:
        fp = str(odir / f"{stem}_5_rgba_full.png")
        Image.fromarray(rgba, "RGBA").save(fp)
        log.info("  [debug] full RGBA -> %s", fp)

    rgba_cropped = crop_content(rgba, padding)
    Image.fromarray(rgba_cropped, "RGBA").save(output_path, optimize=True)
    oh, ow = rgba_cropped.shape[:2]
    elapsed = time.time() - t0

    log.info("[ok] %s (%dx%d) — %d objects — %.2fs [seg=%.2fs mat=%.2fs]",
             output_path, ow, oh, len(kept_masks), elapsed, seg_time, mat_time)

    return {
        "input": image_path, "output": output_path, "status": "success",
        "regions": len(all_masks), "kept": len(kept_masks), **stats,
        "output_size": (ow, oh), "elapsed": round(elapsed, 2),
        "seg_time": round(seg_time, 2), "mat_time": round(mat_time, 2),
    }


# ══════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════
def batch_extract(input_dir: str, output_dir: str, extractor: SAM2AutoExtractor, **kwargs):
    ipath = Path(input_dir)
    if not ipath.is_dir():
        log.error("Not found: %s", input_dir); return []
    opath = Path(output_dir); opath.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ipath.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        log.warning("No images in %s", input_dir); return []
    log.info("Found %d image(s)", len(files))

    results = []
    for idx, f in enumerate(files, 1):
        log.info("--- [%d/%d] %s ---", idx, len(files), f.name)
        r = extract(str(f), str(opath / f"{f.stem}_rgba.png"), extractor, **kwargs)
        results.append(r)

    ok = sum(1 for r in results if r["status"] == "success")
    tt = sum(r.get("elapsed", 0) for r in results)
    log.info("=== BATCH DONE — %d/%d ok — %.1fs (%.1fs/img) ===", ok, len(results), tt, tt / len(results) if results else 0)

    with open(str(opath / "batch_report.json"), "w") as fp:
        json.dump({"total": len(results), "ok": ok, "results": results}, fp, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Script A: Extract objects via SAM2 auto + alpha matting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", type=str, help="Single image path")
    p.add_argument("--output", type=str, help="Output path (single mode)")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--input-dir", type=str)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--device", type=str, default=None)

    # SAM2 filtering
    p.add_argument("--max-area-pct", type=float, default=DEFAULT_MAX_AREA_PCT)
    p.add_argument("--min-area-px", type=int, default=DEFAULT_MIN_AREA_PX)
    p.add_argument("--min-color-std", type=float, default=DEFAULT_MIN_COLOR_STD)
    p.add_argument("--iou-dedup", type=float, default=DEFAULT_IOU_DEDUP)
    p.add_argument("--points-per-side", type=int, default=DEFAULT_POINTS_PER_SIDE)

    # Alpha matting
    p.add_argument("--erode-px", type=int, default=DEFAULT_ERODE_PX,
                   help="Trimap: erode pixels for definite foreground")
    p.add_argument("--dilate-px", type=int, default=DEFAULT_DILATE_PX,
                   help="Trimap: dilate pixels for definite background")

    p.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ext = SAM2AutoExtractor(device=device, points_per_side=args.points_per_side)

    common = dict(
        max_area_pct=args.max_area_pct, min_area_px=args.min_area_px,
        min_color_std=args.min_color_std, iou_dedup=args.iou_dedup,
        erode_px=args.erode_px, dilate_px=args.dilate_px,
        padding=args.padding, debug=args.debug,
    )

    if args.batch:
        if not args.input_dir: p.error("--batch requires --input-dir")
        batch_extract(args.input_dir, args.output_dir, ext, **common)
    elif args.image:
        if not args.output: p.error("--image requires --output")
        extract(args.image, args.output, ext, **common)
    else:
        p.error("Provide --image or --batch")


if __name__ == "__main__":
    main()

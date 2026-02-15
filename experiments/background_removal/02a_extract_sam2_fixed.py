#!/usr/bin/env python3
"""
02a_extract_sam2_fixed.py

Updates:
1. Tighter matting (smaller trimap) for sharper edges.
2. "Target Count" mode: explicitly looks for N items (e.g., 8 plates).
3. Relaxed filters: Removed color_std rejection (was killing smooth plates).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAM2_HF_ID = "facebook/sam2-hiera-large"

# SAM2 defaults
DEFAULT_POINTS_PER_SIDE = 64

# Matting Defaults (UPDATED: Tighter values)
DEFAULT_ERODE_PX = 1   # Was 3
DEFAULT_DILATE_PX = 3  # Was 7 (Total blend width = 4px instead of 10px)

# Output
DEFAULT_PADDING = 20


# ══════════════════════════════════════════════════════════════════
# SAM2 Automatic Mask Generator
# ══════════════════════════════════════════════════════════════════
class SAM2AutoExtractor:
    def __init__(self, device: str = "cuda", points_per_side: int = DEFAULT_POINTS_PER_SIDE):
        self.device = device
        log.info(f"Loading SAM2 ({SAM2_HF_ID})...")
        from sam2.build_sam import build_sam2_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        model = build_sam2_hf(SAM2_HF_ID, device=device)
        
        # We lower thresholds slightly to ensure we catch all 8 plates
        # We will filter them out later using the 'Target Count' logic
        self.generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=64,
            pred_iou_thresh=0.65,        # Lowered slightly to catch difficult plates
            stability_score_thresh=0.65, # Lowered slightly
            box_nms_thresh=0.7,
            min_mask_region_area=50,
        )
        log.info("SAM2 ready.")

    def generate(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        return self.generator.generate(image_np)


# ══════════════════════════════════════════════════════════════════
# Intelligent Filtering (The "Top 8" Logic)
# ══════════════════════════════════════════════════════════════════
def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0

def filter_masks_top_k(
    all_masks: List[Dict],
    image_np: np.ndarray,
    target_count: int,
    min_area_px: int = 100,
    iou_dedup: float = 0.5
) -> Tuple[List[Dict], List[int]]:
    """
    1. Removes tiny noise.
    2. Deduplicates overlapping masks.
    3. Sorts by stability/quality.
    4. Returns exactly the top K masks (if available).
    """
    h, w = image_np.shape[:2]
    total_px = h * w
    
    # 1. Pre-filter noise (Too small or Too huge)
    candidates = []
    for i, m in enumerate(all_masks):
        area = m["area"]
        # Filter 1: Too small?
        if area < min_area_px: 
            continue
        # Filter 2: Too big? (e.g. > 50% of screen is probably background)
        if area > (total_px * 0.50):
            continue
        
        candidates.append(m)

    log.info(f"  Candidates after size filtering: {len(candidates)}")

    # 2. Sort by Stability Score (Confidence)
    # This puts the "cleanest" looking plates at the top
    candidates.sort(key=lambda x: x["stability_score"], reverse=True)

    # 3. Deduplication (Greedy NMS)
    # We iterate through the sorted list. If a mask overlaps heavily with 
    # one we already picked, we skip it (it's likely a duplicate detection).
    kept = []
    for cand in candidates:
        is_dup = False
        for k in kept:
            if compute_iou(cand["segmentation"], k["segmentation"]) > iou_dedup:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)

    log.info(f"  Candidates after dedup: {len(kept)}")

    # 4. Target Count Cutoff
    # If we want 8, we take the top 8. 
    # If we have 5, we take 5.
    final_masks = kept[:target_count]
    
    # Recover indices for debugging visualization
    # (Matches memory addresses of segmentation arrays)
    final_seg_ids = {id(m["segmentation"]) for m in final_masks}
    kept_indices = [i for i, m in enumerate(all_masks) if id(m["segmentation"]) in final_seg_ids]

    return final_masks, kept_indices


# ══════════════════════════════════════════════════════════════════
# Alpha Matting
# ══════════════════════════════════════════════════════════════════
def binary_mask_to_trimap(mask_bool: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8)
    # If erode/dilate is 0, handle gracefully
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, erode_px * 2 + 1), max(1, erode_px * 2 + 1)))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, dilate_px * 2 + 1), max(1, dilate_px * 2 + 1)))
    
    if erode_px > 0:
        fg = cv2.erode(mask_u8, k_e, iterations=1)
    else:
        fg = mask_u8

    if dilate_px > 0:
        outer = cv2.dilate(mask_u8, k_d, iterations=1)
    else:
        outer = mask_u8

    trimap = np.full(mask_u8.shape, 0.5, dtype=np.float64)
    trimap[fg == 1] = 1.0
    trimap[outer == 0] = 0.0
    return trimap

def matting_alpha(image_f64: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    try:
        from pymatting import estimate_alpha_cf
        # "laplacian" set to 1 often helps with cleaner hard objects like plates
        alpha = estimate_alpha_cf(image_f64, trimap, laplacian_kwargs={"epsilon": 1e-7})
        return np.clip(alpha, 0.0, 1.0)
    except Exception as e:
        log.warning(f"PyMatting failed: {e}, falling back to binary")
        return trimap # Fallback

def estimate_foreground(image_f64: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    try:
        from pymatting import estimate_foreground_ml
        return np.clip(estimate_foreground_ml(image_f64, alpha), 0.0, 1.0)
    except:
        return image_f64

def crop_content(rgba: np.ndarray, pad: int = DEFAULT_PADDING) -> np.ndarray:
    a = rgba[:, :, 3]
    if not np.any(a > 0): return rgba
    rows = np.any(a > 0, axis=1)
    cols = np.any(a > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = rgba.shape[:2]
    return rgba[max(0, rmin - pad):min(h, rmax + pad + 1),
                max(0, cmin - pad):min(w, cmax + pad + 1)]

# ══════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════
def viz_debug(image_bgr, all_masks, kept_indices, trimap, alpha, stem, odir):
    # 1. Masks
    overlay = image_bgr.copy()
    for i, m in enumerate(all_masks):
        color = (0, 128, 0) if i in kept_indices else (0, 0, 128)
        mask = m["segmentation"]
        # Fast overlay
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
    cv2.imwrite(str(odir / f"{stem}_1_masks.png"), overlay)
    
    # 2. Trimap
    cv2.imwrite(str(odir / f"{stem}_2_trimap.png"), (trimap * 255).astype(np.uint8))
    
    # 3. Alpha
    cv2.imwrite(str(odir / f"{stem}_3_alpha.png"), (alpha * 255).astype(np.uint8))

# ══════════════════════════════════════════════════════════════════
# Main Logic
# ══════════════════════════════════════════════════════════════════
def extract(
    image_path: str,
    output_path: str,
    extractor: SAM2AutoExtractor,
    target_count: int = 8,  # DEFAULT IS NOW 8
    erode_px: int = DEFAULT_ERODE_PX,
    dilate_px: int = DEFAULT_DILATE_PX,
    debug: bool = False
):
    stem = Path(output_path).stem
    odir = Path(output_path).parent
    os.makedirs(str(odir), exist_ok=True)

    # Load Image
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    
    # 1. Generate All Masks
    all_masks = extractor.generate(img_np)
    
    # 2. Filter using Top-K logic
    kept_masks, kept_indices = filter_masks_top_k(
        all_masks, img_np, target_count=target_count
    )
    
    log.info(f"Targeting {target_count} items -> Found {len(kept_masks)} valid items")

    if not kept_masks:
        log.warning(f"Failed to find objects in {image_path}")
        return

    # 3. Combine Masks & Trimap
    combined_mask = np.zeros((h, w), dtype=bool)
    for m in kept_masks:
        combined_mask |= m["segmentation"]
        
    trimap = binary_mask_to_trimap(combined_mask, erode_px, dilate_px)
    
    # 4. Matting & Foreground
    img_f64 = img_np.astype(np.float64) / 255.0
    alpha = matting_alpha(img_f64, trimap)
    fg = estimate_foreground(img_f64, alpha)
    
    # 5. Save
    rgba = np.dstack([(fg * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    rgba = crop_content(rgba)
    Image.fromarray(rgba).save(output_path)
    
    if debug:
        viz_debug(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), all_masks, set(kept_indices), trimap, alpha, stem, odir)

    log.info(f"Saved: {output_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", action="store_true")
    p.add_argument("--input-dir", type=str)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--image", type=str)
    p.add_argument("--output", type=str)
    p.add_argument("--debug", action="store_true")
    
    # Tweaks
    p.add_argument("--target-count", type=int, default=8, help="How many plates to find?")
    p.add_argument("--erode", type=int, default=DEFAULT_ERODE_PX, help="Inner trimap shrink")
    p.add_argument("--dilate", type=int, default=DEFAULT_DILATE_PX, help="Outer trimap expand")
    
    args = p.parse_args()

    ext = SAM2AutoExtractor()

    if args.batch:
        files = list(Path(args.input_dir).glob("*.*"))
        for f in files:
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                extract(str(f), str(Path(args.output_dir) / f"{f.stem}.png"), ext, 
                        target_count=args.target_count, erode_px=args.erode, dilate_px=args.dilate, debug=args.debug)
    elif args.image:
        extract(args.image, args.output, ext, 
                target_count=args.target_count, erode_px=args.erode, dilate_px=args.dilate, debug=args.debug)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
02d_extract_forced_8.py — The "Always 8" Plate Extractor

Logic:
1. Detects EVERYTHING (Low thresholds).
2. Calculates the MEDIAN object size (The 8 plates define the average).
3. Selects exactly the 8 objects closest to that median size.
4. Removes Background (PyMatting).
5. KEEPS POSITION (No cropping).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ── LOGGING ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAM2_HF_ID = "facebook/sam2-hiera-large"

# ── CONFIG FOR PLATES ─────────────────────────────────────────────
TARGET_COUNT = 8         # WE WANT EXACTLY 8
MIN_AREA_PX = 500        # Ignore tiny specs/dust
MIN_ASPECT_RATIO = 0.4   # Ignore skinny Cutlery
MAX_ASPECT_RATIO = 2.5   # Ignore long Table Runners

# Matting settings (Sharp edges for hard plates)
ERODE_PX = 2
DILATE_PX = 4

class SAM2Loader:
    _instance = None
    
    @classmethod
    def get(cls, device="cuda"):
        if cls._instance is None:
            log.info(f"Loading SAM2 ({SAM2_HF_ID})...")
            from sam2.build_sam import build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                
            model = build_sam2_hf(SAM2_HF_ID, device=device)
            # AGGRESSIVE SETTINGS: Lower thresholds to ensure we catch ALL 8 plates
            # We will filter the bad ones out manually later.
            cls._instance = SAM2AutomaticMaskGenerator(
                model, 
                points_per_side=64, 
                pred_iou_thresh=0.5,        # Lowered from 0.7
                stability_score_thresh=0.5, # Lowered from 0.7
                min_mask_region_area=MIN_AREA_PX
            )
        return cls._instance

def get_bbox_aspect_ratio(mask: np.ndarray) -> float:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any(): return 0.0
    h = np.sum(rows)
    w = np.sum(cols)
    return w / h

def filter_masks_median_consensus(all_masks: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
    h, w = image_shape
    total_px = h * w
    
    # 1. Pre-filter (Remove walls, dust, cutlery)
    candidates = []
    for m in all_masks:
        area = m["area"]
        # Too small?
        if area < MIN_AREA_PX: continue
        # Too big (Wall/Table)? > 50% of screen
        if area > total_px * 0.5: continue 
        
        # Shape Check (Reject cutlery)
        ar = get_bbox_aspect_ratio(m["segmentation"])
        if not (MIN_ASPECT_RATIO <= ar <= MAX_ASPECT_RATIO):
            continue
            
        candidates.append(m)

    log.info(f"  Found {len(candidates)} candidates after cleaning noise/walls.")

    if not candidates:
        return []
    
    # 2. If we have <= 8, just take them all
    if len(candidates) <= TARGET_COUNT:
        log.warning(f"  Only found {len(candidates)} valid objects. Returning all of them.")
        return candidates

    # 3. MEDIAN CONSENSUS LOGIC
    # Calculate the Median Area of the remaining candidates.
    # Since we expect 8 plates and maybe 1-2 other things, the Median will be the Plate Size.
    areas = [m["area"] for m in candidates]
    median_area = np.median(areas)
    
    log.info(f"  Median Object Size: {int(median_area)} px")

    # Score candidates by how close they are to the median
    # We also weigh Stability slightly so we prefer a clean plate over a blurry one of same size
    scored_candidates = []
    for m in candidates:
        dist = abs(m["area"] - median_area)
        # Score: Lower is better. 
        # Distance is dominant. Stability acts as a tie-breaker (higher stability subtracts from score)
        score = dist - (m["stability_score"] * 100) 
        scored_candidates.append((score, m))
    
    # Sort: Lowest score (closest to median) first
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take Top 8
    final_masks = [x[1] for x in scored_candidates[:TARGET_COUNT]]
    
    # Debug info
    sizes = [m["area"] for m in final_masks]
    log.info(f"  Selected 8 sizes: {sizes}")
    
    return final_masks

def process_image(img_path: Path, output_path: Path, generator, debug=False, no_crop=True):
    try:
        log.info(f"Processing: {img_path.name}")
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        
        masks = generator.generate(img_np)
        
        # USE THE NEW MEDIAN LOGIC
        kept_masks = filter_masks_median_consensus(masks, (h, w))
        
        if not kept_masks:
            log.warning(f"  No objects found in {img_path.name}")
            return

        # Combine
        combined_mask = np.zeros((h, w), dtype=bool)
        for m in kept_masks:
            combined_mask |= m["segmentation"]
        
        mask_u8 = combined_mask.astype(np.uint8)
        
        # Trimap / Matting
        k_e = max(1, ERODE_PX * 2 + 1)
        k_d = max(1, DILATE_PX * 2 + 1)
        
        # Create Trimap
        fg = cv2.erode(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_e, k_e)))
        outer = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_d, k_d)))
        
        trimap = np.full(mask_u8.shape, 0.5, dtype=np.float64)
        trimap[fg == 1] = 1.0
        trimap[outer == 0] = 0.0

        # PyMatting
        img_f64 = img_np.astype(np.float64) / 255.0
        try:
            from pymatting import estimate_alpha_cf, estimate_foreground_ml
            alpha = estimate_alpha_cf(img_f64, trimap, laplacian_kwargs={"epsilon": 1e-7})
            foreground = estimate_foreground_ml(img_f64, alpha)
        except ImportError:
            alpha = trimap
            foreground = img_f64

        # Save
        alpha = np.clip(alpha, 0, 1)
        rgba = np.dstack([(foreground * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
        
        # NO CROP LOGIC (Ensures perfect overlay)
        if not no_crop:
            a_channel = rgba[:, :, 3]
            if np.any(a_channel):
                rows = np.any(a_channel, axis=1)
                cols = np.any(a_channel, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                pad = 20
                rgba = rgba[max(0, rmin-pad):min(h, rmax+pad), max(0, cmin-pad):min(w, cmax+pad)]
        
        os.makedirs(output_path.parent, exist_ok=True)
        Image.fromarray(rgba).save(output_path)
        log.info(f"  Saved {output_path.name} ({len(kept_masks)} objects)")
        
        if debug:
            debug_dir = output_path.parent / "debug_vis"
            os.makedirs(debug_dir, exist_ok=True)
            overlay = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            for m in kept_masks:
                overlay[m["segmentation"]] = (overlay[m["segmentation"]]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_selected.png"), overlay)

    except Exception as e:
        log.error(f"  Failed {img_path.name}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--debug", action="store_true")
    # Default is NO CROP now, but keeping flag for compatibility
    p.add_argument("--crop", action="store_true", help="Crop the output image to content (removes shifting)")
    
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = SAM2Loader.get(device)
    
    in_path = Path(args.input_dir)
    files = [f for f in in_path.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    # Enforce NO_CROP unless user specifically asks to crop
    no_crop_setting = not args.crop
    
    for f in files:
        process_image(f, Path(args.output_dir) / f"{f.stem}.png", gen, debug=args.debug, no_crop=no_crop_setting)
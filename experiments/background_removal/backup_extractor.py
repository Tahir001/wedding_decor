#!/usr/bin/env python3
"""
02_extract_items.py  —  Extract ALL objects from table scenes → transparent RGBA

Strategy (no GroundingDINO — just SAM2 automatic mask generation):
  1. SAM2 auto-segments every distinct region in the image
  2. Large regions (table surface, tablecloth, background) are filtered out
  3. Tiny regions (noise, specks) are filtered out
  4. Everything else = the objects sitting ON the table
  5. Combined into a single RGBA PNG with exact original positions preserved

The result can be placed directly on top of any other image/background
and the layout of items is perfectly preserved.

Usage:
  # All images in a directory:
  python 02_extract_items.py --batch --input-dir ./cutlery --output-dir ./outputs/cutlery --debug

  # Single image:
  python 02_extract_items.py --image scene.jpg --output extracted.png --debug

  # Tune filtering thresholds:
  python 02_extract_items.py --image scene.jpg --output out.png --max-area-pct 10 --min-area-px 50 --debug

Debug outputs (with --debug):
  *_1_all_masks.png   → every SAM2 region: green=kept, red=rejected
  *_2_kept_grid.png   → grid of individual kept masks with stats
  *_3_overlay.png     → final mask overlay on original image
  *_4_rgba_full.png   → full-size RGBA (uncropped)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Filtering thresholds
DEFAULT_MAX_AREA_PCT = 15.0     # masks covering >15% of image → table / background
DEFAULT_MIN_AREA_PX = 100       # masks smaller than 100px → noise
DEFAULT_POINTS_PER_SIDE = 64    # SAM2 point grid density (higher = finds smaller items)

# Post-processing
DEFAULT_PADDING = 20
DEFAULT_FEATHER = 2
MORPH_ITERATIONS = 2


# ══════════════════════════════════════════════════════════════════
# SAM2 Automatic Mask Generator
# ══════════════════════════════════════════════════════════════════
class SAM2AutoExtractor:
    """Wraps SAM2 automatic mask generation for table-scene object extraction."""

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
        """Run automatic mask generation. Returns list of mask dicts."""
        return self.generator.generate(image_np)


# ══════════════════════════════════════════════════════════════════
# Post-processing
# ══════════════════════════════════════════════════════════════════
def smooth_mask(mask: np.ndarray, iterations: int = MORPH_ITERATIONS) -> np.ndarray:
    """Morphological smoothing for clean edges."""
    from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
    s = generate_binary_structure(2, 2)
    m = binary_closing(mask, structure=s, iterations=iterations)
    m = binary_opening(m, structure=s, iterations=iterations)
    return m.astype(bool)


def feather_alpha(mask: np.ndarray, radius: int = DEFAULT_FEATHER) -> np.ndarray:
    """Soft edge feathering for natural-looking cutouts."""
    a = mask.astype(np.float64)
    if radius > 0:
        a = ndimage.gaussian_filter(a, sigma=radius * 0.5)
        a = np.where(mask, np.maximum(a, 0.95), a)
    return np.clip(a, 0.0, 1.0)


def crop_content(rgba: np.ndarray, pad: int = DEFAULT_PADDING) -> np.ndarray:
    """Crop to bounding box of non-transparent content + padding."""
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
    """All SAM2 regions: green = kept (objects), red = rejected (table/bg/noise)."""
    overlay = image_bgr.copy()
    # Rejected → red tint
    for i, m in enumerate(all_masks):
        if i in kept_indices:
            continue
        mask = m["segmentation"]
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.5
                         + np.array([0, 0, 128], dtype=np.float32)).clip(0, 255).astype(np.uint8)
    # Kept → green tint
    for i in kept_indices:
        mask = all_masks[i]["segmentation"]
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.5
                         + np.array([0, 128, 0], dtype=np.float32)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, overlay)
    log.info("📸 All masks → %s (green=kept, red=rejected)", output_path)


def viz_masks_grid(masks_info: List[Dict], total_px: int, output_path: str):
    """Grid view of kept masks with area + score labels."""
    if not masks_info:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(masks_info)
    dim = max(1, math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(dim, dim, figsize=(4 * dim, 4 * dim))
    if dim == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, m in enumerate(masks_info):
        axes[i].imshow(m["segmentation"].astype(np.uint8) * 255, cmap="gray")
        pct = m["area"] / total_px * 100
        axes[i].set_title(
            f"#{i}  {m['area']}px ({pct:.2f}%)\n"
            f"iou={m['predicted_iou']:.2f}  stab={m['stability_score']:.2f}",
            fontsize=9, fontweight="bold",
        )
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("📸 Kept masks grid (%d) → %s", n, output_path)


def viz_overlay(image_bgr: np.ndarray, combined_mask: np.ndarray, output_path: str):
    """Final combined mask: bright = kept items, dimmed = removed."""
    ann = image_bgr.copy()
    ann[~combined_mask] = (ann[~combined_mask] * 0.2).astype(np.uint8)
    cv2.imwrite(output_path, ann)
    log.info("📸 Overlay → %s", output_path)


# ══════════════════════════════════════════════════════════════════
# Core Extraction
# ══════════════════════════════════════════════════════════════════
def extract(
    image_path: str,
    output_path: str,
    extractor: SAM2AutoExtractor,
    max_area_pct: float = DEFAULT_MAX_AREA_PCT,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    padding: int = DEFAULT_PADDING,
    feather_radius: int = DEFAULT_FEATHER,
    debug: bool = False,
) -> Dict[str, Any]:
    """Extract all objects from a single table scene image."""
    t0 = time.time()
    stem = Path(output_path).stem
    odir = Path(output_path).parent
    os.makedirs(str(odir), exist_ok=True)

    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]
    total_px = h * w
    log.info("=== %s (%dx%d, %d px) ===", Path(image_path).name, w, h, total_px)

    # ── 1. SAM2 automatic mask generation ──
    t_seg = time.time()
    all_masks = extractor.generate(image_np)
    seg_time = time.time() - t_seg
    # Sort by area descending
    all_masks.sort(key=lambda m: m["area"], reverse=True)
    log.info("SAM2 auto: %d region(s) found in %.1fs", len(all_masks), seg_time)

    # ── 2. Filter: reject table/background (big) and noise (tiny) ──
    kept_indices: List[int] = []
    rejected_large = 0
    rejected_small = 0
    for i, m in enumerate(all_masks):
        area_pct = m["area"] / total_px * 100
        if area_pct > max_area_pct:
            log.info("  🚫 region %2d: %6d px (%5.1f%%) — REJECTED (table/background)", i, m["area"], area_pct)
            rejected_large += 1
            continue
        if m["area"] < min_area_px:
            rejected_small += 1
            continue
        log.info("  ✅ region %2d: %6d px (%5.1f%%)  stab=%.2f  iou=%.2f",
                 i, m["area"], area_pct, m["stability_score"], m["predicted_iou"])
        kept_indices.append(i)

    kept_masks = [all_masks[i] for i in kept_indices]
    log.info("Result: %d kept, %d rejected-large, %d rejected-small (of %d total)",
             len(kept_masks), rejected_large, rejected_small, len(all_masks))

    if not kept_masks:
        log.warning("⚠️  No objects found in %s", image_path)
        return {"input": image_path, "output": None, "status": "no_objects",
                "total_regions": len(all_masks), "kept": 0, "elapsed": round(time.time() - t0, 2)}

    # ── 3. Combine kept masks ──
    combined = np.zeros((h, w), dtype=bool)
    for m in kept_masks:
        combined |= m["segmentation"]

    mask_area = int(combined.sum())
    mask_pct = mask_area / total_px * 100
    log.info("Combined object mask: %d px (%.1f%% of image)", mask_area, mask_pct)

    # ── 4. Debug visualizations ──
    if debug:
        viz_all_masks(image_bgr, all_masks, set(kept_indices),
                      str(odir / f"{stem}_1_all_masks.png"))
        viz_masks_grid(kept_masks, total_px, str(odir / f"{stem}_2_kept_grid.png"))
        viz_overlay(image_bgr, combined, str(odir / f"{stem}_3_overlay.png"))

    # ── 5. Smooth edges + feather alpha ──
    combined = smooth_mask(combined)
    alpha = feather_alpha(combined, radius=feather_radius)

    # ── 6. Build RGBA output ──
    rgba = np.dstack([image_np, (alpha * 255).astype(np.uint8)])

    if debug:
        fp = str(odir / f"{stem}_4_rgba_full.png")
        Image.fromarray(rgba, "RGBA").save(fp)
        log.info("📸 Full RGBA → %s", fp)

    # Cropped output (tight bounding box around objects)
    rgba_cropped = crop_content(rgba, padding)
    Image.fromarray(rgba_cropped, "RGBA").save(output_path, optimize=True)
    oh, ow = rgba_cropped.shape[:2]

    elapsed = time.time() - t0
    log.info("✅ %s (%dx%d) — %d objects kept — %.2fs [seg=%.2fs]",
             output_path, ow, oh, len(kept_masks), elapsed, seg_time)

    return {
        "input": image_path, "output": output_path, "status": "success",
        "total_regions": len(all_masks), "kept": len(kept_masks),
        "rejected_large": rejected_large, "rejected_small": rejected_small,
        "mask_area_px": mask_area, "mask_pct": round(mask_pct, 2),
        "output_size": (ow, oh), "elapsed": round(elapsed, 2),
        "seg_time": round(seg_time, 2),
    }


# ══════════════════════════════════════════════════════════════════
# Batch Processing
# ══════════════════════════════════════════════════════════════════
def batch_extract(input_dir: str, output_dir: str, extractor: SAM2AutoExtractor,
                  **kwargs) -> List[Dict[str, Any]]:
    """Process all images in a directory."""
    ipath = Path(input_dir)
    if not ipath.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return []

    opath = Path(output_dir)
    opath.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in ipath.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        log.warning("No images in %s", input_dir)
        return []
    log.info("Found %d image(s) in %s", len(files), input_dir)

    results = []
    for idx, f in enumerate(files, 1):
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("  [%d/%d] %s", idx, len(files), f.name)
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        out_name = f"{f.stem}_rgba.png"
        r = extract(str(f), str(opath / out_name), extractor, **kwargs)
        results.append(r)

    ok = sum(1 for r in results if r["status"] == "success")
    total_time = sum(r.get("elapsed", 0) for r in results)

    log.info("")
    log.info("═" * 60)
    log.info("  BATCH COMPLETE — %d/%d successful", ok, len(results))
    log.info("  Total: %.1fs (%.1fs/img)", total_time, total_time / len(results) if results else 0)
    log.info("═" * 60)

    # Save report
    rp = str(opath / "batch_report.json")
    with open(rp, "w") as fp:
        json.dump({"total": len(results), "succeeded": ok, "results": results}, fp, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Extract objects from table scenes → transparent RGBA PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --batch --input-dir ./cutlery --output-dir ./outputs/cutlery --debug
  %(prog)s --image scene.jpg --output extracted.png --debug
  %(prog)s --image scene.jpg --output out.png --max-area-pct 10 --min-area-px 50 --debug
        """,
    )
    # Mode
    p.add_argument("--image", type=str, help="Single image path")
    p.add_argument("--output", type=str, help="Output path (for --image mode)")
    p.add_argument("--batch", action="store_true", help="Process all images in --input-dir")
    p.add_argument("--input-dir", type=str, help="Input directory (for --batch mode)")
    p.add_argument("--output-dir", type=str, default="./outputs", help="Output directory (default: ./outputs)")

    # Filtering
    p.add_argument("--max-area-pct", type=float, default=DEFAULT_MAX_AREA_PCT,
                   help=f"Reject masks larger than this %% of image — increase if items are being dropped, "
                        f"decrease if table leaks through (default: {DEFAULT_MAX_AREA_PCT})")
    p.add_argument("--min-area-px", type=int, default=DEFAULT_MIN_AREA_PX,
                   help=f"Reject masks smaller than this pixel count (default: {DEFAULT_MIN_AREA_PX})")

    # SAM2 settings
    p.add_argument("--points-per-side", type=int, default=DEFAULT_POINTS_PER_SIDE,
                   help=f"SAM2 grid density — higher finds smaller items (default: {DEFAULT_POINTS_PER_SIDE})")

    # Output tuning
    p.add_argument("--padding", type=int, default=DEFAULT_PADDING, help=f"Padding around crop (default: {DEFAULT_PADDING})")
    p.add_argument("--feather", type=int, default=DEFAULT_FEATHER, help=f"Edge feather radius (default: {DEFAULT_FEATHER})")
    p.add_argument("--debug", action="store_true", help="Save debug visualizations")
    p.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")

    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SAM2AutoExtractor(device=device, points_per_side=args.points_per_side)

    common = dict(
        max_area_pct=args.max_area_pct,
        min_area_px=args.min_area_px,
        padding=args.padding,
        feather_radius=args.feather,
        debug=args.debug,
    )

    if args.batch:
        if not args.input_dir:
            p.error("--batch requires --input-dir")
        batch_extract(args.input_dir, args.output_dir, extractor, **common)
    elif args.image:
        if not args.output:
            p.error("--image requires --output")
        extract(args.image, args.output, extractor, **common)
    else:
        p.error("Provide --image or --batch")


if __name__ == "__main__":
    main()

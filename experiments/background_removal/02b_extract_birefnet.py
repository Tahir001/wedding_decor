#!/usr/bin/env python3
"""
02b_extract_birefnet.py  —  Extract objects via RMBG-2.0 / BiRefNet

Production pipeline:
  1. BiRefNet (RMBG-2.0) single forward pass -> soft alpha matte
  2. Foreground color estimation (eliminates compositing halos)
  3. Final RGBA PNG with items in exact original positions

No SAM2, no filtering heuristics, no trimaps.
The model is trained specifically for foreground/background separation
and outputs continuous alpha values (0.0 - 1.0) directly.

Usage:
  python 02b_extract_birefnet.py --batch --input-dir ./cutlery --output-dir ./outputs/cutlery_B --debug
  python 02b_extract_birefnet.py --image scene.jpg --output extracted.png --debug

Debug outputs (--debug):
  *_1_masks.png      -> raw model alpha output (grayscale)
  *_2_trimap.png     -> thresholded trimap view (for comparison with Script A)
  *_3_alpha.png      -> final alpha after optional thresholding
  *_4_overlay.png    -> alpha overlay on original image
  *_5_rgba_full.png  -> full-size RGBA (uncropped)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
RMBG_MODEL_ID = "briaai/RMBG-2.0"

# Processing
DEFAULT_MODEL_SIZE = 1024       # BiRefNet input resolution
DEFAULT_ALPHA_THRESHOLD = 0.0   # 0 = use raw alpha, >0 = hard-threshold weak regions
DEFAULT_PADDING = 20


# ══════════════════════════════════════════════════════════════════
# BiRefNet / RMBG-2.0 Model
# ══════════════════════════════════════════════════════════════════
class BiRefNetExtractor:
    """RMBG-2.0 background removal — single forward pass -> alpha matte."""

    def __init__(self, device: str = "cuda", model_size: int = DEFAULT_MODEL_SIZE):
        self.device = device
        self.model_size = model_size
        log.info("Loading RMBG-2.0 (BiRefNet)...")

        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        self.model = AutoModelForImageSegmentation.from_pretrained(
            RMBG_MODEL_ID, trust_remote_code=True
        )
        self.model.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((model_size, model_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        log.info("RMBG-2.0 ready (input size=%d)", model_size)

    @torch.no_grad()
    def predict(self, image_pil: Image.Image) -> np.ndarray:
        """Run model, return alpha matte at original resolution (float64, 0-1)."""
        orig_w, orig_h = image_pil.size
        inp = self.transform(image_pil).unsqueeze(0).to(self.device)
        pred = self.model(inp)[-1].sigmoid().cpu()
        alpha = pred[0].squeeze().numpy().astype(np.float64)
        # Resize back to original dimensions
        alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha_pil = alpha_pil.resize((orig_w, orig_h), Image.LANCZOS)
        return np.array(alpha_pil).astype(np.float64) / 255.0


# ══════════════════════════════════════════════════════════════════
# Foreground Estimation
# ══════════════════════════════════════════════════════════════════
def estimate_foreground(image_f64: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Estimate true foreground color to eliminate halos on compositing."""
    try:
        from pymatting import estimate_foreground_ml
        fg = estimate_foreground_ml(image_f64, alpha)
        return np.clip(fg, 0.0, 1.0)
    except ImportError:
        log.warning("pymatting not installed, using raw image (may have halos)")
        return image_f64
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
def viz_raw_alpha(alpha: np.ndarray, output_path: str):
    """Raw model alpha output as grayscale."""
    vis = (alpha * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, vis)
    log.info("  [debug] raw alpha -> %s", output_path)


def viz_trimap_view(alpha: np.ndarray, output_path: str):
    """Pseudo-trimap for comparison: fg(>0.9)=white, bg(<0.1)=black, unknown=gray."""
    vis = np.full(alpha.shape, 128, dtype=np.uint8)
    vis[alpha > 0.9] = 255
    vis[alpha < 0.1] = 0
    cv2.imwrite(output_path, vis)
    log.info("  [debug] trimap view -> %s", output_path)


def viz_alpha(alpha: np.ndarray, output_path: str):
    vis = (alpha * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, vis)
    log.info("  [debug] final alpha -> %s", output_path)


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
    extractor: BiRefNetExtractor,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
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
    log.info("=== %s (%dx%d) ===", Path(image_path).name, w, h)

    # 1. BiRefNet alpha prediction
    t_pred = time.time()
    raw_alpha = extractor.predict(image_pil)
    pred_time = time.time() - t_pred
    log.info("BiRefNet prediction: %.2fs", pred_time)

    if debug:
        viz_raw_alpha(raw_alpha, str(odir / f"{stem}_1_masks.png"))
        viz_trimap_view(raw_alpha, str(odir / f"{stem}_2_trimap.png"))

    # Optional: hard-threshold very weak alpha regions
    alpha = raw_alpha.copy()
    if alpha_threshold > 0:
        alpha[alpha < alpha_threshold] = 0.0
        log.info("Applied alpha threshold: %.2f", alpha_threshold)

    fg_pct = (alpha > 0.5).sum() / (h * w) * 100
    log.info("Foreground coverage: %.1f%%", fg_pct)

    if debug:
        viz_alpha(alpha, str(odir / f"{stem}_3_alpha.png"))
        viz_overlay(image_bgr, alpha, str(odir / f"{stem}_4_overlay.png"))

    # 2. Foreground estimation (anti-halo)
    foreground = estimate_foreground(image_f64, alpha)

    # 3. Build RGBA
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

    log.info("[ok] %s (%dx%d) — fg=%.1f%% — %.2fs [pred=%.2fs]",
             output_path, ow, oh, fg_pct, elapsed, pred_time)

    return {
        "input": image_path, "output": output_path, "status": "success",
        "fg_pct": round(fg_pct, 2),
        "output_size": (ow, oh), "elapsed": round(elapsed, 2),
        "pred_time": round(pred_time, 2),
    }


# ══════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════
def batch_extract(input_dir: str, output_dir: str, extractor: BiRefNetExtractor, **kwargs):
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
        description="Script B: Extract objects via RMBG-2.0 / BiRefNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", type=str, help="Single image path")
    p.add_argument("--output", type=str, help="Output path (single mode)")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--input-dir", type=str)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--device", type=str, default=None)

    # BiRefNet settings
    p.add_argument("--model-size", type=int, default=DEFAULT_MODEL_SIZE,
                   help="Model input resolution (default: 1024)")
    p.add_argument("--alpha-threshold", type=float, default=DEFAULT_ALPHA_THRESHOLD,
                   help="Zero-out alpha below this value (0=keep raw, try 0.1 to cut noise)")

    p.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ext = BiRefNetExtractor(device=device, model_size=args.model_size)

    common = dict(
        alpha_threshold=args.alpha_threshold,
        padding=args.padding,
        debug=args.debug,
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

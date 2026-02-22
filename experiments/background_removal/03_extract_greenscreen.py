#!/usr/bin/env python3
"""
03_extract_greenscreen.py — Green Screen Chroma Key Extractor

Extracts objects (plates, cutlery, etc.) from green screen photos with
semi-transparent shadow preservation for compositing.

Algorithm:
1. Auto-detect background green color from image edges
2. Compute per-pixel color distance from background in LAB space
3. Map distance to alpha with smooth falloff (captures shadows)
4. Despill green color contamination from object edges
5. Output full-size RGBA PNG (no cropping, preserves position)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

DEFAULT_SHADOW_STRENGTH = 0.6
DEFAULT_DESPILL_STRENGTH = 0.7
DEFAULT_EDGE_SOFTNESS = 1.5


def sample_background_color(img_bgr: np.ndarray, margin_pct: float = 0.05) -> np.ndarray:
    """Auto-detect background color by sampling image edges (robust to objects near borders)."""
    h, w = img_bgr.shape[:2]
    mh = max(10, int(h * margin_pct))
    mw = max(10, int(w * margin_pct))

    samples = np.vstack([
        img_bgr[:mh, :].reshape(-1, 3),
        img_bgr[-mh:, :].reshape(-1, 3),
        img_bgr[:, :mw].reshape(-1, 3),
        img_bgr[:, -mw:].reshape(-1, 3),
    ])

    bg_color = np.median(samples, axis=0).astype(np.uint8)
    log.info(f"  Background color (BGR): [{bg_color[0]}, {bg_color[1]}, {bg_color[2]}]")
    return bg_color


def compute_alpha(img_bgr: np.ndarray, bg_color: np.ndarray,
                  shadow_strength: float) -> np.ndarray:
    """
    Build alpha matte via LAB color distance from the background.

    Two adaptive thresholds split the distance range into three zones:
      [0, bg_thresh)           -> background  (alpha = 0)
      [bg_thresh, fg_thresh)   -> shadow/edge (alpha ramps 0 to 1)
      [fg_thresh, inf)         -> foreground  (alpha = 1)

    shadow_strength (0-1) widens the shadow zone by pulling bg_thresh lower.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    bg_lab = cv2.cvtColor(
        bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
    ).astype(np.float64).ravel()

    distance = np.sqrt(np.sum((img_lab - bg_lab) ** 2, axis=2))

    p5 = np.percentile(distance, 5)
    p95 = np.percentile(distance, 95)
    span = p95 - p5

    bg_thresh = p5 + span * (0.12 - 0.09 * shadow_strength)
    fg_thresh = p5 + span * (0.35 - 0.10 * shadow_strength)

    log.info(f"  LAB distance — p5={p5:.1f}  p95={p95:.1f}  "
             f"bg_thresh={bg_thresh:.1f}  fg_thresh={fg_thresh:.1f}")

    alpha = np.clip(
        (distance - bg_thresh) / (fg_thresh - bg_thresh + 1e-6), 0.0, 1.0
    )
    return alpha


def refine_alpha_hsv(img_bgr: np.ndarray, alpha: np.ndarray,
                     shadow_strength: float) -> np.ndarray:
    """
    HSV pass to sharpen the background/shadow boundary.

    Pure green-screen pixels (high saturation, green hue, bright) are forced
    to alpha 0.  Shadow pixels (green hue but lower S/V) keep partial alpha
    proportional to their desaturation.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    green_hue = (h_ch >= 30) & (h_ch <= 90)

    # Pure background: green hue + high saturation + bright
    pure_bg = green_hue & (s_ch > 80) & (v_ch > 80)
    greenness = np.where(pure_bg, s_ch / 255.0, 0.0)
    alpha[pure_bg] = np.minimum(alpha[pure_bg], 1.0 - greenness[pure_bg])

    # Shadow boost: green hue but desaturated/darker -> add partial alpha
    shadow_zone = green_hue & (s_ch < 180) & (v_ch < 200) & ~pure_bg
    desat = 1.0 - s_ch[shadow_zone] / 255.0
    alpha[shadow_zone] = np.clip(
        alpha[shadow_zone] + desat * shadow_strength * 0.5, 0.0, 1.0
    )

    return alpha


def despill_green(img_bgr: np.ndarray, alpha: np.ndarray,
                  strength: float) -> np.ndarray:
    """Suppress green-channel excess on edge / semi-transparent pixels."""
    if strength <= 0:
        return img_bgr.copy()

    out = img_bgr.astype(np.float64)
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]

    spill_weight = np.clip(1.0 - alpha, 0.0, 1.0) * strength
    green_excess = np.maximum(0.0, g - (r + b) / 2.0)
    out[:, :, 1] = np.clip(g - green_excess * spill_weight, 0, 255)

    return out.astype(np.uint8)


def soften_edges(alpha: np.ndarray, radius: float) -> np.ndarray:
    """Light Gaussian blur on edge pixels only (keeps hard fg/bg intact)."""
    if radius <= 0:
        return alpha

    ksize = max(3, int(radius * 2) * 2 + 1)
    blurred = cv2.GaussianBlur(alpha, (ksize, ksize), radius)

    edge = (alpha > 0.01) & (alpha < 0.99)
    out = alpha.copy()
    out[edge] = blurred[edge]
    return out


# ── Debug visualisation helpers ──────────────────────────────────────

def _make_checkerboard(h: int, w: int, sq: int = 16) -> np.ndarray:
    """Fast numpy checkerboard (light/dark gray)."""
    rows = np.arange(h) // sq
    cols = np.arange(w) // sq
    grid = (rows[:, None] + cols[None, :]) % 2
    board = np.where(grid[:, :, None] == 0, 200, 255).astype(np.uint8)
    return np.broadcast_to(board, (h, w, 3)).copy()


def save_debug(img_bgr: np.ndarray, img_rgb: np.ndarray,
               alpha: np.ndarray, alpha_u8: np.ndarray,
               output_path: Path):
    debug_dir = output_path.parent / "debug_vis"
    os.makedirs(debug_dir, exist_ok=True)
    stem = output_path.stem

    # Alpha as grayscale
    cv2.imwrite(str(debug_dir / f"{stem}_alpha.png"), alpha_u8)

    # Composite on checkerboard
    h, w = alpha.shape
    checker = _make_checkerboard(h, w)
    a3 = alpha[:, :, None]
    comp = (img_rgb.astype(np.float64) * a3
            + checker.astype(np.float64) * (1.0 - a3))
    cv2.imwrite(
        str(debug_dir / f"{stem}_checker.png"),
        cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    # Shadow highlight (orange tint on shadow pixels)
    shadow_vis = img_rgb.copy()
    smask = (alpha > 0.02) & (alpha < 0.5)
    shadow_vis[smask] = (
        shadow_vis[smask].astype(np.float64) * 0.5
        + np.array([255, 140, 0], dtype=np.float64) * 0.5
    ).astype(np.uint8)
    cv2.imwrite(
        str(debug_dir / f"{stem}_shadows.png"),
        cv2.cvtColor(shadow_vis, cv2.COLOR_RGB2BGR),
    )

    log.info(f"  Debug images -> {debug_dir}/")


# ── Per-image pipeline ───────────────────────────────────────────────

def process_image(img_path: Path, output_path: Path,
                  shadow_strength: float, despill_strength: float,
                  edge_softness: float, debug: bool = False):
    try:
        log.info(f"Processing: {img_path.name}")
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.error(f"  Could not read {img_path}")
            return

        h, w = img_bgr.shape[:2]
        log.info(f"  Size: {w}x{h}")

        bg_color = sample_background_color(img_bgr)
        alpha = compute_alpha(img_bgr, bg_color, shadow_strength)
        alpha = refine_alpha_hsv(img_bgr, alpha, shadow_strength)
        alpha = soften_edges(alpha, edge_softness)

        img_clean = despill_green(img_bgr, alpha, despill_strength)
        img_rgb = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)

        alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([img_rgb, alpha_u8])

        os.makedirs(output_path.parent, exist_ok=True)
        Image.fromarray(rgba).save(output_path)

        opaque = np.mean(alpha > 0.5) * 100
        shadow = np.mean((alpha > 0.01) & (alpha <= 0.5)) * 100
        log.info(f"  Saved {output_path.name} — "
                 f"{opaque:.1f}% opaque, {shadow:.1f}% shadow")

        if debug:
            save_debug(img_bgr, img_rgb, alpha, alpha_u8, output_path)

    except Exception as e:
        log.error(f"  Failed {img_path.name}: {e}", exc_info=True)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Green screen chroma key extractor with shadow preservation",
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory with green screen images")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for RGBA PNGs")
    p.add_argument("--debug", action="store_true",
                   help="Save alpha, checkerboard, and shadow debug images")
    p.add_argument("--shadow-strength", type=float,
                   default=DEFAULT_SHADOW_STRENGTH,
                   help=f"Shadow capture intensity 0.0-1.0 (default {DEFAULT_SHADOW_STRENGTH})")
    p.add_argument("--despill-strength", type=float,
                   default=DEFAULT_DESPILL_STRENGTH,
                   help=f"Green spill removal 0.0-1.0 (default {DEFAULT_DESPILL_STRENGTH})")
    p.add_argument("--edge-softness", type=float,
                   default=DEFAULT_EDGE_SOFTNESS,
                   help=f"Alpha edge blur radius (default {DEFAULT_EDGE_SOFTNESS})")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        log.error(f"Input directory not found: {in_dir}")
        sys.exit(1)

    files = sorted(f for f in in_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        log.error(f"No images found in {in_dir}")
        sys.exit(1)

    log.info(f"Found {len(files)} image(s) in {in_dir}")
    log.info(f"Settings: shadow={args.shadow_strength}  "
             f"despill={args.despill_strength}  softness={args.edge_softness}")

    out_dir = Path(args.output_dir)

    for f in files:
        process_image(
            f, out_dir / f"{f.stem}.png",
            shadow_strength=args.shadow_strength,
            despill_strength=args.despill_strength,
            edge_softness=args.edge_softness,
            debug=args.debug,
        )

    log.info("Done.")

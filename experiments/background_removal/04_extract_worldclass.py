#!/usr/bin/env python3
"""
04_extract_worldclass.py — Two-Stage Professional Background Removal

Pipeline:
  Stage 1: BiRefNet (or RMBG-2.0) → high-quality binary mask
  Stage 2: ViTMatte (Vision Transformer) → refined alpha matte with shadows
  Post:    Green despill (auto-detected) → clean RGBA output

Works on any background: green screen, white, complex scenes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"
RMBG_MODEL_ID = "briaai/RMBG-2.0"
VITMATTE_MODEL_ID = "hustvl/vitmatte-small-distinctions-646"

SEG_INPUT_SIZE = (1024, 1024)
SEG_MEAN = [0.485, 0.456, 0.406]
SEG_STD = [0.229, 0.224, 0.225]

DEFAULT_SHADOW_RADIUS = 40
DEFAULT_ERODE_RADIUS = 5
DEFAULT_DESPILL_STRENGTH = 0.7


# ── Model loaders (singleton) ───────────────────────────────────────

class SegModelLoader:
    _model = None
    _transform = None
    _name = None

    @classmethod
    def get(cls, device: str, prefer: str | None = None):
        if cls._model is not None:
            return cls._model, cls._transform

        from transformers import AutoModelForImageSegmentation

        candidates = [prefer] if prefer else [BIREFNET_MODEL_ID, RMBG_MODEL_ID]

        for model_id in candidates:
            try:
                log.info(f"Loading segmentation model: {model_id} ...")
                cls._model = (
                    AutoModelForImageSegmentation
                    .from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                    )
                    .float()
                    .eval()
                    .to(device)
                )
                cls._name = model_id
                cls._transform = transforms.Compose([
                    transforms.Resize(SEG_INPUT_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(SEG_MEAN, SEG_STD),
                ])
                log.info(f"  {model_id} loaded successfully.")
                return cls._model, cls._transform
            except Exception as e:
                log.warning(f"  Could not load {model_id}: {e}")

        raise RuntimeError(
            "No segmentation model could be loaded. "
            "Try: pip install timm safetensors transformers"
        )


class ViTMatteLoader:
    _model = None
    _processor = None

    @classmethod
    def get(cls, device: str):
        if cls._model is None:
            log.info(f"Loading ViTMatte ({VITMATTE_MODEL_ID})...")
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            cls._processor = VitMatteImageProcessor.from_pretrained(VITMATTE_MODEL_ID)
            cls._model = (
                VitMatteForImageMatting
                .from_pretrained(VITMATTE_MODEL_ID)
                .eval()
                .to(device)
            )
            log.info("  ViTMatte loaded.")
        return cls._model, cls._processor


# ── Stage 1: Segmentation (BiRefNet / RMBG-2.0) ─────────────────────

def run_segmentation(img_pil: Image.Image, device: str,
                     model_id: str | None = None) -> np.ndarray:
    """Run segmentation model and return a float32 mask in [0, 1] at original resolution."""
    model, transform = SegModelLoader.get(device, prefer=model_id)
    orig_size = img_pil.size  # (W, H)

    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)[-1].sigmoid().cpu()

    mask_pil = transforms.ToPILImage()(pred[0].squeeze())
    mask_pil = mask_pil.resize(orig_size, Image.BILINEAR)
    return np.array(mask_pil).astype(np.float32) / 255.0


# ── Stage 2: Trimap generation + ViTMatte ────────────────────────────

def make_trimap(mask: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
    """
    Convert a soft mask to a trimap with three zones:
      255 = definite foreground (eroded mask)
      128 = unknown (between erode and dilate — shadows/edges live here)
        0 = definite background (outside dilated mask)
    """
    binary = (mask > 0.5).astype(np.uint8)

    kernel_e = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    kernel_d = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
    )

    fg = cv2.erode(binary, kernel_e)
    outer = cv2.dilate(binary, kernel_d)

    trimap = np.zeros_like(binary, dtype=np.uint8)
    trimap[outer == 1] = 128
    trimap[fg == 1] = 255

    return trimap


def run_vitmatte(img_pil: Image.Image, trimap: np.ndarray,
                 device: str) -> np.ndarray:
    """Run ViTMatte and return a float64 alpha matte in [0, 1]."""
    model, processor = ViTMatteLoader.get(device)

    trimap_pil = Image.fromarray(trimap)
    inputs = processor(images=img_pil, trimaps=trimap_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    alpha = output.alphas[0, 0].cpu().numpy()

    # ViTMatte may pad the image; crop back to original size
    h, w = np.array(img_pil).shape[:2]
    alpha = alpha[:h, :w]

    return np.clip(alpha, 0.0, 1.0).astype(np.float64)


# ── Post-processing ─────────────────────────────────────────────────

def detect_green_screen(img_bgr: np.ndarray, margin_pct: float = 0.05) -> bool:
    """Check if the image background is a green screen."""
    h, w = img_bgr.shape[:2]
    mh, mw = max(10, int(h * margin_pct)), max(10, int(w * margin_pct))

    samples = np.vstack([
        img_bgr[:mh, :].reshape(-1, 3),
        img_bgr[-mh:, :].reshape(-1, 3),
        img_bgr[:, :mw].reshape(-1, 3),
        img_bgr[:, -mw:].reshape(-1, 3),
    ])
    bg_color = np.median(samples, axis=0).astype(np.uint8)
    bg_hsv = cv2.cvtColor(
        bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2HSV
    ).ravel()

    return bg_hsv[1] > 50 and 30 <= bg_hsv[0] <= 90


def despill_green(img_bgr: np.ndarray, alpha: np.ndarray,
                  strength: float) -> np.ndarray:
    """Suppress green-channel excess on semi-transparent edge pixels."""
    if strength <= 0:
        return img_bgr.copy()

    out = img_bgr.astype(np.float64)
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]

    spill_weight = np.clip(1.0 - alpha, 0.0, 1.0) * strength
    green_excess = np.maximum(0.0, g - (r + b) / 2.0)
    out[:, :, 1] = np.clip(g - green_excess * spill_weight, 0, 255)

    return out.astype(np.uint8)


# ── Debug visualisation ─────────────────────────────────────────────

def _make_checkerboard(h: int, w: int, sq: int = 16) -> np.ndarray:
    rows = np.arange(h) // sq
    cols = np.arange(w) // sq
    grid = (rows[:, None] + cols[None, :]) % 2
    board = np.where(grid[:, :, None] == 0, 200, 255).astype(np.uint8)
    return np.broadcast_to(board, (h, w, 3)).copy()


def save_debug(img_rgb: np.ndarray, mask: np.ndarray, trimap: np.ndarray,
               alpha: np.ndarray, output_path: Path):
    debug_dir = output_path.parent / "debug_vis"
    os.makedirs(debug_dir, exist_ok=True)
    stem = output_path.stem

    # 1. Binary mask from RMBG
    cv2.imwrite(
        str(debug_dir / f"{stem}_1_mask.png"),
        (mask * 255).astype(np.uint8),
    )

    # 2. Trimap
    cv2.imwrite(str(debug_dir / f"{stem}_2_trimap.png"), trimap)

    # 3. Final alpha
    alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{stem}_3_alpha.png"), alpha_u8)

    # 4. Checkerboard composite
    h, w = alpha.shape
    checker = _make_checkerboard(h, w)
    a3 = alpha[:, :, None]
    comp = (img_rgb.astype(np.float64) * a3
            + checker.astype(np.float64) * (1.0 - a3))
    cv2.imwrite(
        str(debug_dir / f"{stem}_4_checker.png"),
        cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    # 5. Shadow highlight (orange tint on shadow-alpha pixels)
    shadow_vis = img_rgb.copy()
    smask = (alpha > 0.02) & (alpha < 0.5)
    shadow_vis[smask] = (
        shadow_vis[smask].astype(np.float64) * 0.5
        + np.array([255, 140, 0], dtype=np.float64) * 0.5
    ).astype(np.uint8)
    cv2.imwrite(
        str(debug_dir / f"{stem}_5_shadows.png"),
        cv2.cvtColor(shadow_vis, cv2.COLOR_RGB2BGR),
    )

    log.info(f"  Debug images -> {debug_dir}/")


# ── Per-image pipeline ───────────────────────────────────────────────

def process_image(img_path: Path, output_path: Path, device: str,
                  shadow_radius: int, erode_radius: int,
                  despill_strength: float,
                  no_matting: bool, debug: bool,
                  seg_model: str | None = None):
    try:
        log.info(f"Processing: {img_path.name}")
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        log.info(f"  Size: {w}x{h}")

        # ── Stage 1: Segmentation ────────────────────────────────────
        mask = run_segmentation(img_pil, device, model_id=seg_model)
        log.info(f"  Segmentation mask: {np.mean(mask > 0.5) * 100:.1f}% foreground")

        if no_matting:
            alpha = mask.astype(np.float64)
            trimap = np.zeros_like(mask, dtype=np.uint8)
        else:
            # ── Stage 2: Trimap + ViTMatte ───────────────────────────
            trimap = make_trimap(mask, erode_radius, shadow_radius)
            unknown_pct = np.mean(trimap == 128) * 100
            log.info(f"  Trimap: {unknown_pct:.1f}% unknown zone "
                     f"(erode={erode_radius}, dilate={shadow_radius})")

            alpha = run_vitmatte(img_pil, trimap, device)

        # ── Post-processing ──────────────────────────────────────────
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        is_green = detect_green_screen(img_bgr)
        if is_green:
            log.info("  Green screen detected — applying despill")
            img_bgr = despill_green(img_bgr, alpha, despill_strength)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([img_rgb, alpha_u8])

        os.makedirs(output_path.parent, exist_ok=True)
        Image.fromarray(rgba).save(output_path)

        opaque = np.mean(alpha > 0.5) * 100
        shadow = np.mean((alpha > 0.01) & (alpha <= 0.5)) * 100
        log.info(f"  Saved {output_path.name} — "
                 f"{opaque:.1f}% opaque, {shadow:.1f}% shadow")

        if debug:
            save_debug(img_rgb, mask, trimap, alpha, output_path)

    except Exception as e:
        log.error(f"  Failed {img_path.name}: {e}", exc_info=True)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="World-class background removal: BiRefNet + ViTMatte",
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory with input images")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for RGBA PNGs")
    p.add_argument("--debug", action="store_true",
                   help="Save mask, trimap, alpha, checkerboard, shadow debug images")
    p.add_argument("--shadow-radius", type=int,
                   default=DEFAULT_SHADOW_RADIUS,
                   help=f"Dilation radius for shadow capture zone in px "
                        f"(default {DEFAULT_SHADOW_RADIUS})")
    p.add_argument("--erode-radius", type=int,
                   default=DEFAULT_ERODE_RADIUS,
                   help=f"Erosion radius for definite foreground in px "
                        f"(default {DEFAULT_ERODE_RADIUS})")
    p.add_argument("--despill-strength", type=float,
                   default=DEFAULT_DESPILL_STRENGTH,
                   help=f"Green spill removal strength 0.0-1.0 "
                        f"(default {DEFAULT_DESPILL_STRENGTH})")
    p.add_argument("--no-matting", action="store_true",
                   help="Skip ViTMatte; use segmentation mask directly (faster, no shadows)")
    p.add_argument("--seg-model", default=None,
                   help=f"HuggingFace model ID for segmentation "
                        f"(default: tries {BIREFNET_MODEL_ID}, then {RMBG_MODEL_ID})")
    p.add_argument("--device", default=None,
                   help="Device override (default: auto-detect cuda/mps/cpu)")
    args = p.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    log.info(f"Device: {device}")

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        log.error(f"Input directory not found: {in_dir}")
        sys.exit(1)

    files = sorted(f for f in in_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        log.error(f"No images found in {in_dir}")
        sys.exit(1)

    log.info(f"Found {len(files)} image(s) in {in_dir}")
    log.info(f"Settings: shadow_radius={args.shadow_radius}  "
             f"erode_radius={args.erode_radius}  "
             f"despill={args.despill_strength}  "
             f"matting={'OFF' if args.no_matting else 'ON'}")

    out_dir = Path(args.output_dir)

    for f in files:
        process_image(
            f, out_dir / f"{f.stem}.png",
            device=device,
            shadow_radius=args.shadow_radius,
            erode_radius=args.erode_radius,
            despill_strength=args.despill_strength,
            no_matting=args.no_matting,
            debug=args.debug,
            seg_model=args.seg_model,
        )

    log.info("Done.")

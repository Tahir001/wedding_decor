#!/usr/bin/env python3
"""
background_remover_main_v2.py — Production Background Removal v2

Same core pipeline as v1 (RMBG-2.0 → ViTMatte → clamped alpha →
foreground decontamination → green despill → sharpen), plus three
new stages focused on making the RGBA cutout itself as clean as
possible — eliminating halos, jaggies, and edge color bleed.

  Stage 5: Alpha refinement — morphological cleanup + median
           anti-aliasing to remove alpha noise and single-pixel jaggies
  Stage 6: Edge defringing — propagate confident interior foreground
           colors outward to replace background-contaminated edge colors
  Stage 7: Edge feathering — edge-aware Gaussian smoothing on alpha
           transitions for natural anti-aliased edges

Output: clean RGBA PNGs with transparent backgrounds, optimized for
compositing onto any scene.
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

RMBG_MODEL_ID = "briaai/RMBG-2.0"
BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"
VITMATTE_MODEL_ID = "hustvl/vitmatte-base-distinctions-646"

SEG_INPUT_SIZE = (1024, 1024)
SEG_MEAN = [0.485, 0.456, 0.406]
SEG_STD = [0.229, 0.224, 0.225]

DEFAULT_SHADOW_RADIUS = 20
DEFAULT_ERODE_RADIUS = 5
DEFAULT_DESPILL_STRENGTH = 0.7
DEFAULT_SHARPEN_AMOUNT = 0.7
DEFAULT_CONTRAST_STRENGTH = 1.0
DEFAULT_SATURATION_BOOST = 1.0

DEFAULT_ALPHA_CLEANUP_SIZE = 3
DEFAULT_DEFRINGE_RADIUS = 15
DEFAULT_DEFRINGE_STRENGTH = 0.8
DEFAULT_EDGE_FEATHER = 1.2


# ── Model loaders ───────────────────────────────────────────────────

class SegModelLoader:
    _model = None
    _transform = None
    _name = None

    @classmethod
    def get(cls, device: str, prefer: str | None = None,
            hf_token: str | None = None):
        if cls._model is not None:
            return cls._model, cls._transform

        from transformers import AutoModelForImageSegmentation

        model_id = prefer or RMBG_MODEL_ID

        log.info(f"Loading segmentation model: {model_id} ...")
        load_kwargs = dict(
            trust_remote_code=True,
            token=hf_token,
        )

        strategies = [
            {},
            {"low_cpu_mem_usage": False, "device_map": None},
            {"low_cpu_mem_usage": False, "device_map": None,
             "torch_dtype": torch.float32},
        ]

        last_err = None
        for extra in strategies:
            try:
                cls._model = (
                    AutoModelForImageSegmentation
                    .from_pretrained(model_id, **load_kwargs, **extra)
                    .float()
                    .eval()
                    .to(device)
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            log.warning(f"  All standard strategies failed for {model_id}, "
                        f"retrying with force_download=True to bust stale cache...")
            try:
                cls._model = (
                    AutoModelForImageSegmentation
                    .from_pretrained(
                        model_id, **load_kwargs,
                        low_cpu_mem_usage=False, device_map=None,
                        force_download=True,
                    )
                    .float()
                    .eval()
                    .to(device)
                )
                last_err = None
            except Exception as e:
                last_err = e

        if last_err is not None:
            import transformers
            raise RuntimeError(
                f"Could not load {model_id}.\n"
                f"  Error: {last_err}\n"
                f"  transformers version: {transformers.__version__}\n\n"
                f"  This is a known transformers version incompatibility. Try:\n"
                f"    1. pip install transformers==4.48.3\n"
                f"    2. Clear HF cache: rm -rf ~/.cache/huggingface/modules/"
                f"transformers_modules/briaai/\n"
                f"    3. Or use --seg-model ZhengPeng7/BiRefNet as an alternative"
            )

        cls._name = model_id
        cls._transform = transforms.Compose([
            transforms.Resize(SEG_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(SEG_MEAN, SEG_STD),
        ])
        log.info(f"  {model_id} loaded successfully.")
        return cls._model, cls._transform


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


# ── Stage 1: Segmentation ───────────────────────────────────────────

def run_segmentation(img_pil: Image.Image, device: str,
                     model_id: str | None = None,
                     hf_token: str | None = None) -> np.ndarray:
    model, transform = SegModelLoader.get(device, prefer=model_id,
                                          hf_token=hf_token)
    orig_size = img_pil.size

    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)[-1].sigmoid().cpu()

    mask_pil = transforms.ToPILImage()(pred[0].squeeze())
    mask_pil = mask_pil.resize(orig_size, Image.BILINEAR)
    return np.array(mask_pil).astype(np.float32) / 255.0


# ── Stage 2: Trimap + ViTMatte ───────────────────────────────────────

def make_trimap(mask: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
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
    model, processor = ViTMatteLoader.get(device)

    trimap_pil = Image.fromarray(trimap)
    inputs = processor(images=img_pil, trimaps=trimap_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    alpha = output.alphas[0, 0].cpu().numpy()
    h, w = np.array(img_pil).shape[:2]
    alpha = alpha[:h, :w]
    return np.clip(alpha, 0.0, 1.0).astype(np.float64)


# ── Alpha clamping ──────────────────────────────────────────────────

def clamp_alpha_to_trimap(alpha: np.ndarray,
                          trimap: np.ndarray) -> np.ndarray:
    out = alpha.copy()
    out[trimap == 255] = 1.0
    out[trimap == 0] = 0.0
    return out


# ── Foreground estimation ────────────────────────────────────────────

def estimate_background_color(img_rgb: np.ndarray, alpha: np.ndarray,
                              trimap: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    bg_mask = (trimap == 0) & (alpha < 0.05)

    if np.sum(bg_mask) < 100:
        m = max(10, min(h, w) // 20)
        border = np.zeros((h, w), dtype=bool)
        border[:m, :] = True
        border[-m:, :] = True
        border[:, :m] = True
        border[:, -m:] = True
        bg_mask = border

    bg_pixels = img_rgb[bg_mask].astype(np.float64)
    global_bg = np.median(bg_pixels, axis=0)
    bg_map = np.full_like(img_rgb, global_bg, dtype=np.float64)

    bg_float = np.where(bg_mask[:, :, None], img_rgb.astype(np.float64), 0.0)
    bg_count = bg_mask.astype(np.float64)

    ksize = min(h, w) // 4
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    ksize = max(ksize, 31)

    bg_sum = cv2.GaussianBlur(bg_float, (ksize, ksize), 0)
    bg_cnt = cv2.GaussianBlur(bg_count, (ksize, ksize), 0)

    valid = bg_cnt > 0.001
    for c in range(3):
        bg_map[:, :, c] = np.where(valid, bg_sum[:, :, c] / bg_cnt, global_bg[c])

    log.info(f"  Background color (median): "
             f"R={global_bg[0]:.0f} G={global_bg[1]:.0f} B={global_bg[2]:.0f}")
    return bg_map


def estimate_foreground(img_rgb: np.ndarray, alpha: np.ndarray,
                        bg_map: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float64)
    a = alpha[:, :, np.newaxis]
    a_safe = np.maximum(a, 1e-3)

    fg = (img - (1.0 - a) * bg_map) / a_safe
    fg = np.clip(fg, 0.0, 255.0)

    confidence = np.clip(alpha, 0.0, 1.0)
    fg_weighted = fg * confidence[:, :, np.newaxis]

    ksize = 15
    conf_blur = cv2.GaussianBlur(confidence, (ksize, ksize), 0)
    conf_blur = np.maximum(conf_blur, 1e-6)

    fg_stable = np.zeros_like(fg)
    for c in range(3):
        fg_stable[:, :, c] = (
            cv2.GaussianBlur(fg_weighted[:, :, c], (ksize, ksize), 0)
            / conf_blur
        )

    blend = np.clip((confidence - 0.05) / 0.25, 0.0, 1.0)[:, :, np.newaxis]
    fg_final = fg * blend + fg_stable * (1.0 - blend)

    return np.clip(fg_final, 0.0, 255.0).astype(np.uint8)


# ── Post-processing (v1) ────────────────────────────────────────────

def detect_green_screen(img_bgr: np.ndarray, margin_pct: float = 0.05) -> bool:
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


def despill_green(img_rgb: np.ndarray, alpha: np.ndarray,
                  strength: float) -> np.ndarray:
    if strength <= 0:
        return img_rgb.copy()

    out = img_rgb.astype(np.float64)
    r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]

    green_hue = ((hue >= 35) & (hue <= 85)).astype(np.float64)
    saturated = (sat > 30).astype(np.float64)
    hue_weight = green_hue * saturated

    spill_weight = np.clip(1.0 - alpha, 0.0, 1.0) * strength * hue_weight
    green_excess = np.maximum(0.0, g - (r + b) / 2.0)
    out[:, :, 1] = np.clip(g - green_excess * spill_weight, 0, 255)
    return out.astype(np.uint8)


def sharpen_foreground(img_rgb: np.ndarray, alpha: np.ndarray,
                       amount: float) -> np.ndarray:
    if amount <= 0:
        return img_rgb
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(img_rgb, 1.0 + amount, blurred, -amount, 0)
    mask3 = (alpha > 0.5)[:, :, np.newaxis].astype(np.float64)
    out = img_rgb.astype(np.float64) * (1.0 - mask3) + sharpened.astype(np.float64) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_contrast(img_rgb: np.ndarray, alpha: np.ndarray,
                     strength: float) -> np.ndarray:
    if strength <= 1.0:
        return img_rgb
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]

    clip_limit = 1.0 + (strength - 1.0) * 5.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)

    mask = (alpha > 0.5).astype(np.uint8)
    lab[:, :, 0] = np.where(mask, l_enhanced, l_chan)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def boost_saturation(img_rgb: np.ndarray, alpha: np.ndarray,
                     factor: float) -> np.ndarray:
    if factor <= 1.0:
        return img_rgb
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    mask = (alpha > 0.5)
    hsv[:, :, 1] = np.where(mask, np.clip(hsv[:, :, 1] * factor, 0, 255), hsv[:, :, 1])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ═══════════════════════════════════════════════════════════════════════
# v2 ENHANCEMENTS — Clean RGBA Cutout
# ═══════════════════════════════════════════════════════════════════════

# ── Stage 5: Alpha Refinement ────────────────────────────────────────

def refine_alpha(alpha: np.ndarray, morph_size: int) -> np.ndarray:
    """
    Clean up alpha channel noise and anti-alias jagged edges.

    1. Morphological CLOSE fills tiny holes inside the foreground
       (single-pixel dropouts from matting)
    2. Morphological OPEN removes tiny foreground specks in the
       background (isolated noise pixels)
    3. Median filter in the edge zone kills single-pixel jaggies
       while preserving the overall edge shape
    """
    if morph_size <= 0:
        return alpha

    alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_size, morph_size)
    )
    closed = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    edge_band = (cleaned > 5) & (cleaned < 250)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_zone = cv2.dilate(
        edge_band.astype(np.uint8), dilate_k
    ).astype(bool)

    median = cv2.medianBlur(cleaned, 3)
    result = cleaned.copy()
    result[edge_zone] = median[edge_zone]

    holes_filled = int(np.sum((alpha_u8 < 128) & (result >= 128)))
    specks_removed = int(np.sum((alpha_u8 >= 128) & (result < 128)))
    log.info(f"  Alpha refinement: {holes_filled} hole px filled, "
             f"{specks_removed} speck px removed (morph={morph_size})")

    return result.astype(np.float64) / 255.0


# ── Stage 6: Edge Defringing ─────────────────────────────────────────

def defringe_edges(fg_rgb: np.ndarray, alpha: np.ndarray,
                   radius: int, strength: float) -> np.ndarray:
    """
    Remove background color contamination from edge pixels.

    Even after foreground estimation, edge pixels often retain a faint
    halo of the original background color (white fringe on white bg,
    green tint on green screen). This is the #1 visual difference
    between our output and ChatGPT's.

    Fix: build a "clean interior color map" by blurring only the
    high-confidence foreground pixels, then blend edge pixel colors
    toward this clean reference. The blend strength scales with how
    close to the edge the pixel is (lower alpha = more correction).
    """
    if radius <= 0 or strength <= 0:
        return fg_rgb

    fg = fg_rgb.astype(np.float64)
    conf = np.clip(alpha, 0.0, 1.0)

    high_conf = (conf > 0.9).astype(np.float64)
    fg_weighted = fg * high_conf[:, :, np.newaxis]

    ksize = radius * 2 + 1
    conf_blur = cv2.GaussianBlur(high_conf, (ksize, ksize), 0)
    conf_blur = np.maximum(conf_blur, 1e-6)

    clean_color = np.zeros_like(fg)
    for c in range(3):
        clean_color[:, :, c] = (
            cv2.GaussianBlur(fg_weighted[:, :, c], (ksize, ksize), 0)
            / conf_blur
        )

    edge_mask = ((conf > 0.02) & (conf < 0.9)).astype(np.float64)
    blend = np.clip((0.9 - conf) / 0.85, 0.0, 1.0) * edge_mask * strength

    out = fg * (1.0 - blend[:, :, np.newaxis]) + \
          clean_color * blend[:, :, np.newaxis]

    n_defringed = int(np.sum(edge_mask > 0))
    avg_blend = float(np.mean(blend[blend > 0])) if n_defringed > 0 else 0
    log.info(f"  Edge defringing: {n_defringed} edge px corrected, "
             f"avg blend={avg_blend:.2f} (radius={radius}, "
             f"strength={strength:.1f})")

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Stage 7: Edge Feathering ─────────────────────────────────────────

def feather_alpha_edges(alpha: np.ndarray, radius: float) -> np.ndarray:
    """
    Smooth alpha transitions at object edges for natural anti-aliasing.

    Detects the narrow band where alpha transitions between 0 and 1,
    applies targeted Gaussian blur only in that band. Interior (alpha=1)
    and exterior (alpha=0) are completely untouched — this only affects
    the edge transition profile, making it smoother and more natural
    for compositing onto any background.
    """
    if radius <= 0:
        return alpha

    edge_band = (alpha > 0.02) & (alpha < 0.98)
    dilate_k = max(3, int(radius * 4) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)
    )
    edge_zone = cv2.dilate(
        edge_band.astype(np.uint8), kernel
    ).astype(bool)

    sigma = max(0.5, radius)
    ksize = int(sigma * 6) | 1
    ksize = max(ksize, 3)
    smoothed = cv2.GaussianBlur(alpha, (ksize, ksize), sigma)

    out = alpha.copy()
    out[edge_zone] = smoothed[edge_zone]
    out = np.clip(out, 0.0, 1.0)

    n_feathered = int(np.sum(edge_zone))
    log.info(f"  Edge feathering: {n_feathered} pixels smoothed "
             f"(radius={radius:.1f})")
    return out


# ── Debug ────────────────────────────────────────────────────────────

def _make_checkerboard(h: int, w: int, sq: int = 16) -> np.ndarray:
    rows = np.arange(h) // sq
    cols = np.arange(w) // sq
    grid = (rows[:, None] + cols[None, :]) % 2
    board = np.where(grid[:, :, None] == 0, 200, 255).astype(np.uint8)
    return np.broadcast_to(board, (h, w, 3)).copy()


def save_debug(img_rgb: np.ndarray, fg_rgb: np.ndarray,
               mask: np.ndarray, trimap: np.ndarray,
               alpha: np.ndarray, output_path: Path):
    debug_dir = output_path.parent / "debug_vis"
    os.makedirs(debug_dir, exist_ok=True)
    stem = output_path.stem

    cv2.imwrite(str(debug_dir / f"{stem}_1_mask.png"),
                (mask * 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{stem}_2_trimap.png"), trimap)
    cv2.imwrite(str(debug_dir / f"{stem}_3_alpha.png"),
                (np.clip(alpha, 0, 1) * 255).astype(np.uint8))

    h, w = alpha.shape
    a3 = alpha[:, :, None]

    checker = _make_checkerboard(h, w)
    comp = (fg_rgb.astype(np.float64) * a3
            + checker.astype(np.float64) * (1.0 - a3))
    cv2.imwrite(str(debug_dir / f"{stem}_4a_checker.png"),
                cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR))

    white = np.full_like(checker, 255, dtype=np.float64)
    comp_w = (fg_rgb.astype(np.float64) * a3 + white * (1.0 - a3))
    cv2.imwrite(str(debug_dir / f"{stem}_4b_on_white.png"),
                cv2.cvtColor(comp_w.astype(np.uint8), cv2.COLOR_RGB2BGR))

    black = np.zeros_like(checker, dtype=np.float64)
    comp_b = (fg_rgb.astype(np.float64) * a3 + black * (1.0 - a3))
    cv2.imwrite(str(debug_dir / f"{stem}_4c_on_black.png"),
                cv2.cvtColor(comp_b.astype(np.uint8), cv2.COLOR_RGB2BGR))

    log.info(f"  Debug images -> {debug_dir}/")


# ── Per-image pipeline ───────────────────────────────────────────────

def process_image(img_path: Path, output_path: Path, device: str,
                  shadow_radius: int, erode_radius: int,
                  despill_strength: float,
                  no_matting: bool, debug: bool,
                  seg_model: str | None = None,
                  hf_token: str | None = None,
                  sharpen: float = DEFAULT_SHARPEN_AMOUNT,
                  contrast: float = DEFAULT_CONTRAST_STRENGTH,
                  saturation: float = DEFAULT_SATURATION_BOOST,
                  alpha_cleanup: int = DEFAULT_ALPHA_CLEANUP_SIZE,
                  defringe_radius: int = DEFAULT_DEFRINGE_RADIUS,
                  defringe_strength: float = DEFAULT_DEFRINGE_STRENGTH,
                  edge_feather: float = DEFAULT_EDGE_FEATHER):
    try:
        log.info(f"Processing: {img_path.name}")
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        log.info(f"  Size: {w}x{h}")

        # ── Stage 1: Segmentation ────────────────────────────────────
        mask = run_segmentation(img_pil, device, model_id=seg_model,
                                hf_token=hf_token)
        log.info(f"  Segmentation: "
                 f"{np.mean(mask > 0.5) * 100:.1f}% foreground")

        if no_matting:
            alpha = mask.astype(np.float64)
            trimap = np.zeros_like(mask, dtype=np.uint8)
            alpha[alpha > 0.95] = 1.0
            alpha[alpha < 0.05] = 0.0
        else:
            # ── Stage 2: ViTMatte ────────────────────────────────────
            trimap = make_trimap(mask, erode_radius, shadow_radius)
            unknown_pct = np.mean(trimap == 128) * 100
            log.info(f"  Trimap: {unknown_pct:.1f}% unknown "
                     f"(erode={erode_radius}, dilate={shadow_radius})")

            alpha = run_vitmatte(img_pil, trimap, device)
            alpha = clamp_alpha_to_trimap(alpha, trimap)

            clamped_fg = np.sum(trimap == 255)
            clamped_bg = np.sum(trimap == 0)
            edge_zone = np.sum(trimap == 128)
            log.info(f"  Alpha clamped: {clamped_fg} fg px -> 1.0, "
                     f"{clamped_bg} bg px -> 0.0, "
                     f"{edge_zone} edge px from ViTMatte")

        # ── Stage 3: Foreground estimation (edge zone only) ──────────
        bg_map = estimate_background_color(img_np, alpha, trimap)
        fg_rgb = estimate_foreground(img_np, alpha, bg_map)

        touched = np.sum((alpha > 0.001) & (alpha < 0.999))
        log.info(f"  Foreground estimation: {touched} edge pixels "
                 f"decontaminated")

        # ── Post: Green despill ──────────────────────────────────────
        is_green = detect_green_screen(
            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        )
        if is_green:
            log.info("  Green screen detected — hue-aware despill")
            fg_rgb = despill_green(fg_rgb, alpha, despill_strength)

        # ── Stage 4: Enhancement ─────────────────────────────────────
        if sharpen > 0:
            fg_rgb = sharpen_foreground(fg_rgb, alpha, sharpen)
            log.info(f"  Sharpening: amount={sharpen:.2f}")
        if contrast > 1.0:
            fg_rgb = enhance_contrast(fg_rgb, alpha, contrast)
            log.info(f"  Contrast: CLAHE strength={contrast:.2f}")
        if saturation > 1.0:
            fg_rgb = boost_saturation(fg_rgb, alpha, saturation)
            log.info(f"  Saturation: boost={saturation:.2f}")

        # ── Stage 5: Alpha refinement (v2) ───────────────────────────
        alpha = refine_alpha(alpha, alpha_cleanup)

        # ── Stage 6: Edge defringing (v2) ────────────────────────────
        fg_rgb = defringe_edges(fg_rgb, alpha, defringe_radius,
                                defringe_strength)

        # ── Stage 7: Edge feathering (v2) ────────────────────────────
        alpha = feather_alpha_edges(alpha, edge_feather)

        # ── Save RGBA ─────────────────────────────────────────────────
        alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([fg_rgb, alpha_u8])

        os.makedirs(output_path.parent, exist_ok=True)
        Image.fromarray(rgba).save(output_path)

        opaque = np.mean(alpha > 0.5) * 100
        semi = np.mean((alpha > 0.01) & (alpha <= 0.5)) * 100
        log.info(f"  Saved {output_path.name} — "
                 f"{opaque:.1f}% opaque, {semi:.1f}% semi-transparent")

        if debug:
            save_debug(img_np, fg_rgb, mask, trimap, alpha, output_path)

    except Exception as e:
        log.error(f"  Failed {img_path.name}: {e}", exc_info=True)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Production background removal v2: "
                    "v1 pipeline + alpha refinement + edge defringing "
                    "+ edge feathering for clean RGBA cutouts",
    )
    p.add_argument("--input-dir",
                   default="experiments/background_removal/plates")
    p.add_argument("--output-dir",
                   default="experiments/background_removal/outputs/bg_remover_v2")
    p.add_argument("--debug", action="store_true", default=True)
    p.add_argument("--shadow-radius", type=int,
                   default=DEFAULT_SHADOW_RADIUS)
    p.add_argument("--erode-radius", type=int,
                   default=DEFAULT_ERODE_RADIUS)
    p.add_argument("--despill-strength", type=float,
                   default=DEFAULT_DESPILL_STRENGTH)
    p.add_argument("--no-matting", action="store_true",
                   help="Use RMBG-2.0's native alpha directly, no ViTMatte")
    p.add_argument("--seg-model", default=None,
                   help="Override segmentation model HF ID")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for gated models (RMBG-2.0)")
    p.add_argument("--sharpen", type=float,
                   default=DEFAULT_SHARPEN_AMOUNT,
                   help="Unsharp mask amount (0=off)")
    p.add_argument("--contrast", type=float,
                   default=DEFAULT_CONTRAST_STRENGTH,
                   help="CLAHE contrast strength (1.0=off)")
    p.add_argument("--saturation", type=float,
                   default=DEFAULT_SATURATION_BOOST,
                   help="Saturation boost factor (1.0=off)")
    p.add_argument("--device", default="cpu")

    # v2 cutout quality enhancements
    p.add_argument("--alpha-cleanup", type=int,
                   default=DEFAULT_ALPHA_CLEANUP_SIZE,
                   help="Morphological kernel size for alpha cleanup (0=off)")
    p.add_argument("--defringe-radius", type=int,
                   default=DEFAULT_DEFRINGE_RADIUS,
                   help="Radius for interior color propagation to kill halos")
    p.add_argument("--defringe-strength", type=float,
                   default=DEFAULT_DEFRINGE_STRENGTH,
                   help="How aggressively to replace edge colors (0-1)")
    p.add_argument("--edge-feather", type=float,
                   default=DEFAULT_EDGE_FEATHER,
                   help="Alpha edge smoothing radius in px (0=off)")
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

    files = sorted(
        f for f in in_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        log.error(f"No images found in {in_dir}")
        sys.exit(1)

    seg_name = args.seg_model or RMBG_MODEL_ID
    log.info(f"Found {len(files)} image(s) in {in_dir}")
    log.info(f"Settings: seg={seg_name}  "
             f"matting={'OFF' if args.no_matting else VITMATTE_MODEL_ID}  "
             f"shadow_radius={args.shadow_radius}  "
             f"erode_radius={args.erode_radius}  "
             f"sharpen={args.sharpen}  contrast={args.contrast}  "
             f"saturation={args.saturation}  "
             f"alpha_cleanup={args.alpha_cleanup}  "
             f"defringe={args.defringe_radius}@{args.defringe_strength}  "
             f"feather={args.edge_feather}")

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
            hf_token=args.hf_token,
            sharpen=args.sharpen,
            contrast=args.contrast,
            saturation=args.saturation,
            alpha_cleanup=args.alpha_cleanup,
            defringe_radius=args.defringe_radius,
            defringe_strength=args.defringe_strength,
            edge_feather=args.edge_feather,
        )

    log.info("Done.")

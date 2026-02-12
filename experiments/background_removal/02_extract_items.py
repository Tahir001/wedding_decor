#!/usr/bin/env python3
"""
02_extract_items.py -- Extract wedding decor items from table scene photos.

Pipeline:
  1. GroundingDINO  -- text-prompted object detection  -> bounding box(es)
  2. SAM 2          -- precise segmentation mask from bounding box
  3. Alpha matting   -- PyMatting refines edges & captures natural shadows
  4. Shadow fallback -- Gaussian drop shadow when no natural shadow exists

Outputs RGBA PNG files with transparent backgrounds, cropped to content.

Usage examples:

  # Single item
  python 02_extract_items.py \
      --image scene.jpg --prompt "gold charger plate" --output plate_rgba.png

  # Batch -- all cutlery images
  python 02_extract_items.py --batch \
      --input-dir ./cutlery \
      --item-type cutlery \
      --output-dir ./outputs/cutlery

  # List available item presets
  python 02_extract_items.py --list-presets
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR = "."
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_PADDING = 20
DEFAULT_SHADOW_OPACITY = 0.30
DEFAULT_SHADOW_BLUR = 12
DEFAULT_SHADOW_OFFSET: Tuple[int, int] = (4, 6)  # (x, y) pixels
TRIMAP_MARGIN = 25  # pixels to expand mask for trimap "unknown" zone

# GroundingDINO thresholds
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Model checkpoints (auto-downloaded from HuggingFace Hub)
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_HF_ID = "facebook/sam2-hiera-large"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ---------------------------------------------------------------------------
# Determine SAM2 backend: prefer the standalone sam2 package, fall back to
# HuggingFace transformers which ships SAM2 since v4.44+.
# ---------------------------------------------------------------------------
try:
    from sam2.build_sam import build_sam2_hf  # noqa: F401
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401

    _SAM2_BACKEND = "sam2"
except ImportError:
    _SAM2_BACKEND = "transformers"

# ---------------------------------------------------------------------------
# Item presets -- maps item type to detection prompt, shadow strategy, and
# tuning parameters.
# ---------------------------------------------------------------------------
ITEM_PRESETS: Dict[str, Dict[str, Any]] = {
    "charger_plates": {
        "prompt": "charger plate",
        "shadow_mode": "generate",
        "trimap_margin": 20,
        "padding": 20,
        "notes": "Preserve all detected charger plates; remove everything else",
    },
    "cutlery": {
        "prompt": "cutlery set fork knife spoon",
        "shadow_mode": "generate",
        "trimap_margin": 30,
        "padding": 25,
        "notes": "Preserve all detected cutlery; remove table/background",
    },
    "glassware": {
        "prompt": "wine glass water glass",
        "shadow_mode": "generate",
        "trimap_margin": 15,
        "padding": 20,
        "notes": "Preserve all detected glasses",
    },
    "napkins": {
        "prompt": "folded napkin",
        "shadow_mode": "generate",
        "trimap_margin": 20,
        "padding": 15,
        "notes": "Preserve all detected napkins",
    },
    "centerpieces": {
        "prompt": "floral centerpiece",
        "shadow_mode": "generate",
        "trimap_margin": 30,
        "padding": 30,
        "notes": "Preserve all detected centerpieces",
    },
    "dinner_plates": {
        "prompt": "dinner plate",
        "shadow_mode": "generate",
        "trimap_margin": 20,
        "padding": 20,
        "notes": "Preserve all detected dinner plates",
    },
}


# ===================================================================
# Model management
# ===================================================================
class ModelManager:
    """Lazy-load and cache GroundingDINO + SAM2 models on first access."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._gdino_model = None
        self._gdino_processor = None
        self._sam2_predictor = None
        log.info(
            "ModelManager initialised -- device: %s, SAM2 backend: %s",
            self.device,
            _SAM2_BACKEND,
        )

    # -- GroundingDINO ------------------------------------------------------
    @property
    def grounding_dino(self):
        """Return (model, processor) for GroundingDINO, loading on first call."""
        if self._gdino_model is None:
            self._load_grounding_dino()
        return self._gdino_model, self._gdino_processor

    def _load_grounding_dino(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        log.info("Loading GroundingDINO from %s ...", GROUNDING_DINO_ID)
        self._gdino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
        self._gdino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID)
            .to(self.device)
        )
        self._gdino_model.eval()
        log.info("GroundingDINO ready.")

    # -- SAM 2 --------------------------------------------------------------
    @property
    def sam2(self):
        """Return a SAM2 predictor, loading on first call."""
        if self._sam2_predictor is None:
            self._load_sam2()
        return self._sam2_predictor

    def _load_sam2(self):
        log.info("Loading SAM2 from %s (backend: %s) ...", SAM2_HF_ID, _SAM2_BACKEND)

        if _SAM2_BACKEND == "sam2":
            self._load_sam2_native()
        else:
            self._load_sam2_transformers()

        log.info("SAM2 ready.")

    def _load_sam2_native(self):
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2_hf(SAM2_HF_ID, device=self.device)
        self._sam2_predictor = _SAM2NativePredictor(SAM2ImagePredictor(sam2_model))

    def _load_sam2_transformers(self):
        from transformers import AutoModelForMaskGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(SAM2_HF_ID)
        model = (
            AutoModelForMaskGeneration.from_pretrained(SAM2_HF_ID).to(self.device)
        )
        model.eval()
        self._sam2_predictor = _SAM2TransformersPredictor(
            model, processor, self.device
        )


# ---------------------------------------------------------------------------
# Unified SAM2 predictor interface -- wraps both backends so the rest of the
# code can call  predictor.predict_mask(image_np, box) -> (mask, score)
# ---------------------------------------------------------------------------
class _SAM2NativePredictor:
    """Wraps the Meta sam2 package predictor."""

    def __init__(self, predictor):
        self._predictor = predictor

    def predict_mask(
        self, image_np: np.ndarray, box: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        self._predictor.set_image(image_np)
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],  # (1, 4)
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


class _SAM2TransformersPredictor:
    """Wraps HuggingFace transformers SAM2 models."""

    def __init__(self, model, processor, device: str):
        self._model = model
        self._processor = processor
        self._device = device

    def predict_mask(
        self, image_np: np.ndarray, box: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        image_pil = Image.fromarray(image_np)
        inputs = self._processor(
            images=image_pil,
            input_boxes=[[[float(c) for c in box]]],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        iou_scores = outputs.iou_scores.cpu().squeeze()
        if iou_scores.dim() == 0:
            best = 0
            score = float(iou_scores)
        else:
            best = int(torch.argmax(iou_scores))
            score = float(iou_scores[best])

        mask_np = masks[0][best].numpy().astype(bool)
        return mask_np, score


# ===================================================================
# Detection  (GroundingDINO)
# ===================================================================
def detect_objects(
    model_mgr: ModelManager,
    image: Image.Image,
    prompt: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Detect objects matching *prompt* via GroundingDINO."""
    model, processor = model_mgr.grounding_dino

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model_mgr.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # transformers API compatibility: box_threshold vs threshold
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results.get("text_labels", results.get("labels", []))

    # sort by score descending
    if len(scores) > 1:
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        labels = [labels[i] for i in order]

    log.info("GroundingDINO: %d detection(s) for prompt '%s'", len(boxes), prompt)
    for i, (b, s, lbl) in enumerate(zip(boxes, scores, labels)):
        log.info("  [%d] %s  score=%.3f  box=%s", i, lbl, s, b.astype(int).tolist())

    return boxes, scores, labels


# ===================================================================
# Segmentation  (SAM 2)
# ===================================================================
def segment_object(
    model_mgr: ModelManager,
    image_np: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """Segment the object within *box* using SAM2."""
    mask, score = model_mgr.sam2.predict_mask(image_np, box)
    log.info("SAM2 mask: score=%.3f  area=%d px", score, int(mask.sum()))
    return mask


# ===================================================================
# Trimap + alpha matting
# ===================================================================
def build_trimap(mask: np.ndarray, margin: int = TRIMAP_MARGIN) -> np.ndarray:
    erode_iters = max(1, margin // 3)
    fg = ndimage.binary_erosion(mask, iterations=erode_iters)

    dilated = ndimage.binary_dilation(mask, iterations=margin)

    shift = int(margin * 0.8)
    if shift > 0:
        shadow_extra = np.zeros_like(mask)
        shadow_extra[shift:, :] = mask[:-shift, :]
        shadow_extra = ndimage.binary_dilation(shadow_extra, iterations=margin // 2)
        dilated = dilated | shadow_extra

    trimap = np.full(mask.shape, 0.5, dtype=np.float64)
    trimap[~dilated] = 0.0
    trimap[fg.astype(bool)] = 1.0
    return trimap


def alpha_matte(
    image_np: np.ndarray,
    mask: np.ndarray,
    trimap_margin: int = TRIMAP_MARGIN,
) -> np.ndarray:
    import pymatting

    trimap = build_trimap(mask, margin=trimap_margin)
    img_f64 = image_np.astype(np.float64) / 255.0

    alpha = pymatting.estimate_alpha_lkm(img_f64, trimap)
    alpha = np.clip(alpha, 0.0, 1.0)

    log.info("Alpha matting done -- range [%.3f, %.3f]", float(alpha.min()), float(alpha.max()))
    return alpha


# ===================================================================
# Shadow detection & generation
# ===================================================================
def has_natural_shadow(
    alpha: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.05,
) -> bool:
    shadow_zone = (alpha > 0.02) & (~mask.astype(bool))
    shadow_ratio = shadow_zone.sum() / max(mask.sum(), 1)
    log.info("Shadow ratio (alpha outside mask / mask area): %.4f", float(shadow_ratio))
    return shadow_ratio > threshold


def generate_drop_shadow(
    alpha: np.ndarray,
    offset: Tuple[int, int] = DEFAULT_SHADOW_OFFSET,
    blur_radius: int = DEFAULT_SHADOW_BLUR,
    opacity: float = DEFAULT_SHADOW_OPACITY,
) -> np.ndarray:
    h, w = alpha.shape
    silhouette = (alpha > 0.5).astype(np.float64)

    shadow = np.zeros_like(silhouette)
    ox, oy = offset

    src_y = slice(max(0, -oy), min(h, h - oy))
    dst_y = slice(max(0, oy), min(h, h + oy))
    src_x = slice(max(0, -ox), min(w, w - ox))
    dst_x = slice(max(0, ox), min(w, w + ox))
    shadow[dst_y, dst_x] = silhouette[src_y, src_x]

    shadow = ndimage.gaussian_filter(shadow, sigma=blur_radius)
    shadow *= opacity
    shadow = np.where(alpha > 0.5, 0.0, shadow)

    log.info(
        "Generated drop shadow (offset=%s, blur=%d, opacity=%.2f)",
        offset,
        blur_radius,
        opacity,
    )
    return shadow


def combine_alpha_with_shadow(alpha: np.ndarray, shadow_alpha: np.ndarray) -> np.ndarray:
    combined = alpha + shadow_alpha * (1.0 - alpha)
    return np.clip(combined, 0.0, 1.0)


# ===================================================================
# Crop to content
# ===================================================================
def crop_to_content(
    rgba: np.ndarray,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    a = rgba[:, :, 3]
    rows = np.any(a > 0, axis=1)
    cols = np.any(a > 0, axis=0)

    if not rows.any():
        log.warning("No non-transparent pixels found -- returning original array.")
        return rgba

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = rgba.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)

    return rgba[rmin : rmax + 1, cmin : cmax + 1]


# ===================================================================
# Core: extract a single item
# ===================================================================
def extract_item(
    image_path: str,
    item_prompt: str,
    output_path: str,
    model_mgr: ModelManager,
    shadow_mode: str = "auto",
    trimap_margin: int = TRIMAP_MARGIN,
    padding: int = DEFAULT_PADDING,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    max_objects: Optional[int] = None,
) -> Optional[str]:
    log.info("=== Extracting '%s' from %s", item_prompt, image_path)

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    log.info("Image size: %d x %d", w, h)

    boxes, scores, labels = detect_objects(
        model_mgr, image_pil, item_prompt, box_threshold, text_threshold
    )
    if len(boxes) == 0:
        log.warning("No objects detected for prompt '%s' -- skipping.", item_prompt)
        return None

    # Keep all instances of the target object class (or top N if requested)
    selected_boxes = boxes[:max_objects] if max_objects else boxes
    log.info("Using %d detection(s) for segmentation", len(selected_boxes))

    mask = np.zeros(image_np.shape[:2], dtype=bool)
    for i, box in enumerate(selected_boxes, 1):
        instance_mask = segment_object(model_mgr, image_np, box)
        mask |= instance_mask
        log.info("Merged instance %d/%d -- combined area=%d px", i, len(selected_boxes), int(mask.sum()))

    # Fast path: skip alpha matting in generate mode
    if shadow_mode == "generate":
        hard_alpha = mask.astype(np.float64)
        shadow = generate_drop_shadow(hard_alpha)
        final_alpha = combine_alpha_with_shadow(hard_alpha, shadow)
        log.info("Shadow mode: generate (hard mask + artificial shadow, no alpha matting)")

    elif shadow_mode == "preserve":
        alpha = alpha_matte(image_np, mask, trimap_margin=trimap_margin)
        final_alpha = alpha
        log.info("Shadow mode: preserve (using alpha matte as-is)")

    elif shadow_mode == "auto":
        alpha = alpha_matte(image_np, mask, trimap_margin=trimap_margin)
        if has_natural_shadow(alpha, mask):
            final_alpha = alpha
            log.info("Shadow mode: auto -> preserve (natural shadow detected)")
        else:
            shadow = generate_drop_shadow(alpha)
            final_alpha = combine_alpha_with_shadow(alpha, shadow)
            log.info("Shadow mode: auto -> generate (no natural shadow)")

    else:
        raise ValueError(f"Unknown shadow_mode: {shadow_mode!r}")

    alpha_uint8 = (final_alpha * 255).astype(np.uint8)
    rgba = np.dstack([image_np, alpha_uint8])

    rgba_cropped = crop_to_content(rgba, padding=padding)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(rgba_cropped, "RGBA").save(output_path)
    log.info(
        "Saved: %s (%d x %d)",
        output_path,
        rgba_cropped.shape[1],
        rgba_cropped.shape[0],
    )

    return output_path


# ===================================================================
# Batch processing
# ===================================================================
def batch_extract(
    input_dir: str,
    item_type: str,
    output_dir: str,
    model_mgr: ModelManager,
    shadow_mode: Optional[str] = None,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    max_objects: Optional[int] = None,
) -> List[str]:
    if item_type not in ITEM_PRESETS:
        raise ValueError(
            f"Unknown item_type '{item_type}'. "
            f"Choose from: {', '.join(sorted(ITEM_PRESETS.keys()))}"
        )

    preset = ITEM_PRESETS[item_type]
    prompt = preset["prompt"]
    effective_shadow = shadow_mode or preset["shadow_mode"]
    trimap_margin = preset["trimap_margin"]
    padding = preset["padding"]

    log.info(
        "Batch extract: type=%s  prompt='%s'  shadow=%s  input=%s  output=%s",
        item_type,
        prompt,
        effective_shadow,
        input_dir,
        output_dir,
    )

    input_path = Path(input_dir)
    if not input_path.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        log.warning("No images found in %s", input_dir)
        return []

    log.info("Found %d image(s) to process.", len(image_files))

    results: List[str] = []
    for idx, img_file in enumerate(image_files, 1):
        log.info("-- [%d/%d] %s", idx, len(image_files), img_file.name)
        stem = img_file.stem
        out_name = f"{item_type}_{stem}_rgba.png"
        out_file = str(output_path / out_name)

        result = extract_item(
            image_path=str(img_file),
            item_prompt=prompt,
            output_path=out_file,
            model_mgr=model_mgr,
            shadow_mode=effective_shadow,
            trimap_margin=trimap_margin,
            padding=padding,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_objects=max_objects,
        )
        if result:
            results.append(result)

    log.info("Batch complete: %d / %d succeeded.", len(results), len(image_files))
    return results


# ===================================================================
# CLI
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract wedding decor items from table scene photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available item-type presets and exit.",
    )

    # -- Single-image mode --------------------------------------------------
    single = p.add_argument_group("Single-image mode")
    single.add_argument("--image", type=str, help="Path to input image.")
    single.add_argument("--prompt", type=str, help="Text prompt for object detection.")
    single.add_argument("--output", type=str, help="Output RGBA PNG path.")

    # -- Batch mode ---------------------------------------------------------
    batch = p.add_argument_group("Batch mode")
    batch.add_argument("--batch", action="store_true", help="Enable batch processing.")
    batch.add_argument("--input-dir", type=str, help="Directory of input images.")
    batch.add_argument(
        "--item-type",
        type=str,
        choices=sorted(ITEM_PRESETS.keys()),
        help="Item type preset for batch mode.",
    )
    batch.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )

    # -- Shared options -----------------------------------------------------
    shared = p.add_argument_group("Shared options")
    shared.add_argument(
        "--shadow-mode",
        type=str,
        choices=["preserve", "generate", "auto"],
        help="Shadow handling mode (overrides preset default).",
    )
    shared.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Limit number of detections to keep (default: keep all detections).",
    )
    shared.add_argument(
        "--padding",
        type=int,
        default=None,
        help=f"Padding around cropped output in px (default: per-preset or {DEFAULT_PADDING}).",
    )
    shared.add_argument(
        "--trimap-margin",
        type=int,
        default=None,
        help=f"Trimap unknown-zone margin in px (default: per-preset or {TRIMAP_MARGIN}).",
    )
    shared.add_argument(
        "--box-threshold",
        type=float,
        default=BOX_THRESHOLD,
        help=f"GroundingDINO box confidence threshold (default: {BOX_THRESHOLD}).",
    )
    shared.add_argument(
        "--text-threshold",
        type=float,
        default=TEXT_THRESHOLD,
        help=f"GroundingDINO text match threshold (default: {TEXT_THRESHOLD}).",
    )
    shared.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cuda', 'cpu' (default: auto-detect).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable item-type presets:\n")
        hdr = f"  {'Type':<18} {'Prompt':<34} {'Shadow':<12} Notes"
        print(hdr)
        print(f"  {'---'*6:<18} {'---'*11:<34} {'---'*4:<12} {'---'*13}")
        for name, cfg in ITEM_PRESETS.items():
            print(
                f"  {name:<18} {cfg['prompt']:<34} "
                f"{cfg['shadow_mode']:<12} {cfg['notes']}"
            )
        print()
        return

    if not args.batch and not args.image:
        parser.error("Provide --image (single mode) or --batch (batch mode).")

    if args.batch:
        if not args.item_type:
            parser.error("--batch requires --item-type.")
        if not args.input_dir:
            parser.error("--batch requires --input-dir.")
    else:
        if not args.prompt:
            parser.error("Single mode requires --prompt.")
        if not args.output:
            parser.error("Single mode requires --output.")

    model_mgr = ModelManager(device=args.device)

    if args.batch:
        batch_extract(
            input_dir=args.input_dir,
            item_type=args.item_type,
            output_dir=args.output_dir,
            model_mgr=model_mgr,
            shadow_mode=args.shadow_mode,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_objects=args.max_objects,
        )
    else:
        extract_item(
            image_path=args.image,
            item_prompt=args.prompt,
            output_path=args.output,
            model_mgr=model_mgr,
            shadow_mode=args.shadow_mode or "generate",
            trimap_margin=args.trimap_margin or TRIMAP_MARGIN,
            padding=args.padding or DEFAULT_PADDING,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_objects=args.max_objects,
        )

if __name__ == "__main__":
    main()
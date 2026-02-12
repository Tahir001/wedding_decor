#!/usr/bin/env python3
"""
02_extract_items.py — Extract wedding decor items from table scene photos.

Architecture:
  1. GroundingDINO  → text-prompted object detection  → bounding boxes
  2. SAM2 / FastSAM → precise segmentation mask from each box
  3. Edge cleanup   → morphological smoothing for clean cutouts
  4. Supervision    → visualization, annotation, QA grid views

Integrates key techniques from Roboflow's Grounded-SAM notebook:
  - enhance_class_name() prompt engineering for better multi-instance detection
  - supervision library for professional annotation visualizations
  - Grid view for QA of all detected masks
  - Class-based detection with per-class counting

Outputs:
  - RGBA PNG with FULLY transparent background (only detected items visible)

  With --debug, also saves:
  - *_1_detections.png  → bounding boxes + labels via supervision
  - *_2_masks_grid.png  → grid of individual instance masks (QA view)
  - *_3_overlay.png     → mask overlay (green = kept, dim = removed)
  - *_4_rgba_full.png   → full-size transparent PNG (uncropped)

Usage:
  python 02_extract_items.py --batch \
      --input-dir ./cutlery --item-type cutlery \
      --output-dir ./outputs/cutlery \
      --segmentor sam2 --debug

  python 02_extract_items.py \
      --image scene.jpg --prompt "fork,knife,spoon" --output cutlery.png --debug

  python 02_extract_items.py --list-presets
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Constants
DEFAULT_PADDING = 20
DEFAULT_SHADOW_OPACITY = 0.25
DEFAULT_SHADOW_BLUR = 10
DEFAULT_SHADOW_OFFSET = (3, 5)
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
MIN_MASK_RATIO = 0.001
MAX_MASK_RATIO = 0.85
EDGE_SMOOTH_ITERATIONS = 2
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_HF_ID = "facebook/sam2-hiera-large"
FASTSAM_MODEL = "FastSAM-x.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Backend detection
try:
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False

try:
    from fastsam import FastSAM, FastSAMPrompt
    _FASTSAM_AVAILABLE = True
except ImportError:
    _FASTSAM_AVAILABLE = False

try:
    import supervision as sv
    _SV = True
    log.info("supervision %s loaded", sv.__version__)
except ImportError:
    _SV = False

# ===================================================================
# Item presets — classes list (not raw prompts)
# ===================================================================
ITEM_PRESETS: Dict[str, Dict[str, Any]] = {
    "charger_plates": {"classes": ["charger plate"], "shadow": False, "padding": 20, "notes": "Large base plates"},
    "dinner_plates":  {"classes": ["dinner plate"], "shadow": False, "padding": 20, "notes": "Standard dinner plates"},
    "cutlery":        {"classes": ["fork", "knife", "spoon"], "shadow": False, "padding": 25, "notes": "Per-utensil detection"},
    "glassware":      {"classes": ["wine glass", "champagne flute", "water glass"], "shadow": False, "padding": 20, "notes": "All glass types"},
    "napkins":        {"classes": ["folded napkin", "cloth napkin"], "shadow": False, "padding": 15, "notes": "Folded napkins"},
    "centerpieces":   {"classes": ["floral centerpiece", "flower arrangement"], "shadow": False, "padding": 30, "notes": "Center arrangements"},
}

# ===================================================================
# Prompt engineering — from Roboflow's Grounded-SAM notebook
# ===================================================================
def enhance_class_name(class_names: List[str]) -> List[str]:
    """'fork' → 'all forks' — helps GroundingDINO find ALL instances."""
    return [f"all {name}s" for name in class_names]

def classes_to_prompt(class_names: List[str], enhance: bool = True) -> str:
    """Build period-separated prompt: 'all forks . all knives . all spoons'"""
    names = enhance_class_name(class_names) if enhance else class_names
    return " . ".join(names)

# ===================================================================
@dataclass
class ExtractionResult:
    input_path: str; output_path: Optional[str]; prompt: str; classes: List[str]
    num_detections: int; detections_per_class: Dict[str, int]
    mask_area_px: int; mask_ratio: float
    output_size: Optional[Tuple[int, int]] = None
    elapsed_s: float = 0.0; detection_time_s: float = 0.0; segmentation_time_s: float = 0.0
    status: str = "success"; warning: Optional[str] = None
    segmentor_used: str = "sam2"; debug_paths: Optional[Dict[str, str]] = None

# ===================================================================
# Visualization
# ===================================================================
def viz_detections(image_bgr, boxes, scores, class_ids, class_names, output_path):
    if _SV:
        dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        labels = [f"{class_names[cid]} {c:.2f}" for cid, c in zip(class_ids, scores)]
        ann = sv.BoxAnnotator(thickness=3).annotate(scene=image_bgr.copy(), detections=dets)
        ann = sv.LabelAnnotator(text_scale=0.6, text_thickness=2).annotate(scene=ann, detections=dets, labels=labels)
        cv2.imwrite(output_path, ann)
    else:
        from PIL import ImageDraw, ImageFont
        COLORS = [(255,82,82),(76,175,80),(33,150,243),(255,193,7),(156,39,176),(0,188,212)]
        viz = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(viz)
        try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except: font = ImageFont.load_default()
        for i, (box, score) in enumerate(zip(boxes, scores)):
            c = COLORS[i % len(COLORS)]; x1,y1,x2,y2 = box.astype(int)
            for o in range(3): draw.rectangle([x1-o,y1-o,x2+o,y2+o], outline=c)
            txt = f"{class_names[class_ids[i]]} {score:.2f}"
            bb = draw.textbbox((x1,y1-22), txt, font=font)
            draw.rectangle([bb[0]-2,bb[1]-2,bb[2]+2,bb[3]+2], fill=c)
            draw.text((x1,y1-22), txt, fill=(255,255,255), font=font)
        viz.save(output_path)
    log.info("📸 Detections → %s", output_path)


def viz_masks_grid(masks, labels, output_path):
    if not masks: return
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = len(masks); dim = math.ceil(math.sqrt(n))
    fig, axes = plt.subplots(dim, dim, figsize=(4*dim, 4*dim))
    if dim == 1: axes = np.array([axes])
    axes = axes.flatten()
    for i, (m, l) in enumerate(zip(masks, labels)):
        axes[i].imshow(m.astype(np.uint8)*255, cmap='gray')
        axes[i].set_title(l, fontsize=12, fontweight='bold'); axes[i].axis('off')
    for i in range(n, len(axes)): axes[i].axis('off')
    plt.tight_layout(); plt.savefig(output_path, dpi=100, bbox_inches='tight'); plt.close(fig)
    log.info("📸 Mask grid (%d) → %s", n, output_path)


def viz_overlay(image_bgr, mask, boxes, scores, class_ids, class_names, output_path):
    ann = image_bgr.copy()
    ann[~mask] = (ann[~mask] * 0.3).astype(np.uint8)
    if _SV and len(boxes) > 0:
        dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        labels = [f"{class_names[cid]} {c:.2f}" for cid, c in zip(class_ids, scores)]
        ann = sv.BoxAnnotator(thickness=2).annotate(scene=ann, detections=dets)
        ann = sv.LabelAnnotator(text_scale=0.5).annotate(scene=ann, detections=dets, labels=labels)
    cv2.imwrite(output_path, ann)
    log.info("📸 Overlay → %s", output_path)


# ===================================================================
# Model Manager
# ===================================================================
class ModelManager:
    def __init__(self, device=None, segmentor="sam2"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentor_type = segmentor
        self._gdino_model = self._gdino_processor = self._sam2_predictor = self._fastsam_model = None
        log.info("ModelManager — device: %s, segmentor: %s", self.device, segmentor)

    @property
    def grounding_dino(self):
        if self._gdino_model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            log.info("Loading GroundingDINO...")
            self._gdino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
            self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID).to(self.device)
            self._gdino_model.eval(); log.info("GroundingDINO ready.")
        return self._gdino_model, self._gdino_processor

    @property
    def sam2(self):
        if self._sam2_predictor is None:
            log.info("Loading SAM2...")
            if _SAM2_AVAILABLE:
                m = build_sam2_hf(SAM2_HF_ID, device=self.device)
                self._sam2_predictor = _SAM2Native(SAM2ImagePredictor(m))
            else:
                from transformers import AutoModelForMaskGeneration, AutoProcessor
                p = AutoProcessor.from_pretrained(SAM2_HF_ID)
                m = AutoModelForMaskGeneration.from_pretrained(SAM2_HF_ID).to(self.device); m.eval()
                self._sam2_predictor = _SAM2HF(m, p, self.device)
            log.info("SAM2 ready.")
        return self._sam2_predictor

    @property
    def fastsam(self):
        if self._fastsam_model is None:
            if not _FASTSAM_AVAILABLE:
                raise ImportError("FastSAM not installed. pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git")
            log.info("Loading FastSAM..."); self._fastsam_model = FastSAM(FASTSAM_MODEL); log.info("FastSAM ready.")
        return self._fastsam_model

class _SAM2Native:
    def __init__(self, p): self._p = p
    def predict(self, img, box):
        self._p.set_image(img)
        masks, scores, _ = self._p.predict(point_coords=None, point_labels=None, box=box[None,:], multimask_output=True)
        b = int(np.argmax(scores)); return masks[b].astype(bool), float(scores[b])

class _SAM2HF:
    def __init__(self, model, proc, dev): self._m, self._p, self._d = model, proc, dev
    def predict(self, img, box):
        pil = Image.fromarray(img)
        inp = self._p(images=pil, input_boxes=[[[float(c) for c in box]]], return_tensors="pt").to(self._d)
        with torch.no_grad(): out = self._m(**inp)
        masks = self._p.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())
        iou = out.iou_scores.cpu().squeeze()
        b = 0 if iou.dim()==0 else int(torch.argmax(iou))
        s = float(iou) if iou.dim()==0 else float(iou[b])
        return masks[0][b].numpy().astype(bool), s

# ===================================================================
# Detection
# ===================================================================
def detect_objects(mgr, image_pil, classes, box_thr=BOX_THRESHOLD, text_thr=TEXT_THRESHOLD, enhance=True):
    model, proc = mgr.grounding_dino
    prompt = classes_to_prompt(classes, enhance=enhance)
    log.info("Prompt: '%s'", prompt)
    inp = proc(images=image_pil, text=prompt, return_tensors="pt").to(mgr.device)
    with torch.no_grad(): out = model(**inp)
    try:
        res = proc.post_process_grounded_object_detection(out, inp["input_ids"], box_threshold=box_thr, text_threshold=text_thr, target_sizes=[image_pil.size[::-1]])[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=box_thr, text_threshold=text_thr, target_sizes=[image_pil.size[::-1]])[0]

    boxes = res["boxes"].cpu().numpy(); scores = res["scores"].cpu().numpy()
    raw_labels = res.get("text_labels", res.get("labels", []))
    enhanced = enhance_class_name(classes) if enhance else classes

    class_ids, clean_labels = [], []
    for lbl in raw_labels:
        ll = lbl.lower().strip(); mid = 0
        for i, (o, e) in enumerate(zip(classes, enhanced)):
            if o.lower() in ll or e.lower() in ll: mid = i; break
        class_ids.append(mid); clean_labels.append(classes[mid])
    class_ids = np.array(class_ids, dtype=int)

    if len(scores) > 1:
        order = np.argsort(scores)[::-1]
        boxes, scores, class_ids = boxes[order], scores[order], class_ids[order]
        clean_labels = [clean_labels[i] for i in order]

    log.info("GroundingDINO: %d detection(s)", len(boxes))
    for i,(b,s,l) in enumerate(zip(boxes,scores,clean_labels)):
        log.info("  [%d] %s score=%.3f box=%s", i, l, s, b.astype(int).tolist())
    return boxes, scores, clean_labels, class_ids

# ===================================================================
# Segmentation
# ===================================================================
def segment_all(mgr, image_np, image_pil, boxes):
    h, w = image_np.shape[:2]; results = []
    for i, box in enumerate(boxes):
        log.info("  Segment %d/%d", i+1, len(boxes))
        if mgr.segmentor_type == "fastsam":
            m = mgr.fastsam; d = mgr.device
            everything = m(image_pil, device=d, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            pp = FastSAMPrompt(image_pil, everything, device=d)
            mask = pp.box_prompt(bbox=box.tolist())
            if isinstance(mask, list) and mask: mask = mask[0]
            if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
            mask = mask.astype(bool)
            if mask.ndim == 3: mask = mask.squeeze(0)
            if mask.shape != (h,w):
                mask = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize((w,h), Image.NEAREST)) > 127
            score = 0.9
        else:
            mask, score = mgr.sam2.predict(image_np, box)
            if mask.shape != (h,w):
                mask = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize((w,h), Image.NEAREST)) > 127
        log.info("    %s: score=%.3f area=%d px", mgr.segmentor_type, score, int(mask.sum()))
        results.append((mask, score))
    return results

# ===================================================================
# Post-processing helpers
# ===================================================================
def smooth_mask(mask, iters=EDGE_SMOOTH_ITERATIONS):
    from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
    s = generate_binary_structure(2,2)
    return binary_opening(binary_closing(mask, structure=s, iterations=iters), structure=s, iterations=iters).astype(bool)

def feather_alpha(mask, radius=2):
    a = mask.astype(np.float64)
    if radius > 0:
        a = ndimage.gaussian_filter(a, sigma=radius*0.5)
        a = np.where(mask, np.maximum(a, 0.95), a)
    return np.clip(a, 0.0, 1.0)

def validate_mask(mask, shape):
    r = mask.sum() / (shape[0]*shape[1])
    if r < MIN_MASK_RATIO: return False, f"Too small ({r:.4f})"
    if r > MAX_MASK_RATIO: return False, f"Too large ({r:.4f})"
    return True, None

def gen_shadow(alpha, offset=DEFAULT_SHADOW_OFFSET, blur=DEFAULT_SHADOW_BLUR, opacity=DEFAULT_SHADOW_OPACITY):
    h,w = alpha.shape; sil = (alpha>0.5).astype(np.float64); sh = np.zeros_like(sil)
    ox,oy = offset
    sh[max(0,oy):min(h,h+oy), max(0,ox):min(w,w+ox)] = sil[max(0,-oy):min(h,h-oy), max(0,-ox):min(w,w-ox)]
    sh = ndimage.gaussian_filter(sh, sigma=blur) * opacity
    return np.where(alpha>0.5, 0.0, sh)

def crop_content(rgba, pad=DEFAULT_PADDING):
    a = rgba[:,:,3]; rows = np.any(a>0, axis=1); cols = np.any(a>0, axis=0)
    if not rows.any(): return rgba
    rmin,rmax = np.where(rows)[0][[0,-1]]; cmin,cmax = np.where(cols)[0][[0,-1]]
    h,w = rgba.shape[:2]
    return rgba[max(0,rmin-pad):min(h,rmax+pad+1), max(0,cmin-pad):min(w,cmax+pad+1)]

# ===================================================================
# Core extraction
# ===================================================================
def extract_item(image_path, classes, output_path, model_mgr, add_shadow=False,
                 padding=DEFAULT_PADDING, feather_radius=2, box_threshold=BOX_THRESHOLD,
                 text_threshold=TEXT_THRESHOLD, max_objects=None, smooth_edges=True,
                 debug=False, enhance_prompts=True):
    t0 = time.time()
    log.info("=== Extracting %s from %s", classes, image_path)

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil); image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]
    debug_paths = {}; stem = Path(output_path).stem; odir = Path(output_path).parent
    prompt_str = classes_to_prompt(classes, enhance=enhance_prompts)

    # 1. Detect
    t_det = time.time()
    boxes, scores, labels, cids = detect_objects(model_mgr, image_pil, classes, box_threshold, text_threshold, enhance_prompts)
    det_time = time.time() - t_det

    if len(boxes) == 0:
        return ExtractionResult(input_path=image_path, output_path=None, prompt=prompt_str, classes=classes,
                                num_detections=0, detections_per_class={}, mask_area_px=0, mask_ratio=0.0,
                                elapsed_s=time.time()-t0, status="no_detections", segmentor_used=model_mgr.segmentor_type)

    det_per_class = {}
    for l in labels: det_per_class[l] = det_per_class.get(l, 0) + 1
    log.info("Per class: %s", det_per_class)

    sel = slice(None, max_objects) if max_objects else slice(None)
    sb, ss, sl, sc = boxes[sel], scores[sel], labels[:max_objects] if max_objects else labels, cids[sel]

    if debug:
        dp = str(odir / f"{stem}_1_detections.png")
        viz_detections(image_bgr, sb, ss, sc, classes, dp); debug_paths["detections"] = dp

    # 2. Segment
    t_seg = time.time()
    mask_results = segment_all(model_mgr, image_np, image_pil, sb)
    ind_masks = [m for m,_ in mask_results]
    combined = np.zeros((h,w), dtype=bool)
    for m,_ in mask_results: combined |= m
    seg_time = time.time() - t_seg

    if debug and ind_masks:
        gp = str(odir / f"{stem}_2_masks_grid.png"); viz_masks_grid(ind_masks, sl, gp); debug_paths["masks_grid"] = gp
    if debug:
        op = str(odir / f"{stem}_3_overlay.png"); viz_overlay(image_bgr, combined, sb, ss, sc, classes, op); debug_paths["overlay"] = op

    # 3. Validate + cleanup
    valid, warning = validate_mask(combined, (h,w))
    if not valid: log.warning("⚠️  %s", warning)
    mask_area = int(combined.sum()); mask_ratio = mask_area / (h*w)

    if smooth_edges: combined = smooth_mask(combined)
    alpha = feather_alpha(combined, radius=feather_radius)
    if add_shadow:
        alpha = np.clip(alpha + gen_shadow(alpha) * (1.0 - alpha), 0.0, 1.0)

    # 4. RGBA output — fully transparent except items
    rgba = np.dstack([image_np, (alpha*255).astype(np.uint8)])
    rgba_cropped = crop_content(rgba, padding)

    os.makedirs(str(odir), exist_ok=True)
    Image.fromarray(rgba_cropped, "RGBA").save(output_path, optimize=True)
    elapsed = time.time() - t0; oh, ow = rgba_cropped.shape[:2]

    if debug:
        fp = str(odir / f"{stem}_4_rgba_full.png")
        Image.fromarray(rgba, "RGBA").save(fp); debug_paths["rgba_full"] = fp

    log.info("✅ %s (%dx%d) — %.2fs [det=%.2fs %s=%.2fs]", output_path, ow, oh, elapsed, det_time, model_mgr.segmentor_type, seg_time)

    return ExtractionResult(
        input_path=image_path, output_path=output_path, prompt=prompt_str, classes=classes,
        num_detections=len(sb), detections_per_class=det_per_class, mask_area_px=mask_area,
        mask_ratio=mask_ratio, output_size=(ow,oh), elapsed_s=elapsed, detection_time_s=det_time,
        segmentation_time_s=seg_time, status="success" if valid else "warning", warning=warning,
        segmentor_used=model_mgr.segmentor_type, debug_paths=debug_paths if debug else None)


# ===================================================================
# Batch
# ===================================================================
def batch_extract(input_dir, item_type, output_dir, model_mgr, add_shadow=False,
                  box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                  max_objects=None, smooth_edges=True, debug=False, enhance_prompts=True):
    if item_type not in ITEM_PRESETS:
        raise ValueError(f"Unknown '{item_type}'. Choose from: {', '.join(sorted(ITEM_PRESETS))}")
    preset = ITEM_PRESETS[item_type]; classes = preset["classes"]; padding = preset["padding"]
    log.info("Batch: %s classes=%s seg=%s", item_type, classes, model_mgr.segmentor_type)

    ipath = Path(input_dir)
    if not ipath.is_dir(): log.error("Not found: %s", input_dir); return []
    opath = Path(output_dir); opath.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ipath.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files: log.warning("No images in %s", input_dir); return []
    log.info("Found %d image(s)", len(files))

    results, td, ts = [], 0.0, 0.0
    for idx, f in enumerate(files, 1):
        log.info("━━ [%d/%d] %s ━━", idx, len(files), f.name)
        r = extract_item(str(f), classes, str(opath / f"{item_type}_{f.stem}_rgba.png"), model_mgr,
                         add_shadow, padding, 2, box_threshold, text_threshold, max_objects, smooth_edges, debug, enhance_prompts)
        results.append(r); td += r.detection_time_s; ts += r.segmentation_time_s

    n = len(results); ok = sum(1 for r in results if r.status in ("success","warning"))
    total_time = sum(r.elapsed_s for r in results)
    total_cls = {}
    for r in results:
        for c, cnt in r.detections_per_class.items(): total_cls[c] = total_cls.get(c,0)+cnt

    log.info("\n" + "═"*60)
    log.info("  BATCH COMPLETE — %s (%s)", item_type, model_mgr.segmentor_type.upper())
    log.info("  %d/%d ok | Detections: %s", ok, n, total_cls)
    log.info("  Det: %.2fs avg | Seg: %.2fs avg | Total: %.2fs (%.2fs/img)", td/n, ts/n, total_time, total_time/n)
    log.info("═"*60)

    rp = str(opath / f"batch_report_{item_type}.json")
    with open(rp, "w") as fp:
        json.dump({"item_type": item_type, "classes": classes, "segmentor": model_mgr.segmentor_type,
                    "enhance": enhance_prompts, "total": n, "succeeded": ok, "per_class": total_cls,
                    "timing": {"total": round(total_time,2), "avg": round(total_time/n,2)},
                    "results": [{"in": r.input_path, "out": r.output_path, "dets": r.num_detections,
                                 "cls": r.detections_per_class, "ratio": round(r.mask_ratio,4),
                                 "size": r.output_size, "t": round(r.elapsed_s,2), "status": r.status}
                                for r in results]}, fp, indent=2)
    return results


# ===================================================================
# CLI
# ===================================================================
def main():
    p = argparse.ArgumentParser(description="Extract wedding decor → transparent RGBA PNGs")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--image", type=str); p.add_argument("--prompt", type=str, help="Comma-separated classes")
    p.add_argument("--output", type=str)
    p.add_argument("--batch", action="store_true"); p.add_argument("--input-dir", type=str)
    p.add_argument("--item-type", type=str, choices=sorted(ITEM_PRESETS)); p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--segmentor", default="sam2", choices=["sam2","fastsam"])
    p.add_argument("--debug", action="store_true"); p.add_argument("--shadow", action="store_true")
    p.add_argument("--no-enhance", action="store_true"); p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--feather", type=int, default=2); p.add_argument("--max-objects", type=int)
    p.add_argument("--padding", type=int); p.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    p.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD); p.add_argument("--device", type=str)
    args = p.parse_args()

    if args.list_presets:
        print("\nPresets:\n")
        for name, cfg in sorted(ITEM_PRESETS.items()):
            print(f"  {name:<18} {', '.join(cfg['classes']):<42} {cfg['notes']}")
        return

    mgr = ModelManager(device=args.device, segmentor=args.segmentor)
    enh = not args.no_enhance

    if args.batch:
        batch_extract(args.input_dir, args.item_type, args.output_dir, mgr, args.shadow,
                      args.box_threshold, args.text_threshold, args.max_objects,
                      not args.no_smooth, args.debug, enh)
    elif args.image:
        classes = [c.strip() for c in args.prompt.split(",")]
        extract_item(args.image, classes, args.output, mgr, args.shadow,
                     args.padding or DEFAULT_PADDING, args.feather, args.box_threshold,
                     args.text_threshold, args.max_objects, not args.no_smooth, args.debug, enh)
    else:
        p.error("Provide --image or --batch")

if __name__ == "__main__":
    main()
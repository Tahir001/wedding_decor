#!/usr/bin/env python3
"""
batch_remove_bg.py - Batch background removal for all product images.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "pipeline_config.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
BG_REMOVER_DIR = Path(__file__).parent.parent / "background_removal"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def find_bg_remover():
    v2_path = BG_REMOVER_DIR / "background_remover_main_v2.py"
    if not v2_path.exists():
        log.error("Background remover not found at %s", v2_path)
        sys.exit(1)
    sys.path.insert(0, str(BG_REMOVER_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location("bg_remover_v2", str(v2_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_images(input_dir):
    if not input_dir.is_dir():
        return []
    return sorted(f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def process_category(category, input_dir, output_dir, device, force, bg_module, **kwargs):
    images = collect_images(input_dir)
    if not images:
        log.warning("  [%s] No images found in %s", category, input_dir)
        return {"category": category, "processed": 0, "skipped": 0, "failed": 0}

    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    failed = 0

    for img_path in images:
        out_path = output_dir / (img_path.stem + ".png")

        if out_path.exists() and not force:
            log.info("  [%s] SKIP %s (already processed)", category, img_path.name)
            skipped += 1
            continue

        try:
            bg_module.process_image(
                img_path=img_path,
                output_path=out_path,
                device=device,
                shadow_radius=20,
                erode_radius=5,
                despill_strength=0.7,
                no_matting=False,
                debug=False,
                sharpen=0.7,
                contrast=1.0,
                saturation=1.0,
                alpha_cleanup=3,
                defringe_radius=15,
                defringe_strength=0.8,
                edge_feather=1.2,
            )
            processed += 1
            log.info("  [%s] OK %s -> %s", category, img_path.name, out_path.name)
        except Exception as e:
            failed += 1
            log.error("  [%s] FAIL %s: %s", category, img_path.name, e)

    return {
        "category": category,
        "total": len(images),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch background removal.")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--no-matting", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg["paths"]

    images_root = Path(args.images_root or paths["images_root"])
    processed_root = images_root / paths["processed_dir"]
    product_dirs = paths["product_dirs"]

    if args.device:
        device = args.device
    else:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Device: %s", device)
    log.info("Images root: %s", images_root)
    log.info("Processed output: %s", processed_root)

    bg_module = find_bg_remover()

    if args.category:
        if args.category not in product_dirs:
            avail = ", ".join(product_dirs.keys())
            log.error("Unknown category. Available: %s", avail)
            sys.exit(1)
        categories = {args.category: product_dirs[args.category]}
    else:
        categories = product_dirs

    total_start = time.time()
    results = []

    for cat_name, cat_folder in categories.items():
        input_dir = images_root / cat_folder
        output_dir = processed_root / cat_name
        log.info("Processing category: %s (%s)", cat_name, input_dir)
        result = process_category(
            category=cat_name, input_dir=input_dir,
            output_dir=output_dir, device=device,
            force=args.force, bg_module=bg_module,
        )
        results.append(result)

    elapsed = time.time() - total_start

    print("\nBATCH BACKGROUND REMOVAL COMPLETE")
    print("-" * 47)
    print("%-15s %-8s %-8s %-8s %-8s" % ("Category", "Total", "Done", "Skip", "Fail"))
    print("-" * 47)
    for r in results:
        print("%-15s %-8s %-8s %-8s %-8s" % (
            r["category"], r.get("total", 0),
            r["processed"], r["skipped"], r["failed"]))
    print("-" * 47)
    total_p = sum(r["processed"] for r in results)
    total_s = sum(r["skipped"] for r in results)
    total_f = sum(r["failed"] for r in results)
    print("%-15s %-8s %-8s %-8s %-8s" % ("TOTAL", "", total_p, total_s, total_f))
    print("Elapsed: %.1fs" % elapsed)

    report_path = processed_root / "batch_report.json"
    os.makedirs(processed_root, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"results": results, "elapsed_seconds": elapsed, "device": device}, f, indent=2)
    print("Report: %s" % report_path)


if __name__ == "__main__":
    main()

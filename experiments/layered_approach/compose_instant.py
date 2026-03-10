#!/usr/bin/env python3
"""
compose_instant.py - Fast table setting compositor.

Composites pre-processed product images onto the master template with
perspective transforms and shadows. Generates any combination in <0.5s.

Usage:
    # Single combination
    python compose_instant.py \\
        --charger "anna_blush" \\
        --plate "white_gold_rim" \\
        --napkin "satin_pink" \\
        --cutlery "gold_luxe"

    # Batch from CSV
    python compose_instant.py --batch combos.csv

    # List available products
    python compose_instant.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from perspective_and_shadow import prepare_item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "pipeline_config.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

def find_product_image(
    product_name: str,
    category: str,
    images_root: Path,
    processed_dir: str,
    product_dirs: dict,
) -> Optional[Path]:
    """Find a product image, preferring processed (RGBA) over raw."""
    processed_path = images_root / processed_dir / category
    raw_path = images_root / product_dirs.get(category, category)

    for search_dir in [processed_path, raw_path]:
        if not search_dir.is_dir():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = search_dir / f"{product_name}{ext}"
            if candidate.exists():
                return candidate
        for f in search_dir.iterdir():
            if f.stem.lower() == product_name.lower() and f.suffix.lower() in IMAGE_EXTENSIONS:
                return f

    return None
def list_available_products(images_root: Path, processed_dir: str, product_dirs: dict):
    """Print all available products by category."""
    for cat_name, cat_folder in product_dirs.items():
        print(f"\n{cat_name.upper()}:")

        processed_path = images_root / processed_dir / cat_name
        raw_path = images_root / cat_folder

        found = set()
        for search_dir in [processed_path, raw_path]:
            if search_dir.is_dir():
                for f in sorted(search_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        status = "RGBA" if search_dir == processed_path else "raw"
                        if f.stem not in found:
                            print(f"  {f.stem} ({status})")
                            found.add(f.stem)

        if not found:
            print(f"  (no images found in {raw_path} or {processed_path})")


def load_rgba(path: Path) -> Image.Image:
    """Load image and ensure RGBA mode."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def composite_at(
    canvas: Image.Image,
    item: Image.Image,
    center_x: float,
    center_y: float,
) -> Image.Image:
    """Paste an RGBA item onto canvas at the given center position (fractions)."""
    cw, ch = canvas.size
    px = int(center_x * cw) - item.width // 2
    py = int(center_y * ch) - item.height // 2
    canvas.paste(item, (px, py), item)
    return canvas

def compose_table_setting(
    template_path: Path,
    selections: Dict[str, str],
    config: dict,
    output_path: Path,
) -> float:
    """Compose a complete table setting and return elapsed time.

    Args:
        template_path: path to the master template image
        selections: mapping of category -> product name
        config: pipeline config dict
        output_path: where to save the result

    Returns:
        elapsed time in seconds
    """
    start = time.time()

    template = load_rgba(template_path)
    canvas = template.copy()
    cw, ch = canvas.size

    zones = config["zones"]
    scales = config["layer_scales"]
    offsets = config["layer_offsets"]
    z_orders = config["layer_z_order"]
    rotations = config["rotate_with_zone"]
    shadow_cfg = config["shadow"]
    camera_angle = config["camera"]["angle_degrees"]
    paths_cfg = config["paths"]

    images_root = Path(paths_cfg["images_root"])
    processed_dir = paths_cfg["processed_dir"]
    product_dirs = paths_cfg["product_dirs"]

    layers_to_render = []
    for cat in config["layer_order"]:
        if cat not in selections:
            continue

        product_name = selections[cat]
        img_path = find_product_image(
            product_name, cat, images_root, processed_dir, product_dirs,
        )

        if img_path is None:
            log.warning("Product not found: " + str(cat) + "/" + str(product_name))
            continue

        raw_img = load_rgba(img_path)
        scale = scales.get(cat, 0.10)
        offset = offsets.get(cat, [0, 0])
        z = z_orders.get(cat, 0)
        should_rotate = rotations.get(cat, False)
        if cat == "centerpiece":
            center = config["table"]["center"]
            item, shadow = prepare_item(
                raw_img, cw, ch, scale, camera_angle,
                rotation_deg=0, should_rotate=False,
                shadow_config=shadow_cfg,
            )
            layers_to_render.append((z, center[0], center[1], item, shadow))
        else:
            for zone in zones:
                zone_rot = zone["rotation_deg"]
                angle_rad = math.radians(zone_rot - 90)

                ox = offset[0] * math.cos(angle_rad) - offset[1] * math.sin(angle_rad)
                oy = offset[0] * math.sin(angle_rad) + offset[1] * math.cos(angle_rad)

                cx = zone["x"] + ox
                cy = zone["y"] + oy

                item, shadow = prepare_item(
                    raw_img, cw, ch, scale, camera_angle,
                    rotation_deg=zone_rot,
                    should_rotate=should_rotate,
                    shadow_config=shadow_cfg,
                )
                layers_to_render.append((z, cx, cy, item, shadow))

    layers_to_render.sort(key=lambda x: x[0])

    for z, cx, cy, item, shadow in layers_to_render:
        if shadow is not None:
            composite_at(canvas, shadow, cx, cy)
        composite_at(canvas, item, cx, cy)

    final = Image.new("RGB", canvas.size, (255, 255, 255))
    final.paste(canvas, mask=canvas.split()[3])

    os.makedirs(output_path.parent, exist_ok=True)
    final.save(output_path, quality=95)

    elapsed = time.time() - start
    return elapsed

def run_batch(csv_path: Path, config: dict, output_dir: Path):
    """Run compositions from a CSV file.

    CSV columns: output_name, charger, plate, napkin, cutlery, glassware, centerpiece
    """
    images_root = Path(config["paths"]["images_root"])
    template_path = images_root / config["template"]["image"]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    log.info(f"Batch: {len(rows)} combinations from {csv_path}")
    total_time = 0

    for i, row in enumerate(rows, 1):
        output_name = row.get("output_name", f"combo_{i:04d}")
        selections = {}
        for cat in config["layer_order"]:
            val = row.get(cat, "").strip()
            if val:
                selections[cat] = val

        out_path = output_dir / f"{output_name}.png"
        elapsed = compose_table_setting(template_path, selections, config, out_path)
        total_time += elapsed

        log.info("  [" + str(i) + "/" + str(len(rows)) + "] " + output_name + " -> " + str(round(elapsed,3)) + "s")

    avg = total_time / len(rows) if rows else 0
    log.info("Batch complete: " + str(len(rows)) + " images, avg " + str(round(avg,3)) + "s each, total " + str(round(total_time,1)) + "s")

def main():
    parser = argparse.ArgumentParser(description="Instant table setting compositor.")

    parser.add_argument("--charger", type=str, help="Charger plate product name.")
    parser.add_argument("--plate", type=str, help="Dinner plate product name.")
    parser.add_argument("--napkin", type=str, help="Napkin product name.")
    parser.add_argument("--cutlery", type=str, help="Cutlery product name.")
    parser.add_argument("--glassware", type=str, help="Glassware product name.")
    parser.add_argument("--centerpiece", type=str, help="Centerpiece product name.")

    parser.add_argument("--batch", type=str, help="CSV file with combinations.")
    parser.add_argument("--output", type=str, default=None, help="Output image path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for batch.")
    parser.add_argument("--list", action="store_true", help="List available products.")
    parser.add_argument("--images-root", type=str, default=None, help="Override images root.")

    args = parser.parse_args()
    config = load_config()

    if args.images_root:
        config["paths"]["images_root"] = args.images_root

    images_root = Path(config["paths"]["images_root"])

    if args.list:
        list_available_products(
            images_root,
            config["paths"]["processed_dir"],
            config["paths"]["product_dirs"],
        )
        return

    if args.batch:
        out_dir = Path(args.output_dir or str(images_root / config["paths"]["output_dir"]))
        run_batch(Path(args.batch), config, out_dir)
        return
    selections = {}
    if args.charger:     selections["charger"] = args.charger
    if args.plate:       selections["plate"] = args.plate
    if args.napkin:      selections["napkin"] = args.napkin
    if args.cutlery:     selections["cutlery"] = args.cutlery
    if args.glassware:   selections["glassware"] = args.glassware
    if args.centerpiece: selections["centerpiece"] = args.centerpiece

    if not selections:
        parser.error("Provide at least one product.")
    template_path = images_root / config["template"]["image"]
    if not template_path.exists():
        log.error("Template not found: " + str(template_path))
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        combo_name = "_".join(f"{k}-{v}" for k, v in sorted(selections.items()))
        out_dir = images_root / config["paths"]["output_dir"]
        output_path = out_dir / f"{combo_name}.png"

    elapsed = compose_table_setting(template_path, selections, config, output_path)

    log.info("Output: " + str(output_path))
    log.info("Time:   " + str(round(elapsed,3)) + "s")
    log.info("Size:   " + str(Image.open(output_path).size))

if __name__ == "__main__":
    main()

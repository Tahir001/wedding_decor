#!/usr/bin/env python3
"""
03_compose_table.py — Compose extracted decor items onto table backgrounds.

Takes transparent RGBA PNGs from 02_extract_items.py and composites them
onto new table/tablecloth backgrounds with configurable placement.

Supports two modes:
  1. Manual: explicit position/scale per layer
  2. Auto: circular place-setting layout around a round table

Usage:
  # Quick composite — single item on a background
  python 03_compose_table.py \
      --background tablecloth_red.jpg \
      --layer plate.png --pos 0.5,0.5 --scale 0.3 \
      --output composed.png

  # Full table from config file
  python 03_compose_table.py --config table_config.json --output composed.png

  # Auto-layout: evenly space items around a round table
  python 03_compose_table.py \
      --background tablecloth_red.jpg \
      --auto-layout \
      --plate plate_rgba.png \
      --napkin napkin_rgba.png \
      --cutlery cutlery_rgba.png \
      --glass glass_rgba.png \
      --centerpiece centerpiece_rgba.png \
      --num-settings 8 \
      --output full_table.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_SIZE = (1024, 1024)
DEFAULT_NUM_SETTINGS = 8
DEFAULT_TABLE_RADIUS = 0.35  # fraction of image width


# ===================================================================
# Data classes
# ===================================================================
@dataclass
class Layer:
    """A single compositing layer."""
    image_path: str
    position: Tuple[float, float]  # (x, y) as fraction of canvas [0, 1]
    scale: float = 0.15            # scale relative to canvas width
    rotation: float = 0.0          # degrees
    z_order: int = 0               # higher = on top


# ===================================================================
# Compositing engine
# ===================================================================
def load_rgba(path: str) -> Image.Image:
    """Load image and ensure RGBA mode."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def composite_layer(
    canvas: Image.Image,
    layer_img: Image.Image,
    position: Tuple[float, float],
    scale: float,
    rotation: float = 0.0,
) -> Image.Image:
    """Paste a single RGBA layer onto the canvas.
    
    Args:
        canvas: background image (RGBA)
        layer_img: item to composite (RGBA, transparent bg)
        position: (x, y) center position as fraction of canvas size
        scale: item width as fraction of canvas width
        rotation: degrees to rotate the item
    """
    cw, ch = canvas.size

    # Scale item
    target_w = int(cw * scale)
    aspect = layer_img.height / layer_img.width
    target_h = int(target_w * aspect)
    resized = layer_img.resize((target_w, target_h), Image.LANCZOS)

    # Rotate if needed
    if rotation != 0:
        resized = resized.rotate(-rotation, expand=True, resample=Image.BICUBIC)

    # Position (center-based)
    cx = int(position[0] * cw) - resized.width // 2
    cy = int(position[1] * ch) - resized.height // 2

    # Alpha composite
    canvas.paste(resized, (cx, cy), resized)
    return canvas


def compose_table(
    background_path: str,
    layers: List[Layer],
    output_path: str,
    output_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Compose all layers onto a background image."""
    log.info("Composing table: %d layers onto %s", len(layers), background_path)

    # Load and prepare background
    bg = Image.open(background_path).convert("RGB")
    if output_size:
        bg = bg.resize(output_size, Image.LANCZOS)
    canvas = bg.convert("RGBA")

    # Sort by z-order
    sorted_layers = sorted(layers, key=lambda l: l.z_order)

    for i, layer in enumerate(sorted_layers):
        if not os.path.exists(layer.image_path):
            log.warning("Layer not found: %s — skipping", layer.image_path)
            continue

        layer_img = load_rgba(layer.image_path)
        canvas = composite_layer(
            canvas, layer_img,
            position=layer.position,
            scale=layer.scale,
            rotation=layer.rotation,
        )
        log.info("  [%d] %s → pos=(%.2f, %.2f) scale=%.2f rot=%.1f°",
                 i + 1, Path(layer.image_path).name,
                 layer.position[0], layer.position[1],
                 layer.scale, layer.rotation)

    # Save as RGB (flatten alpha)
    final = Image.new("RGB", canvas.size, (255, 255, 255))
    final.paste(canvas, mask=canvas.split()[3])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.save(output_path, quality=95)
    log.info("Saved: %s (%d x %d)", output_path, final.width, final.height)

    return output_path


# ===================================================================
# Auto-layout: circular place settings
# ===================================================================
def generate_circular_layout(
    num_settings: int = DEFAULT_NUM_SETTINGS,
    table_radius: float = DEFAULT_TABLE_RADIUS,
    center: Tuple[float, float] = (0.5, 0.5),
    start_angle: float = -90,  # top of table
    items: Optional[Dict[str, str]] = None,
) -> List[Layer]:
    """Generate layers for evenly-spaced place settings around a round table.
    
    Each setting can include: plate, napkin, cutlery, glass.
    Centerpiece goes in the middle.
    
    Args:
        num_settings: number of place settings
        table_radius: radius as fraction of canvas width
        center: center of table as (x, y) fraction
        start_angle: angle of first setting in degrees (-90 = top)
        items: dict mapping item type → RGBA file path
            Keys: "plate", "napkin", "cutlery", "glass", "centerpiece"
    """
    if items is None:
        items = {}

    layers: List[Layer] = []
    angle_step = 360 / num_settings

    # Scale factors per item type (relative to canvas width)
    ITEM_SCALES = {
        "plate": 0.12,
        "napkin": 0.06,
        "cutlery": 0.08,
        "glass": 0.05,
    }

    # Offset from plate center (as fraction of canvas width)
    ITEM_OFFSETS = {
        "napkin": (0.0, 0.0),      # on plate
        "cutlery": (0.07, 0.0),    # right of plate
        "glass": (0.05, -0.04),    # above-right of plate
    }

    for i in range(num_settings):
        angle_deg = start_angle + i * angle_step
        angle_rad = math.radians(angle_deg)

        # Place setting center position
        px = center[0] + table_radius * math.cos(angle_rad)
        py = center[1] + table_radius * math.sin(angle_rad)

        # Rotation: items face toward table center
        face_angle = angle_deg + 90

        # Plate (base layer for each setting)
        if "plate" in items:
            layers.append(Layer(
                image_path=items["plate"],
                position=(px, py),
                scale=ITEM_SCALES["plate"],
                rotation=0,  # plates are rotationally symmetric
                z_order=10,
            ))

        # Napkin (on plate)
        if "napkin" in items:
            ox, oy = ITEM_OFFSETS["napkin"]
            layers.append(Layer(
                image_path=items["napkin"],
                position=(px + ox, py + oy),
                scale=ITEM_SCALES["napkin"],
                rotation=face_angle,
                z_order=20,
            ))

        # Cutlery (beside plate, rotated to face center)
        if "cutlery" in items:
            ox, oy = ITEM_OFFSETS["cutlery"]
            # Rotate offset around the setting
            rot_ox = ox * math.cos(angle_rad) - oy * math.sin(angle_rad)
            rot_oy = ox * math.sin(angle_rad) + oy * math.cos(angle_rad)
            layers.append(Layer(
                image_path=items["cutlery"],
                position=(px + rot_ox, py + rot_oy),
                scale=ITEM_SCALES["cutlery"],
                rotation=face_angle,
                z_order=15,
            ))

        # Glass (above-right of plate)
        if "glass" in items:
            ox, oy = ITEM_OFFSETS["glass"]
            rot_ox = ox * math.cos(angle_rad) - oy * math.sin(angle_rad)
            rot_oy = ox * math.sin(angle_rad) + oy * math.cos(angle_rad)
            layers.append(Layer(
                image_path=items["glass"],
                position=(px + rot_ox, py + rot_oy),
                scale=ITEM_SCALES["glass"],
                rotation=0,  # glasses are upright
                z_order=25,
            ))

    # Centerpiece
    if "centerpiece" in items:
        layers.append(Layer(
            image_path=items["centerpiece"],
            position=center,
            scale=0.18,
            rotation=0,
            z_order=30,
        ))

    log.info("Generated %d layers for %d place settings", len(layers), num_settings)
    return layers


# ===================================================================
# Config file support
# ===================================================================
def load_config(config_path: str) -> Tuple[str, List[Layer], Optional[Tuple[int, int]]]:
    """Load composition config from JSON.
    
    Example config:
    {
        "background": "tablecloth_red.jpg",
        "output_size": [1024, 1024],
        "layers": [
            {
                "image": "plate_rgba.png",
                "position": [0.5, 0.5],
                "scale": 0.2,
                "rotation": 0,
                "z_order": 10
            }
        ]
    }
    """
    with open(config_path) as f:
        cfg = json.load(f)

    background = cfg["background"]
    output_size = tuple(cfg["output_size"]) if "output_size" in cfg else None

    layers = []
    for lc in cfg.get("layers", []):
        layers.append(Layer(
            image_path=lc["image"],
            position=tuple(lc["position"]),
            scale=lc.get("scale", 0.15),
            rotation=lc.get("rotation", 0),
            z_order=lc.get("z_order", 0),
        ))

    return background, layers, output_size


# ===================================================================
# CLI
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compose extracted decor items onto table backgrounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--output", type=str, required=True, help="Output image path.")

    # Config mode
    p.add_argument("--config", type=str, help="JSON config file for composition.")

    # Manual mode
    manual = p.add_argument_group("Manual mode")
    manual.add_argument("--background", type=str, help="Background image path.")
    manual.add_argument("--layer", type=str, action="append", help="Layer RGBA PNG (can repeat).")
    manual.add_argument("--pos", type=str, action="append", help="Position as x,y fraction (matches --layer order).")
    manual.add_argument("--scale", type=float, action="append", help="Scale per layer (matches --layer order).")

    # Auto-layout mode
    auto = p.add_argument_group("Auto-layout mode")
    auto.add_argument("--auto-layout", action="store_true", help="Use circular place-setting layout.")
    auto.add_argument("--plate", type=str, help="Plate RGBA PNG.")
    auto.add_argument("--napkin", type=str, help="Napkin RGBA PNG.")
    auto.add_argument("--cutlery", type=str, help="Cutlery RGBA PNG.")
    auto.add_argument("--glass", type=str, help="Glass RGBA PNG.")
    auto.add_argument("--centerpiece", type=str, help="Centerpiece RGBA PNG.")
    auto.add_argument("--num-settings", type=int, default=DEFAULT_NUM_SETTINGS, help="Number of place settings.")
    auto.add_argument("--table-radius", type=float, default=DEFAULT_TABLE_RADIUS, help="Table radius (fraction).")

    # Shared
    p.add_argument("--size", type=str, default=None, help="Output size as WxH (e.g. 1024x1024).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_size = None
    if args.size:
        w, h = args.size.split("x")
        output_size = (int(w), int(h))

    if args.config:
        # Config mode
        background, layers, cfg_size = load_config(args.config)
        output_size = output_size or cfg_size
        compose_table(background, layers, args.output, output_size)

    elif args.auto_layout:
        # Auto-layout mode
        if not args.background:
            parser.error("--auto-layout requires --background.")

        items = {}
        if args.plate: items["plate"] = args.plate
        if args.napkin: items["napkin"] = args.napkin
        if args.cutlery: items["cutlery"] = args.cutlery
        if args.glass: items["glass"] = args.glass
        if args.centerpiece: items["centerpiece"] = args.centerpiece

        if not items:
            parser.error("--auto-layout requires at least one item (--plate, --napkin, etc.).")

        layers = generate_circular_layout(
            num_settings=args.num_settings,
            table_radius=args.table_radius,
            items=items,
        )
        compose_table(args.background, layers, args.output, output_size or DEFAULT_OUTPUT_SIZE)

    elif args.layer:
        # Manual mode
        if not args.background:
            parser.error("Manual mode requires --background.")

        layers = []
        for i, layer_path in enumerate(args.layer):
            pos = (0.5, 0.5)
            if args.pos and i < len(args.pos):
                x, y = args.pos[i].split(",")
                pos = (float(x), float(y))
            scale = 0.15
            if args.scale and i < len(args.scale):
                scale = args.scale[i]
            layers.append(Layer(image_path=layer_path, position=pos, scale=scale, z_order=i))

        compose_table(args.background, layers, args.output, output_size)

    else:
        parser.error("Provide --config, --auto-layout, or --layer/--background.")


if __name__ == "__main__":
    main()
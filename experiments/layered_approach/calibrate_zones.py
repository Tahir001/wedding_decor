#!/usr/bin/env python3
"""
calibrate_zones.py - Interactive zone calibration for the layered pipeline.

Opens the template image and lets you click to mark place-setting centers.
Saves zone positions to pipeline_config.json.

Usage:
    python calibrate_zones.py                           # uses config defaults
    python calibrate_zones.py --template my_table.png   # custom template
    python calibrate_zones.py --num-settings 6          # 6 place settings
    python calibrate_zones.py --auto                    # auto-generate circular zones (no clicking)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "pipeline_config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Config saved to {CONFIG_PATH}")


def auto_generate_zones(
    num_settings: int,
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.32,
    start_angle: float = -90,
) -> list[dict]:
    """Generate evenly-spaced zones in a circle."""
    zones = []
    step = 360.0 / num_settings
    for i in range(num_settings):
        angle_deg = start_angle + i * step
        angle_rad = math.radians(angle_deg)
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        rot = (angle_deg + 90) % 360
        zones.append({
            "id": i,
            "x": round(x, 3),
            "y": round(y, 3),
            "rotation_deg": round(rot, 1),
        })
    return zones


def interactive_calibrate(template_path: str, num_settings: int) -> list[dict]:
    """Open template image and collect clicks for zone centers."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("ERROR: matplotlib and Pillow are required for interactive mode.")
        print("  pip install matplotlib Pillow")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    img = Image.open(template_path)
    w, h = img.size

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(
        f"Click {num_settings} place-setting centers in order.\n"
        f"Right-click to undo last point. Close window when done."
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    points = []
    markers = []

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.button == 3 and points:
            points.pop()
            m = markers.pop()
            m.remove()
            ax.set_title(f"Clicked {len(points)}/{num_settings}. Right-click to undo.")
            fig.canvas.draw()
            return

        if event.button == 1 and len(points) < num_settings:
            px, py = event.xdata, event.ydata
            points.append((px / w, py / h))
            m = ax.plot(px, py, "r+", markersize=15, markeredgewidth=2)[0]
            markers.append(m)
            ax.annotate(
                str(len(points)),
                (px, py),
                textcoords="offset points",
                xytext=(10, -10),
                fontsize=12,
                color="red",
                fontweight="bold",
            )
            fig.canvas.draw()

            if len(points) >= num_settings:
                ax.set_title(f"All {num_settings} zones marked. Close window to save.")
                fig.canvas.draw()
            else:
                ax.set_title(f"Clicked {len(points)}/{num_settings}. Right-click to undo.")
                fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(points) < num_settings:
        print(f"WARNING: Only {len(points)} of {num_settings} zones clicked.")

    table_cx, table_cy = 0.5, 0.5
    zones = []
    for i, (fx, fy) in enumerate(points):
        angle_to_center = math.degrees(math.atan2(table_cy - fy, table_cx - fx))
        rot = (angle_to_center + 90) % 360
        zones.append({
            "id": i,
            "x": round(fx, 3),
            "y": round(fy, 3),
            "rotation_deg": round(rot, 1),
        })

    return zones


def main():
    parser = argparse.ArgumentParser(description="Calibrate place-setting zones on template.")
    parser.add_argument("--template", type=str, help="Path to template image.")
    parser.add_argument("--num-settings", type=int, default=8, help="Number of place settings.")
    parser.add_argument("--auto", action="store_true", help="Auto-generate circular layout (no GUI).")
    parser.add_argument("--radius", type=float, default=None, help="Table radius as fraction of canvas.")
    parser.add_argument("--camera-angle", type=float, default=None, help="Camera angle in degrees.")
    args = parser.parse_args()

    cfg = load_config()

    if args.auto:
        radius = args.radius or cfg.get("table", {}).get("radius", 0.32)
        center = tuple(cfg.get("table", {}).get("center", [0.5, 0.5]))
        start = cfg.get("table", {}).get("start_angle_degrees", -90)

        zones = auto_generate_zones(args.num_settings, center, radius, start)
        print(f"Auto-generated {len(zones)} zones (radius={radius}):")
        for z in zones:
            print(f"  Zone {z["id"]}: ({z["x"]:.3f}, {z["y"]:.3f}) rot={z["rotation_deg"]}")
    else:
        template = args.template
        if not template:
            images_root = cfg.get("paths", {}).get("images_root", ".")
            template_name = cfg.get("template", {}).get("image", "base_image_table.png")
            template = os.path.join(images_root, template_name)

        print(f"Template: {template}")
        print(f"Settings: {args.num_settings}")
        zones = interactive_calibrate(template, args.num_settings)

    cfg["zones"] = zones
    cfg.setdefault("table", {})["num_settings"] = len(zones)

    if args.camera_angle is not None:
        cfg.setdefault("camera", {})["angle_degrees"] = args.camera_angle

    if args.radius is not None:
        cfg.setdefault("table", {})["radius"] = args.radius

    save_config(cfg)
    print(f"\nSaved {len(zones)} zones to {CONFIG_PATH}")


if __name__ == "__main__":
    main()

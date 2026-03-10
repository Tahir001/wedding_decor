#!/usr/bin/env python3
"""
run_pipeline.py - Main orchestrator for the layered compositing pipeline.

Usage:
    python run_pipeline.py setup [--device cuda] [--force]
    python run_pipeline.py calibrate [--auto] [--num-settings 8]
    python run_pipeline.py compose --plate X --napkin Y [--cutlery Z] ...
    python run_pipeline.py batch --combos combos.csv
    python run_pipeline.py catalog [--max 100]
    python run_pipeline.py list
"""

from __future__ import annotations
import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "pipeline_config.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)
def cmd_setup(args):
    log.info("=== SETUP: Background removal ===")

    from batch_remove_bg import main as bg_main
    sys.argv = ["batch_remove_bg.py"]
    if args.device:
        sys.argv.extend(["--device", args.device])

    if args.category:
        sys.argv.extend(["--category", args.category])

    if args.force:
        sys.argv.append("--force")

    if args.images_root:
        sys.argv.extend(["--images-root", args.images_root])

    bg_main()
def cmd_calibrate(args):
    log.info("=== CALIBRATE: Zone positions ===")

    from calibrate_zones import main as cal_main
    sys.argv = ["calibrate_zones.py"]
    if args.auto:
        sys.argv.append("--auto")

    if args.num_settings:
        sys.argv.extend(["--num-settings", str(args.num_settings)])

    if args.template:
        sys.argv.extend(["--template", args.template])

    if args.radius:
        sys.argv.extend(["--radius", str(args.radius)])

    if args.camera_angle:
        sys.argv.extend(["--camera-angle", str(args.camera_angle)])

    cal_main()
def cmd_compose(args):
    from compose_instant import compose_table_setting, load_config as load_cfg
    config = load_cfg()

    if args.images_root:
        config["paths"]["images_root"] = args.images_root
    images_root = Path(config["paths"]["images_root"])

    template_path = images_root / config["template"]["image"]
    if not template_path.exists():
        log.error("Template not found: " + str(template_path))

        sys.exit(1)

    selections = {}
    if args.charger: selections["charger"] = args.charger
    if args.plate: selections["plate"] = args.plate
    if args.napkin: selections["napkin"] = args.napkin
    if args.cutlery: selections["cutlery"] = args.cutlery
    if args.glassware: selections["glassware"] = args.glassware
    if args.centerpiece: selections["centerpiece"] = args.centerpiece
    if not selections:
        log.error("Provide at least one product.")

        sys.exit(1)

    if args.output:
        output_path = Path(args.output)

    else:
        combo = "_".join(k + "-" + v for k, v in sorted(selections.items()))

        out_dir = images_root / config["paths"]["output_dir"]
        output_path = out_dir / (combo + ".png")

    elapsed = compose_table_setting(template_path, selections, config, output_path)

    log.info("Output: " + str(output_path))

    log.info("Time: " + str(round(elapsed, 3)) + "s")
def cmd_batch(args):
    from compose_instant import run_batch, load_config as load_cfg
    config = load_cfg()

    if args.images_root:
        config["paths"]["images_root"] = args.images_root
    images_root = Path(config["paths"]["images_root"])

    out_dir = Path(args.output_dir) if args.output_dir else images_root / config["paths"]["output_dir"]["paths"]["output_dir"]
    run_batch(Path(args.combos), config, out_dir)
def cmd_catalog(args):
    config = load_config()

    if args.images_root:
        config["paths"]["images_root"] = args.images_root
    images_root = Path(config["paths"]["images_root"])

    processed_dir = config["paths"]["processed_dir"]
    product_dirs = config["paths"]["product_dirs"]
    products_by_cat = {}
    for cat, folder in product_dirs.items():
        proc_path = images_root / processed_dir / cat
        raw_path = images_root / folder
        items = set()

        for d in [proc_path, raw_path]:
            if d.is_dir():
                for f in d.iterdir():
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        items.add(f.stem)

        if items:
            products_by_cat[cat] = sorted(items)

    for cat, items in products_by_cat.items():
        log.info("{}".format(cat) + ": " + str(len(items)) + " products")

    for cat, items in products_by_cat.items():
        log.info("{0}: {1} products".format(cat, len(items)))

    cats = [c for c in config["layer_order"] if c in products_by_cat]
    lists = [products_by_cat[c] for c in cats]
    total = 1
    for l in lists:
        total = total * len(l)

    log.info("Total combinations: {0}".format(total))

    limit = args.max or total
    if limit < total:
        log.info("Limiting to {0} combinations".format(limit))

    from compose_instant import compose_table_setting
    template_path = images_root / config["template"]["image"]
    out_dir = Path(args.output_dir) if args.output_dir else images_root / config["paths"]["output_dir"]["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    t0 = time.time()

    for combo in itertools.product(*lists):
        if count >= limit:
            break
        selections = dict(zip(cats, combo))

        name = "_".join(k + "-" + v for k, v in selections.items())
        out_path = out_dir / (name + ".png")

        elapsed = compose_table_setting(template_path, selections, config, out_path)

        count = count + 1
        if count % 10 == 0 or count == 1:
            log.info("[{0}/{1}] {2} {3}s".format(count, min(limit, total), name, round(elapsed, 3)))

    wall = time.time() - t0
    avg = wall / count if count > 0 else 0
    log.info("Generated {0} images in {1}s (avg {2}s each)".format(count, round(wall, 1), round(avg, 3)))

def cmd_list(args):
    from compose_instant import list_available_products, load_config as load_cfg
    config = load_cfg()

    if args.images_root:
        config["paths"]["images_root"] = args.images_root
    images_root = Path(config["paths"]["images_root"])

    list_available_products(images_root, config["paths"]["processed_dir"], config["paths"]["product_dirs"])
def main():
    parser = argparse.ArgumentParser(description="Layered compositing pipeline orchestrator.")

    parser.add_argument("--images-root", type=str, default=None, help="Override images root.")

    sub = parser.add_subparsers(dest="command", help="Pipeline command")


    p_setup = sub.add_parser("setup", help="Run background removal on all products.")

    p_setup.add_argument("--device", type=str, default=None)

    p_setup.add_argument("--category", type=str, default=None)

    p_setup.add_argument("--force", action="store_true")


    p_cal = sub.add_parser("calibrate", help="Calibrate zone positions.")

    p_cal.add_argument("--auto", action="store_true")

    p_cal.add_argument("--num-settings", type=int, default=None)

    p_cal.add_argument("--template", type=str, default=None)

    p_cal.add_argument("--radius", type=float, default=None)

    p_cal.add_argument("--camera-angle", type=float, default=None)


    p_comp = sub.add_parser("compose", help="Compose a single table setting.")

    p_comp.add_argument("--charger", type=str)

    p_comp.add_argument("--plate", type=str)

    p_comp.add_argument("--napkin", type=str)

    p_comp.add_argument("--cutlery", type=str)

    p_comp.add_argument("--glassware", type=str)

    p_comp.add_argument("--centerpiece", type=str)

    p_comp.add_argument("--output", type=str, default=None)


    p_batch = sub.add_parser("batch", help="Batch compose from CSV.")

    p_batch.add_argument("--combos", type=str, required=True)

    p_batch.add_argument("--output-dir", type=str, default=None)


    p_cat = sub.add_parser("catalog", help="Generate all possible combos.")

    p_cat.add_argument("--max", type=int, default=None, help="Max combinations.")

    p_cat.add_argument("--output-dir", type=str, default=None)


    sub.add_parser("list", help="List available products.")


    args = parser.parse_args()

    if not args.command:
        parser.print_help()

        sys.exit(1)


    os.chdir(SCRIPT_DIR)


    commands = {
        "setup": cmd_setup,
        "calibrate": cmd_calibrate,
        "compose": cmd_compose,
        "batch": cmd_batch,
        "catalog": cmd_catalog,
        "list": cmd_list,
    }
    commands[args.command](args)

if __name__ == "__main__":
    main()
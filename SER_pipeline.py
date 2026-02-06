"""
===============================================================================
WEDDING DECOR VISUALIZATION - MODULAR COMBINATORIAL PIPELINE
===============================================================================
Based on V12C (Qwen-Image-Edit-2511 + Lightning LoRA)

MODES:
  1. batch_tablecloths  - Apply N tablecloths to base image
  2. combo_preview       - Small combinatorial test (2x2x2)
  3. combo_full          - Full combinatorial run (all items)

USAGE:
  python wedding_decor_pipeline.py --mode batch_tablecloths --limit 10
  python wedding_decor_pipeline.py --mode combo_preview
  python wedding_decor_pipeline.py --mode combo_full
  python wedding_decor_pipeline.py --mode batch_tablecloths --items "Velvet Black,Velvet Blush,Woven Navy"
===============================================================================
"""

import os
import gc
import sys
import time
import math
import glob
import argparse
import itertools
import torch
from PIL import Image
from datetime import datetime
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = "/workspace/wedding_decor/images"
RENTALS_DIR = os.path.join(INPUT_DIR, "SpecialEventsRentals")
BASE_IMAGE_PATH = os.path.join(INPUT_DIR, "base_image_table.png")
OUTPUT_BASE = os.path.join(INPUT_DIR, "output")

# Subdirectories for each layer's output
OUTPUT_TABLECLOTHS = os.path.join(OUTPUT_BASE, "tablecloths")
OUTPUT_RUNNERS = os.path.join(OUTPUT_BASE, "table_runners")
OUTPUT_CHAIRS = os.path.join(OUTPUT_BASE, "chairs")
OUTPUT_COMBOS = os.path.join(OUTPUT_BASE, "combos")

# Image dimensions
FIXED_WIDTH = 768
FIXED_HEIGHT = 768
REF_SIZE = 384

# Model config
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42
NUM_STEPS = 4

# =============================================================================
# LAYER DEFINITIONS - Prompt templates for each decor layer
# =============================================================================
# Each layer defines how to compose the edit prompt.
# {item_name} gets replaced with a descriptive name from the filename.
# =============================================================================

LAYER_CONFIG = {
    "tablecloth": {
        "source_dir": os.path.join(RENTALS_DIR, "table_cloths"),
        "output_dir": OUTPUT_TABLECLOTHS,
        "prompt_template": "Replace the tablecloth in image 1 with the {item_name} tablecloth from image 2. Keep everything else the same.",
    },
    "table_runner": {
        "source_dir": os.path.join(RENTALS_DIR, "table_runners"),
        "output_dir": OUTPUT_RUNNERS,
        "prompt_template": "Add the {item_name} table runner from image 2 on top of the tablecloth in image 1. Keep everything else the same.",
    },
    "chair": {
        "source_dir": os.path.join(RENTALS_DIR, "Chairs"),
        "output_dir": OUTPUT_CHAIRS,
        "prompt_template": "Replace all chairs in image 1 with the {item_name} chair from image 2. Place chairs evenly around the round table.",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def print_banner(text, char="="):
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def slugify(name):
    """Convert a filename to a clean slug for output naming."""
    # Strip extension, replace spaces/special chars
    name = os.path.splitext(name)[0]
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()


def item_display_name(filename):
    """Extract a human-readable name from the filename for prompts."""
    name = os.path.splitext(filename)[0]
    # Remove common prefixes like "120 Round - ", "Rectangle Velvet - "
    for prefix in ["120 Round - ", "120 Round ", "Rectangle Velvet - ", "Rectangle Woven - ",
                    "Chair - "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip()


def resize_to_fixed(img, width=FIXED_WIDTH, height=FIXED_HEIGHT):
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img, size=REF_SIZE):
    return img.resize((size, size), Image.LANCZOS)


def discover_items(source_dir):
    """Find all image files in a directory, return sorted list of (filename, filepath)."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.avif'}
    items = []
    if not os.path.exists(source_dir):
        print(f"⚠️  Directory not found: {source_dir}")
        return items
    for f in sorted(os.listdir(source_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in extensions:
            items.append((f, os.path.join(source_dir, f)))
    return items


def select_items(items, limit=None, names_filter=None):
    """
    Filter items by:
      - limit: take first N items
      - names_filter: comma-separated substrings to match against filenames
    """
    if names_filter:
        keywords = [k.strip().lower() for k in names_filter.split(",")]
        filtered = [(f, p) for f, p in items if any(k in f.lower() for k in keywords)]
        if not filtered:
            print(f"⚠️  No items matched filter: {names_filter}")
            print(f"   Available: {[f for f, _ in items[:10]]}...")
        return filtered

    if limit and limit < len(items):
        return items[:limit]

    return items


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL: Qwen-Image-Edit-2511")

    gc.collect()
    torch.cuda.empty_cache()

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    print(f"📦 Loading {MODEL_NAME}...")
    load_start = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_NAME,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(f"✅ Base model loaded in {time.time() - load_start:.1f}s")

    print(f"⚡ Loading 4-step Lightning LoRA...")
    pipeline.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS)
    print("✅ LoRA loaded")

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def warmup(pipeline):
    print("\n🔥 Warmup...")
    dummy = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), 'white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), 'gray')

    with torch.inference_mode():
        _ = pipeline(
            image=[dummy, dummy_ref],
            prompt="warmup",
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
        )

    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Warmup complete")


# =============================================================================
# CORE EDIT FUNCTION
# =============================================================================

def apply_edit(pipeline, base_img, ref_img, prompt, step_seed=SEED):
    """Apply a single Qwen image edit. Returns (result_image, elapsed_seconds)."""
    torch.cuda.synchronize()
    start = time.time()

    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(step_seed),
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)

    return result, elapsed


def apply_layer(pipeline, base_img, layer_name, ref_filepath, ref_filename, seed=SEED):
    """
    Apply a single layer (tablecloth/runner/chair) to a base image.
    Returns (result_image, elapsed, prompt_used).
    """
    config = LAYER_CONFIG[layer_name]
    display_name = item_display_name(ref_filename)
    prompt = config["prompt_template"].format(item_name=display_name)

    ref_img = resize_reference(Image.open(ref_filepath).convert("RGB"))
    result, elapsed = apply_edit(pipeline, base_img, ref_img, prompt, seed)

    return result, elapsed, prompt


# =============================================================================
# MODE 1: BATCH SINGLE LAYER
# =============================================================================

def batch_single_layer(pipeline, base_img, layer_name, limit=None, names_filter=None):
    """Apply all (or filtered) items of a single layer to the base image."""
    config = LAYER_CONFIG[layer_name]
    all_items = discover_items(config["source_dir"])
    items = select_items(all_items, limit=limit, names_filter=names_filter)

    if not items:
        print(f"❌ No {layer_name} items found!")
        return {}

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print_banner(f"BATCH: {len(items)} {layer_name}(s) @ {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"📂 Source: {config['source_dir']}")
    print(f"📂 Output: {output_dir}")
    print(f"📋 Items: {[item_display_name(f) for f, _ in items]}\n")

    results = {}
    for i, (filename, filepath) in enumerate(items, 1):
        slug = slugify(filename)
        display = item_display_name(filename)
        print(f"\n[{i}/{len(items)}] 🎨 {display}")

        result, elapsed, prompt = apply_layer(
            pipeline, base_img, layer_name, filepath, filename, seed=SEED + i
        )

        out_path = os.path.join(output_dir, f"{slug}.png")
        result.save(out_path)
        print(f"   ⏱️  {elapsed:.2f}s → {out_path}")

        results[slug] = {
            "image": result,
            "path": out_path,
            "source": filename,
            "display_name": display,
            "prompt": prompt,
            "time": elapsed,
        }

    # Summary
    print_banner(f"BATCH COMPLETE: {len(results)} {layer_name}(s)")
    total_time = sum(r["time"] for r in results.values())
    for slug, r in results.items():
        print(f"  {r['display_name']:<40} {r['time']:.2f}s")
    print(f"\n  Total inference: {total_time:.2f}s")
    print(f"  Avg per item:    {total_time / len(results):.2f}s")

    return results


# =============================================================================
# MODE 2: COMBINATORIAL PIPELINE
# =============================================================================

def combo_pipeline(pipeline, base_img, layer_specs):
    """
    Run combinatorial pipeline across multiple layers.

    layer_specs is a list of dicts:
      [
        {"layer": "tablecloth", "limit": 2, "names_filter": None},
        {"layer": "table_runner", "limit": 2, "names_filter": None},
        {"layer": "chair", "limit": 2, "names_filter": None},
      ]

    Each layer's outputs feed as base images into the next layer.
    """
    os.makedirs(OUTPUT_COMBOS, exist_ok=True)

    # Discover items for each layer
    layer_items = []
    for spec in layer_specs:
        config = LAYER_CONFIG[spec["layer"]]
        all_items = discover_items(config["source_dir"])
        items = select_items(all_items, limit=spec.get("limit"), names_filter=spec.get("names_filter"))
        layer_items.append((spec["layer"], items))
        print(f"📋 {spec['layer']}: {len(items)} items → {[item_display_name(f) for f, _ in items]}")

    # Calculate total combos
    total = 1
    for _, items in layer_items:
        total *= len(items)
    print(f"\n🔢 Total combinations: {total}")

    # Build combos recursively: at each layer, take all current base images
    # and apply each item, producing new base images for the next layer.
    # current_states: list of (combo_name_parts, image)
    current_states = [( [], base_img )]

    for layer_idx, (layer_name, items) in enumerate(layer_items):
        print_banner(f"LAYER {layer_idx + 1}: {layer_name} ({len(items)} items × {len(current_states)} bases)")
        next_states = []

        for state_idx, (name_parts, state_img) in enumerate(current_states):
            for item_idx, (filename, filepath) in enumerate(items):
                slug = slugify(filename)
                display = item_display_name(filename)
                new_parts = name_parts + [slug]
                combo_name = "__".join(new_parts)

                global_idx = state_idx * len(items) + item_idx + 1
                print(f"  [{global_idx}/{len(current_states) * len(items)}] "
                      f"{' + '.join([item_display_name(p) for p in name_parts] + [display]) if name_parts else display}")

                result, elapsed, prompt = apply_layer(
                    pipeline, state_img, layer_name, filepath, filename,
                    seed=SEED + layer_idx * 100 + item_idx
                )

                # Save intermediate
                layer_dir = os.path.join(OUTPUT_COMBOS, layer_name)
                os.makedirs(layer_dir, exist_ok=True)
                out_path = os.path.join(layer_dir, f"{combo_name}.png")
                result.save(out_path)
                print(f"     ⏱️  {elapsed:.2f}s → {out_path}")

                next_states.append((new_parts, result))

        current_states = next_states

    # Save final combos
    final_dir = os.path.join(OUTPUT_COMBOS, "final")
    os.makedirs(final_dir, exist_ok=True)
    print_banner(f"SAVING {len(current_states)} FINAL COMBINATIONS")
    for name_parts, img in current_states:
        combo_name = "__".join(name_parts)
        out_path = os.path.join(final_dir, f"{combo_name}.png")
        img.save(out_path)
        print(f"  💾 {out_path}")

    print(f"\n✅ {len(current_states)} combinations saved to {final_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Wedding Decor Combinatorial Pipeline")

    parser.add_argument("--mode", choices=["batch_tablecloths", "batch_runners", "batch_chairs",
                                           "combo_preview", "combo_full", "list"],
                        default="batch_tablecloths",
                        help="Pipeline mode")

    parser.add_argument("--limit", type=int, default=None,
                        help="Max items per layer (for batch modes)")

    parser.add_argument("--items", type=str, default=None,
                        help="Comma-separated name filters (e.g. 'Velvet Black,Woven Navy')")

    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup inference")

    return parser.parse_args()


def list_available():
    """Print all available items per layer."""
    for layer_name, config in LAYER_CONFIG.items():
        items = discover_items(config["source_dir"])
        print_banner(f"{layer_name.upper()} ({len(items)} items)")
        for i, (f, _) in enumerate(items, 1):
            print(f"  {i:3d}. {item_display_name(f):<45} [{f}]")


def main():
    args = parse_args()

    if args.mode == "list":
        list_available()
        return

    print(f"🚀 Wedding Decor Pipeline @ {datetime.now().strftime('%H:%M:%S')}")
    print(f"🖥️  {torch.cuda.get_device_name(0)}")
    print(f"📋 Mode: {args.mode}")

    # Load model
    pipeline = load_pipeline()

    if not args.no_warmup:
        warmup(pipeline)

    # Load base image
    base_img = resize_to_fixed(Image.open(BASE_IMAGE_PATH).convert("RGB"))

    # ---- BATCH MODES ----
    if args.mode == "batch_tablecloths":
        batch_single_layer(pipeline, base_img, "tablecloth",
                           limit=args.limit or 10, names_filter=args.items)

    elif args.mode == "batch_runners":
        batch_single_layer(pipeline, base_img, "table_runner",
                           limit=args.limit or 10, names_filter=args.items)

    elif args.mode == "batch_chairs":
        batch_single_layer(pipeline, base_img, "chair",
                           limit=args.limit or 10, names_filter=args.items)

    # ---- COMBO PREVIEW (2 x 2 x 2 = 8 combos) ----
    elif args.mode == "combo_preview":
        combo_pipeline(pipeline, base_img, [
            {"layer": "tablecloth", "limit": 2},
            {"layer": "table_runner", "limit": 2},
            {"layer": "chair", "limit": 2},
        ])

    # ---- COMBO FULL (all items, all combos) ----
    elif args.mode == "combo_full":
        combo_pipeline(pipeline, base_img, [
            {"layer": "tablecloth"},
            {"layer": "table_runner"},
            {"layer": "chair"},
        ])

    print(f"\n✨ Done @ {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
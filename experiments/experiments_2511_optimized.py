"""
===============================================================================
WEDDING DECOR - OPTIMIZED TABLECLOTH EXPERIMENTS (QWEN-IMAGE-EDIT-2511 + LIGHTNING)
===============================================================================
Optimized experiment runner based on Experiment 15 baseline config.

Two tests sweeping step counts x CFG scales (4 tablecloths each):
  Test 1 – Positive-only prompts + step/CFG sweep, default base image
  Test 2 – Same prompts, but base image matched to fabric type

Usage:
    python experiments_2511_optimized.py                  # run both tests
    python experiments_2511_optimized.py --dry-run        # print config, don't run
    python experiments_2511_optimized.py --test 1         # run only Test 1
    python experiments_2511_optimized.py --test 2         # run only Test 2
    python experiments_2511_optimized.py --output-dir /custom/path
===============================================================================
"""

import os
import gc
import sys
import time
import math
import argparse
import torch
from PIL import Image
from datetime import datetime
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# PATHS (relative to this script's directory)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "tablecloths", "optimized_10")

# Base images
BASE_IMAGES = {
    "default": os.path.join(INPUT_DIR, "base_image.png"),
    "damask": os.path.join(INPUT_DIR, "base_image_damask.png"),
    "woven": os.path.join(INPUT_DIR, "base_image_woven.png"),
    "pintuck": os.path.join(INPUT_DIR, "base_image_pintuck.png"),
    "velvet": os.path.join(INPUT_DIR, "base_image_velvet.png"),
}

# =============================================================================
# TABLECLOTH DEFINITIONS
# =============================================================================

TABLECLOTH_DIR = os.path.join(INPUT_DIR, "tablecloths")

TABLECLOTHS = [
    {
        "name": "Tan Buffalo Check",
        "filename": "120 Round - Tan Buffalo Check.jpg",
        "color": "tan",
        "material": "buffalo check",
        "fabric_type": "default",
    },
    {
        "name": "Damask Black",
        "filename": "120 Round Damask - Black.png",
        "color": "black",
        "material": "damask",
        "fabric_type": "damask",
    },
    {
        "name": "Pintuck Taffeta Red",
        "filename": "120 Round Pintuck Taffeta - Red.jpg",
        "color": "red",
        "material": "pintuck taffeta",
        "fabric_type": "pintuck",
    },
    {
        "name": "Woven Celery",
        "filename": "120 Round Woven Celery.png",
        "color": "celery",
        "material": "woven",
        "fabric_type": "woven",
    },
]

# =============================================================================
# MODEL CONFIG (held constant)
# =============================================================================

MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384
GUIDANCE_SCALE = 1.0  # placeholder, no effect
SEED = 42

# =============================================================================
# EXPERIMENT 15 BASELINE CONFIG
# =============================================================================

TRUE_CFG_SCALES = [1.5, 2.0, 3.0]
STEP_COUNTS = [4, 8, 12]

# Negative prompts are non-functional in Qwen-Image-Edit-2511 (flow-matching
# architecture with Qwen2.5-VL encoder, never trained for negative conditioning).
# A single space is required to prevent pipeline errors.
TARGETED_NEGATIVE_PROMPT = " "


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_prompt(tablecloth):
    """Generate a positive-only, natural-language prompt for a tablecloth swap.

    Uses vivid descriptive language that the Qwen2.5-VL text encoder handles
    well.  All negations ("no", "not", "without") are avoided because the
    model can latch onto the unwanted concept tokens.
    """
    color = tablecloth["color"]
    material = tablecloth["material"]
    return (
        f"Replace only the tablecloth in image 1 with the {color} {material} "
        f"tablecloth shown in image 2. Use the exact color, texture, and pattern "
        f"from image 2. The tablecloth covers the round table completely, draping "
        f"smoothly over the edges with soft, natural fabric folds. The color is "
        f"even and consistent across the entire surface under uniform, balanced "
        f"lighting. The fabric looks like real {material} cloth with authentic "
        f"textile texture. Preserve the chairs, background, and all surroundings "
        f"exactly as they appear in image 1. Photorealistic, professional event "
        f"photography quality."
    )


# =============================================================================
# BASE IMAGE ROUTING
# =============================================================================

def get_base_image(filename):
    """Route to the appropriate base image based on fabric keywords in filename.

    Priority order (checked against lowercase filename):
        1. "damask"        -> base_image_damask.png
        2. "woven"         -> base_image_woven.png
        3. "pintuck"       -> base_image_pintuck.png
        4. "velvet"        -> base_image_velvet.png
        5. default         -> base_image.png
    """
    lower = filename.lower()
    if "damask" in lower:
        return BASE_IMAGES["damask"]
    elif "woven" in lower:
        return BASE_IMAGES["woven"]
    elif "pintuck" in lower:
        return BASE_IMAGES["pintuck"]
    elif "velvet" in lower:
        return BASE_IMAGES["velvet"]
    else:
        return BASE_IMAGES["default"]


def slugify(name):
    """Convert a tablecloth name to a filesystem-safe slug."""
    return name.lower().replace(" ", "_")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def resize_to_fixed(img, width=FIXED_WIDTH, height=FIXED_HEIGHT):
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img, size=REF_SIZE):
    return img.resize((size, size), Image.LANCZOS)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL: Qwen-Image-Edit-2511 + Lightning LoRA")

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

    print(f"Loading {MODEL_NAME}...")
    load_start = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_NAME,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(f"Base model loaded in {time.time() - load_start:.1f}s")

    print("Loading 4-step Lightning LoRA...")
    pipeline.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS)
    print("LoRA loaded")

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


# =============================================================================
# WARMUP
# =============================================================================

def warmup(pipeline):
    print("\nWarmup run...")
    dummy = Image.new("RGB", (FIXED_WIDTH, FIXED_HEIGHT), "white")
    dummy_ref = Image.new("RGB", (REF_SIZE, REF_SIZE), "gray")

    with torch.inference_mode():
        _ = pipeline(
            image=[dummy, dummy_ref],
            prompt="warmup",
            num_inference_steps=4,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
        )

    gc.collect()
    torch.cuda.empty_cache()
    print("Warmup complete\n")


# =============================================================================
# SINGLE-IMAGE EDIT
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, negative_prompt, num_steps,
             true_cfg_scale):
    """Run a single tablecloth swap and return (result_image, elapsed_seconds)."""
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED),
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)

    return result, elapsed


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(pipeline, test_name, tablecloths, step_counts, cfg_scales,
                   use_fabric_base, output_dir, ref_images, base_img_default,
                   base_img_cache):
    """Run a full test across all tablecloths, step counts, and CFG scales.

    Args:
        pipeline:         Loaded diffusion pipeline.
        test_name:        Name of the test (e.g., "test1_realism").
        tablecloths:      List of tablecloth dicts.
        step_counts:      List of step counts to sweep (e.g., [4, 6, 8, 12]).
        cfg_scales:       List of true_cfg_scale values to sweep (e.g., [1.5, 3.0]).
        use_fabric_base:  If True, route base image by fabric type; else use default.
        output_dir:       Root output directory.
        ref_images:       Dict mapping filename -> PIL reference image.
        base_img_default: PIL base image (default).
        base_img_cache:   Dict mapping base image path -> PIL image.

    Returns:
        Dict with test_name, use_fabric_base, results list, and total_time.
    """
    test_dir = os.path.join(output_dir, test_name)
    os.makedirs(test_dir, exist_ok=True)

    print_banner(f"TEST: {test_name}  |  fabric_base={use_fabric_base}")

    all_results = []
    test_start = time.time()

    for cfg_scale in cfg_scales:
        cfg_dir = os.path.join(test_dir, f"cfg_{cfg_scale}")
        os.makedirs(cfg_dir, exist_ok=True)

        for steps in step_counts:
            step_dir = os.path.join(cfg_dir, f"steps_{steps}")
            os.makedirs(step_dir, exist_ok=True)

            print_banner(f"{test_name} / cfg={cfg_scale} / steps={steps}", char="-")

            step_results = []
            step_start = time.time()

            for tc_idx, tc in enumerate(tablecloths, start=1):
                prompt = build_prompt(tc)
                slug = slugify(tc["name"])

                # Select base image
                if use_fabric_base:
                    base_path = get_base_image(tc["filename"])
                    base_img = base_img_cache[base_path]
                    base_label = os.path.basename(base_path)
                else:
                    base_img = base_img_default
                    base_label = "base_image.png"

                ref_img = ref_images[tc["filename"]]

                print(f"  [{tc_idx}/{len(tablecloths)}] {tc['name']}")
                print(f"         cfg:    {cfg_scale}")
                print(f"         steps:  {steps}")
                print(f"         base:   {base_label}")
                print(f"         prompt: {prompt}")

                result_img, elapsed = run_edit(
                    pipeline, base_img, ref_img, prompt,
                    TARGETED_NEGATIVE_PROMPT, steps, cfg_scale
                )

                out_path = os.path.join(step_dir, f"tablecloth_{tc_idx}_{slug}.png")
                result_img.save(out_path)

                result_entry = {
                    "tablecloth": tc["name"],
                    "filename": tc["filename"],
                    "slug": slug,
                    "cfg_scale": cfg_scale,
                    "steps": steps,
                    "prompt": prompt,
                    "base_image": base_label,
                    "output_path": out_path,
                    "time": elapsed,
                }
                step_results.append(result_entry)
                print(f"         time:   {elapsed:.2f}s  ->  {out_path}")

            step_total = time.time() - step_start
            print(f"\n  cfg={cfg_scale} steps={steps} complete: {step_total:.2f}s")

            all_results.extend(step_results)

            gc.collect()
            torch.cuda.empty_cache()

    test_total = time.time() - test_start

    # Write per-test report
    write_test_report(test_name, test_dir, all_results, test_total, use_fabric_base)

    return {
        "test_name": test_name,
        "use_fabric_base": use_fabric_base,
        "results": all_results,
        "total_time": test_total,
    }

# =============================================================================
# PER-TEST REPORT
# =============================================================================

def write_test_report(test_name, test_dir, results, total_time, use_fabric_base):
    """Write report.txt for a single test."""
    report_path = os.path.join(test_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test: {test_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write(f"true_cfg_scales:    {TRUE_CFG_SCALES}\n")
        f.write(f"step_counts:        {STEP_COUNTS}\n")
        f.write(f"guidance_scale:     {GUIDANCE_SCALE} (placeholder)\n")
        f.write(f"seed:               {SEED}\n")
        f.write(f"resolution:         {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
                f"{REF_SIZE}x{REF_SIZE} (ref)\n")
        f.write(f"fabric_base_images: {use_fabric_base}\n")
        f.write(f"negative_prompt:    {TARGETED_NEGATIVE_PROMPT!r}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 60 + "\n")

        # Group by cfg_scale, then by steps
        for cfg_scale in TRUE_CFG_SCALES:
            for steps in STEP_COUNTS:
                group = [r for r in results
                         if r["cfg_scale"] == cfg_scale and r["steps"] == steps]
                if not group:
                    continue
                f.write(f"\n  cfg={cfg_scale}  steps={steps}:\n")
                for r in group:
                    f.write(f"    {r['tablecloth']:30s}  base={r['base_image']:25s}  "
                            f"time={r['time']:.2f}s\n")
                    f.write(f"      prompt: {r['prompt']}\n")
                    f.write(f"      output: {r['output_path']}\n")

        f.write("\n\nTIMING\n")
        f.write("-" * 60 + "\n")
        inference_total = sum(r["time"] for r in results)
        f.write(f"  Inference total:  {inference_total:.2f}s\n")
        f.write(f"  Wall-clock total: {total_time:.2f}s\n")

    print(f"  Report saved: {report_path}")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary(test_results, output_dir, model_load_time, warmup_time, total_runtime):
    """Write summary.txt with all experiment configs, timing, and outputs."""
    summary_path = os.path.join(output_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write("Optimized Tablecloth Experiments - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n")
        f.write(f"Seed:               {SEED}\n")
        f.write(f"Resolution:         {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
                f"{REF_SIZE}x{REF_SIZE} (ref)\n")
        f.write(f"true_cfg_scales:    {TRUE_CFG_SCALES}\n")
        f.write(f"Step counts:        {STEP_COUNTS}\n\n")

        # Negative prompt
        f.write("NEGATIVE PROMPT\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {TARGETED_NEGATIVE_PROMPT!r}\n\n")

        # Tablecloths tested
        f.write("TABLECLOTHS TESTED\n")
        f.write("-" * 70 + "\n")
        for i, tc in enumerate(TABLECLOTHS, 1):
            f.write(f"  {i}. {tc['name']}\n")
            f.write(f"     file:     {tc['filename']}\n")
            f.write(f"     color:    {tc['color']}\n")
            f.write(f"     material: {tc['material']}\n")
            f.write(f"     fabric:   {tc['fabric_type']}\n")
        f.write("\n")

        # Base image routing
        f.write("BASE IMAGE ROUTING (Test 2)\n")
        f.write("-" * 70 + "\n")
        for tc in TABLECLOTHS:
            base_file = os.path.basename(get_base_image(tc["filename"]))
            f.write(f"  {tc['filename']:45s} -> {base_file}\n")
        f.write("\n")

        # Per-test results table
        for test_data in test_results:
            test_name = test_data["test_name"]
            results = test_data["results"]
            test_total = test_data["total_time"]

            f.write(f"\n{'=' * 70}\n")
            f.write(f"  {test_name.upper()}  "
                    f"(fabric_base={'Yes' if test_data['use_fabric_base'] else 'No'})\n")
            f.write(f"{'=' * 70}\n\n")

            # Table header
            hdr = (
                f"{'CFG':>5}  {'Steps':>5}  {'Tablecloth':>25}  {'Base Image':>25}  "
                f"{'Time (s)':>9}  {'Output':>s}"
            )
            f.write(hdr + "\n")
            f.write("-" * len(hdr) + "\n")

            for cfg_scale in TRUE_CFG_SCALES:
                for steps in STEP_COUNTS:
                    group = [r for r in results
                             if r["cfg_scale"] == cfg_scale and r["steps"] == steps]
                    for r in group:
                        out_rel = os.path.relpath(r["output_path"], output_dir)
                        line = (
                            f"{r['cfg_scale']:5.1f}  {r['steps']:5d}  "
                            f"{r['tablecloth']:>25s}  "
                            f"{r['base_image']:>25s}  {r['time']:9.2f}  {out_rel}"
                        )
                        f.write(line + "\n")

            step_inference = sum(r["time"] for r in results)
            f.write(f"\n  Inference total:  {step_inference:.2f}s\n")
            f.write(f"  Wall-clock total: {test_total:.2f}s\n")

        # Aggregate timing
        f.write(f"\n\n{'=' * 70}\n")
        f.write(f"  AGGREGATE TIMING\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"  Model load:       {model_load_time:.2f}s\n")
        f.write(f"  Warmup:           {warmup_time:.2f}s\n")
        for test_data in test_results:
            f.write(f"  {test_data['test_name']:20s} {test_data['total_time']:.2f}s\n")

        total_inference = sum(
            r["time"] for td in test_results for r in td["results"]
        )
        total_images = sum(len(td["results"]) for td in test_results)
        f.write(f"\n  Total inference:  {total_inference:.2f}s "
                f"({total_images} images)\n")
        f.write(f"  Total runtime:    {total_runtime:.2f}s "
                f"({total_runtime / 60:.1f} min)\n")

    print_banner(f"SUMMARY SAVED: {summary_path}")
    return summary_path


# =============================================================================
# DRY RUN
# =============================================================================

def print_dry_run():
    """Print the full experiment config without running anything."""
    print_banner("DRY RUN - Optimized Experiment Config")

    print(f"Baseline:  Experiment 15 config")
    print(f"CFG:       {TRUE_CFG_SCALES}")
    print(f"Steps:     {STEP_COUNTS}")
    print(f"Seed:      {SEED}")
    print(f"Resolution: {FIXED_WIDTH}x{FIXED_HEIGHT} (main), {REF_SIZE}x{REF_SIZE} (ref)")
    print()

    print(f"Negative prompt (disabled — single space):")
    print(f"  {TARGETED_NEGATIVE_PROMPT!r}")
    print()

    print(f"Tablecloths ({len(TABLECLOTHS)}):")
    for i, tc in enumerate(TABLECLOTHS, 1):
        print(f"  {i}. {tc['name']}")
        print(f"     file:     {tc['filename']}")
        print(f"     color:    {tc['color']}")
        print(f"     material: {tc['material']}")
        print(f"     fabric:   {tc['fabric_type']}")
        print(f"     prompt:   \"{build_prompt(tc)}\"")
    print()

    runs_per_test = len(TABLECLOTHS) * len(STEP_COUNTS) * len(TRUE_CFG_SCALES)

    # Test 1
    print_banner("TEST 1: Realism Enhancement + Step/CFG Sweep", char="-")
    print(f"  Base image: base_image.png (default for all)")
    print(f"  CFG sweep:  {TRUE_CFG_SCALES}")
    print(f"  Step sweep: {STEP_COUNTS}")
    print(f"  Runs: {len(TABLECLOTHS)} tablecloths x {len(STEP_COUNTS)} steps x "
          f"{len(TRUE_CFG_SCALES)} cfg = {runs_per_test}")
    print()

    # Test 2
    print_banner("TEST 2: Fabric-Specific Base Images + Step/CFG Sweep", char="-")
    print(f"  Base image routing:")
    for tc in TABLECLOTHS:
        base_file = os.path.basename(get_base_image(tc["filename"]))
        print(f"    {tc['filename']:45s} -> {base_file}")
    print(f"  CFG sweep:  {TRUE_CFG_SCALES}")
    print(f"  Step sweep: {STEP_COUNTS}")
    print(f"  Runs: {len(TABLECLOTHS)} tablecloths x {len(STEP_COUNTS)} steps x "
          f"{len(TRUE_CFG_SCALES)} cfg = {runs_per_test}")
    print()

    total_runs = 2 * runs_per_test
    print(f"Total runs: {total_runs}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized tablecloth experiments (Qwen-Image-Edit-2511 + Lightning)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment config without running any inference.",
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=[1, 2],
        default=None,
        metavar="N",
        help="Run only Test N (1 or 2). Default: run both.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Custom output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR

    # --dry-run: print config and exit
    if args.dry_run:
        print_dry_run()
        return

    # Determine which tests to run
    run_test1 = args.test is None or args.test == 1
    run_test2 = args.test is None or args.test == 2

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    # Always need the default base image
    if not os.path.exists(BASE_IMAGES["default"]):
        print(f"Error: default base image not found: {BASE_IMAGES['default']}")
        sys.exit(1)

    # If running Test 2, validate all fabric-specific base images
    if run_test2:
        for key, path in BASE_IMAGES.items():
            if not os.path.exists(path):
                print(f"Error: base image '{key}' not found: {path}")
                sys.exit(1)

    for tc in TABLECLOTHS:
        tc_path = os.path.join(TABLECLOTH_DIR, tc["filename"])
        if not os.path.exists(tc_path):
            print(f"Error: tablecloth not found: {tc_path}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Print run info
    # -------------------------------------------------------------------------
    runs_per_test = len(TABLECLOTHS) * len(STEP_COUNTS) * len(TRUE_CFG_SCALES)
    total_runs = (
        (runs_per_test if run_test1 else 0) +
        (runs_per_test if run_test2 else 0)
    )

    print_banner("OPTIMIZED TABLECLOTH EXPERIMENTS")
    print(f"Date:            {datetime.now()}")
    print(f"Model:           {MODEL_NAME}")
    print(f"LoRA:            {LORA_WEIGHTS}")
    print(f"Output dir:      {output_dir}")
    print(f"Resolution:      {FIXED_WIDTH}x{FIXED_HEIGHT} (main), {REF_SIZE}x{REF_SIZE} (ref)")
    print(f"Seed:            {SEED}")
    print(f"CFG scales:      {TRUE_CFG_SCALES}")
    print(f"Step counts:     {STEP_COUNTS}")
    print(f"Tests to run:    {'1, 2' if (run_test1 and run_test2) else ('1' if run_test1 else '2')}")
    print(f"Total runs:      {total_runs}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print()

    # -------------------------------------------------------------------------
    # Load reference tablecloth images
    # -------------------------------------------------------------------------
    print("Loading reference tablecloth images...")
    ref_images = {}
    for tc in TABLECLOTHS:
        tc_path = os.path.join(TABLECLOTH_DIR, tc["filename"])
        ref_images[tc["filename"]] = resize_reference(
            Image.open(tc_path).convert("RGB")
        )
    print(f"  {len(ref_images)} reference tablecloths loaded")

    # -------------------------------------------------------------------------
    # Load base images
    # -------------------------------------------------------------------------
    print("Loading base images...")
    base_img_cache = {}
    for key, path in BASE_IMAGES.items():
        if os.path.exists(path):
            base_img_cache[path] = resize_to_fixed(
                Image.open(path).convert("RGB")
            )
            print(f"  Loaded: {os.path.basename(path)}")

    base_img_default = base_img_cache[BASE_IMAGES["default"]]

    # -------------------------------------------------------------------------
    # Load model (once)
    # -------------------------------------------------------------------------
    model_load_start = time.time()
    pipeline = load_pipeline()
    model_load_time = time.time() - model_load_start

    # Warmup (once)
    warmup_start = time.time()
    warmup(pipeline)
    warmup_time = time.time() - warmup_start

    # -------------------------------------------------------------------------
    # Run experiments
    # -------------------------------------------------------------------------
    all_test_results = []
    run_start = time.time()

    # Test 1: Realism Enhancement + Step Count Sweep
    if run_test1:
        print(f"\n>>> Starting Test 1: Realism Enhancement <<<")
        test1_result = run_experiment(
            pipeline=pipeline,
            test_name="test1_realism",
            tablecloths=TABLECLOTHS,
            step_counts=STEP_COUNTS,
            cfg_scales=TRUE_CFG_SCALES,
            use_fabric_base=False,
            output_dir=output_dir,
            ref_images=ref_images,
            base_img_default=base_img_default,
            base_img_cache=base_img_cache,
        )
        all_test_results.append(test1_result)

    # Test 2: Fabric-Specific Base Images + Step Count Sweep
    if run_test2:
        print(f"\n>>> Starting Test 2: Fabric-Specific Base Images <<<")
        test2_result = run_experiment(
            pipeline=pipeline,
            test_name="test2_fabric_base",
            tablecloths=TABLECLOTHS,
            step_counts=STEP_COUNTS,
            cfg_scales=TRUE_CFG_SCALES,
            use_fabric_base=True,
            output_dir=output_dir,
            ref_images=ref_images,
            base_img_default=base_img_default,
            base_img_cache=base_img_cache,
        )
        all_test_results.append(test2_result)

    total_runtime = time.time() - run_start

    # -------------------------------------------------------------------------
    # Generate summary report
    # -------------------------------------------------------------------------
    summary_path = generate_summary(
        all_test_results, output_dir, model_load_time, warmup_time, total_runtime
    )

    # -------------------------------------------------------------------------
    # Final stats
    # -------------------------------------------------------------------------
    total_images = sum(len(td["results"]) for td in all_test_results)
    print_banner("ALL EXPERIMENTS COMPLETE")
    print(f"Tests run:        {len(all_test_results)}")
    print(f"Images generated: {total_images}")
    print(f"Total runtime:    {total_runtime:.2f}s ({total_runtime / 60:.1f} min)")
    print(f"Output dir:       {output_dir}")
    print(f"Summary:          {summary_path}")
    print()


if __name__ == "__main__":
    main()

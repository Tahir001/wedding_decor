"""
===============================================================================
WEDDING DECOR - CHAIR REPLACEMENT EXPERIMENTS (QWEN-IMAGE-EDIT-2511 + LIGHTNING)
===============================================================================
Experiment runner for chair swaps, using the same model configuration and
positive-only prompt strategy as the tablecloth and table runner experiments.

Sweeps across:
  1. Prompt style: minimal vs detailed
  2. Chair type: 3 chairs
  3. true_cfg_scale: 1.5 vs 2.0
  4. Step count: 4 vs 8 vs 12

Total: 2 cfgs x 2 prompts x 3 chairs x 3 steps = 36 runs

Usage:
    python experiments_chairs.py                    # run all 36 experiments
    python experiments_chairs.py --dry-run          # print config, don't run
    python experiments_chairs.py --base-image /path/to/image.png
    python experiments_chairs.py --output-dir /custom/path
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
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "chairs", "optimized_2")

# Base image: table scene with existing chairs to be replaced
DEFAULT_BASE_IMAGE = os.path.join(INPUT_DIR, "base_image_tablerunner.png")

# =============================================================================
# CHAIR DEFINITIONS
# =============================================================================

CHAIR_DIR = os.path.join(INPUT_DIR, "chairs")

CHAIRS = [
    {
        "name": "Gold Chiavari Black Cushion",
        "filename": "Gold Chiavari with Black Cushion.jpg",
        "color": "gold",
        "material": "chiavari",
        "cushion": "black",
        "description": "gold chiavari chair with a black cushion",
    },
    {
        "name": "Sonoma",
        "filename": "Sonoma.png",
        "color": "natural wood",
        "material": "french louis",
        "cushion": "cream linen",
        "description": "Sonoma French Louis XVI chair with a cane back and cream linen seat",
    },
    {
        "name": "Walnut Folding Chair",
        "filename": "Walnut Folding Chair.jpg",
        "color": "walnut",
        "material": "wood folding",
        "cushion": "none",
        "description": "walnut wood folding chair with a dark brown finish",
    },
]

# =============================================================================
# MODEL CONFIG (held constant)
# =============================================================================

MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

FIXED_WIDTH = 784
FIXED_HEIGHT = 784
REF_SIZE = 512
GUIDANCE_SCALE = 1.0  # placeholder, no effect
SEED = 42

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

TRUE_CFG_SCALES = [1.0, 1.25]
STEP_COUNTS = [4, 8, 12]

# Negative prompt: Qwen-Image-Edit-2511 does NOT support negative conditioning
# (flow-matching architecture, not trained for it). The parameter must be present
# to avoid pipeline errors, but only a single space is needed.
# See: https://blog.promptmaster.pro/posts/qwen-image-negative-prompts
NEGATIVE_PROMPT = " "

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
#
# PROMPT STRATEGY (positive-only, natural language):
# - Qwen-Image-Edit-2511 uses Qwen2.5-VL (7B VLM) as its text encoder,
#   so rich descriptive sentences work far better than keyword lists.
# - ALL negations ("do not", "no", "without") are avoided because the
#   model can latch onto the unwanted concept tokens and produce them.
#   Instead, preservation intent is stated positively.
# =============================================================================

PROMPT_TEMPLATES = {
    "minimal": (
        "Replace all the chairs in image 1 with the chair shown in image 2. "
        "Place the same style of chair in every seat position around the table. "
        "The table, tablecloth, table runner, background and everything else stay exactly the same."
    ),
    "detailed": (
        "Replace every chair around the round table in image 1 with the "
        "chair shown in image 2. Use the exact chair design, color, "
        "and materials from image 2. Place one identical chair at each seating "
        "position around the table, maintaining the same circular arrangement "
        "and spacing. Each chair faces the table naturally. The chairs look like "
        "real furniture with authentic materials and proportions, matching the "
        "perspective and lighting of the scene. The round table, brown damask "
        "tablecloth, ivory table runner, and the white background all remain "
        "exactly as they appear in image 1. Photorealistic, professional event "
        "photography quality."
    ),
}


# =============================================================================
# EXPERIMENT GRID BUILDER
# =============================================================================

def build_experiment_grid():
    """Build the full experiment grid.

    cfg_scale x prompt_style x chair x step_count.

    Returns a list of experiment dicts, one per run.
    """
    experiments = []
    exp_id = 1

    for cfg_scale in TRUE_CFG_SCALES:
        for prompt_key in PROMPT_TEMPLATES:
            for chair in CHAIRS:
                # Format the prompt with chair-specific details
                prompt_text = PROMPT_TEMPLATES[prompt_key].format(
                    description=chair["description"],
                    color=chair["color"],
                    material=chair["material"],
                    cushion=chair["cushion"],
                )

                for num_steps in STEP_COUNTS:
                    experiments.append({
                        "id": exp_id,
                        "prompt_style": prompt_key,
                        "chair": chair,
                        "prompt_text": prompt_text,
                        "num_steps": num_steps,
                        "cfg_scale": cfg_scale,
                    })
                    exp_id += 1

    return experiments


# =============================================================================
# PROMPT BUILDING (for display / reports)
# =============================================================================

def format_prompt(prompt_key, chair):
    """Format a prompt template with chair details."""
    return PROMPT_TEMPLATES[prompt_key].format(
        description=chair["description"],
        color=chair["color"],
        material=chair["material"],
        cushion=chair["cushion"],
    )


def slugify(name):
    """Convert a name to a filesystem-safe slug."""
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
             cfg_scale):
    """Run a single chair replacement and return (result_image, elapsed_seconds).

    Args:
        pipeline:         Loaded diffusion pipeline.
        base_img:         PIL base image (table scene with existing chairs).
        ref_img:          PIL reference image (target chair).
        prompt:           Text prompt.
        negative_prompt:  Negative prompt (single space).
        num_steps:        Number of inference steps.
        cfg_scale:        True CFG scale value.

    Returns:
        (result_image, elapsed_seconds)
    """
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
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

def run_experiments(pipeline, experiments, base_img, chair_images, output_dir):
    """Run the full experiment grid.

    Args:
        pipeline:       Loaded diffusion pipeline.
        experiments:    List of experiment dicts from build_experiment_grid().
        base_img:       PIL base image (table scene).
        chair_images:   Dict mapping chair filename -> PIL reference image.
        output_dir:     Root output directory.

    Returns:
        (results_list, total_time)
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    total_start = time.time()

    for exp in experiments:
        exp_id = exp["id"]
        prompt_style = exp["prompt_style"]
        chair = exp["chair"]
        prompt_text = exp["prompt_text"]
        num_steps = exp["num_steps"]
        cfg_scale = exp["cfg_scale"]

        chair_name = chair["name"]
        chair_slug = slugify(chair_name)

        # Create experiment output directory
        exp_dir = os.path.join(output_dir, f"experiment_{exp_id:02d}")
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n  [{exp_id:2d}/{len(experiments)}] "
              f"{prompt_style} / {chair_name} / "
              f"cfg={cfg_scale} / steps={num_steps}")
        print(f"         prompt: {prompt_text}")
        print(f"         neg:    (disabled, single space)")
        print(f"         cfg:    {cfg_scale}")
        print(f"         steps:  {num_steps}")

        # Get reference chair image
        ref_img = chair_images[chair["filename"]]

        # Run inference
        result_img, elapsed = run_edit(
            pipeline, base_img, ref_img, prompt_text, NEGATIVE_PROMPT,
            num_steps, cfg_scale
        )

        # Save output image
        out_path = os.path.join(exp_dir, f"chair_{chair_slug}.png")
        result_img.save(out_path)

        print(f"         time:   {elapsed:.2f}s  ->  {out_path}")

        # Write per-experiment report
        report_path = os.path.join(exp_dir, "report.txt")
        write_experiment_report(report_path, exp, elapsed, out_path)

        result_entry = {
            "experiment_id": exp_id,
            "prompt_style": prompt_style,
            "chair_name": chair_name,
            "chair_filename": chair["filename"],
            "prompt_text": prompt_text,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "output_path": out_path,
            "time": elapsed,
        }
        all_results.append(result_entry)

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start
    return all_results, total_time


# =============================================================================
# PER-EXPERIMENT REPORT
# =============================================================================

def write_experiment_report(report_path, exp, elapsed, out_path):
    """Write report.txt for a single experiment."""
    chair = exp["chair"]
    num_steps = exp["num_steps"]
    cfg_scale = exp["cfg_scale"]

    with open(report_path, "w") as f:
        f.write(f"Experiment {exp['id']:02d}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n\n")

        f.write("CHAIR\n")
        f.write("-" * 60 + "\n")
        f.write(f"name:               {chair['name']}\n")
        f.write(f"filename:           {chair['filename']}\n")
        f.write(f"color:              {chair['color']}\n")
        f.write(f"material:           {chair['material']}\n")
        f.write(f"cushion:            {chair['cushion']}\n")
        f.write(f"description:        {chair['description']}\n\n")

        f.write("PROMPT\n")
        f.write("-" * 60 + "\n")
        f.write(f"style:              {exp['prompt_style']}\n")
        f.write(f"text:               {exp['prompt_text']}\n")
        f.write(f"negative:           {NEGATIVE_PROMPT!r}\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write(f"true_cfg_scale:     {cfg_scale}\n")
        f.write(f"num_inference_steps:{num_steps}\n")
        f.write(f"guidance_scale:     {GUIDANCE_SCALE} (placeholder)\n")
        f.write(f"seed:               {SEED}\n")
        f.write(f"resolution:         {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
                f"{REF_SIZE}x{REF_SIZE} (ref)\n\n")

        f.write("TIMING\n")
        f.write("-" * 60 + "\n")
        f.write(f"inference_time:     {elapsed:.2f}s\n")
        f.write(f"output:             {out_path}\n")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary(results, output_dir, base_image_path, model_load_time,
                     warmup_time, experiment_time, total_runtime):
    """Write summary.txt with all experiment configs, timing, and outputs."""
    summary_path = os.path.join(output_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write("Chair Replacement Experiments - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n")
        f.write(f"Seed:               {SEED}\n")
        f.write(f"Resolution:         {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
                f"{REF_SIZE}x{REF_SIZE} (ref)\n")
        f.write(f"true_cfg_scales:    {TRUE_CFG_SCALES}\n")
        f.write(f"step_counts:        {STEP_COUNTS}\n")
        f.write(f"Base image:         {base_image_path}\n")
        f.write(f"Total experiments:  {len(results)}\n\n")

        # Negative prompt
        f.write("NEGATIVE PROMPT\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {NEGATIVE_PROMPT!r}\n\n")

        # Chairs tested
        f.write("CHAIRS TESTED\n")
        f.write("-" * 70 + "\n")
        for i, chair in enumerate(CHAIRS, 1):
            f.write(f"  {i}. {chair['name']}\n")
            f.write(f"     file:        {chair['filename']}\n")
            f.write(f"     color:       {chair['color']}\n")
            f.write(f"     material:    {chair['material']}\n")
            f.write(f"     cushion:     {chair['cushion']}\n")
            f.write(f"     description: {chair['description']}\n")
        f.write("\n")

        # Prompt templates
        f.write("PROMPT TEMPLATES\n")
        f.write("-" * 70 + "\n")
        for key, template in PROMPT_TEMPLATES.items():
            f.write(f"  {key}:\n")
            f.write(f"    \"{template}\"\n\n")

        # Results table
        f.write("RESULTS\n")
        f.write("=" * 70 + "\n\n")

        # Table header
        hdr = (f"{'Exp':>3}  {'Prompt Style':>14}  "
               f"{'Chair':>30}  {'CFG':>5}  {'Steps':>5}  {'Time (s)':>9}")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")

        for r in results:
            line = (
                f"{r['experiment_id']:3d}  "
                f"{r['prompt_style']:>14s}  "
                f"{r['chair_name']:>30s}  "
                f"{r['cfg_scale']:5.1f}  "
                f"{r['num_steps']:5d}  "
                f"{r['time']:9.2f}"
            )
            f.write(line + "\n")

        f.write("\n")

        # Per-prompt-style averages
        f.write("PER-PROMPT-STYLE AVERAGES\n")
        f.write("-" * 70 + "\n")
        for prompt_key in PROMPT_TEMPLATES:
            style_results = [r for r in results
                             if r["prompt_style"] == prompt_key]
            if style_results:
                avg_time = sum(r["time"] for r in style_results) / len(style_results)
                total_time = sum(r["time"] for r in style_results)
                f.write(f"  {prompt_key:20s}  "
                        f"runs={len(style_results):2d}  "
                        f"avg={avg_time:.2f}s  "
                        f"total={total_time:.2f}s\n")
        f.write("\n")

        # Aggregate timing
        f.write("AGGREGATE TIMING\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Model load:       {model_load_time:.2f}s\n")
        f.write(f"  Warmup:           {warmup_time:.2f}s\n")
        f.write(f"  Experiments:      {experiment_time:.2f}s\n")

        total_inference = sum(r["time"] for r in results)
        f.write(f"\n  Total inference:  {total_inference:.2f}s "
                f"({len(results)} images)\n")
        f.write(f"  Total runtime:    {total_runtime:.2f}s "
                f"({total_runtime / 60:.1f} min)\n")

    print_banner(f"SUMMARY SAVED: {summary_path}")
    return summary_path


# =============================================================================
# DRY RUN
# =============================================================================

def print_dry_run(base_image_path):
    """Print the full experiment config without running anything."""
    experiments = build_experiment_grid()

    print_banner("DRY RUN - Chair Replacement Experiment Config")

    print(f"CFG:         {TRUE_CFG_SCALES}")
    print(f"Steps:       {STEP_COUNTS}")
    print(f"Seed:        {SEED}")
    print(f"Resolution:  {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
          f"{REF_SIZE}x{REF_SIZE} (ref)")
    print(f"Base image:  {base_image_path}")
    print()

    print(f"Negative prompt (disabled -- single space):")
    print(f"  {NEGATIVE_PROMPT!r}")
    print()

    print(f"Chairs ({len(CHAIRS)}):")
    for i, chair in enumerate(CHAIRS, 1):
        print(f"  {i}. {chair['name']}")
        print(f"     file:        {chair['filename']}")
        print(f"     description: {chair['description']}")
    print()

    print(f"Prompt templates ({len(PROMPT_TEMPLATES)}):")
    for key, template in PROMPT_TEMPLATES.items():
        print(f"  {key}:")
        example = template.format(
            description=CHAIRS[0]["description"],
            color=CHAIRS[0]["color"],
            material=CHAIRS[0]["material"],
            cushion=CHAIRS[0]["cushion"],
        )
        print(f"    \"{example}\"")
    print()

    print_banner("EXPERIMENT GRID", char="-")
    for exp in experiments:
        print(f"  Exp {exp['id']:2d}: "
              f"{exp['prompt_style']:14s}  {exp['chair']['name']:30s}  "
              f"cfg={exp['cfg_scale']}  steps={exp['num_steps']}")
    print()
    print(f"Total experiments: {len(experiments)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chair replacement experiments - "
                    "Qwen-Image-Edit-2511 + Lightning"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment config without running any inference.",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        help=f"Path to base image (table scene with chairs). "
             f"Default: {DEFAULT_BASE_IMAGE}",
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
    base_image_path = args.base_image if args.base_image else DEFAULT_BASE_IMAGE

    # --dry-run: print config and exit
    if args.dry_run:
        print_dry_run(base_image_path)
        return

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    print("Validating input files...")

    if not os.path.exists(base_image_path):
        print(f"Error: base image not found: {base_image_path}")
        sys.exit(1)

    for chair in CHAIRS:
        chair_path = os.path.join(CHAIR_DIR, chair["filename"])
        if not os.path.exists(chair_path):
            print(f"Error: chair image not found: {chair_path}")
            sys.exit(1)

    print("  All input files found.")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Build experiment grid
    # -------------------------------------------------------------------------
    experiments = build_experiment_grid()

    # -------------------------------------------------------------------------
    # Print run info
    # -------------------------------------------------------------------------
    print_banner("CHAIR REPLACEMENT EXPERIMENTS")
    print(f"Date:            {datetime.now()}")
    print(f"Model:           {MODEL_NAME}")
    print(f"LoRA:            {LORA_WEIGHTS}")
    print(f"Output dir:      {output_dir}")
    print(f"Base image:      {base_image_path}")
    print(f"Resolution:      {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
          f"{REF_SIZE}x{REF_SIZE} (ref)")
    print(f"Seed:            {SEED}")
    print(f"CFG:             {TRUE_CFG_SCALES}")
    print(f"Step counts:     {STEP_COUNTS}")
    print(f"Total runs:      {len(experiments)}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print()

    # -------------------------------------------------------------------------
    # Load images
    # -------------------------------------------------------------------------
    print("Loading base image...")
    base_img = resize_to_fixed(
        Image.open(base_image_path).convert("RGB")
    )
    print(f"  Loaded: {os.path.basename(base_image_path)}")

    print("Loading chair reference images...")
    chair_images = {}
    for chair in CHAIRS:
        chair_path = os.path.join(CHAIR_DIR, chair["filename"])
        chair_images[chair["filename"]] = resize_reference(
            Image.open(chair_path).convert("RGB")
        )
        print(f"  Loaded: {chair['name']} ({chair['filename']})")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model_load_start = time.time()
    pipeline = load_pipeline()
    model_load_time = time.time() - model_load_start

    # Warmup
    warmup_start = time.time()
    warmup(pipeline)
    warmup_time = time.time() - warmup_start

    # -------------------------------------------------------------------------
    # Run experiments
    # -------------------------------------------------------------------------
    run_start = time.time()

    print_banner(f"RUNNING {len(experiments)} EXPERIMENTS")
    results, experiment_time = run_experiments(
        pipeline, experiments, base_img, chair_images, output_dir
    )

    total_runtime = time.time() - run_start

    # -------------------------------------------------------------------------
    # Generate summary report
    # -------------------------------------------------------------------------
    summary_path = generate_summary(
        results, output_dir, base_image_path,
        model_load_time, warmup_time, experiment_time, total_runtime
    )

    # -------------------------------------------------------------------------
    # Final stats
    # -------------------------------------------------------------------------
    total_inference = sum(r["time"] for r in results)
    print_banner("ALL EXPERIMENTS COMPLETE")
    print(f"Experiments run:  {len(results)}")
    print(f"Total inference:  {total_inference:.2f}s")
    print(f"Total runtime:    {total_runtime:.2f}s ({total_runtime / 60:.1f} min)")
    print(f"Output dir:       {output_dir}")
    print(f"Summary:          {summary_path}")
    print()


if __name__ == "__main__":
    main()

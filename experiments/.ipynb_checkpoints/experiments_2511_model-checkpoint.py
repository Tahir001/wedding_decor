"""
===============================================================================
WEDDING DECOR - TABLECLOTH EXPERIMENT GRID (QWEN-IMAGE-EDIT-2511 + LIGHTNING)
===============================================================================
Grid-search experiment runner that tests tablecloth swaps with different
hyperparameter combinations.

32 experiments total (4 true_cfg_scale x 2 num_inference_steps x 2 negative
prompts x 2 prompt styles), each producing 4 tablecloth images = 128 images.

Usage:
    python experiments_2511_model.py                  # run all 32 experiments
    python experiments_2511_model.py --dry-run        # print grid, don't run
    python experiments_2511_model.py --experiment 5   # run only experiment 5
    python experiments_2511_model.py --output-dir /custom/path
===============================================================================
"""

import os
import gc
import sys
import time
import math
import argparse
import itertools
import torch
from PIL import Image
from datetime import datetime
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# PATHS (relative to this script's directory)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BASE_IMAGE = os.path.join(INPUT_DIR, "base_image.png")

# =============================================================================
# TABLECLOTH REFERENCES
# =============================================================================

TABLECLOTH_DIR = os.path.join(INPUT_DIR, "tablecloths")
TABLECLOTHS = [
    {"file": "120 Round - Tan Buffalo Check.jpg", "name": "Tan Buffalo Check"},
    {"file": "120 Round Damask - Black.png", "name": "Damask Black"},
    {"file": "120 Round Pintuck Taffeta - Red.jpg", "name": "Pintuck Taffeta Red"},
    {"file": "120 Round Woven Celery.png", "name": "Woven Celery"},
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
# EXPERIMENT GRID DEFINITION
# =============================================================================

TRUE_CFG_SCALES = [1.0, 1.5, 2.0, 2.5]
NUM_INFERENCE_STEPS = [4, 8]

NEGATIVE_PROMPTS = {
    "minimal": " ",
    "targeted": (
        "wrinkles, creases, folds, shadows, dark spots, uneven color, "
        "uneven lighting, lighting artifacts, changed furniture, "
        "altered background, distortion, blurry"
    ),
}

PROMPT_STYLES = {
    "minimal": lambda name: (
        "Replace the tablecloth in image 1 with the tablecloth from image 2. The tablecloth should be uniform, lay flat with no creases."
    ),
    "detailed": lambda name: (
        f"Replace the tablecloth in image 1 with the {name} tablecloth from "
        f"image 2. Match the exact color, texture, and pattern. Apply the "
        f"color uniformly across the entire tablecloth with even lighting "
        f"and no dark spots. Keep everything else unchanged."
    ),
}

# Total: 4 x 2 x 2 x 2 = 32 experiments


def build_experiment_grid():
    """Generate all hyperparameter combinations using itertools.product."""
    grid = []
    for exp_idx, (cfg, steps, neg_key, prompt_key) in enumerate(
        itertools.product(
            TRUE_CFG_SCALES,
            NUM_INFERENCE_STEPS,
            sorted(NEGATIVE_PROMPTS.keys()),
            sorted(PROMPT_STYLES.keys()),
        ),
        start=1,
    ):
        grid.append({
            "experiment_num": exp_idx,
            "true_cfg_scale": cfg,
            "num_inference_steps": steps,
            "negative_prompt_key": neg_key,
            "negative_prompt": NEGATIVE_PROMPTS[neg_key],
            "prompt_style_key": prompt_key,
            "prompt_fn": PROMPT_STYLES[prompt_key],
        })
    return grid


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


def experiment_label(exp):
    """Short human-readable label for an experiment."""
    return (
        f"cfg={exp['true_cfg_scale']}_steps={exp['num_inference_steps']}_"
        f"neg={exp['negative_prompt_key']}_prompt={exp['prompt_style_key']}"
    )


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

def run_edit(pipeline, base_img, ref_img, prompt, exp):
    """Run a single tablecloth swap and return (result_image, elapsed_seconds)."""
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=exp["negative_prompt"],
            num_inference_steps=exp["num_inference_steps"],
            true_cfg_scale=exp["true_cfg_scale"],
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

def run_experiment(pipeline, exp, base_img, ref_images, output_dir):
    """Run one experiment across all 4 tablecloths and save outputs."""
    exp_num = exp["experiment_num"]
    exp_dir = os.path.join(output_dir, f"experiment_{exp_num:02d}")
    os.makedirs(exp_dir, exist_ok=True)

    label = experiment_label(exp)
    print_banner(f"EXPERIMENT {exp_num:02d}/32: {label}")

    image_times = []
    prompts_used = []
    exp_start = time.time()

    for tc_idx, tablecloth in enumerate(TABLECLOTHS, start=1):
        tc_name = tablecloth["name"]
        prompt = exp["prompt_fn"](tc_name)
        prompts_used.append(prompt)

        print(f"  [{tc_idx}/4] {tc_name}")
        print(f"         prompt: {prompt}")
        print(f"         neg:    {exp['negative_prompt_key']}")

        result, elapsed = run_edit(pipeline, base_img, ref_images[tc_idx - 1], prompt, exp)

        out_path = os.path.join(exp_dir, f"tablecloth_{tc_idx}.png")
        result.save(out_path)
        image_times.append({"name": tc_name, "time": elapsed})
        print(f"         time:   {elapsed:.2f}s  ->  {out_path}")

    total_time = time.time() - exp_start

    # Write per-experiment report
    write_experiment_report(exp, exp_dir, prompts_used, image_times, total_time)

    return {
        "experiment_num": exp_num,
        "label": label,
        "image_times": image_times,
        "total_time": total_time,
        "params": {
            "true_cfg_scale": exp["true_cfg_scale"],
            "num_inference_steps": exp["num_inference_steps"],
            "negative_prompt_key": exp["negative_prompt_key"],
            "prompt_style_key": exp["prompt_style_key"],
        },
    }


def write_experiment_report(exp, exp_dir, prompts_used, image_times, total_time):
    """Write report.txt for a single experiment."""
    report_path = os.path.join(exp_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Experiment {exp['experiment_num']:02d}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write(f"true_cfg_scale:     {exp['true_cfg_scale']}\n")
        f.write(f"num_inference_steps:{exp['num_inference_steps']}\n")
        f.write(f"guidance_scale:     {GUIDANCE_SCALE} (placeholder)\n")
        f.write(f"seed:               {SEED}\n")
        f.write(f"resolution:         {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
                f"{REF_SIZE}x{REF_SIZE} (ref)\n")
        f.write(f"negative_prompt:    [{exp['negative_prompt_key']}] "
                f"{exp['negative_prompt']}\n")
        f.write(f"prompt_style:       {exp['prompt_style_key']}\n\n")

        f.write("PROMPTS USED\n")
        f.write("-" * 60 + "\n")
        for i, (tc, prompt) in enumerate(zip(TABLECLOTHS, prompts_used), 1):
            f.write(f"  tablecloth_{i} ({tc['name']}):\n")
            f.write(f"    {prompt}\n\n")

        f.write("TIMING\n")
        f.write("-" * 60 + "\n")
        for i, t in enumerate(image_times, 1):
            f.write(f"  tablecloth_{i} ({t['name']}): {t['time']:.2f}s\n")
        inference_total = sum(t["time"] for t in image_times)
        f.write(f"\n  Inference total:  {inference_total:.2f}s\n")
        f.write(f"  Wall-clock total: {total_time:.2f}s\n")

    print(f"  Report saved: {report_path}")


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(results, output_dir):
    """Write top-level summary.txt comparing all experiments."""
    summary_path = os.path.join(output_dir, "summary.txt")

    # Column widths
    hdr = (
        f"{'Exp':>4}  {'CFG':>5}  {'Steps':>5}  {'Neg Prompt':>12}  "
        f"{'Prompt Style':>14}  {'Img1':>6}  {'Img2':>6}  {'Img3':>6}  "
        f"{'Img4':>6}  {'Total':>7}"
    )
    sep = "-" * len(hdr)

    with open(summary_path, "w") as f:
        f.write("Tablecloth Experiment Grid - Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date:  {datetime.now()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"LoRA:  {LORA_WEIGHTS}\n")
        f.write(f"Seed:  {SEED}\n\n")

        f.write("Tablecloths tested:\n")
        for i, tc in enumerate(TABLECLOTHS, 1):
            f.write(f"  {i}. {tc['name']} ({tc['file']})\n")
        f.write("\n")

        f.write(hdr + "\n")
        f.write(sep + "\n")

        for r in sorted(results, key=lambda x: x["experiment_num"]):
            times = r["image_times"]
            img_strs = [f"{t['time']:6.2f}" for t in times]
            # pad if fewer than 4
            while len(img_strs) < 4:
                img_strs.append(f"{'N/A':>6}")
            line = (
                f"{r['experiment_num']:4d}  "
                f"{r['params']['true_cfg_scale']:5.1f}  "
                f"{r['params']['num_inference_steps']:5d}  "
                f"{r['params']['negative_prompt_key']:>12}  "
                f"{r['params']['prompt_style_key']:>14}  "
                f"{img_strs[0]}  {img_strs[1]}  {img_strs[2]}  {img_strs[3]}  "
                f"{r['total_time']:7.2f}"
            )
            f.write(line + "\n")

        f.write(sep + "\n")

        total_runtime = sum(r["total_time"] for r in results)
        f.write(f"\nTotal runtime across all experiments: {total_runtime:.2f}s "
                f"({total_runtime / 60:.1f} min)\n")

    print_banner(f"SUMMARY SAVED: {summary_path}")


# =============================================================================
# DRY RUN
# =============================================================================

def print_dry_run(grid):
    """Print the full experiment grid without running anything."""
    print_banner("DRY RUN - Experiment Grid")
    print(f"Total experiments: {len(grid)}")
    print(f"Images per experiment: {len(TABLECLOTHS)}")
    print(f"Total images: {len(grid) * len(TABLECLOTHS)}\n")

    hdr = f"{'Exp':>4}  {'CFG':>5}  {'Steps':>5}  {'Neg Prompt':>12}  {'Prompt Style':>14}"
    print(hdr)
    print("-" * len(hdr))

    for exp in grid:
        print(
            f"{exp['experiment_num']:4d}  "
            f"{exp['true_cfg_scale']:5.1f}  "
            f"{exp['num_inference_steps']:5d}  "
            f"{exp['negative_prompt_key']:>12}  "
            f"{exp['prompt_style_key']:>14}"
        )

    print(f"\nTablecloths:")
    for i, tc in enumerate(TABLECLOTHS, 1):
        print(f"  {i}. {tc['name']} ({tc['file']})")

    print(f"\nNegative prompts:")
    for key, val in sorted(NEGATIVE_PROMPTS.items()):
        print(f"  {key}: \"{val}\"")

    print(f"\nPrompt styles (example with '{TABLECLOTHS[0]['name']}'):")
    for key, fn in sorted(PROMPT_STYLES.items()):
        print(f"  {key}: \"{fn(TABLECLOTHS[0]['name'])}\"")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tablecloth experiment grid runner (Qwen-Image-Edit-2511 + Lightning)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment grid without running any inference.",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=None,
        metavar="N",
        help="Run only experiment N (1-32).",
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

    grid = build_experiment_grid()

    # --dry-run: print grid and exit
    if args.dry_run:
        print_dry_run(grid)
        return

    # --experiment N: filter to single experiment
    if args.experiment is not None:
        matches = [e for e in grid if e["experiment_num"] == args.experiment]
        if not matches:
            print(f"Error: experiment {args.experiment} not found (valid: 1-{len(grid)})")
            sys.exit(1)
        grid = matches

    # Validate inputs
    if not os.path.exists(BASE_IMAGE):
        print(f"Error: base image not found: {BASE_IMAGE}")
        sys.exit(1)

    for tc in TABLECLOTHS:
        tc_path = os.path.join(TABLECLOTH_DIR, tc["file"])
        if not os.path.exists(tc_path):
            print(f"Error: tablecloth not found: {tc_path}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Print run info
    print_banner("TABLECLOTH EXPERIMENT GRID")
    print(f"Date:            {datetime.now()}")
    print(f"Model:           {MODEL_NAME}")
    print(f"LoRA:            {LORA_WEIGHTS}")
    print(f"Base image:      {BASE_IMAGE}")
    print(f"Output dir:      {output_dir}")
    print(f"Resolution:      {FIXED_WIDTH}x{FIXED_HEIGHT} (main), {REF_SIZE}x{REF_SIZE} (ref)")
    print(f"Seed:            {SEED}")
    print(f"Experiments:     {len(grid)}")
    print(f"Images/exp:      {len(TABLECLOTHS)}")
    print(f"Total images:    {len(grid) * len(TABLECLOTHS)}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print()

    # Load base and reference images
    print("Loading images...")
    base_img = resize_to_fixed(Image.open(BASE_IMAGE).convert("RGB"))
    ref_images = []
    for tc in TABLECLOTHS:
        tc_path = os.path.join(TABLECLOTH_DIR, tc["file"])
        ref_images.append(resize_reference(Image.open(tc_path).convert("RGB")))
    print(f"  Base image loaded: {BASE_IMAGE}")
    print(f"  {len(ref_images)} reference tablecloths loaded")

    # Load model (once)
    pipeline = load_pipeline()

    # Warmup (once)
    warmup(pipeline)

    # Run experiments
    all_results = []
    run_start = time.time()

    for i, exp in enumerate(grid, 1):
        print(f"\n>>> Running experiment {i}/{len(grid)} <<<")
        result = run_experiment(pipeline, exp, base_img, ref_images, output_dir)
        all_results.append(result)

        gc.collect()
        torch.cuda.empty_cache()

    total_runtime = time.time() - run_start

    # Write summary
    write_summary(all_results, output_dir)

    # Final stats
    print_banner("ALL EXPERIMENTS COMPLETE")
    print(f"Experiments run:  {len(all_results)}")
    print(f"Images generated: {sum(len(r['image_times']) for r in all_results)}")
    print(f"Total runtime:    {total_runtime:.2f}s ({total_runtime / 60:.1f} min)")
    print(f"Output dir:       {output_dir}")
    print()


if __name__ == "__main__":
    main()

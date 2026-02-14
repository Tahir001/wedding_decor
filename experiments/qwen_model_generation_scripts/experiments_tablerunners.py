"""
===============================================================================
WEDDING DECOR - TABLE RUNNER EXPERIMENTS (LEVEL 2)
===============================================================================
Focused experiment runner for table runners, using optimized Level 1
hyperparameters as baseline.

Tests the critical variables for table runner generation:
  1. Image configuration: 2-image vs 3-image (with pose reference)
  2. Prompt style: minimal vs detailed (2img) vs detailed (3img)
  3. Pose reference type: wireframe blueprint vs realistic photo
  4. Pose image size (3-image configs only)
  5. Step count: 4 vs 8
  6. true_cfg_scale: 1.5 vs 2.0

Total: 96 runs
  - 2img:           2 cfgs x 2 prompts x 3 runners x 2 steps                  = 24
  - 3img_wireframe: 2 cfgs x 3 prompts x 3 runners x 1 pose_size x 2 steps   = 36
  - 3img_realistic: 2 cfgs x 3 prompts x 3 runners x 1 pose_size x 2 steps   = 36

Usage:
    python experiments_tablerunners.py                    # run all 96 experiments
    python experiments_tablerunners.py --dry-run          # print config, don't run
    python experiments_tablerunners.py --test             # sanity test (1 run, 3img)
    python experiments_tablerunners.py --base-image /path/to/tablecloth_result.png
    python experiments_tablerunners.py --output-dir /custom/path
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
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "tablerunners", "optimized")

# Default base image: a level 1 tablecloth result (Pintuck Red from Experiment 15)
# Override with --base-image CLI flag
DEFAULT_BASE_IMAGE = os.path.join(
    SCRIPT_DIR, "input", "base_image_damask.png"
)

# Pose reference images (image 3 candidates)
POSE_WIREFRAME = os.path.join(INPUT_DIR, "pose_wireframe.png")
POSE_REALISTIC = os.path.join(INPUT_DIR, "pose_realistic.png")

# =============================================================================
# TABLE RUNNER DEFINITIONS
# =============================================================================

RUNNER_DIR = os.path.join(INPUT_DIR, "tablerunners")

TABLE_RUNNERS = [
    {
        "name": "Velvet Cashmere",
        "filename": "Rectangle Velvet - Cashmere.jpeg",
        "color": "cashmere",
        "material": "velvet",
    },
    {
        "name": "Velvet Hot Pink",
        "filename": "Rectangle Velvet - Hot Pink.png",
        "color": "hot pink",
        "material": "velvet",
    },
    {
        "name": "Woven Ivory",
        "filename": "Rectangle Woven - Ivory.png",
        "color": "ivory",
        "material": "woven",
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
# FIXED HYPERPARAMETERS (from Level 1 optimization)
# =============================================================================

TRUE_CFG_SCALES = [1.5, 2.0]
STEP_COUNTS = [4, 8, 12]

# Pose image sizes to test (3-image configs only)
POSE_SIZES = [512]

# Negative prompt: Qwen-Image-Edit-2511 does NOT support negative conditioning
# (flow-matching architecture, not trained for it). The parameter must be present
# to avoid pipeline errors, but only a single space is needed.
# See: https://blog.promptmaster.pro/posts/qwen-image-negative-prompts
NEGATIVE_PROMPT = " "

# =============================================================================
# IMAGE CONFIGURATIONS
# =============================================================================
# Each config defines which images to pass and which prompt styles are valid.
# =============================================================================

IMAGE_CONFIGS = [
    {
        "name": "2img",
        "description": "Standard 2-image (base + runner swatch)",
        "use_pose": False,
        "pose_path": None,
        "valid_prompts": ["minimal", "detailed_2img"],
    },
    {
        "name": "3img_wireframe",
        "description": "3-image with wireframe/blueprint pose reference",
        "use_pose": True,
        "pose_path": POSE_WIREFRAME,
        "valid_prompts": ["minimal", "detailed_2img", "detailed_3img"],
    },
    {
        "name": "3img_realistic",
        "description": "3-image with realistic photo pose reference",
        "use_pose": True,
        "pose_path": POSE_REALISTIC,
        "valid_prompts": ["minimal", "detailed_2img", "detailed_3img"],
    },
]

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
# - For 3-image configs, the 3rd image (pose/layout) provides implicit
#   visual guidance through concatenation. We say "the reference" rather
#   than "image 3" because Qwen has a known image ordering issue (#169)
#   and doesn't reliably map numbered references to inputs.
# =============================================================================

PROMPT_TEMPLATES = {
    "minimal": (
        "Add only a table runner on top of the tablecloth in image 1, "
        "using the fabric shown in image 2. The runner is centered vertically "
        "across the round table from one edge to the opposite edge. "
        "The existing tablecloth, chairs, and background stay exactly the same."
    ),
    "detailed_2img": (
        "Add only a {color} {material} table runner on top of the tablecloth "
        "on the round table in image 1. Use the exact fabric color, texture, "
        "and pattern from image 2 for the runner. The runner is centered "
        "vertically across the table, running straight from the 12 o'clock "
        "edge to the 6 o'clock edge. It lies flat on the existing tablecloth "
        "surface and drapes smoothly over both edges with soft, natural fabric "
        "folds. The runner fabric looks like real {material} with authentic "
        "textile texture. The existing tablecloth underneath remains fully "
        "visible with its original color and pattern intact. The chairs, "
        "background, and all surroundings stay exactly as they appear in "
        "image 1. Photorealistic, professional event photography quality."
    ),
    "detailed_3img": (
        "Add only a {color} {material} table runner on top of the tablecloth "
        "on the round table. Use the exact fabric color, texture, and pattern "
        "from the runner swatch for the runner material. The runner is centered "
        "vertically across the round table, running straight from one edge to "
        "the opposite edge, matching the placement and layout shown in the "
        "reference. It lies flat on the existing tablecloth and drapes smoothly "
        "over both edges with soft, natural fabric folds. The runner fabric "
        "looks like real {material} with authentic textile texture. The existing "
        "tablecloth underneath remains fully visible with its original color "
        "and pattern intact. The chairs, background, and all surroundings stay "
        "exactly as they appear in the original scene. Photorealistic, "
        "professional event photography quality."
    ),
}


# =============================================================================
# EXPERIMENT GRID BUILDER
# =============================================================================

def build_experiment_grid():
    """Build the full experiment grid.

    For 2-image configs: cfg_scale x prompt_style x runner x step_count.
    For 3-image configs: cfg_scale x prompt_style x runner x pose_size x step_count.

    Returns a list of experiment dicts, one per run.
    """
    experiments = []
    exp_id = 1

    for cfg_scale in TRUE_CFG_SCALES:
        for img_config in IMAGE_CONFIGS:
            for prompt_key in img_config["valid_prompts"]:
                for runner in TABLE_RUNNERS:
                    # Format the prompt with runner-specific details
                    prompt_text = PROMPT_TEMPLATES[prompt_key].format(
                        color=runner["color"],
                        material=runner["material"],
                    )

                    # Pose sizes: only sweep for 3-image configs
                    pose_sizes = POSE_SIZES if img_config["use_pose"] else [None]

                    for pose_size in pose_sizes:
                        for num_steps in STEP_COUNTS:
                            experiments.append({
                                "id": exp_id,
                                "image_config": img_config,
                                "prompt_style": prompt_key,
                                "runner": runner,
                                "prompt_text": prompt_text,
                                "pose_size": pose_size,
                                "num_steps": num_steps,
                                "cfg_scale": cfg_scale,
                            })
                            exp_id += 1

    return experiments


# =============================================================================
# PROMPT BUILDING (for display / reports)
# =============================================================================

def format_prompt(prompt_key, runner):
    """Format a prompt template with runner details."""
    return PROMPT_TEMPLATES[prompt_key].format(
        color=runner["color"],
        material=runner["material"],
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

def run_edit(pipeline, image_list, prompt, negative_prompt, num_steps, cfg_scale):
    """Run a single table runner edit and return (result_image, elapsed_seconds).

    Args:
        pipeline:         Loaded diffusion pipeline.
        image_list:       List of PIL images [base, ref] or [base, ref, pose].
        prompt:           Text prompt.
        negative_prompt:  Negative prompt.
        num_steps:        Number of inference steps.
        cfg_scale:        True CFG scale value.

    Returns:
        (result_image, elapsed_seconds)
    """
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.inference_mode():
        output = pipeline(
            image=image_list,
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

def run_experiments(pipeline, experiments, base_img, runner_images, pose_images,
                    output_dir):
    """Run the full experiment grid.

    Args:
        pipeline:       Loaded diffusion pipeline.
        experiments:    List of experiment dicts from build_experiment_grid().
        base_img:       PIL base image (tablecloth result from level 1).
        runner_images:  Dict mapping runner filename -> PIL reference image.
        pose_images:    Dict mapping (pose_path, pose_size) -> PIL pose image.
        output_dir:     Root output directory.

    Returns:
        List of result dicts with timing and output paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    total_start = time.time()

    for exp in experiments:
        exp_id = exp["id"]
        img_config = exp["image_config"]
        prompt_style = exp["prompt_style"]
        runner = exp["runner"]
        prompt_text = exp["prompt_text"]
        num_steps = exp["num_steps"]
        pose_size = exp["pose_size"]
        cfg_scale = exp["cfg_scale"]

        config_name = img_config["name"]
        runner_name = runner["name"]
        runner_slug = slugify(runner_name)

        # Create experiment output directory
        exp_dir = os.path.join(output_dir, f"experiment_{exp_id:02d}")
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n  [{exp_id:2d}/{len(experiments)}] "
              f"{config_name} / {prompt_style} / {runner_name} / "
              f"cfg={cfg_scale} / steps={num_steps}"
              f"{f' / pose={pose_size}px' if pose_size else ''}")
        print(f"         prompt: {prompt_text}")
        print(f"         neg:    (disabled, single space)")

        # Build image list
        ref_img = runner_images[runner["filename"]]
        image_list = [base_img, ref_img]

        if img_config["use_pose"]:
            pose_img = pose_images[(img_config["pose_path"], pose_size)]
            image_list.append(pose_img)
            print(f"         pose:   {os.path.basename(img_config['pose_path'])} "
                  f"@ {pose_size}x{pose_size}")
        else:
            print(f"         pose:   (none, 2-image mode)")

        print(f"         cfg:    {cfg_scale}")
        print(f"         steps:  {num_steps}")

        # Run inference
        result_img, elapsed = run_edit(
            pipeline, image_list, prompt_text, NEGATIVE_PROMPT, num_steps,
            cfg_scale
        )

        # Save output image
        out_path = os.path.join(exp_dir, f"runner_{runner_slug}.png")
        result_img.save(out_path)

        print(f"         time:   {elapsed:.2f}s  ->  {out_path}")

        # Write per-experiment report
        report_path = os.path.join(exp_dir, "report.txt")
        write_experiment_report(report_path, exp, elapsed, out_path)

        result_entry = {
            "experiment_id": exp_id,
            "image_config": config_name,
            "prompt_style": prompt_style,
            "runner_name": runner_name,
            "runner_filename": runner["filename"],
            "prompt_text": prompt_text,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "pose_size": pose_size if pose_size else "n/a",
            "pose_image": (os.path.basename(img_config["pose_path"])
                           if img_config["use_pose"] else "none"),
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
    img_config = exp["image_config"]
    runner = exp["runner"]
    num_steps = exp["num_steps"]
    pose_size = exp["pose_size"]
    cfg_scale = exp["cfg_scale"]

    with open(report_path, "w") as f:
        f.write(f"Experiment {exp['id']:02d}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n\n")

        f.write("IMAGE CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"config:             {img_config['name']}\n")
        f.write(f"description:        {img_config['description']}\n")
        f.write(f"num_images:         {'3' if img_config['use_pose'] else '2'}\n")
        if img_config["use_pose"]:
            f.write(f"pose_image:         "
                    f"{os.path.basename(img_config['pose_path'])}\n")
            f.write(f"pose_size:          {pose_size}x{pose_size}\n")
        f.write("\n")

        f.write("TABLE RUNNER\n")
        f.write("-" * 60 + "\n")
        f.write(f"name:               {runner['name']}\n")
        f.write(f"filename:           {runner['filename']}\n")
        f.write(f"color:              {runner['color']}\n")
        f.write(f"material:           {runner['material']}\n\n")

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
                f"{REF_SIZE}x{REF_SIZE} (ref)\n")
        if pose_size:
            f.write(f"pose_resolution:    {pose_size}x{pose_size}\n")
        f.write("\n")

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
        f.write("Table Runner Experiments (Level 2) - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date:               {datetime.now()}\n")
        f.write(f"Model:              {MODEL_NAME}\n")
        f.write(f"LoRA:               {LORA_WEIGHTS}\n")
        f.write(f"Seed:               {SEED}\n")
        f.write(f"Resolution:         {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
                f"{REF_SIZE}x{REF_SIZE} (ref)\n")
        f.write(f"true_cfg_scales:    {TRUE_CFG_SCALES}\n")
        f.write(f"step_counts:        {STEP_COUNTS}\n")
        f.write(f"pose_sizes:         {POSE_SIZES}\n")
        f.write(f"Base image:         {base_image_path}\n")
        f.write(f"Total experiments:  {len(results)}\n\n")

        # Negative prompt
        f.write("NEGATIVE PROMPT\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {NEGATIVE_PROMPT}\n\n")

        # Table runners tested
        f.write("TABLE RUNNERS TESTED\n")
        f.write("-" * 70 + "\n")
        for i, runner in enumerate(TABLE_RUNNERS, 1):
            f.write(f"  {i}. {runner['name']}\n")
            f.write(f"     file:     {runner['filename']}\n")
            f.write(f"     color:    {runner['color']}\n")
            f.write(f"     material: {runner['material']}\n")
        f.write("\n")

        # Image configurations
        f.write("IMAGE CONFIGURATIONS\n")
        f.write("-" * 70 + "\n")
        for cfg in IMAGE_CONFIGS:
            f.write(f"  {cfg['name']:20s} {cfg['description']}\n")
            f.write(f"  {'':20s} prompts: {cfg['valid_prompts']}\n")
            if cfg["use_pose"]:
                f.write(f"  {'':20s} pose:    "
                        f"{os.path.basename(cfg['pose_path'])}\n")
        f.write("\n")

        # Prompt templates
        f.write("PROMPT TEMPLATES\n")
        f.write("-" * 70 + "\n")
        for key, template in PROMPT_TEMPLATES.items():
            f.write(f"  {key}:\n")
            f.write(f"    \"{template}\"\n\n")

        # Results table grouped by image config
        f.write("RESULTS\n")
        f.write("=" * 70 + "\n\n")

        # Table header
        hdr = (f"{'Exp':>3}  {'Image Config':>18}  {'Prompt Style':>14}  "
               f"{'Runner':>18}  {'CFG':>5}  {'Steps':>5}  {'Pose Size':>9}  "
               f"{'Pose':>20}  {'Time (s)':>9}")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")

        for r in results:
            pose_size_str = (f"{r['pose_size']}px"
                             if r['pose_size'] != "n/a" else "n/a")
            line = (
                f"{r['experiment_id']:3d}  "
                f"{r['image_config']:>18s}  "
                f"{r['prompt_style']:>14s}  "
                f"{r['runner_name']:>18s}  "
                f"{r['cfg_scale']:5.1f}  "
                f"{r['num_steps']:5d}  "
                f"{pose_size_str:>9s}  "
                f"{r['pose_image']:>20s}  "
                f"{r['time']:9.2f}"
            )
            f.write(line + "\n")

        f.write("\n")

        # Per-config averages
        f.write("PER-CONFIG AVERAGES\n")
        f.write("-" * 70 + "\n")
        for cfg in IMAGE_CONFIGS:
            cfg_results = [r for r in results
                           if r["image_config"] == cfg["name"]]
            if cfg_results:
                avg_time = sum(r["time"] for r in cfg_results) / len(cfg_results)
                total_time = sum(r["time"] for r in cfg_results)
                f.write(f"  {cfg['name']:20s}  "
                        f"runs={len(cfg_results):2d}  "
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

    print_banner("DRY RUN - Table Runner Experiment Config")

    print(f"Baseline:    Level 1 optimized config (Experiment 15)")
    print(f"CFG:         {TRUE_CFG_SCALES}")
    print(f"Steps:       {STEP_COUNTS}")
    print(f"Pose sizes:  {POSE_SIZES} (3-image configs only)")
    print(f"Seed:        {SEED}")
    print(f"Resolution:  {FIXED_WIDTH}x{FIXED_HEIGHT} (main), "
          f"{REF_SIZE}x{REF_SIZE} (ref)")
    print(f"Base image:  {base_image_path}")
    print()

    print(f"Negative prompt:")
    print(f"  \"{NEGATIVE_PROMPT}\"")
    print()

    print(f"Table runners ({len(TABLE_RUNNERS)}):")
    for i, runner in enumerate(TABLE_RUNNERS, 1):
        print(f"  {i}. {runner['name']}")
        print(f"     file:     {runner['filename']}")
        print(f"     color:    {runner['color']}")
        print(f"     material: {runner['material']}")
    print()

    print(f"Image configurations ({len(IMAGE_CONFIGS)}):")
    for cfg in IMAGE_CONFIGS:
        print(f"  {cfg['name']:20s} {cfg['description']}")
        print(f"  {'':20s} prompts: {cfg['valid_prompts']}")
        if cfg["use_pose"]:
            print(f"  {'':20s} pose: "
                  f"{os.path.basename(cfg['pose_path'])}")
    print()

    print(f"Prompt templates ({len(PROMPT_TEMPLATES)}):")
    for key, template in PROMPT_TEMPLATES.items():
        print(f"  {key}:")
        example = template.format(color="cashmere", material="velvet")
        print(f"    \"{example}\"")
    print()

    print_banner("EXPERIMENT GRID", char="-")
    for exp in experiments:
        cfg_name = exp["image_config"]["name"]
        pose_info = (f"pose={exp['pose_size']}px"
                     if exp["pose_size"] else "no_pose")
        print(f"  Exp {exp['id']:2d}: {cfg_name:18s}  "
              f"{exp['prompt_style']:14s}  {exp['runner']['name']:18s}  "
              f"cfg={exp['cfg_scale']}  steps={exp['num_steps']}  {pose_info}")
    print()
    print(f"Total experiments: {len(experiments)}")


# =============================================================================
# SANITY TEST
# =============================================================================

def run_sanity_test(pipeline, base_img, runner_images, pose_images, output_dir):
    """Run a single 3-image experiment to verify the pipeline works."""
    print_banner("SANITY TEST: 3-image pipeline verification")

    test_dir = os.path.join(output_dir, "sanity_test")
    os.makedirs(test_dir, exist_ok=True)

    runner = TABLE_RUNNERS[0]  # Velvet Cashmere
    ref_img = runner_images[runner["filename"]]
    test_steps = STEP_COUNTS[-1]  # use highest step count for best quality
    test_pose_size = POSE_SIZES[0]  # 384
    test_cfg = TRUE_CFG_SCALES[0]  # use first CFG scale

    # Test with wireframe pose (3 images)
    pose_img = pose_images[(POSE_WIREFRAME, test_pose_size)]
    image_list = [base_img, ref_img, pose_img]

    prompt = format_prompt("detailed_3img", runner)

    print(f"  Runner:  {runner['name']}")
    print(f"  Config:  3img_wireframe")
    print(f"  CFG:     {test_cfg}")
    print(f"  Steps:   {test_steps}")
    print(f"  Pose:    {test_pose_size}x{test_pose_size}")
    print(f"  Prompt:  {prompt}")
    print(f"  Images:  {len(image_list)} (base + runner + wireframe pose)")
    print()

    result_img, elapsed = run_edit(
        pipeline, image_list, prompt, NEGATIVE_PROMPT, test_steps, test_cfg
    )

    out_path = os.path.join(test_dir, f"sanity_3img_{slugify(runner['name'])}.png")
    result_img.save(out_path)

    print(f"  Result:  {out_path}")
    print(f"  Time:    {elapsed:.2f}s")
    print()
    print(f"  3-image pipeline works! The model accepted 3 input images.")
    print(f"  Inspect the output to verify quality before running full grid.")

    return out_path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Table runner experiments (Level 2) - "
                    "Qwen-Image-Edit-2511 + Lightning"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment config without running any inference.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single sanity test (1 run with 3-image input) "
             "to verify the pipeline accepts 3 images.",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        help=f"Path to base image (tablecloth result from level 1). "
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
        print(f"  Use --base-image to specify a tablecloth result from level 1.")
        print(f"  Example: --base-image output/tablecloths/experiment_15/"
              f"tablecloth_3.png")
        sys.exit(1)

    for runner in TABLE_RUNNERS:
        runner_path = os.path.join(RUNNER_DIR, runner["filename"])
        if not os.path.exists(runner_path):
            print(f"Error: table runner not found: {runner_path}")
            sys.exit(1)

    for pose_path in [POSE_WIREFRAME, POSE_REALISTIC]:
        if not os.path.exists(pose_path):
            print(f"Error: pose image not found: {pose_path}")
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
    print_banner("TABLE RUNNER EXPERIMENTS (LEVEL 2)")
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
    print(f"Pose sizes:      {POSE_SIZES}")
    print(f"Mode:            {'sanity test' if args.test else 'full grid'}")
    print(f"Total runs:      {1 if args.test else len(experiments)}")
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

    print("Loading table runner reference images...")
    runner_images = {}
    for runner in TABLE_RUNNERS:
        runner_path = os.path.join(RUNNER_DIR, runner["filename"])
        runner_images[runner["filename"]] = resize_reference(
            Image.open(runner_path).convert("RGB")
        )
        print(f"  Loaded: {runner['name']} ({runner['filename']})")

    print("Loading pose reference images...")
    pose_images = {}
    for pose_path in [POSE_WIREFRAME, POSE_REALISTIC]:
        raw_pose = Image.open(pose_path).convert("RGB")
        for pose_size in POSE_SIZES:
            resized = raw_pose.resize((pose_size, pose_size), Image.LANCZOS)
            pose_images[(pose_path, pose_size)] = resized
            print(f"  Loaded: {os.path.basename(pose_path)} "
                  f"@ {pose_size}x{pose_size}")

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

    if args.test:
        # Sanity test: single 3-image run
        run_sanity_test(pipeline, base_img, runner_images, pose_images,
                        output_dir)
        total_runtime = time.time() - run_start
        print_banner("SANITY TEST COMPLETE")
        print(f"Total runtime: {total_runtime:.2f}s")
        return

    # Full experiment grid
    print_banner(f"RUNNING {len(experiments)} EXPERIMENTS")
    results, experiment_time = run_experiments(
        pipeline, experiments, base_img, runner_images, pose_images,
        output_dir
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

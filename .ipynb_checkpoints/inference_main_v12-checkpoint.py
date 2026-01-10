"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V12 (QWEN-IMAGE-EDIT-2511)
===============================================================================
UPGRADE from V11 (2509) to V12 (2511) - Latest Model Release (Dec 23, 2025)

KEY IMPROVEMENTS IN 2511:
1. MITIGATED IMAGE DRIFT - Previous edits are better preserved across steps
   (Critical for our 7-step sequential pipeline!)
2. Better object consistency - Elements maintain appearance through edits
3. Enhanced multi-reference handling
4. Stronger geometric reasoning - Better spatial placement
5. 4-step Lightning LoRA (vs 8-step) - Faster inference

WHAT'S THE SAME (for direct comparison with V11):
- Same base image
- Same reference images  
- Same prompts
- Same pipeline order (big → small)
- Same dimensions (1024x1024 / 384x384)
- Same seed

===============================================================================
"""

import os
import gc
import time
import math
import torch
from PIL import Image
from datetime import datetime
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = "/workspace/wedding_decor/images"
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v12_2511"
BASE_IMAGE = "base_image_table.png"

# === IMAGE DIMENSIONS (same as V11) ===
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# === MODEL CHANGE: 2509 → 2511 ===
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"

# === LORA CHANGE: 8-step → 4-step ===
# The 2511 Lightning LoRA uses 4-step distillation (faster than 2509's 8-step)
# Options:
#   - bf16 version (850 MB, recommended for memory)
#   - fp32 version (1.7 GB, slightly higher quality)
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# CFG settings (same as V11)
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# PIPELINE STEPS - BIG → SMALL 
# =============================================================================
# 
# STEP COUNT STRATEGY FOR 2511:
# The 4-step LoRA was distilled for optimal quality at 4 steps.
# We use 4-5 steps for most layers (vs 6-8 in V11).
# The model's improved consistency should maintain quality with fewer steps.
#
# Early steps (structural): 4 steps
# Medium steps (place settings): 4 steps  
# Later steps (details): 5 steps (slightly more for fine detail)
#

PIPELINE_STEPS = [
    # === STRUCTURAL (foundation) ===
    {
        "name": "chairs",
        "steps": 4,  # Was 6 in V11
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs with elegant gold chiavari chairs with white cushions matching the reference. 8 chairs evenly spaced around the round table with white tablecloth.",
    },
    {
        "name": "tablecloth",
        "steps": 4,  # Was 6 in V11
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "The round table now has a luxurious deep red satin tablecloth with elegant draping matching the reference. 8 gold chiavari chairs with white cushions surround the table.",
    },
    # === PLACE SETTINGS (medium detail) ===
    {
        "name": "plates",
        "steps": 4,  # Was 6 in V11
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 white dinner plates with gold rim matching the reference. One plate at each place setting on the red tablecloth. Gold chiavari chairs around table.",
    },
    {
        "name": "napkins",
        "steps": 4,  # Was 8 in V11
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Add pink satin napkins folded in elegant fan shapes on each plate, matching the reference. 8 place settings with plates on red tablecloth. Gold chiavari chairs.",
    },
    # === FINE DETAILS (precision) ===
    {
        "name": "cutlery",
        "steps": 5,  # Was 8 in V11
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add gold cutlery beside each plate - fork on left, knife and spoon on right, matching the reference. Complete place settings with plates and pink napkins on red tablecloth. Gold chiavari chairs.",
    },
    {
        "name": "glassware",
        "steps": 5,  # Was 8 in V11
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add crystal wine glasses at each place setting above the knife, matching the reference. Realistic glass transparency. Complete settings with plates, napkins, cutlery on red tablecloth. Gold chiavari chairs.",
    },
    {
        "name": "centerpiece",
        "steps": 5,  # Was 8 in V11
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add a stunning pink rose centerpiece on gold stand to the center of the table, matching the reference. Spherical arrangement of fresh roses. All 8 place settings surround it with plates, napkins, cutlery, glasses on red tablecloth. Gold chiavari chairs.",
    },
]

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


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL: Qwen-Image-Edit-2511")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Scheduler config (same as 2509)
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
    
    # Load 4-step Lightning LoRA (faster than 2509's 8-step)
    print(f"⚡ Loading 4-step Lightning LoRA from {LORA_REPO}...")
    pipeline.load_lora_weights(
        LORA_REPO,
        weight_name=LORA_WEIGHTS
    )
    print("✅ LoRA loaded")
    
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, step_config, step_num):
    steps = step_config['steps']
    name = step_config['name']
    prompt = step_config['prompt']
    
    print(f"\n🎨 Step {step_num}: {name} ({steps} inference steps)")
    print(f"   Prompt: {prompt[:70]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, missing items, wrong count",
            num_inference_steps=steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED + step_num),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    print(f"   ⏱️  Done in {elapsed:.2f}s")
    return result, elapsed


# =============================================================================
# WARMUP
# =============================================================================

def warmup(pipeline):
    print("\n🔥 Warmup run...")
    dummy = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), 'white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), 'gray')
    
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
    print("✅ Warmup complete")


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(pipeline):
    print_banner("WEDDING PIPELINE V12: QWEN-IMAGE-EDIT-2511", "🎨")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate expected steps
    total_steps = sum(s['steps'] for s in PIPELINE_STEPS)
    v11_steps = 6 + 6 + 6 + 8 + 8 + 8 + 8  # 50 steps in V11
    
    print(f"📋 Pipeline: {len(PIPELINE_STEPS)} layers")
    print(f"📊 Total inference steps: {total_steps} (vs {v11_steps} in V11)")
    print(f"📐 Fixed size: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"📷 Reference size: {REF_SIZE}x{REF_SIZE}")
    print(f"🔄 Model: {MODEL_NAME}")
    print(f"⚡ LoRA: 4-step Lightning (vs 8-step in V11)")
    
    # Load base
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Missing: {base_path}")
        return
    
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    print(f"\n💾 Saved original")
    
    # Warmup
    warmup(pipeline)
    
    # Run pipeline
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Missing ref: {step['ref_image']}")
            continue
        
        ref_img = resize_reference(Image.open(ref_path).convert("RGB"))
        
        result, elapsed = run_edit(pipeline, current_image, ref_img, step, i)
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"   💾 Saved: {output_path}")
        
        step_times.append({"name": step["name"], "steps": step["steps"], "time": elapsed})
        current_image = result
    
    # Final
    total_time = time.time() - pipeline_start
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # Summary
    print_banner("COMPLETE", "✅")
    print("📊 TIMING COMPARISON:")
    print("-" * 60)
    print(f"{'Layer':<15} {'Steps':<8} {'Time':<10} {'V11 Steps'}")
    print("-" * 60)
    v11_step_counts = [6, 6, 6, 8, 8, 8, 8]
    for i, s in enumerate(step_times):
        v11_steps_i = v11_step_counts[i] if i < len(v11_step_counts) else "?"
        print(f"{s['name']:<15} {s['steps']:<8} {s['time']:.2f}s     ({v11_steps_i} in V11)")
    print("-" * 60)
    
    inference_total = sum(s['time'] for s in step_times)
    print(f"   Inference: {inference_total:.2f}s ({total_steps} steps)")
    print(f"   Total:     {total_time:.2f}s")
    print(f"\n🏁 Result: {final_path}")
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("V12 - QWEN-IMAGE-EDIT-2511\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"LoRA: {LORA_WEIGHTS}\n")
        f.write(f"Total inference steps: {total_steps}\n\n")
        f.write("TIMING:\n")
        f.write("-" * 40 + "\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['steps']} steps, {s['time']:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total inference: {inference_total:.2f}s\n")
        f.write(f"Total wall time: {total_time:.2f}s\n\n")
        f.write("COMPARISON WITH V11 (2509):\n")
        f.write(f"- V11 steps: 50 | V12 steps: {total_steps}\n")
        f.write(f"- V11 LoRA: 8-step | V12 LoRA: 4-step\n")
        f.write("- Expected: Better consistency, less image drift\n")


if __name__ == "__main__":
    print(f"🚀 Starting V12 (2511) at {datetime.now().strftime('%H:%M:%S')}")
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    
    pipeline = load_pipeline()
    run_pipeline(pipeline)
    
    print(f"\n✨ Done at {datetime.now().strftime('%H:%M:%S')}")
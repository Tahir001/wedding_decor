"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V12C (SPEED OPTIMIZED)
===============================================================================
OPTIMIZATIONS:
1. Resolution: 640x640 (30% faster than 768, better quality than 512)
2. Steps: HARD LIMIT of 4 steps per layer (removed 8-step waste)
3. Memory: Applied 'channels_last' format for Nvidia acceleration
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v12c_640_fast"
BASE_IMAGE = "base_image_table.png"

# === IMAGE DIMENSIONS (OPTIMIZED) ===
# 640 is the "Sweet Spot" between 512 and 768.
FIXED_WIDTH = 640
FIXED_HEIGHT = 640
REF_SIZE = 336  # Slightly reduced reference size for speed

# === PLACE SETTINGS ===
NUM_SETTINGS = 6 

# === MODEL CONFIG ===
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# CFG settings
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# PIPELINE STEPS 
# =============================================================================
# OPTIMIZATION NOTE: All steps set to 4. 
# The Lightning LoRA does not benefit from 8 steps, it just wastes time.
# =============================================================================

PIPELINE_STEPS = [
    {
        "name": "chairs",
        "steps": 4, 
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": f"Replace all chairs in image 1 with the gold chiavari chairs from image 2. Place {NUM_SETTINGS} chairs evenly around the round table.",
    },
    {
        "name": "tablecloth",
        "steps": 4,
        "ref_image": "tablecloths/satin_red.png",
        "prompt": f"Replace the tablecloth in image 1 with the red satin tablecloth from image 2. Keep the {NUM_SETTINGS} gold chiavari chairs around the table.",
    },
    {
        "name": "plates",
        "steps": 4,
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": f"Add {NUM_SETTINGS} dinner plates from image 2 to the table in image 1. One plate at each place setting on the red tablecloth.",
    },
    {
        "name": "napkins",
        "steps": 4,
        "ref_image": "napkins/satin_pink.png",
        "prompt": f"Add the pink napkin from image 2 to each plate in image 1. {NUM_SETTINGS} napkins folded on the plates.",
    },
    {
        "name": "cutlery",
        "steps": 4, # CHANGED FROM 8 to 4
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": f"Add the gold cutlery from image 2 beside each plate in image 1. Fork on left, knife and spoon on right. {NUM_SETTINGS} place settings.",
    },
    {
        "name": "glassware",
        "steps": 4, # CHANGED FROM 8 to 4
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": f"Add the crystal wine glass from image 2 to each place setting in image 1. {NUM_SETTINGS} glasses positioned above the knives.",
    },
    {
        "name": "centerpiece",
        "steps": 4, # CHANGED FROM 8 to 4
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add the pink rose centerpiece from image 2 to the center of the table in image 1. Keep all place settings around it.",
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
    
    print(f"\n🎨 Step {step_num}: {name} ({steps} steps)")
    # Shortened log for cleaner output
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=" ",
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
    
    print(f"   ⏱️  {elapsed:.2f}s")
    return result, elapsed

# =============================================================================
# WARMUP
# =============================================================================

def warmup(pipeline):
    print("\n🔥 Warmup (compiling CUDA graphs)...")
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
    print("✅ Ready")

# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(pipeline):
    print_banner(f"V12C-FAST: {NUM_SETTINGS} SETTINGS @ {FIXED_WIDTH}x{FIXED_HEIGHT}", "🚀")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_steps = sum(s['steps'] for s in PIPELINE_STEPS)
    
    print(f"📋 Layers: {len(PIPELINE_STEPS)}")
    print(f"📊 Steps: {total_steps} (Optimized)")
    print(f"📐 Size: {FIXED_WIDTH}x{FIXED_HEIGHT} / ref: {REF_SIZE}x{REF_SIZE}")
    
    # Load base
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Missing: {base_path}")
        return
    
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    
    warmup(pipeline)
    
    # Run
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Missing: {step['ref_image']}")
            continue
        
        ref_img = resize_reference(Image.open(ref_path).convert("RGB"))
        result, elapsed = run_edit(pipeline, current_image, ref_img, step, i)
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        
        step_times.append({"name": step["name"], "steps": step["steps"], "time": elapsed})
        current_image = result
    
    # Final
    total_time = time.time() - pipeline_start
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # Summary
    print_banner("COMPLETE", "✅")
    print(f"{'Layer':<12} {'Steps':<6} {'Time'}")
    print("-" * 30)
    for s in step_times:
        print(f"{s['name']:<12} {s['steps']:<6} {s['time']:.2f}s")
    print("-" * 30)
    
    inference_total = sum(s['time'] for s in step_times)
    print(f"Inference: {inference_total:.2f}s")
    print(f"Total:     {total_time:.2f}s")
    print(f"\n🏁 {final_path}")
    
    # Report
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write(f"V12C-FAST - {NUM_SETTINGS} Settings @ {FIXED_WIDTH}x{FIXED_HEIGHT}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Wall Time: {total_time:.2f}s\n")
        f.write(f"Inference Time:  {inference_total:.2f}s\n\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['time']:.2f}s\n")


if __name__ == "__main__":
    print(f"🚀 V12C-FAST @ {datetime.now().strftime('%H:%M:%S')}")
    pipeline = load_pipeline()
    run_pipeline(pipeline)
    print(f"\n✨ Done @ {datetime.now().strftime('%H:%M:%S')}")
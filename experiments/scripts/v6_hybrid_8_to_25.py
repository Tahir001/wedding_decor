"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V6 (HYBRID)
===============================================================================
Strategy: HYBRID STEPS + BIG TO SMALL
1. BIG CHANGES FIRST (Chairs, Tablecloth) -> SMALL LAST (Cutlery, Centerpiece)
2. VARIABLE STEPS:
   - Early stages (Big items): 8 Steps (Fast, forgiving)
   - Late stages (Fine details): 25 Steps (Precise, preserves details)
3. Fixed dimensions (1024x1024)
4. Small reference images (384px)

Hypothesis: 8 steps is fine for changing a tablecloth color, but 25 steps
is needed to draw consistent fork tines or glass stems without blurring.
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v6_hybrid_big_to_small"
BASE_IMAGE = "base_image_table.png"

# Fixed dimensions - square for consistency
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024

# Smaller reference images (faster processing, less interference)
REF_SIZE = 384

# Model Weights
LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

# Lightning settings
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# PIPELINE STEPS - BIG CHANGES FIRST → FINE DETAILS LAST
# =============================================================================
#
# Strategy:
# - Steps 1-4 (Big/Med): 8 Inference Steps (Speed)
# - Steps 5-7 (Fine):   25 Inference Steps (Quality/Precision)
#

PIPELINE_STEPS = [
    # === PHASE 1: BIG STRUCTURAL CHANGES (8 STEPS) ===
    {
        "name": "chairs",
        "steps": 8,
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs with elegant gold chiavari chairs with white cushions, matching the reference. 8 chairs evenly spaced around the round table. White tablecloth, simple setting.",
    },
    {
        "name": "tablecloth",
        "steps": 8,
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "Change the tablecloth to luxurious deep red satin with elegant draping, matching the reference. Gold chiavari chairs around the table. Simple white plates on table.",
    },
    {
        "name": "plates",
        "steps": 8,
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 white dinner plates with gold rim to the table, one at each seat, matching the reference. Red satin tablecloth, gold chiavari chairs surrounding.",
    },
    {
        "name": "napkins",
        "steps": 8,
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Add 8 pink satin napkins folded in elegant fan shapes, placed on top of the white plates, matching the reference. Red satin tablecloth, gold chiavari chairs.",
    },
    
    # === PHASE 2: FINE DETAILS (25 STEPS) ===
    # We slow down here to ensure cutlery doesn't merge and glass looks clear
    {
        "name": "cutlery",
        "steps": 25,
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add 8 sets of gold luxury cutlery at each place setting - fork on left, knife and spoon on right, matching the reference. Pink napkins on plates. Red tablecloth, gold chairs.",
    },
    {
        "name": "glassware",
        "steps": 25,
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add 8 crystal wine glasses around the table, one at each seat position, matching the reference. Gold cutlery, pink napkins, white plates. Red tablecloth, gold chairs.",
    },
    {
        "name": "centerpiece",
        "steps": 25,
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add a stunning pink rose centerpiece on a tall gold stand to the exact center of the table, matching the reference. Surrounded by wine glasses, gold cutlery, pink napkins. Red tablecloth, gold chairs.",
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

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"📊 GPU Memory: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total")

def resize_to_fixed(img, width=FIXED_WIDTH, height=FIXED_HEIGHT):
    return img.resize((width, height), Image.LANCZOS)

def resize_reference(img, size=REF_SIZE):
    return img.resize((size, size), Image.LANCZOS)

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL: Qwen-Image-Edit-2509 + Lightning LoRA")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("🔧 Configuring Lightning scheduler...")
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    print("📦 Loading base model...")
    load_start = time.time()
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"\n⚡ Loading Lightning LoRA...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    
    pipeline.set_progress_bar_config(disable=True)
    total_load_time = time.time() - load_start
    print(f"🚀 MODEL READY in {format_time(total_load_time)}")
    print_gpu_memory()
    
    return pipeline

# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_config, step_num):
    steps = step_config['steps']
    name = step_config['name']
    
    print(f"\n🎨 Generating Step {step_num}: {name} ({steps} steps)...")
    print(f"   Prompt: {prompt[:80]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, missing items, wrong count",
            num_inference_steps=steps,  # Dynamic step count
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED + step_num),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    print(f"   ⏱️  Completed in {elapsed:.2f}s")
    return result, elapsed

# =============================================================================
# WARMUP
# =============================================================================

def warmup_pipeline(pipeline):
    print("\n🔥 Running warmup (8 steps)...")
    dummy_img = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), color='gray')
    
    with torch.inference_mode():
        _ = pipeline(
            image=[dummy_img, dummy_ref],
            prompt="Test warmup",
            num_inference_steps=8,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(0),
        )
    torch.cuda.synchronize()
    print("✅ Warmup complete")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    print_banner("WEDDING PIPELINE V6: HYBRID STEPS", "🚀")
    print("Strategy: Big (8 steps) → Small (25 steps)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load base image
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Base image missing: {base_path}")
        return
        
    original_image = Image.open(base_path).convert("RGB")
    current_image = resize_to_fixed(original_image)
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    
    warmup_pipeline(pipeline)
    
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Ref missing: {step['ref_image']}")
            continue
            
        ref_img = resize_reference(Image.open(ref_path).convert("RGB"))
        
        result, elapsed = run_edit(
            pipeline=pipeline,
            base_img=current_image,
            ref_img=ref_img,
            prompt=step["prompt"],
            step_config=step,
            step_num=i
        )
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        
        step_times.append({"step": i, "name": step["name"], "time": elapsed, "steps": step["steps"]})
        current_image = result
    
    total_time = time.time() - pipeline_start
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    print_banner("PIPELINE COMPLETE", "=")
    print(f"{'STEP':<15} {'STEPS':<8} {'TIME':<10}")
    print("-" * 35)
    for s in step_times:
        print(f"{s['name']:<15} {s['steps']:<8} {s['time']:.2f}s")
    print("-" * 35)
    print(f"Total Time: {format_time(total_time)}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit(1)
    pipeline = load_pipeline()
    run_full_pipeline(pipeline)
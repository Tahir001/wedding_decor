"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V8 (ANTI-GRAVITY & DEPTH)
===============================================================================
Fixes:
1. "Floating Glassware" -> Enforced by "casting shadows" and "inner circle" prompts.
2. Step 4 Quality -> Increased Napkins to 20 steps to prevent geometry loss.
3. Perspective Lock -> Added "High-angle view" to all prompts to keep camera steady.
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v8_perspective_fix"
BASE_IMAGE = "base_image_table.png"

FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# ROBUST PIPELINE STEPS
# =============================================================================

PIPELINE_STEPS = [
    # === PHASE 1: STRUCTURAL FOUNDATION (8 STEPS) ===
    # Fast updates for big geometry
    {
        "name": "chairs",
        "steps": 8,
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": (
            "High-angle view of a round wedding table. "
            "Replace existing chairs with 8 Gold Chiavari Chairs arranged in a perfect circle. "
            "The chairs have gold bamboo frames and white cushions. "
            "They are evenly spaced and facing the center. "
            "White background, clean lighting."
        ),
    },
    {
        "name": "tablecloth",
        "steps": 8,
        "ref_image": "tablecloths/satin_red.png",
        "prompt": (
            "High-angle view. Change the tablecloth to luxurious Deep Red Satin. "
            "The fabric drapes naturally with soft folds and a glossy sheen. "
            "The 8 Gold Chiavari chairs surround the red table. "
            "The table surface is a flat, red circular plane ready for setting."
        ),
    },
    {
        "name": "plates",
        "steps": 8,
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": (
            "High-angle view. Place 8 White Porcelain Plates with gold rims on the red tablecloth. "
            "One plate centered in front of each chair. "
            "The plates are lying flat on the table surface. "
            "Red satin tablecloth visible between plates. "
            "Gold chairs arranged around the edge."
        ),
    },
    
    # === PHASE 2: CRITICAL ANCHORING (INCREASED STEPS) ===
    # We bumped this to 20 steps. If the napkin looks fake, the glass will float.
    {
        "name": "napkins",
        "steps": 20, # INCREASED from 8 to 20 for stability
        "ref_image": "napkins/satin_pink.png",
        "prompt": (
            "High-angle view. On top of each of the 8 white plates, place a Pink Satin Napkin folded in a fan shape. "
            "The napkin sits physically on the ceramic plate. "
            "Realistic fabric texture, casting small shadows onto the white plate. "
            "Do not blend with the red tablecloth. "
            "The geometry of the plates must remain sharp."
        ),
    },
    
    # === PHASE 3: FINE DETAILS & PREVENTING FLOAT (25 STEPS) ===
    {
        "name": "cutlery",
        "steps": 25,
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": (
            "High-angle view. Place Gold Luxury Cutlery flat on the red tablecloth next to each plate. "
            "Fork on the left, knife and spoon on the right. "
            "The cutlery is resting on the table surface, casting realistic contact shadows. "
            "Metallic reflection. "
            "Maintain the pink napkins on the white plates. "
            "Ensure distinct separation between gold cutlery and red cloth."
        ),
    },
    {
        "name": "glassware",
        "steps": 25,
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": (
            "High-angle view. Place 8 Crystal Wine Glasses on the table surface. "
            "Position strictly: Inside the circle of plates, slightly to the right of each setting. "
            "The glasses must stand vertically on the red tablecloth base. "
            "Clear glass with reflections. "
            "WARNING: Do not place glasses in the foreground or floating in the air. "
            "All items must be physically supported by the table."
        ),
    },
    {
        "name": "centerpiece",
        "steps": 25,
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": (
            "High-angle view. In the center of the table, place a Pink Rose Floral Centerpiece on a tall gold stand. "
            "The base of the stand sits firmly on the center of the red tablecloth. "
            "The flower sphere is elevated. "
            "The surrounding place settings (glasses, cutlery, plates, napkins) remain undisturbed. "
            "Depth of field focuses on the table arrangement."
        ),
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
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL: V8 ANTI-GRAVITY")
    
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
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_config, step_num):
    steps = step_config['steps']
    name = step_config['name']
    
    print(f"\n🎨 Generating Step {step_num}: {name} ({steps} steps)...")
    print(f"   Prompt: {prompt[:100]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Updated Negative Prompt to fight floating objects
    negative_prompt = (
        "floating objects, levitating, flying glasses, foreground overlay, "
        "distorted perspective, out of scale, oversized, blurry, low quality, "
        "artifacts, missing items, fused objects, watermark, text"
    )
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=negative_prompt,
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
    
    print(f"   ⏱️  Completed in {elapsed:.2f}s")
    return result, elapsed

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    print_banner("WEDDING PIPELINE V8: PERSPECTIVE FIX", "🚀")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Base image missing: {base_path}")
        return
        
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    
    # Warmup
    print("\n🔥 Warming up...")
    dummy_img = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    with torch.inference_mode():
        _ = pipeline(image=[dummy_img, dummy_img], prompt="warmup", num_inference_steps=4, true_cfg_scale=1.0, guidance_scale=1.0)
    
    step_times = []
    
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
        step_times.append({"step": i, "name": step["name"], "time": elapsed})
        current_image = result
    
    current_image.save(os.path.join(OUTPUT_DIR, "FINAL_RESULT.png"))
    print("\n✅ DONE")

if __name__ == "__main__":
    if not torch.cuda.is_available(): exit(1)
    pipeline = load_pipeline()
    run_full_pipeline(pipeline)
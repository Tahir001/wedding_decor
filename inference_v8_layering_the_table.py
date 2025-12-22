"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V9 (THE BUILDER)
===============================================================================
Goal: "Building" a table layer-by-layer (Additive Logic).
Method: 
1. Strict "Physical Layering" order (Cloth -> Chairs -> Plates -> etc.)
2. ADDITIVE PROMPTS ONLY: "Add X to the existing Y." (No "Replace/Change")
3. PRESERVATION TAGS: "Existing X remains unchanged."
4. Hybrid Steps: 10 steps for layers (fast), 25 steps for fine details (safe).
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v9_layering_builder"
BASE_IMAGE = "base_image_table.png"

FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# PIPELINE STEPS - THE "SETTING THE TABLE" ORDER
# =============================================================================
# Logic: You cannot place a napkin until the plate exists. 
# You cannot place a plate until the tablecloth exists.
# We build from the ground up.

PIPELINE_STEPS = [
    # === LAYER 1: THE FOUNDATION ===
    {
        "name": "tablecloth",
        "steps": 10, # Fast layer
        "ref_image": "tablecloths/satin_red.png",
        "prompt": (
            "Drape the empty table with a luxurious Deep Red Satin Tablecloth. "
            "The fabric should be smooth with natural hanging folds. "
            "The surface is a clean, flat red plane ready for setting. "
            "High-angle view, bright lighting."
        ),
    },
    {
        "name": "chairs",
        "steps": 10, # Fast layer
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": (
            "Arrange 8 Gold Chiavari Chairs around the EXISTING red satin covered table. "
            "The chairs face the center. "
            "KEEP the red satin tablecloth exactly as it is. "
            "Just add the chairs around the perimeter."
        ),
    },
    
    # === LAYER 2: THE ANCHORS ===
    {
        "name": "plates",
        "steps": 12, # Need slightly more precision so they aren't flat circles
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": (
            "Place 8 White Porcelain Plates with gold rims onto the EXISTING red satin tablecloth. "
            "Position one plate in front of each EXISTING gold chair. "
            "The plates rest flat on the table surface. "
            "KEEP the red tablecloth and gold chairs unchanged."
        ),
    },
    
    # === LAYER 3: THE FABRIC DETAIL ===
    {
        "name": "napkins",
        "steps": 25, # HIGH STEPS: prevents blending into the plate
        "ref_image": "napkins/satin_pink.png",
        "prompt": (
            "Place a Pink Satin Napkin folded in a fan shape directly ON TOP of each EXISTING white plate. "
            "The napkin sits securely on the plate. "
            "KEEP the white plates, red tablecloth, and gold chairs exactly as they are. "
            "Do not modify the furniture, just add the napkins."
        ),
    },
    
    # === LAYER 4: THE HARDWARE (Fine Details) ===
    {
        "name": "cutlery",
        "steps": 25, # HIGH STEPS: prevents "melting" metal
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": (
            "Set the table with Gold Luxury Cutlery. "
            "Place a fork to the left and knife/spoon to the right of each EXISTING plate/napkin setting. "
            "The cutlery rests on the EXISTING red tablecloth. "
            "KEEP the pink napkins, white plates, and gold chairs unchanged."
        ),
    },
    {
        "name": "glassware",
        "steps": 25, # HIGH STEPS: prevents "floating" or ghosting
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": (
            "Place Crystal Wine Glasses on the table. "
            "Position one glass at each setting, just above the cutlery. "
            "The glasses rest on the EXISTING red tablecloth. "
            "KEEP the gold cutlery, pink napkins, white plates, and chairs unchanged. "
            "No floating objects."
        ),
    },
    
    # === LAYER 5: THE FINISHING TOUCH ===
    {
        "name": "centerpiece",
        "steps": 25,
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": (
            "Place a Floral Centerpiece in the empty center of the table. "
            "Pink roses on a gold stand. "
            "The base sits on the red tablecloth. "
            "KEEP all surrounding place settings (glasses, cutlery, plates, napkins, chairs) EXACTLY as they are. "
            "Do not disrupt the table setting."
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
    print_banner("LOADING MODEL: V9 'THE BUILDER'")
    
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
    
    print(f"\n🏗️  Building Layer {step_num}: {name} ({steps} steps)...")
    print(f"   Prompt: {prompt[:100]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Negative prompt focuses on preserving the "Construction" logic
    negative_prompt = (
        "modifying existing items, removing items, replacing items, "
        "floating objects, distorted perspective, blurry, low quality, "
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
    
    print(f"   ⏱️  Layer added in {elapsed:.2f}s")
    return result, elapsed

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    print_banner("WEDDING PIPELINE V9: THE BUILDER", "🔨")
    
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
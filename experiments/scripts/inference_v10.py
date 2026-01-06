"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V10 (SPEED DEMON)
===============================================================================
Optimizations:
1. Flash Attention 2 (Hardware acceleration)
2. Reduced Steps (Aligned with Lightning-8step LoRA capabilities)
3. torch.compile (Graph optimization)
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v10_speed_builder"
BASE_IMAGE = "base_image_table.png"

FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# OPTIMIZED PIPELINE STEPS
# =============================================================================
# The LoRA is trained for 8 steps. 
# We use 4 for simple background layers and 8 for detailed layers.
# This reduces total steps from ~130 to ~40.

PIPELINE_STEPS = [
    # === LAYER 1 ===
    {
        "name": "tablecloth",
        "steps": 4, # SPEED: 4 steps is enough for texture
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "Drape the empty table with a luxurious Deep Red Satin Tablecloth. Smooth fabric, natural folds. High-angle view.",
    },
    {
        "name": "chairs",
        "steps": 4, # SPEED: Simple geometry
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Arrange 8 Gold Chiavari Chairs around the EXISTING red satin covered table. Chairs face center. KEEP red tablecloth.",
    },
    # === LAYER 2 ===
    {
        "name": "plates",
        "steps": 6, 
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Place 8 White Porcelain Plates with gold rims onto the EXISTING red satin tablecloth in front of chairs. Flat on surface.",
    },
    # === LAYER 3 ===
    {
        "name": "napkins",
        "steps": 8, # Max fidelity for fabric folds
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Place a Pink Satin Napkin fan-folded ON TOP of each EXISTING white plate. KEEP plates and chairs.",
    },
    # === LAYER 4 ===
    {
        "name": "cutlery",
        "steps": 8, # Max fidelity for metal details
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Set Gold Luxury Cutlery. Fork left, knife/spoon right of EXISTING plate. KEEP napkins and plates.",
    },
    {
        "name": "glassware",
        "steps": 8, # Max fidelity for transparency
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Place Crystal Wine Glasses above cutlery. KEEP cutlery, napkins, plates. No floating objects.",
    },
    # === LAYER 5 ===
    {
        "name": "centerpiece",
        "steps": 8,
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Place a Floral Centerpiece in the center. Pink roses on gold stand. KEEP all surrounding place settings exactly as they are.",
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
    print_banner("LOADING MODEL: V10 'SPEED DEMON'")
    
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
    
    # OPTIMIZATION 1: Flash Attention 2
    print("⚡ Enabling Flash Attention 2...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" 
    ).to("cuda")
    
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    
    pipeline.set_progress_bar_config(disable=True)

    # OPTIMIZATION 2: Torch Compile
    # This takes about 60 seconds on the FIRST run, but makes every subsequent run 20-30% faster.
    print("🚀 Compiling Transformer Graph (This takes a minute, please wait)...")
    pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead", fullgraph=True)
    
    return pipeline

# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_config, step_num):
    steps = step_config['steps']
    name = step_config['name']
    
    print(f"\n🏗️  Layer {step_num}: {name} ({steps} steps)...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Shorter negative prompt saves a tiny bit of token processing
    negative_prompt = "removing items, replacing items, floating objects, distorted, blurry, artifacts"
    
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
    
    print(f"   ⚡ Layer added in {elapsed:.2f}s")
    return result, elapsed

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    print_banner("WEDDING PIPELINE V10: SPEED DEMON", "🔨")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Base image missing: {base_path}")
        return
        
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    
    # Warmup is CRITICAL for torch.compile
    print("\n🔥 Warming up & Compiling Kernels...")
    dummy_img = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    with torch.inference_mode():
        _ = pipeline(image=[dummy_img, dummy_img], prompt="warmup", num_inference_steps=2, true_cfg_scale=1.0, guidance_scale=1.0)
    print("   ...Compile Complete.")
    
    total_start = time.time()
    
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
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        current_image = result
    
    total_time = time.time() - total_start
    current_image.save(os.path.join(OUTPUT_DIR, "FINAL_RESULT.png"))
    
    print_banner(f"DONE. Total generation time: {total_time:.2f}s")

if __name__ == "__main__":
    if not torch.cuda.is_available(): exit(1)
    pipeline = load_pipeline()
    run_full_pipeline(pipeline)
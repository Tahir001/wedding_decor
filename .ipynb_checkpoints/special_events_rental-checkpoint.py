"""
===============================================================================
WEDDING DECOR - BATCH TABLECLOTH GENERATOR (QWEN-IMAGE-EDIT)
===============================================================================
GOAL: Apply different tablecloth references to a single BASE image.
INPUT: /workspace/wedding_decor/images/table_cloths
OUTPUT: /workspace/wedding_decor/images/output/tablecloths
===============================================================================
"""

import os
import gc
import time
import math
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# CONFIGURATION
# =============================================================================

# DIRS
ROOT_DIR = "/workspace/wedding_decor/images/SpecialEventsRentals"
TABLECLOTH_DIR = os.path.join(ROOT_DIR, "table_cloths_red")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/table_cloths_red")

# IMAGES
BASE_IMAGE_PATH = os.path.join(ROOT_DIR, "base_image_table.png")

# DIMENSIONS
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# MODEL CONFIG
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# INFERENCE SETTINGS
STEPS = 4
TRUE_CFG = 3.0
GUIDANCE = 1.0
SEED = 42

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}\n")

def resize_to_fixed(img):
    return img.resize((FIXED_WIDTH, FIXED_HEIGHT), Image.LANCZOS)

def resize_reference(img):
    return img.resize((REF_SIZE, REF_SIZE), Image.LANCZOS)

def get_clean_name(filename):
    # Removes extension and cleans up the name for potential logging
    return os.path.splitext(filename)[0].replace("120 Round - ", "").replace("120 Round", "").strip()

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_pipeline():
    print_banner("LOADING QWEN MODEL")
    
    # Clean memory
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
        MODEL_NAME,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Load Lightning LoRA for speed
    pipeline.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS)
    pipeline.set_progress_bar_config(disable=True)
    
    print("✅ Model & LoRA Loaded Successfully")
    return pipeline

# =============================================================================
# MAIN BATCH PROCESS
# =============================================================================

def run_batch_tablecloths():
    pipeline = load_pipeline()
    
    # 1. Setup Directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Created output directory: {OUTPUT_DIR}")

    # 2. Load Base Image
    if not os.path.exists(BASE_IMAGE_PATH):
        print(f"❌ CRITICAL ERROR: Base image not found at {BASE_IMAGE_PATH}")
        return

    print(f"🖼️  Loading Base Image: {BASE_IMAGE_PATH}")
    base_image_orig = Image.open(BASE_IMAGE_PATH).convert("RGB")
    base_image_fixed = resize_to_fixed(base_image_orig)

    # 3. Get list of tablecloths
    supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.avif')
    files = [f for f in os.listdir(TABLECLOTH_DIR) if f.lower().endswith(supported_ext)]
    files.sort() # Sort alphabetically
    
    print_banner(f"STARTING BATCH PROCESS: {len(files)} Tablecloths")

    # 4. Warmup Model
    print("🔥 Warming up model...")
    with torch.inference_mode():
        _ = pipeline(
            image=[base_image_fixed, resize_reference(base_image_fixed)],
            prompt="warmup",
            num_inference_steps=STEPS,
            true_cfg_scale=TRUE_CFG,
            guidance_scale=GUIDANCE
        )

    # 5. Iterate
    for i, filename in enumerate(files, 1):
        ref_path = os.path.join(TABLECLOTH_DIR, filename)
        clean_name = get_clean_name(filename)
        output_filename = f"fused_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Skip if already exists (optional, uncomment if you want to resume)
        # if os.path.exists(output_path):
        #     print(f"⏭️  Skipping {filename} (already done)")
        #     continue

        print(f"[{i}/{len(files)}] Processing: {clean_name}")
        
        try:
            # Load Reference
            ref_image = Image.open(ref_path).convert("RGB")
            ref_image = resize_reference(ref_image)

            # Construct Prompt
            # We use a generic prompt relying on Image 2 to provide the texture/color info
            # prompt = "Replace the tablecloth in image 1 with the tablecloth texture and color from image 2. Keep the 8 gold chiavari chairs around the table."
            prompt = (
                " Replace the tablecloth in image 1 with the tablecloth from image 2. Match image 2's exact color, texture, and pattern. Nothing else should change in image 1."
            )

            # Run Inference
            start_time = time.time()
            with torch.inference_mode():
                result = pipeline(
                    image=[base_image_fixed, ref_image],
                    prompt=prompt,
                    negative_prompt="distortion, blurry, low quality",
                    num_inference_steps=STEPS,
                    true_cfg_scale=TRUE_CFG,
                    guidance_scale=GUIDANCE,
                    generator=torch.Generator("cuda").manual_seed(SEED)
                ).images[0]
            
            # Save
            if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
                result = resize_to_fixed(result)
                
            result.save(output_path)
            elapsed = time.time() - start_time
            print(f"   ✅ Saved to: {output_filename} ({elapsed:.2f}s)")

        except Exception as e:
            print(f"   ❌ Failed to process {filename}: {str(e)}")

    print_banner("BATCH COMPLETE")
    print(f"Outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_batch_tablecloths()
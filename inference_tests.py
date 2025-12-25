"""
===============================================================================
WEDDING DECOR VISUALIZATION - TEST SCRIPT
===============================================================================
Based on Pipeline V11 (Clean & Fast)

Test 1: Image Fusion
- Fuse Image 2 (chairs) + Image 4.5 (tabletop) → Image A
- Fuse Image A + Image 6.5 (centerpiece with glasses) → Image B

Test 2: Sequential Edits
- Start with 4.5 (tabletop with table)
- Change charger plates → classic charger plates
- Change cutlery → silver cutlery  
- Change napkin → pink satin napkin
- Remove table (products float)
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
OUTPUT_BASE = "/workspace/wedding_decor/images/output/Tests"
TEST1_OUTPUT = os.path.join(OUTPUT_BASE, "test1")
TEST2_OUTPUT = os.path.join(OUTPUT_BASE, "test2")

# Image paths
IMAGE_2_CHAIRS = "2 - chairs chiavari white and gold, white tablecloth,.png"
IMAGE_4_5_NO_TABLE = "4.5 - NO TABLE - tabletop - gold cutlery, gold charger plate.jpeg"
IMAGE_4_5_WITH_TABLE = "4.5 - tabletop - gold cutlery, gold charger plate.png"
IMAGE_6_5 = "6.5 - NO TABLE - Centerpiece with glasses.jpeg"

# Reference images for Test 2
CHARGER_PLATE_CLASSIC = "charger_plate_classic.png"
CUTLERY_SILVER = "cutlery/classic_silver.png"
NAPKIN_PINK = "napkins/satin_pink.png"

# Fixed dimensions
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# Model settings
LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"
NUM_STEPS = 8  # 8 steps for each generation as requested
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

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


def load_image(path, resize_func=resize_to_fixed):
    """Load and resize an image."""
    full_path = os.path.join(INPUT_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    img = Image.open(full_path).convert("RGB")
    return resize_func(img)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL")
    
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
    
    print("📦 Loading Qwen-Image-Edit-2509...")
    load_start = time.time()
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"✅ Base model loaded in {time.time() - load_start:.1f}s")
    
    print("⚡ Loading Lightning 8-step LoRA...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    print("✅ LoRA loaded")
    
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_name, step_num):
    """Run a single edit step."""
    print(f"\n🎨 Step {step_num}: {step_name} ({NUM_STEPS} inference steps)")
    print(f"   Prompt: {prompt[:80]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts",
            num_inference_steps=NUM_STEPS,
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
# TEST 1: IMAGE FUSION
# =============================================================================

def run_test1(pipeline):
    print_banner("TEST 1: IMAGE FUSION", "🧪")
    
    os.makedirs(TEST1_OUTPUT, exist_ok=True)
    step_times = []
    
    # Load source images
    print("📂 Loading source images...")
    image_2 = load_image(IMAGE_2_CHAIRS)
    image_4_5 = load_image(IMAGE_4_5_NO_TABLE)
    image_6_5 = load_image(IMAGE_6_5)
    
    # Save originals for reference
    image_2.save(os.path.join(TEST1_OUTPUT, "step_0a_image2_chairs.png"))
    image_4_5.save(os.path.join(TEST1_OUTPUT, "step_0b_image4.5_tabletop.png"))
    image_6_5.save(os.path.join(TEST1_OUTPUT, "step_0c_image6.5_centerpiece.png"))
    print("💾 Saved original images")
    
    # Step 1: Fuse Image 2 + Image 4.5 → Image A
    prompt_1 = "Infuse these two pictures together. Combine the elegant chairs and table setup with the gold cutlery and charger plate tabletop arrangement seamlessly."
    
    # Use image_2 as base, image_4_5 as reference
    image_a, time_1 = run_edit(
        pipeline, 
        image_2,  # base image
        resize_reference(image_4_5),  # reference image
        prompt_1,
        "Fuse chairs + tabletop → Image A",
        1
    )
    
    image_a.save(os.path.join(TEST1_OUTPUT, "step_1_image_A_fused.png"))
    step_times.append({"name": "Image A (chairs + tabletop)", "time": time_1})
    print(f"   💾 Saved Image A")
    
    # Step 2: Fuse Image A + Image 6.5 → Image B
    prompt_2 = "Infuse these two pictures together. Combine the current table setup with the centerpiece and glasses arrangement seamlessly."
    
    image_b, time_2 = run_edit(
        pipeline,
        image_a,  # base image (result from step 1)
        resize_reference(image_6_5),  # reference image
        prompt_2,
        "Fuse Image A + centerpiece → Image B",
        2
    )
    
    image_b.save(os.path.join(TEST1_OUTPUT, "step_2_image_B_final.png"))
    image_b.save(os.path.join(TEST1_OUTPUT, "FINAL_RESULT.png"))
    step_times.append({"name": "Image B (A + centerpiece)", "time": time_2})
    print(f"   💾 Saved Image B (Final)")
    
    # Summary
    print("\n" + "-" * 50)
    print("📊 TEST 1 SUMMARY:")
    for i, s in enumerate(step_times, 1):
        print(f"   {i}. {s['name']}: {s['time']:.2f}s")
    total_time = sum(s['time'] for s in step_times)
    print(f"   Total: {total_time:.2f}s")
    print("-" * 50)
    
    # Save report
    with open(os.path.join(TEST1_OUTPUT, "report.txt"), "w") as f:
        f.write("TEST 1: IMAGE FUSION\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Steps per generation: {NUM_STEPS}\n\n")
        f.write("Process:\n")
        f.write("1. Fuse Image 2 (chairs) + Image 4.5 (tabletop) → Image A\n")
        f.write("2. Fuse Image A + Image 6.5 (centerpiece) → Image B\n\n")
        f.write("Timing:\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['time']:.2f}s\n")
        f.write(f"\nTotal: {total_time:.2f}s\n")
    
    return image_b


# =============================================================================
# TEST 2: SEQUENTIAL EDITS
# =============================================================================

def run_test2(pipeline):
    print_banner("TEST 2: SEQUENTIAL EDITS", "🧪")
    
    os.makedirs(TEST2_OUTPUT, exist_ok=True)
    step_times = []
    
    # Load base image (4.5 with table)
    print("📂 Loading base image...")
    current_image = load_image(IMAGE_4_5_WITH_TABLE)
    current_image.save(os.path.join(TEST2_OUTPUT, "step_0_original.png"))
    print("💾 Saved original")
    
    # Load reference images
    print("📂 Loading reference images...")
    charger_ref = load_image(CHARGER_PLATE_CLASSIC, resize_reference)
    cutlery_ref = load_image(CUTLERY_SILVER, resize_reference)
    napkin_ref = load_image(NAPKIN_PINK, resize_reference)
    
    # Save references for clarity
    charger_ref.save(os.path.join(TEST2_OUTPUT, "ref_charger_classic.png"))
    cutlery_ref.save(os.path.join(TEST2_OUTPUT, "ref_cutlery_silver.png"))
    napkin_ref.save(os.path.join(TEST2_OUTPUT, "ref_napkin_pink.png"))
    
    # Step 1: Change charger plates
    prompt_1 = "Replace the charger plates with these classic charger plates matching the reference. Keep the gold cutlery and all other elements exactly as they are."
    
    result_1, time_1 = run_edit(
        pipeline,
        current_image,
        charger_ref,
        prompt_1,
        "Change charger plates → classic",
        1
    )
    result_1.save(os.path.join(TEST2_OUTPUT, "step_1_charger_plates.png"))
    step_times.append({"name": "Charger plates → classic", "time": time_1})
    current_image = result_1
    
    # Step 2: Change cutlery to silver
    prompt_2 = "Replace the gold cutlery with elegant silver cutlery matching the reference. Fork on left, knife and spoon on right. Keep charger plates and all other elements exactly as they are."
    
    result_2, time_2 = run_edit(
        pipeline,
        current_image,
        cutlery_ref,
        prompt_2,
        "Change cutlery → silver",
        2
    )
    result_2.save(os.path.join(TEST2_OUTPUT, "step_2_cutlery_silver.png"))
    step_times.append({"name": "Cutlery → silver", "time": time_2})
    current_image = result_2
    
    # Step 3: Change napkin to pink satin
    prompt_3 = "Add elegant pink satin napkins folded beautifully on each plate, matching the reference. Keep the silver cutlery, charger plates, and all other elements exactly as they are."
    
    result_3, time_3 = run_edit(
        pipeline,
        current_image,
        napkin_ref,
        prompt_3,
        "Add napkins → pink satin",
        3
    )
    result_3.save(os.path.join(TEST2_OUTPUT, "step_3_napkins_pink.png"))
    step_times.append({"name": "Napkins → pink satin", "time": time_3})
    current_image = result_3
    
    # Step 4: Remove table, products float
    # For this step, we use the current image as both base and reference
    prompt_4 = "Remove the table completely. The tabletop items including the charger plates, cutlery, napkins, and all place settings should float elegantly in the air against a clean background. No table visible."
    
    result_4, time_4 = run_edit(
        pipeline,
        current_image,
        resize_reference(current_image),  # Use current as reference
        prompt_4,
        "Remove table → floating products",
        4
    )
    result_4.save(os.path.join(TEST2_OUTPUT, "step_4_floating_products.png"))
    result_4.save(os.path.join(TEST2_OUTPUT, "FINAL_RESULT.png"))
    step_times.append({"name": "Remove table → floating", "time": time_4})
    
    # Summary
    print("\n" + "-" * 50)
    print("📊 TEST 2 SUMMARY:")
    for i, s in enumerate(step_times, 1):
        print(f"   {i}. {s['name']}: {s['time']:.2f}s")
    total_time = sum(s['time'] for s in step_times)
    print(f"   Total: {total_time:.2f}s")
    print("-" * 50)
    
    # Save report
    with open(os.path.join(TEST2_OUTPUT, "report.txt"), "w") as f:
        f.write("TEST 2: SEQUENTIAL EDITS\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Steps per generation: {NUM_STEPS}\n\n")
        f.write("Process:\n")
        f.write("1. Base: 4.5 tabletop with gold cutlery, gold charger plate\n")
        f.write("2. Change charger plates → classic\n")
        f.write("3. Change cutlery → silver\n")
        f.write("4. Add napkins → pink satin\n")
        f.write("5. Remove table → floating products\n\n")
        f.write("Timing:\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['time']:.2f}s\n")
        f.write(f"\nTotal: {total_time:.2f}s\n")
    
    return result_4


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"🚀 Starting Tests at {datetime.now().strftime('%H:%M:%S')}")
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📐 Fixed size: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"📷 Reference size: {REF_SIZE}x{REF_SIZE}")
    print(f"🔢 Steps per generation: {NUM_STEPS}")
    
    # Create output directories
    os.makedirs(TEST1_OUTPUT, exist_ok=True)
    os.makedirs(TEST2_OUTPUT, exist_ok=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Warmup
    warmup(pipeline)
    
    # Run tests
    total_start = time.time()
    
    try:
        print_banner("RUNNING TEST 1", "🧪")
        run_test1(pipeline)
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print_banner("RUNNING TEST 2", "🧪")
        run_test2(pipeline)
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - total_start
    
    # Final summary
    print_banner("ALL TESTS COMPLETE", "✅")
    print(f"📁 Test 1 outputs: {TEST1_OUTPUT}")
    print(f"📁 Test 2 outputs: {TEST2_OUTPUT}")
    print(f"⏱️  Total time: {format_time(total_time)}")
    print(f"\n✨ Done at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
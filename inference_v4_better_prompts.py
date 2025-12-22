"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V4
===============================================================================
Key changes from V3:
1. FORCED CONSISTENT DIMENSIONS - All images stay at exactly the same size
2. Output is resized back to base dimensions after each step
3. Reference images are resized to a consistent size too
4. Using 8-step LoRA for quality

This should help prevent elements from drifting/disappearing between steps.
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v4_fixed_dimensions"
BASE_IMAGE = "base_image_table.png"

# FIXED DIMENSIONS - All images will be forced to this size
# Choose dimensions divisible by 16
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024  # Square format for consistency

# Reference image size (smaller is fine, model uses it for style/appearance)
REF_SIZE = 512

# Using 8-step for better quality
NUM_STEPS = 8
LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

# Lightning settings
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0

SEED = 42

# =============================================================================
# PIPELINE STEPS
# =============================================================================

PIPELINE_STEPS = [
    {
        "name": "chairs",
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs around the round table with elegant gold chiavari chairs with white cushions, matching the reference image style. Keep the white tablecloth and round table. 8 chairs evenly spaced around the table.",
    },
    {
        "name": "tablecloth",
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "The round table has a luxurious deep red satin tablecloth with elegant draping, matching the reference. 8 gold chiavari chairs with white cushions surround the table. Rich fabric sheen and natural folds.",
    },
    {
        "name": "plates",
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 elegant white dinner plates with gold rim matching the reference. One plate centered at each place setting around the red tablecloth. Gold chiavari chairs surround the table.",
    },
    {
        "name": "napkins",
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Each of the 8 white plates now has a pink satin napkin folded decoratively on top, matching the reference style. Red satin tablecloth, gold chiavari chairs. Elegant fan-fold napkin presentation.",
    },
    {
        "name": "cutlery",
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add gold cutlery sets beside each plate - fork on left, knife and spoon on right. Matching the luxurious gold reference style. 8 complete place settings with plates, pink napkins, and gold cutlery on red tablecloth.",
    },
    {
        "name": "glassware",
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add crystal wine glasses at each place setting, positioned above and to the right of each plate. 8 elegant glasses matching the reference with realistic glass transparency. Complete place settings on red tablecloth.",
    },
    {
        "name": "centerpiece",
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add a stunning pink rose centerpiece on a gold stand to the center of the table, matching the reference. Spherical arrangement of fresh pink and white roses. The complete table setting surrounds it - 8 place settings with plates, napkins, cutlery, and glasses on red tablecloth with gold chiavari chairs.",
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
    """
    Resize image to exact fixed dimensions.
    Uses padding/cropping to maintain aspect ratio as much as possible,
    then resizes to exact target.
    """
    # Simple resize to exact dimensions
    # For production, you might want smarter cropping/padding
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img, size=REF_SIZE):
    """Resize reference image to consistent square size"""
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
    print_banner(f"LOADING MODEL: Qwen-Image-Edit-2509 + Lightning 8-step LoRA")
    
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
        "shift_terminal": None,
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
    
    base_load_time = time.time() - load_start
    print(f"✅ Base model loaded in {format_time(base_load_time)}")
    print_gpu_memory()
    
    print(f"\n⚡ Loading Lightning 8-step LoRA...")
    lora_start = time.time()
    
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    
    lora_load_time = time.time() - lora_start
    print(f"✅ Lightning LoRA loaded in {format_time(lora_load_time)}")
    
    pipeline.set_progress_bar_config(disable=True)
    print_gpu_memory()
    
    total_load_time = time.time() - load_start
    print(f"\n🚀 TOTAL LOAD TIME: {format_time(total_load_time)}")
    
    return pipeline


# =============================================================================
# EDIT FUNCTION WITH DIMENSION ENFORCEMENT
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_name, step_num):
    """
    Run edit and FORCE output back to fixed dimensions
    """
    print(f"\n🎨 Generating: {step_name}...")
    print(f"   Base size: {base_img.size} (fixed: {FIXED_WIDTH}x{FIXED_HEIGHT})")
    print(f"   Ref size: {ref_img.size}")
    print(f"   Prompt: {prompt[:80]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, extra items, missing items, wrong count",
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED + step_num),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    original_size = result.size
    
    # FORCE back to fixed dimensions
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        print(f"   📐 Resizing output from {result.size} → ({FIXED_WIDTH}, {FIXED_HEIGHT})")
        result = resize_to_fixed(result)
    
    print(f"   ⏱️  Completed in {elapsed:.2f} seconds")
    
    return result, elapsed


# =============================================================================
# WARMUP
# =============================================================================

def warmup_pipeline(pipeline):
    print("\n🔥 Running warmup pass...")
    warmup_start = time.time()
    
    # Use fixed dimensions for warmup too
    dummy_img = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), color='gray')
    
    with torch.inference_mode():
        _ = pipeline(
            image=[dummy_img, dummy_ref],
            prompt="Test warmup",
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(0),
        )
    
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f"✅ Warmup complete in {format_time(warmup_time)}")
    
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    print_banner("WEDDING TABLE STYLING PIPELINE V4", "🎨")
    print("Key Feature: FIXED DIMENSIONS throughout pipeline")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n📋 CONFIGURATION:")
    print(f"   • Model: Qwen-Image-Edit-2509 + Lightning 8-step LoRA")
    print(f"   • Inference Steps: {NUM_STEPS}")
    print(f"   • FIXED Image Size: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"   • Reference Size: {REF_SIZE}x{REF_SIZE}")
    print(f"   • Seed: {SEED}")
    print(f"   • Output: {OUTPUT_DIR}")
    
    # Load and prepare base image
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"\n❌ ERROR: Base image not found: {base_path}")
        return None
    
    print(f"\n📷 Loading base image: {base_path}")
    original_image = Image.open(base_path).convert("RGB")
    print(f"   Original size: {original_image.size}")
    
    # Force to fixed dimensions
    current_image = resize_to_fixed(original_image)
    print(f"   Fixed size: {current_image.size}")
    
    original_path = os.path.join(OUTPUT_DIR, "step_0_original.png")
    current_image.save(original_path)
    print(f"💾 Saved: {original_path}")
    
    # Warmup with fixed dimensions
    warmup_pipeline(pipeline)
    
    step_times = []
    pipeline_start = time.time()
    
    # Run each step
    for i, step in enumerate(PIPELINE_STEPS, 1):
        print_banner(f"STEP {i}/{len(PIPELINE_STEPS)}: {step['name'].upper()}", "-")
        
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Reference image not found: {ref_path}")
            continue
        
        # Load and resize reference to consistent size
        ref_img = Image.open(ref_path).convert("RGB")
        ref_img = resize_reference(ref_img)
        print(f"📷 Reference: {ref_path} → {ref_img.size}")
        
        # Verify base image is still correct size
        assert current_image.size == (FIXED_WIDTH, FIXED_HEIGHT), \
            f"Image size drift detected! Got {current_image.size}, expected ({FIXED_WIDTH}, {FIXED_HEIGHT})"
        
        result, elapsed = run_edit(
            pipeline=pipeline,
            base_img=current_image,
            ref_img=ref_img,
            prompt=step["prompt"],
            step_name=step["name"],
            step_num=i
        )
        
        # Verify output is correct size
        assert result.size == (FIXED_WIDTH, FIXED_HEIGHT), \
            f"Output size wrong! Got {result.size}"
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"💾 Saved: {output_path} (verified: {result.size})")
        
        step_times.append({
            "step": i,
            "name": step["name"],
            "time": elapsed
        })
        
        current_image = result
        
        total_so_far = sum(s["time"] for s in step_times)
        print(f"📊 Running total: {format_time(total_so_far)}")
    
    # Final result
    pipeline_end = time.time()
    total_pipeline_time = pipeline_end - pipeline_start
    
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # Summary
    print_banner("PIPELINE COMPLETE! 🎉", "=")
    
    print("📊 TIMING BREAKDOWN:")
    print("-" * 50)
    
    for s in step_times:
        bar_len = int(s["time"] * 3)
        bar = "█" * min(bar_len, 40)
        print(f"   Step {s['step']}: {s['name']:<15} {s['time']:>6.2f}s  {bar}")
    
    print("-" * 50)
    
    inference_time = sum(s["time"] for s in step_times)
    avg_per_step = inference_time / len(step_times) if step_times else 0
    
    print(f"   {'INFERENCE TOTAL':<20} {inference_time:>6.2f}s")
    print(f"   {'AVERAGE PER STEP':<20} {avg_per_step:>6.2f}s")
    print(f"   {'PIPELINE TOTAL':<20} {total_pipeline_time:>6.2f}s")
    
    print(f"\n📐 All images maintained at: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"🏁 FINAL RESULT: {final_path}")
    print(f"📁 ALL OUTPUTS: {OUTPUT_DIR}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "timing_report.txt")
    with open(report_path, "w") as f:
        f.write("WEDDING DECOR PIPELINE V4 - FIXED DIMENSIONS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Qwen-Image-Edit-2509 + Lightning 8-step\n")
        f.write(f"Fixed Dimensions: {FIXED_WIDTH}x{FIXED_HEIGHT}\n")
        f.write(f"Reference Size: {REF_SIZE}x{REF_SIZE}\n\n")
        f.write("STEP TIMES:\n")
        f.write("-" * 50 + "\n")
        for s in step_times:
            f.write(f"  Step {s['step']}: {s['name']:<15} {s['time']:.2f}s\n")
        f.write("-" * 50 + "\n")
        f.write(f"  INFERENCE TOTAL:    {inference_time:.2f}s\n")
        f.write(f"  AVERAGE PER STEP:   {avg_per_step:.2f}s\n")
    
    print(f"📄 Timing report: {report_path}")
    
    return current_image


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_banner("QWEN-IMAGE-EDIT-2509 PIPELINE V4 - FIXED DIMENSIONS", "🚀")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    print(f"🔧 PyTorch: {torch.__version__}")
    
    pipeline = load_pipeline()
    final_image = run_full_pipeline(pipeline)
    
    print_banner("ALL DONE! ✨", "=")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
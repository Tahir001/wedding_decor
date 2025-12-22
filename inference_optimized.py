"""
===============================================================================
WEDDING DECOR VISUALIZATION - OPTIMIZED PIPELINE V2
===============================================================================
Changes from V1:
1. Reordered pipeline: Fine details FIRST, big changes LAST
2. Speed optimizations: torch.compile, better memory management
3. Improved prompts with explicit preservation instructions
4. Option to use 8-step LoRA for better quality

GPU: A100 80GB recommended
Model: Qwen-Image-Edit-2509 + Lightning LoRA
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/optimized_v2"
BASE_IMAGE = "base_image_table.png"

# Lightning settings - CHOOSE YOUR MODE:
# Mode 1: 4-step (fastest, ~3-5 sec/image on A100)
# Mode 2: 8-step (better quality, ~6-10 sec/image on A100)
USE_8_STEP = False  # Set to True for better quality

NUM_STEPS = 8 if USE_8_STEP else 4
LORA_WEIGHTS = (
    "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"
    if USE_8_STEP else
    "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32.safetensors"
)

# These are REQUIRED for Lightning - don't change
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0

SEED = 42
MAX_SIZE = 1024  # Can reduce to 768 for faster inference

# Enable torch.compile for speed (requires PyTorch 2.0+)
USE_TORCH_COMPILE = False  # Set True if you have time for warmup

# =============================================================================
# PIPELINE STEPS - REORDERED: Fine details FIRST, big changes LAST
# =============================================================================
# 
# Rationale: Start with the cleanest base image for precise work.
# Small items (cutlery, glasses) need precision - do them early.
# Big changes (chairs, tablecloth) are more forgiving of accumulated drift.
#

PIPELINE_STEPS = [
    # === PHASE 1: Fine details on clean base ===
    {
        "name": "centerpiece",
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add one elegant pink floral centerpiece with gold stand to the exact center of the table, matching the reference image. Fresh pink and white roses in a spherical arrangement. Do NOT modify the table, tablecloth, or chairs.",
    },
    {
        "name": "plates",
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 white dinner plates with gold rim matching the reference image. Place one plate centered at each seat position around the table. Do NOT modify the centerpiece, table, tablecloth, or chairs.",
    },
    {
        "name": "napkins",
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Add 8 pink satin napkins folded in a decorative fan shape matching the reference. Place one napkin on each plate. Do NOT modify plates, centerpiece, table, tablecloth, or chairs.",
    },
    {
        "name": "cutlery",
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add gold cutlery sets matching the reference beside each plate. Fork on left, knife and spoon on right. Do NOT modify napkins, plates, centerpiece, table, tablecloth, or chairs.",
    },
    {
        "name": "glassware",
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add crystal wine glasses matching the reference at each place setting, positioned above the knife. Show realistic glass transparency. Do NOT modify cutlery, napkins, plates, centerpiece, table, tablecloth, or chairs.",
    },
    
    # === PHASE 2: Major changes (more forgiving) ===
    {
        "name": "tablecloth",
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "Change the tablecloth to a luxurious red satin tablecloth matching the reference with natural draping and sheen. Keep ALL items on the table exactly as shown: centerpiece, plates, napkins, cutlery, glassware. Do NOT modify the chairs.",
    },
    {
        "name": "chairs",
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs with gold chiavari chairs with white cushions matching the reference style. Maintain the same positions around the table. Keep the entire table setting exactly as shown.",
    },
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    """Print a formatted banner"""
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def print_gpu_memory():
    """Print current GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"📊 GPU Memory: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total")


def resize_image(img, max_side=MAX_SIZE):
    """Resize image maintaining aspect ratio, dimensions divisible by 16"""
    w, h = img.size
    if max(w, h) <= max_side:
        new_w = w - (w % 16) if w % 16 != 0 else w
        new_h = h - (h % 16) if h % 16 != 0 else h
        if new_w != w or new_h != h:
            return img.resize((new_w, new_h), Image.LANCZOS)
        return img
    
    ratio = max_side / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    new_w = new_w - (new_w % 16)
    new_h = new_h - (new_h % 16)
    return img.resize((new_w, new_h), Image.LANCZOS)


def format_time(seconds):
    """Format seconds nicely"""
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
    """Load Qwen-Image-Edit-2509 with Lightning LoRA"""
    
    print_banner(f"LOADING MODEL: Qwen-Image-Edit-2509 + Lightning {NUM_STEPS}-step LoRA")
    
    # Clear any existing GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print("🔧 Configuring Lightning scheduler...")
    
    # Lightning-specific scheduler (REQUIRED)
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
    
    print(f"\n⚡ Loading Lightning {NUM_STEPS}-step LoRA...")
    lora_start = time.time()
    
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    
    lora_load_time = time.time() - lora_start
    print(f"✅ Lightning LoRA loaded in {format_time(lora_load_time)}")
    
    # Optional: Use torch.compile for faster inference (adds warmup time)
    if USE_TORCH_COMPILE:
        print("\n🔥 Compiling model with torch.compile (this takes a few minutes)...")
        pipeline.transformer = torch.compile(
            pipeline.transformer,
            mode="reduce-overhead",
            fullgraph=True
        )
        print("✅ Model compiled")
    
    # Disable progress bar for cleaner batch output
    pipeline.set_progress_bar_config(disable=True)
    
    print_gpu_memory()
    
    total_load_time = time.time() - load_start
    print(f"\n🚀 TOTAL LOAD TIME: {format_time(total_load_time)}")
    
    return pipeline


# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_name, step_num):
    """Run a single edit and return result with timing"""
    
    print(f"\n🎨 Generating: {step_name}...")
    print(f"   Base size: {base_img.size} | Ref size: {ref_img.size}")
    print(f"   Prompt: {prompt[:80]}...")
    
    # Ensure CUDA is synced for accurate timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, wrong count, missing items, extra items, changed background",
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED + step_num),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    
    print(f"   ⏱️  Completed in {elapsed:.2f} seconds")
    
    return result, elapsed


# =============================================================================
# WARMUP FUNCTION
# =============================================================================

def warmup_pipeline(pipeline, sample_img):
    """Run a warmup pass to initialize CUDA kernels"""
    print("\n🔥 Running warmup pass...")
    
    warmup_start = time.time()
    
    # Create a small test image
    small_img = sample_img.resize((512, 512), Image.LANCZOS)
    
    with torch.inference_mode():
        _ = pipeline(
            image=[small_img, small_img],
            prompt="Test warmup",
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(0),
        )
    
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f"✅ Warmup complete in {format_time(warmup_time)}")
    
    # Clear the warmup from memory
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    """Execute the complete wedding table styling pipeline"""
    
    print_banner("WEDDING TABLE STYLING PIPELINE V2 (OPTIMIZED)", "🎨")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print configuration
    print("📋 CONFIGURATION:")
    print(f"   • Model: Qwen-Image-Edit-2509 + Lightning LoRA")
    print(f"   • Steps: {NUM_STEPS} ({'8-step quality mode' if USE_8_STEP else '4-step speed mode'})")
    print(f"   • CFG Scale: {TRUE_CFG_SCALE} (Lightning mode)")
    print(f"   • Max Size: {MAX_SIZE}px")
    print(f"   • Seed: {SEED}")
    print(f"   • Output: {OUTPUT_DIR}")
    print(f"   • Pipeline: {len(PIPELINE_STEPS)} steps")
    print(f"   • Order: Fine details → Major changes")
    
    # Load base image
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"\n❌ ERROR: Base image not found: {base_path}")
        return None
    
    print(f"\n📷 Loading base image: {base_path}")
    current_image = Image.open(base_path).convert("RGB")
    current_image = resize_image(current_image)
    
    # Save original
    original_path = os.path.join(OUTPUT_DIR, "step_0_original.png")
    current_image.save(original_path)
    print(f"💾 Saved: {original_path} ({current_image.size[0]}x{current_image.size[1]})")
    
    # Run warmup
    warmup_pipeline(pipeline, current_image)
    
    # Track timing
    step_times = []
    pipeline_start = time.time()
    
    # Run each step
    for i, step in enumerate(PIPELINE_STEPS, 1):
        print_banner(f"STEP {i}/{len(PIPELINE_STEPS)}: {step['name'].upper()}", "-")
        
        # Load reference image
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Reference image not found: {ref_path}")
            print(f"   Skipping this step...")
            continue
        
        ref_img = Image.open(ref_path).convert("RGB")
        ref_img = resize_image(ref_img)
        print(f"📷 Reference: {ref_path}")
        
        # Run the edit
        result, elapsed = run_edit(
            pipeline=pipeline,
            base_img=current_image,
            ref_img=ref_img,
            prompt=step["prompt"],
            step_name=step["name"],
            step_num=i
        )
        
        # Save intermediate result
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"💾 Saved: {output_path}")
        
        # Record timing
        step_times.append({
            "step": i,
            "name": step["name"],
            "time": elapsed
        })
        
        # Update current image for next step
        current_image = result
        
        # Print running total
        total_so_far = sum(s["time"] for s in step_times)
        print(f"📊 Running total: {format_time(total_so_far)}")
    
    # Save final result
    pipeline_end = time.time()
    total_pipeline_time = pipeline_end - pipeline_start
    
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print_banner("PIPELINE COMPLETE! 🎉", "=")
    
    print("📊 TIMING BREAKDOWN:")
    print("-" * 50)
    
    for s in step_times:
        bar_len = int(s["time"] * 5)
        bar = "█" * min(bar_len, 40)
        print(f"   Step {s['step']}: {s['name']:<15} {s['time']:>6.2f}s  {bar}")
    
    print("-" * 50)
    
    inference_time = sum(s["time"] for s in step_times)
    avg_per_step = inference_time / len(step_times) if step_times else 0
    
    print(f"   {'INFERENCE TOTAL':<20} {inference_time:>6.2f}s")
    print(f"   {'AVERAGE PER STEP':<20} {avg_per_step:>6.2f}s")
    print(f"   {'PIPELINE TOTAL':<20} {total_pipeline_time:>6.2f}s")
    
    print(f"\n🏁 FINAL RESULT: {final_path}")
    print(f"📁 ALL OUTPUTS: {OUTPUT_DIR}")
    
    # Performance analysis
    print("\n📈 PERFORMANCE ANALYSIS:")
    if avg_per_step < 5:
        print("   ✅ Excellent! Running at optimal Lightning speed")
    elif avg_per_step < 8:
        print("   ✅ Good performance")
    elif avg_per_step < 12:
        print("   ⚠️  Slower than expected. Consider:")
        print("      - Reducing MAX_SIZE to 768")
        print("      - Ensuring no other GPU processes")
    else:
        print("   ⚠️  Much slower than expected. Check:")
        print("      - GPU utilization (nvidia-smi)")
        print("      - Memory pressure")
    
    # Save timing report
    report_path = os.path.join(OUTPUT_DIR, "timing_report.txt")
    with open(report_path, "w") as f:
        f.write("WEDDING DECOR PIPELINE V2 - TIMING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Qwen-Image-Edit-2509 + Lightning {NUM_STEPS}-step LoRA\n")
        f.write(f"Steps: {NUM_STEPS}\n")
        f.write(f"CFG: {TRUE_CFG_SCALE}\n")
        f.write(f"Max Size: {MAX_SIZE}px\n\n")
        f.write("STEP TIMES:\n")
        f.write("-" * 50 + "\n")
        for s in step_times:
            f.write(f"  Step {s['step']}: {s['name']:<15} {s['time']:.2f}s\n")
        f.write("-" * 50 + "\n")
        f.write(f"  INFERENCE TOTAL:    {inference_time:.2f}s\n")
        f.write(f"  AVERAGE PER STEP:   {avg_per_step:.2f}s\n")
        f.write(f"  PIPELINE TOTAL:     {total_pipeline_time:.2f}s\n")
    
    print(f"📄 Timing report: {report_path}")
    
    return current_image


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_banner("QWEN-IMAGE-EDIT-2509 LIGHTNING INFERENCE V2", "🚀")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    
    # Print PyTorch version for debugging
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🔧 CUDA: {torch.version.cuda}")
    
    # Load model
    pipeline = load_pipeline()
    
    # Run pipeline
    final_image = run_full_pipeline(pipeline)
    
    print_banner("ALL DONE! ✨", "=")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V3
===============================================================================
Key changes from V2:
1. Back to BIG → SMALL order (this actually worked better!)
2. Using 8-step LoRA for better quality
3. Simplified prompts - less "do not modify" (model ignores them anyway)
4. Focus on describing the COMPLETE desired state at each step

GPU: A100 80GB
Model: Qwen-Image-Edit-2509 + Lightning 8-step LoRA
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v3_bigtosmall_8step"
BASE_IMAGE = "base_image_table.png"

# Using 8-step for better quality
NUM_STEPS = 8
LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

# Lightning settings - REQUIRED
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0

SEED = 42
MAX_SIZE = 1024

# =============================================================================
# PIPELINE STEPS - BIG TO SMALL (what worked in V1!)
# =============================================================================
#
# Strategy: Describe the CUMULATIVE state at each step.
# Instead of "add X, don't change Y", describe what the whole scene should look like.
#

PIPELINE_STEPS = [
    # Step 1: Chairs (biggest structural change)
    {
        "name": "chairs",
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs around the round table with elegant gold chiavari chairs with white cushions, matching the reference image style. Keep the white tablecloth and round table exactly as shown. 8 chairs evenly spaced around the table.",
    },
    # Step 2: Tablecloth (major visual change)
    {
        "name": "tablecloth",
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "The round table has a luxurious deep red satin tablecloth with elegant draping, matching the reference. 8 gold chiavari chairs with white cushions surround the table. Rich fabric sheen and natural folds.",
    },
    # Step 3: Plates (establishing place settings)
    {
        "name": "plates",
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 elegant white dinner plates with gold rim matching the reference. One plate centered at each place setting around the red tablecloth. Gold chiavari chairs surround the table. Clean, evenly spaced arrangement.",
    },
    # Step 4: Napkins (on the plates)
    {
        "name": "napkins",
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Each of the 8 white plates now has a pink satin napkin folded decoratively on top, matching the reference style. Red satin tablecloth, gold chiavari chairs. Elegant fan-fold napkin presentation.",
    },
    # Step 5: Cutlery (beside plates)
    {
        "name": "cutlery",
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add gold cutlery sets beside each plate - fork on left, knife and spoon on right. Matching the luxurious gold reference style. 8 complete place settings with plates, pink napkins, and gold cutlery on red tablecloth.",
    },
    # Step 6: Glassware (completing place settings)
    {
        "name": "glassware",
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add crystal wine glasses at each place setting, positioned above and to the right of each plate. 8 elegant glasses matching the reference, showing realistic glass transparency. Complete place settings on red tablecloth.",
    },
    # Step 7: Centerpiece (final focal point)
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
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, step_name, step_num):
    print(f"\n🎨 Generating: {step_name}...")
    print(f"   Base size: {base_img.size} | Ref size: {ref_img.size}")
    print(f"   Prompt: {prompt[:100]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, extra items, missing items, wrong count, asymmetric",
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
# WARMUP
# =============================================================================

def warmup_pipeline(pipeline, sample_img):
    print("\n🔥 Running warmup pass...")
    warmup_start = time.time()
    
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
    
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(pipeline):
    print_banner("WEDDING TABLE STYLING PIPELINE V3", "🎨")
    print("Strategy: BIG changes first → SMALL details last")
    print("Quality: 8-step Lightning LoRA")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n📋 CONFIGURATION:")
    print(f"   • Model: Qwen-Image-Edit-2509 + Lightning 8-step LoRA")
    print(f"   • Inference Steps: {NUM_STEPS}")
    print(f"   • Max Size: {MAX_SIZE}px")
    print(f"   • Seed: {SEED}")
    print(f"   • Output: {OUTPUT_DIR}")
    print(f"   • Pipeline Order: chairs → tablecloth → plates → napkins → cutlery → glassware → centerpiece")
    
    # Load base image
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"\n❌ ERROR: Base image not found: {base_path}")
        return None
    
    print(f"\n📷 Loading base image: {base_path}")
    current_image = Image.open(base_path).convert("RGB")
    current_image = resize_image(current_image)
    
    original_path = os.path.join(OUTPUT_DIR, "step_0_original.png")
    current_image.save(original_path)
    print(f"💾 Saved: {original_path} ({current_image.size[0]}x{current_image.size[1]})")
    
    # Warmup
    warmup_pipeline(pipeline, current_image)
    
    step_times = []
    pipeline_start = time.time()
    
    # Run each step
    for i, step in enumerate(PIPELINE_STEPS, 1):
        print_banner(f"STEP {i}/{len(PIPELINE_STEPS)}: {step['name'].upper()}", "-")
        
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Reference image not found: {ref_path}")
            continue
        
        ref_img = Image.open(ref_path).convert("RGB")
        ref_img = resize_image(ref_img)
        print(f"📷 Reference: {ref_path}")
        
        result, elapsed = run_edit(
            pipeline=pipeline,
            base_img=current_image,
            ref_img=ref_img,
            prompt=step["prompt"],
            step_name=step["name"],
            step_num=i
        )
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"💾 Saved: {output_path}")
        
        step_times.append({
            "step": i,
            "name": step["name"],
            "time": elapsed
        })
        
        current_image = result
        
        total_so_far = sum(s["time"] for s in step_times)
        print(f"📊 Running total: {format_time(total_so_far)}")
    
    # Save final
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
    
    print(f"\n🏁 FINAL RESULT: {final_path}")
    print(f"📁 ALL OUTPUTS: {OUTPUT_DIR}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "timing_report.txt")
    with open(report_path, "w") as f:
        f.write("WEDDING DECOR PIPELINE V3 - TIMING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Qwen-Image-Edit-2509 + Lightning 8-step LoRA\n")
        f.write(f"Steps: {NUM_STEPS}\n")
        f.write(f"Strategy: Big to Small\n\n")
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
    print_banner("QWEN-IMAGE-EDIT-2509 PIPELINE V3", "🚀")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🔧 CUDA: {torch.version.cuda}")
    
    pipeline = load_pipeline()
    final_image = run_full_pipeline(pipeline)
    
    print_banner("ALL DONE! ✨", "=")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
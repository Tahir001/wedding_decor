"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V12-B (2511 - SAME STEPS AS V11)
===============================================================================
This version uses IDENTICAL step counts to V11 for a true apples-to-apples
quality comparison. Only the MODEL changes (2509 → 2511).

Use this script to directly compare:
- Is 2511 more consistent with the same number of steps?
- Does 2511 preserve previous edits better?
- Is the final quality improved?

Then run V12 (optimized steps) to see if you can get similar quality faster.
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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v12b_2511_same_steps"
BASE_IMAGE = "base_image_table.png"

FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# === MODEL: 2511 ===
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"

# === LORA: 4-step Lightning ===
# Note: Using 4-step distilled LoRA but with higher step counts.
# This may not be optimal but allows direct comparison with V11.
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# PIPELINE STEPS - EXACT SAME AS V11 (for comparison)
# =============================================================================

PIPELINE_STEPS = [
    {
        "name": "chairs",
        "steps": 6,  # SAME AS V11
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs with elegant gold chiavari chairs with white cushions matching the reference. 8 chairs evenly spaced around the round table with white tablecloth.",
    },
    {
        "name": "tablecloth",
        "steps": 6,  # SAME AS V11
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "The round table now has a luxurious deep red satin tablecloth with elegant draping matching the reference. 8 gold chiavari chairs with white cushions surround the table.",
    },
    {
        "name": "plates",
        "steps": 6,  # SAME AS V11
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 white dinner plates with gold rim matching the reference. One plate at each place setting on the red tablecloth. Gold chiavari chairs around table.",
    },
    {
        "name": "napkins",
        "steps": 8,  # SAME AS V11
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Add pink satin napkins folded in elegant fan shapes on each plate, matching the reference. 8 place settings with plates on red tablecloth. Gold chiavari chairs.",
    },
    {
        "name": "cutlery",
        "steps": 8,  # SAME AS V11
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add gold cutlery beside each plate - fork on left, knife and spoon on right, matching the reference. Complete place settings with plates and pink napkins on red tablecloth. Gold chiavari chairs.",
    },
    {
        "name": "glassware",
        "steps": 8,  # SAME AS V11
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add crystal wine glasses at each place setting above the knife, matching the reference. Realistic glass transparency. Complete settings with plates, napkins, cutlery on red tablecloth. Gold chiavari chairs.",
    },
    {
        "name": "centerpiece",
        "steps": 8,  # SAME AS V11
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add a stunning pink rose centerpiece on gold stand to the center of the table, matching the reference. Spherical arrangement of fresh roses. All 8 place settings surround it with plates, napkins, cutlery, glasses on red tablecloth. Gold chiavari chairs.",
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
    print_banner("LOADING MODEL: Qwen-Image-Edit-2511 (Same Steps as V11)")
    
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
    
    print(f"\n🎨 Step {step_num}: {name} ({steps} inference steps)")
    print(f"   Prompt: {prompt[:70]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, missing items, wrong count",
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
# MAIN
# =============================================================================

def run_pipeline(pipeline):
    print_banner("V12-B: 2511 WITH SAME STEPS AS V11 (Direct Comparison)", "🔬")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_steps = sum(s['steps'] for s in PIPELINE_STEPS)
    
    print(f"📋 Pipeline: {len(PIPELINE_STEPS)} layers")
    print(f"📊 Total inference steps: {total_steps} (SAME as V11)")
    print(f"📐 Fixed size: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"📷 Reference size: {REF_SIZE}x{REF_SIZE}")
    print(f"🔄 Model: {MODEL_NAME}")
    print(f"⚡ LoRA: 4-step Lightning")
    print(f"\n🎯 PURPOSE: Direct quality comparison with V11 (2509)")
    
    # Load base
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Missing: {base_path}")
        return
    
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    print(f"\n💾 Saved original")
    
    warmup(pipeline)
    
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path):
            print(f"⚠️  Missing ref: {step['ref_image']}")
            continue
        
        ref_img = resize_reference(Image.open(ref_path).convert("RGB"))
        
        result, elapsed = run_edit(pipeline, current_image, ref_img, step, i)
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"   💾 Saved: {output_path}")
        
        step_times.append({"name": step["name"], "steps": step["steps"], "time": elapsed})
        current_image = result
    
    total_time = time.time() - pipeline_start
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # Summary
    print_banner("COMPLETE - COMPARE WITH V11!", "✅")
    print("📊 TIMING:")
    print("-" * 50)
    for i, s in enumerate(step_times, 1):
        print(f"   {i}. {s['name']:<12} {s['steps']} steps  →  {s['time']:.2f}s")
    print("-" * 50)
    
    inference_total = sum(s['time'] for s in step_times)
    print(f"   Inference: {inference_total:.2f}s")
    print(f"   Total:     {total_time:.2f}s")
    print(f"\n🏁 Result: {final_path}")
    print(f"\n📁 Compare with V11:")
    print(f"   V11: /workspace/wedding_decor/images/output/v11_clean_fast/")
    print(f"   V12-B: {OUTPUT_DIR}/")
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("V12-B: 2511 WITH SAME STEPS AS V11 (Direct Comparison)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"LoRA: {LORA_WEIGHTS}\n")
        f.write(f"Total steps: {total_steps} (SAME as V11)\n\n")
        f.write("TIMING:\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['steps']} steps, {s['time']:.2f}s\n")
        f.write(f"\nTotal: {inference_total:.2f}s\n\n")
        f.write("WHAT TO LOOK FOR:\n")
        f.write("- Are chairs preserved in later steps? (image drift)\n")
        f.write("- Is the tablecloth consistent throughout?\n")
        f.write("- Are plates/napkins/cutlery staying in place?\n")
        f.write("- Overall quality of final result?\n")


if __name__ == "__main__":
    print(f"🚀 Starting V12-B (2511 - Same Steps) at {datetime.now().strftime('%H:%M:%S')}")
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    
    pipeline = load_pipeline()
    run_pipeline(pipeline)
    
    print(f"\n✨ Done at {datetime.now().strftime('%H:%M:%S')}")
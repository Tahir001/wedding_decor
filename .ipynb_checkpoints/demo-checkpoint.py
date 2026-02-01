"""
===============================================================================
WEDDING DECOR VISUALIZATION - INTERACTIVE MODE
===============================================================================
Based on V12C pipeline, modified for interactive use:
- Model stays loaded in VRAM
- Continuously accepts new base images, reference images, and prompts
- Quick iteration for testing different combinations
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

OUTPUT_DIR = "/workspace/wedding_decor/images/output/interactive"

# === IMAGE DIMENSIONS ===
FIXED_WIDTH = 768
FIXED_HEIGHT = 768
REF_SIZE = 384

# === MODEL CONFIG ===
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# === INFERENCE DEFAULTS ===
DEFAULT_STEPS = 4
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0

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


def load_image(path):
    """Load and validate an image file."""
    if not path:
        return None
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    try:
        img = Image.open(path).convert("RGB")
        print(f"✅ Loaded: {path} ({img.size[0]}x{img.size[1]})")
        return img
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL: Qwen-Image-Edit-2511")
    
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
    print("✅ Model warmed up and ready")


# =============================================================================
# EDIT FUNCTION
# =============================================================================

def run_edit(pipeline, base_img, ref_img, prompt, steps=DEFAULT_STEPS, seed=None):
    """Run a single edit operation."""
    if seed is None:
        seed = int(time.time()) % 10000
    
    print(f"\n🎨 Running edit ({steps} steps, seed={seed})")
    print(f"   Prompt: {prompt}")
    
    # Resize images
    base_resized = resize_to_fixed(base_img)
    ref_resized = resize_reference(ref_img)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_resized, ref_resized],
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    print(f"   ⏱️  Completed in {elapsed:.2f}s")
    return result, elapsed


# =============================================================================
# INTERACTIVE LOOP
# =============================================================================

def interactive_mode(pipeline):
    """Main interactive loop."""
    print_banner("INTERACTIVE MODE", "🎨")
    print("Commands:")
    print("  - Enter paths and prompts when asked")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'same' to reuse the last base image")
    print("  - Type 'result' to use the last output as new base")
    print("  - Leave steps blank for default (4)")
    print("  - Leave seed blank for random")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    last_base_img = None
    last_result = None
    edit_count = 0
    
    while True:
        print_banner(f"EDIT #{edit_count + 1}", "-")
        
        # Get base image
        base_input = input("📷 Base image path (or 'same'/'result'/'quit'): ").strip()
        
        if base_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting interactive mode.")
            break
        
        if base_input.lower() == 'same' and last_base_img is not None:
            base_img = last_base_img
            print("   Using previous base image")
        elif base_input.lower() == 'result' and last_result is not None:
            base_img = last_result
            print("   Using last result as base image")
        else:
            base_img = load_image(base_input)
            if base_img is None:
                continue
        
        # Get reference image
        ref_input = input("🖼️  Reference image path: ").strip()
        ref_img = load_image(ref_input)
        if ref_img is None:
            continue
        
        # Get prompt
        prompt = input("✏️  Prompt: ").strip()
        if not prompt:
            print("❌ Prompt cannot be empty")
            continue
        
        # Get optional parameters
        steps_input = input(f"🔢 Steps (default={DEFAULT_STEPS}): ").strip()
        steps = int(steps_input) if steps_input.isdigit() else DEFAULT_STEPS
        
        seed_input = input("🎲 Seed (blank=random): ").strip()
        seed = int(seed_input) if seed_input.isdigit() else None
        
        # Run the edit
        try:
            result, elapsed = run_edit(pipeline, base_img, ref_img, prompt, steps, seed)
            
            # Save result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"edit_{timestamp}.png")
            result.save(output_path)
            print(f"\n💾 Saved: {output_path}")
            
            # Update state
            last_base_img = base_img
            last_result = result
            edit_count += 1
            
            # Show VRAM usage
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"📊 VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
            
        except Exception as e:
            print(f"❌ Error during edit: {e}")
            import traceback
            traceback.print_exc()
        
        print()


# =============================================================================
# BATCH MODE (optional - run predefined steps)
# =============================================================================

def batch_mode(pipeline, steps_config, input_dir, base_image):
    """Run a batch of predefined steps (original pipeline behavior)."""
    print_banner("BATCH MODE", "📋")
    
    batch_output = os.path.join(OUTPUT_DIR, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(batch_output, exist_ok=True)
    
    base_path = os.path.join(input_dir, base_image)
    current_image = load_image(base_path)
    if current_image is None:
        return
    
    current_image = resize_to_fixed(current_image)
    current_image.save(os.path.join(batch_output, "step_0_original.png"))
    
    step_times = []
    
    for i, step in enumerate(steps_config, 1):
        ref_path = os.path.join(input_dir, step["ref_image"])
        ref_img = load_image(ref_path)
        if ref_img is None:
            print(f"⚠️  Skipping step {i}: {step['name']}")
            continue
        
        result, elapsed = run_edit(
            pipeline, current_image, ref_img, 
            step["prompt"], step.get("steps", DEFAULT_STEPS)
        )
        
        output_path = os.path.join(batch_output, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        
        step_times.append({"name": step["name"], "time": elapsed})
        current_image = result
    
    # Save final
    final_path = os.path.join(batch_output, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    print_banner("BATCH COMPLETE", "✅")
    for s in step_times:
        print(f"  {s['name']}: {s['time']:.2f}s")
    print(f"\n🏁 Final: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"🚀 Wedding Decor Interactive @ {datetime.now().strftime('%H:%M:%S')}")
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model once
    pipeline = load_pipeline()
    warmup(pipeline)
    
    # Enter interactive loop
    try:
        interactive_mode(pipeline)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Exiting...")
    
    print(f"\n✨ Session ended @ {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
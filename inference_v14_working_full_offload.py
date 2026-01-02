"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V14 (RTX 4090 INT4 LIGHTNING FAST)
===============================================================================

Optimized for: NVIDIA RTX 4090 (24GB VRAM)
Quantization: INT4 with SVDQuant (Nunchaku)
Inference: Lightning 8-step

KEY OPTIMIZATION: Nunchaku's per-layer offloading with pinned memory
- Keeps ~28 transformer blocks on GPU (tune NUM_BLOCKS_ON_GPU if OOM)
- Uses pinned memory for fast CPU<->GPU transfers
- WAY faster than enable_model_cpu_offload() which shuffles 14GB text encoder

Expected performance:
- Single edit: ~8-12 seconds (vs 31s with full offload)
- 7-layer pipeline: ~60-90 seconds total

If you get OOM, reduce NUM_BLOCKS_ON_GPU in load_pipeline() (try 20, 15, 10)
If you want faster, increase NUM_BLOCKS_ON_GPU (try 30, 32)

===============================================================================
"""

import os
import gc
import time
import math
import torch
from PIL import Image
from datetime import datetime

# Nunchaku imports
from nunchaku import NunchakuQwenImageTransformer2DModel
try:
    from nunchaku.utils import get_precision, get_gpu_memory
    NUNCHAKU_UTILS_AVAILABLE = True
except ImportError:
    NUNCHAKU_UTILS_AVAILABLE = False

from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
INPUT_DIR = "/workspace/wedding_decor/images"
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v14_fast"
BASE_IMAGE = "base_image_table.png"
MODEL_DIR = "/workspace/wedding_decor/models"

# Image dimensions
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# Model configuration
NUNCHAKU_RANK = 128          # 128 = best quality
USE_LIGHTNING = True
LIGHTNING_STEPS = 8

# Inference configuration
# NOTE: For Lightning models, true_cfg_scale should be 1.0 (they're guidance-distilled)
# If using non-Lightning (50 steps), you can increase to 3.0-4.0 for better prompt following
TRUE_CFG_SCALE = 1.0         # Keep at 1.0 for Lightning
GUIDANCE_SCALE = 1.0         # Keep at 1.0 for Lightning
SEED = 42

# Nunchaku offloading - tune this for your GPU
# Higher = faster but more VRAM. Lower if you get OOM.
# RTX 4090 (24GB): try 25-30
# RTX 3090 (24GB): try 20-25  
# RTX 4080 (16GB): try 12-15
NUM_BLOCKS_ON_GPU = 28

# =============================================================================
# PIPELINE STEPS - BULLETPROOF PROMPTS FOR PERFECT RESULTS
# =============================================================================
# 
# PROMPT ENGINEERING PRINCIPLES APPLIED:
# 1. Be EXPLICIT about what to add AND what NOT to add
# 2. Use strong negative prompts to prevent semantic leakage
# 3. Reference the current state of the image to ground the model
# 4. Use specific, concrete language (not vague terms)
# 5. Mention quantities explicitly (8 chairs, 8 plates, etc.)
# 6. Describe spatial relationships clearly
# 7. Each step builds on previous - reference what already exists
#
# =============================================================================

# Global negative prompt for all steps - prevents common artifacts
GLOBAL_NEGATIVE = "blurry, distorted, low quality, deformed, artifacts, bad anatomy, disfigured, poorly drawn, mutation, mutated, ugly, disgusting, watermark, text, logo, signature"

PIPELINE_STEPS = [
    # =========================================================================
    # STEP 1: CHAIRS
    # =========================================================================
    # First step - only chairs, nothing else on the table yet
    {
        "name": "chairs",
        "steps": LIGHTNING_STEPS,
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": """Replace all existing chairs around the round table with elegant clear chiavari chairs exactly matching the reference image. 
Place exactly 8 chiavari chairs evenly spaced in a circle around the table. 
The table has a plain white tablecloth. 
Keep the table surface completely empty - no plates, no cutlery, no decorations, no centerpiece.
Only change the chairs, preserve everything else exactly as it is.""",
        "negative_prompt": "plates, cutlery, fork, knife, spoon, napkin, glasses, wine glass, centerpiece, flowers, candles, place settings, food, tablecloth pattern, colored tablecloth"
    },
    
    # =========================================================================
    # STEP 2: TABLECLOTH
    # =========================================================================
    # Change tablecloth color, chairs are now present
    {
        "name": "tablecloth",
        "steps": LIGHTNING_STEPS,
        "ref_image": "tablecloths/satin_red.png",
        "prompt": """Change the tablecloth on the round table to a luxurious deep red satin tablecloth exactly matching the reference image.
The tablecloth should drape elegantly over the round table with smooth folds.
Keep the 8 clear chiavari chairs exactly as they are around the table.
The table surface must remain completely empty - no plates, no cutlery, no decorations.
Only change the tablecloth color and texture, nothing else.""",
        "negative_prompt": "plates, cutlery, fork, knife, spoon, napkin, glasses, wine glass, centerpiece, flowers, candles, place settings, food, white tablecloth"
    },
    
    # =========================================================================
    # STEP 3: PLATES
    # =========================================================================
    # Add plates ONLY - this is where cutlery was sneaking in before
    {
        "name": "plates",
        "steps": LIGHTNING_STEPS,
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": """Add exactly 8 white dinner plates with gold rim to the table, matching the reference image exactly.
Place one plate centered at each seating position, evenly spaced around the round table.
The plates sit directly on the red satin tablecloth.
IMPORTANT: Add ONLY the plates. Do NOT add any cutlery, forks, knives, spoons, napkins, or glasses.
The plates should be empty with nothing on them.
Keep the 8 clear chiavari chairs and red tablecloth exactly as they are.""",
        "negative_prompt": "cutlery, fork, knife, spoon, silverware, utensils, napkin, napkins, glasses, wine glass, glassware, centerpiece, flowers, food, folded napkin, place setting complete"
    },
    
    # =========================================================================
    # STEP 4: NAPKINS
    # =========================================================================
    # Add napkins on the plates
    {
        "name": "napkins",
        "steps": LIGHTNING_STEPS,
        "ref_image": "napkins/satin_pink.png",
        "prompt": """Add exactly 8 pink satin napkins to the table, one on each white plate with gold rim, matching the reference image.
Fold each napkin elegantly in a fan or decorative fold, placed in the center of each plate.
The napkins should be the same pink satin material as the reference.
IMPORTANT: Add ONLY the napkins on the plates. Do NOT add any cutlery, forks, knives, spoons, or glasses yet.
Keep the 8 clear chiavari chairs, red satin tablecloth, and 8 white plates exactly as they are.""",
        "negative_prompt": "cutlery, fork, knife, spoon, silverware, utensils, glasses, wine glass, glassware, centerpiece, flowers, food, extra plates, charger plate"
    },
    
    # =========================================================================
    # STEP 5: CUTLERY
    # =========================================================================
    # NOW we add cutlery - after plates and napkins
    {
        "name": "cutlery",
        "steps": LIGHTNING_STEPS,
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": """Add gold cutlery to each place setting, matching the reference image exactly.
At each of the 8 place settings: place a gold fork on the left side of the plate, and a gold knife and gold spoon on the right side of the plate.
The cutlery should be elegant gold/brass colored matching the reference.
The cutlery rests directly on the red tablecloth beside each white plate with pink napkin.
IMPORTANT: Add ONLY the cutlery. Do NOT add glasses or centerpiece yet.
Keep the 8 clear chiavari chairs, red tablecloth, 8 plates with pink napkins exactly as they are.""",
        "negative_prompt": "glasses, wine glass, water glass, glassware, goblet, centerpiece, flowers, vase, candles, extra plates, silver cutlery, chrome cutlery"
    },
    
    # =========================================================================
    # STEP 6: GLASSWARE
    # =========================================================================
    # Add wine glasses
    {
        "name": "glassware",
        "steps": LIGHTNING_STEPS,
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": """Add exactly 8 elegant crystal wine glasses to the table, one at each place setting, matching the reference image.
Position each wine glass above and slightly to the right of the knife at each place setting.
The glasses should be clear crystal with realistic transparency and light refraction.
IMPORTANT: Add ONLY the wine glasses. Do NOT add a centerpiece, flowers, or candles yet.
Keep the 8 clear chiavari chairs, red tablecloth, 8 plates with pink napkins, and gold cutlery exactly as they are.""",
        "negative_prompt": "centerpiece, flowers, vase, candles, floral arrangement, extra glasses, champagne flute, water glass, colored glass"
    },
    
    # =========================================================================
    # STEP 7: CENTERPIECE
    # =========================================================================
    # Final step - add the centerpiece
    {
        "name": "centerpiece",
        "steps": LIGHTNING_STEPS,
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": """Add a stunning floral centerpiece to the exact center of the round table, matching the reference image.
The centerpiece features pink roses arranged in a spherical shape on an elegant gold stand.
Position it precisely in the center of the table, surrounded by the 8 place settings.
The centerpiece should be the focal point but not block guests' view of each other.
Keep ALL existing elements exactly as they are: 8 clear chiavari chairs, red satin tablecloth, 8 white plates with gold rim, 8 pink napkins, gold cutlery at each setting, and 8 crystal wine glasses.""",
        "negative_prompt": "candles, extra flowers scattered, petals on table, multiple centerpieces, tall centerpiece blocking view, wilted flowers, artificial looking flowers"
    },
]


def get_negative_prompt(step_config):
    """Combine step-specific negative prompt with global negative prompt"""
    step_negative = step_config.get("negative_prompt", "")
    if step_negative:
        return f"{GLOBAL_NEGATIVE}, {step_negative}"
    return GLOBAL_NEGATIVE

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def get_precision_manual():
    compute_cap = torch.cuda.get_device_capability()
    return "fp4" if compute_cap[0] >= 10 else "int4"


def get_gpu_memory_manual():
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def resize_to_fixed(img, width=FIXED_WIDTH, height=FIXED_HEIGHT):
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img, size=REF_SIZE):
    return img.resize((size, size), Image.LANCZOS)


def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0, 0


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# MODEL LOADING - OPTIMIZED FOR SPEED ON RTX 4090
# =============================================================================

def load_pipeline():
    """
    Load pipeline with SMART memory management for RTX 4090 (24GB):
    
    Strategy: Use Nunchaku's built-in per-block offloading with MANY blocks on GPU
    - With 24GB VRAM, we can keep most transformer blocks on GPU
    - This is MUCH faster than enable_model_cpu_offload()
    
    Memory breakdown:
    - INT4 Transformer: ~5-6GB 
    - VAE: ~0.3GB
    - Text Encoder (Qwen2-VL 7B): ~14GB in bf16
    - Total if all on GPU: ~20GB (too tight)
    
    Solution: Use Nunchaku's smart per-layer offloading with pinned memory
    """
    
    print_banner("LOADING NUNCHAKU INT4 LIGHTNING MODEL (FAST MODE)")
    
    clear_memory()
    
    if NUNCHAKU_UTILS_AVAILABLE:
        precision = get_precision()
        gpu_memory = get_gpu_memory()
    else:
        precision = get_precision_manual()
        gpu_memory = get_gpu_memory_manual()
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
    print(f"🔧 Precision: {precision.upper()}")
    print(f"📊 Rank: {NUNCHAKU_RANK}")
    print(f"⚡ Lightning: {USE_LIGHTNING} ({LIGHTNING_STEPS} steps)")
    print()
    
    # Lightning scheduler
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
    
    # Find model
    nunchaku_dir = f"{MODEL_DIR}/nunchaku-qwen-image-edit-2509"
    model_filename = f"svdq-{precision}_r{NUNCHAKU_RANK}-qwen-image-edit-2509-lightning-{LIGHTNING_STEPS}steps-251115.safetensors"
    local_path = f"{nunchaku_dir}/lightning-251115/{model_filename}"
    
    if os.path.exists(local_path):
        model_path = local_path
        print(f"📦 Loading local model: {model_filename}")
    else:
        model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/{model_filename}"
        print(f"📦 Loading from HuggingFace: {model_path}")
    
    load_start = time.time()
    
    # Load quantized transformer
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
    
    # Load pipeline with bfloat16
    qwen_base = f"{MODEL_DIR}/Qwen-Image-Edit-2509"
    if not os.path.exists(qwen_base):
        qwen_base = "Qwen/Qwen-Image-Edit-2509"
    print(f"   Using base: {qwen_base}")
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        qwen_base,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    
    # ==========================================================================
    # CRITICAL OPTIMIZATION: Smart per-layer offloading
    # ==========================================================================
    # 
    # The Qwen transformer has ~40 blocks. With 24GB VRAM:
    # - We can keep ~20-30 blocks on GPU for speed
    # - Use pinned memory for fast CPU<->GPU transfer
    # - This is WAY faster than enable_model_cpu_offload() which moves
    #   the ENTIRE text encoder (14GB) back and forth
    #
    # num_blocks_on_gpu: Higher = faster but more VRAM
    # - 1 block = ~3-4GB total (minimum)
    # - 10 blocks = ~8GB total
    # - 20 blocks = ~12GB total  
    # - 30 blocks = ~16GB total (good for 24GB GPU)
    # ==========================================================================
    
    print("\n📍 Configuring smart per-layer offloading...")
    
    # Use Nunchaku's optimized offloading
    # NUM_BLOCKS_ON_GPU is set in config section at top of file
    transformer.set_offload(
        True, 
        use_pin_memory=True,  # Pinned memory = faster transfers
        num_blocks_on_gpu=NUM_BLOCKS_ON_GPU
    )
    
    # IMPORTANT: Exclude transformer from Diffusers' offloading
    # (Nunchaku handles it better)
    pipeline._exclude_from_cpu_offload.append("transformer")
    
    # Use sequential offload for text encoder + VAE
    pipeline.enable_sequential_cpu_offload()
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.1f}s")
    print(f"🚀 Mode: Nunchaku per-layer offload ({NUM_BLOCKS_ON_GPU} blocks on GPU)")
    
    allocated, reserved = get_vram_usage()
    print(f"💾 VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


# =============================================================================
# INFERENCE - Simple with Nunchaku handling offloading
# =============================================================================

def run_edit_fast(pipeline, base_img, ref_img, step_config, step_num):
    """Run edit - Nunchaku handles all the offloading internally"""
    
    steps = step_config['steps']
    name = step_config['name']
    prompt = step_config['prompt'].strip()  # Clean up multi-line prompts
    negative = get_negative_prompt(step_config)
    
    print(f"\n🎨 Step {step_num}: {name} ({steps} steps)")
    print(f"   Prompt: {prompt[:80]}...")
    print(f"   Negative: {negative[:60]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=negative,
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
    
    allocated, _ = get_vram_usage()
    print(f"   ⏱️  Done in {elapsed:.2f}s | VRAM: {allocated:.2f}GB")
    
    return result, elapsed


def warmup_fast(pipeline):
    """Warmup to compile CUDA kernels"""
    print("\n🔥 Warmup run (compiling CUDA kernels)...")
    
    dummy_base = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), color='gray')
    
    with torch.inference_mode():
        _ = pipeline(
            image=[dummy_base, dummy_ref],
            prompt="warmup",
            num_inference_steps=4,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
        )
    
    clear_memory()
    print("✅ Warmup complete")


def run_pipeline(pipeline):
    """Run the full pipeline"""
    
    print_banner("🎨 WEDDING PIPELINE V14: INT4 LIGHTNING FAST")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_steps = sum(s['steps'] for s in PIPELINE_STEPS)
    print(f"📋 Pipeline: {len(PIPELINE_STEPS)} layers, {total_steps} total steps")
    print(f"📐 Output: {FIXED_WIDTH}x{FIXED_HEIGHT}, Ref: {REF_SIZE}x{REF_SIZE}")
    print(f"🎯 Steps per layer: {LIGHTNING_STEPS}")
    print()
    
    # Load base image
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Base image not found: {base_path}")
        return
    
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    print(f"📷 Loaded: {BASE_IMAGE}")
    
    warmup_fast(pipeline)
    
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        
        if not os.path.exists(ref_path):
            print(f"⚠️  Missing: {step['ref_image']} - skipping")
            continue
        
        ref_img = resize_reference(Image.open(ref_path).convert("RGB"))
        result, elapsed = run_edit_fast(pipeline, current_image, ref_img, step, i)
        
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"   💾 Saved: step_{i}_{step['name']}.png")
        
        step_times.append({"name": step["name"], "steps": step["steps"], "time": elapsed})
        current_image = result
    
    total_time = time.time() - pipeline_start
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # Summary
    print_banner("✅ PIPELINE COMPLETE")
    
    print("📊 TIMING:")
    print("-" * 50)
    for i, s in enumerate(step_times, 1):
        print(f"   {i}. {s['name']:<12} {s['steps']} steps → {s['time']:.2f}s")
    print("-" * 50)
    
    inference_total = sum(s['time'] for s in step_times)
    print(f"   Inference: {inference_total:.2f}s")
    print(f"   Total:     {total_time:.2f}s")
    print(f"   Avg/layer: {inference_total/len(step_times):.2f}s")
    
    # Cost
    cost = (total_time / 3600) * 0.34
    print(f"\n💰 Cost (RTX 4090 @ $0.34/hr): ${cost:.4f}")
    print(f"🏁 Final: {final_path}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"WEDDING DECOR V14 - FAST MODE\n")
        f.write(f"{'='*40}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n\n")
        f.write("TIMING:\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['time']:.2f}s\n")
        f.write(f"\nTotal inference: {inference_total:.2f}s\n")
        f.write(f"Total pipeline: {total_time:.2f}s\n")
        f.write(f"Cost: ${cost:.4f}\n")


def nuke_gpu_memory():
    """Forcefully clean the GPU memory before loading."""
    print("\n🧹 Cleaning Memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)
        print(f"   Status: {(total_gb-free_gb):.2f}GB Used / {total_gb:.2f}GB Total")

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0, 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_banner("🚀 WEDDING DECOR V14 - INT4 LIGHTNING FAST")
    
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(nuke_gpu_memory())
    print(get_vram_usage())
    
    
    pipeline = load_pipeline()
    run_pipeline(pipeline)
    
    print(f"\n✨ Done at {datetime.now().strftime('%H:%M:%S')}")
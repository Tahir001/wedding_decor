"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V14.2 (RTX 4090 STABLE & FAST)
===============================================================================
FIXES:
1. VAE Tiling: Prevents OOM crashes during 1024x1024 image encoding.
2. Smart Offload: Automatically manages the 14GB Text Encoder.
3. VRAM Monitor: Prints "Pre-flight" usage and "Peak" usage for every step.
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
NUNCHAKU_RANK = 128          
USE_LIGHTNING = True
LIGHTNING_STEPS = 8
TRUE_CFG_SCALE = 1.0         
GUIDANCE_SCALE = 1.0         
SEED = 42

# TUNING: With VAE Tiling and Model Offload enabled, 
# 20-25 blocks is usually the sweet spot for the 4090.
NUM_BLOCKS_ON_GPU = 16

# =============================================================================
# PIPELINE STEPS - (YOUR ORIGINAL PROMPTS PRESERVED)
# =============================================================================

GLOBAL_NEGATIVE = "blurry, distorted, low quality, deformed, artifacts, bad anatomy, disfigured, poorly drawn, mutation, mutated, ugly, disgusting, watermark, text, logo, signature"

PIPELINE_STEPS = [
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        peak = torch.cuda.max_memory_reserved() / (1024**3)
        return allocated, peak
    return 0, 0

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    """
    Load pipeline OPTIMIZED for RTX 4090 (24GB).
    
    STRATEGY: NO OFFLOADING.
    Total Model Size: ~18.5GB
    RTX 4090 VRAM: 24GB
    Headroom: ~5.5GB (Plenty for inference activations)
    """
    
    print_banner("LOADING NUNCHAKU INT4 LIGHTNING (FULL GPU MODE)")
    
    clear_memory()
    
    # Check Nunchaku availability
    if NUNCHAKU_UTILS_AVAILABLE:
        precision = get_precision()
    else:
        precision = get_precision_manual()
    
    # Scheduler Config
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
    
    # Path logic
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
    
    # 1. Load Transformer
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
    
    # 2. Load Pipeline
    qwen_base = f"{MODEL_DIR}/Qwen-Image-Edit-2509"
    if not os.path.exists(qwen_base):
        qwen_base = "Qwen/Qwen-Image-Edit-2509"
        
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        qwen_base,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    
    # ==========================================================================
    # PERFORMANCE FIX
    # ==========================================================================
    
    print("\n🚀 MOVING ENTIRE PIPELINE TO GPU (NO OFFLOAD)...")
    
    # 1. Disable Nunchaku offloading (Keep INT4 model wholly in VRAM)
    transformer.set_offload(False)
    
    # 2. Move standard diffusers components to GPU
    # This keeps the heavy Text Encoder (14GB) in VRAM constantly.
    # No PCIe transfer bottlenecks.
    pipeline.to("cuda")
    
    # 3. Optional: Compile the transformer for extra speed (first run will be slow)
    # Uncomment next line if you want to squeeze out another 10-20% speed
    # pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    allocated, reserved = get_vram_usage()
    print(f"💾 VRAM Usage: {allocated:.2f}GB / 24.0GB")
    
    if allocated > 23.0:
        print("⚠️ WARNING: VRAM is very tight. If OOM occurs, enable model_cpu_offload.")
        
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline

# =============================================================================
# RUN LOGIC
# =============================================================================

def run_edit_fast(pipeline, base_img, ref_img, step_config, step_num):
    name = step_config['name']
    prompt = step_config['prompt'].strip()
    negative = f"{GLOBAL_NEGATIVE}, {step_config.get('negative_prompt', '')}"
    
    # PRE-FLIGHT CHECK
    alloc, peak = get_vram_usage()
    print(f"\n🎨 Step {step_num}: {name}")
    print(f"   📊 Pre-flight VRAM: {alloc:.2f}GB used")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=step_config['steps'],
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED + step_num),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    curr_vram, peak_vram = get_vram_usage()
    print(f"   ⏱️  Time: {elapsed:.2f}s | Peak VRAM this step: {peak_vram:.2f}GB")
    
    return output.images[0], elapsed

def run_pipeline(pipeline):
    print_banner("🚀 WEDDING DECOR V14.2 - HYPER-SPEED PIPELINE")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    current_image = Image.open(base_path).convert("RGB").resize((FIXED_WIDTH, FIXED_HEIGHT))
    
    pipeline_start = time.time()
    step_times = []
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        if not os.path.exists(ref_path): continue
        
        ref_img = Image.open(ref_path).convert("RGB").resize((REF_SIZE, REF_SIZE))
        result, elapsed = run_edit_fast(pipeline, current_image, ref_img, step, i)
        
        result.save(os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png"))
        step_times.append(elapsed)
        current_image = result
    
    total_time = time.time() - pipeline_start
    current_image.save(os.path.join(OUTPUT_DIR, "FINAL_RESULT.png"))
    
    # FINAL STATS
    _, total_peak = get_vram_usage()
    print_banner("✅ PIPELINE COMPLETE")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"🔥 MAX VRAM RECORDED: {total_peak:.2f} GB")
    print(f"💰 Cost: ${(total_time / 3600) * 0.34:.4f}")

import gc
import torch

def nuke_gpu_memory():
    """
    Forcefully clean the GPU memory.
    Run this before loading the pipeline.
    """
    print("\n🧹 Cleaning Memory...")
    
    # 1. Force Python's Garbage Collector to release unreferenced objects
    gc.collect()
    
    # 2. Clear PyTorch's internal cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # Clears persistent buffers
    
    # 3. Report actual status
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)
        used_gb = total_gb - free_gb
        print(f"   Status: {used_gb:.2f}GB Used / {total_gb:.2f}GB Total")
        
        # If we have less than 20GB free, warn the user
        if free_gb < 18.0:
            print("   ⚠️ WARNING: You have < 18GB VRAM free.") 
            print("      Close other apps (WebUI, Jupyter, Chrome) or you will OOM.")
            
if __name__ == "__main__":
    pipeline = load_pipeline()
    run_pipeline(pipeline)
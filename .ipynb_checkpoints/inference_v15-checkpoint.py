"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V15 (MAXIMUM SPEED)
===============================================================================

Optimized for: NVIDIA RTX 4090 (24GB VRAM)
Strategy: KEEP EVERYTHING IN VRAM - No offloading!

KEY CHANGES FROM V14:
1. NO enable_sequential_cpu_offload() - this was the main bottleneck
2. NO Nunchaku block offloading - keep quantized transformer fully in VRAM
3. Preload ALL reference images at startup
4. Optional torch.compile() for even more speed
5. Reuse generator across steps

Memory breakdown (all on GPU):
- INT4 Transformer: ~4-5GB (quantized with Nunchaku)
- Text Encoder (Qwen2-VL 7B bf16): ~14GB  
- VAE: ~0.3GB
- Working memory: ~3-4GB
- Total: ~22GB (fits in 24GB RTX 4090!)

Expected performance:
- Single edit: ~3-5 seconds (vs 10s with block offload, 30s with sequential)
- 7-layer pipeline: ~25-40 seconds total

If you get OOM:
1. First try: Set USE_MODEL_OFFLOAD = True (uses enable_model_cpu_offload)
2. Still OOM: Reduce image size to 768x768
3. Last resort: Go back to v14 with block offloading

===============================================================================
"""

import os
import gc
import time
import math
import torch
from PIL import Image
from datetime import datetime
from typing import Dict, Optional, Tuple

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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v15_fastest"
BASE_IMAGE = "base_image_table.png"
MODEL_DIR = "/workspace/wedding_decor/models"

# Image dimensions
FIXED_WIDTH = 1024
FIXED_HEIGHT = 768
REF_SIZE = 384

# Model configuration
NUNCHAKU_RANK = 128
USE_LIGHTNING = True
LIGHTNING_STEPS = 8

# Inference configuration
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
SEED = 42

# =============================================================================
# SPEED CONFIGURATION - TUNE THESE
# =============================================================================

# Full VRAM mode (fastest) vs fallback to model offload
# Set to True if you get OOM errors
USE_MODEL_OFFLOAD = False

# Use torch.compile for transformer (requires PyTorch 2.0+)
# First run will be slower due to compilation, subsequent runs faster
USE_TORCH_COMPILE = False

# Skip warmup (saves ~5s but first real inference might be slower)
SKIP_WARMUP = False


# =============================================================================
# PIPELINE STEPS - Same bulletproof prompts from v14
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

def print_banner(text: str, char: str = "=") -> None:
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def get_precision_manual() -> str:
    compute_cap = torch.cuda.get_device_capability()
    return "fp4" if compute_cap[0] >= 10 else "int4"


def get_gpu_memory_manual() -> float:
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def resize_to_fixed(img: Image.Image, width: int = FIXED_WIDTH, height: int = FIXED_HEIGHT) -> Image.Image:
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img: Image.Image, size: int = REF_SIZE) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def get_vram_usage() -> Tuple[float, float]:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0.0, 0.0


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_negative_prompt(step_config: dict) -> str:
    step_negative = step_config.get("negative_prompt", "")
    if step_negative:
        return f"{GLOBAL_NEGATIVE}, {step_negative}"
    return GLOBAL_NEGATIVE


# =============================================================================
# PRELOADING - Load all references once at startup
# =============================================================================

def preload_reference_images(pipeline_steps: list, input_dir: str) -> Dict[str, Image.Image]:
    """
    Load and resize ALL reference images at startup.
    Keeps them in CPU memory (PIL Images are lightweight).
    """
    print("📦 Preloading reference images...")
    refs = {}
    
    for step in pipeline_steps:
        ref_path = os.path.join(input_dir, step["ref_image"])
        if os.path.exists(ref_path):
            refs[step["name"]] = resize_reference(Image.open(ref_path).convert("RGB"))
            print(f"   ✓ {step['name']}: {step['ref_image']}")
        else:
            print(f"   ✗ {step['name']}: MISSING - {step['ref_image']}")
    
    print(f"   Loaded {len(refs)}/{len(pipeline_steps)} references\n")
    return refs


# =============================================================================
# MODEL LOADING - MAXIMUM SPEED CONFIGURATION
# =============================================================================

def load_pipeline():
    """
    Load pipeline with EVERYTHING in VRAM for maximum speed.
    
    Strategy:
    1. Load INT4 quantized transformer (small footprint)
    2. Disable ALL offloading
    3. Move entire pipeline to CUDA
    4. Optionally compile transformer with torch.compile
    """
    
    print_banner("LOADING NUNCHAKU INT4 - FULL VRAM MODE")
    
    clear_memory()
    
    # Get GPU info
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
    print(f"🚀 Mode: {'Model Offload' if USE_MODEL_OFFLOAD else 'FULL VRAM (fastest)'}")
    print(f"🔥 Torch Compile: {USE_TORCH_COMPILE}")
    print()
    
    # Lightning scheduler config
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
    
    # Find model path
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
    print("   Loading INT4 quantized transformer...")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
    
    # CRITICAL: Disable Nunchaku's internal offloading
    transformer.set_offload(False)
    
    # Load base pipeline
    qwen_base = f"{MODEL_DIR}/Qwen-Image-Edit-2509"
    if not os.path.exists(qwen_base):
        qwen_base = "Qwen/Qwen-Image-Edit-2509"
    print(f"   Loading base pipeline from: {qwen_base}")
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        qwen_base,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    
    # ==========================================================================
    # MEMORY STRATEGY
    # ==========================================================================
    
    if USE_MODEL_OFFLOAD:
        # Fallback: Use model-level offload (slower but safer)
        print("\n📍 Using enable_model_cpu_offload (fallback mode)...")
        pipeline.enable_model_cpu_offload()
    else:
        # FASTEST: Everything stays in VRAM
        print("\n📍 Moving entire pipeline to CUDA (full VRAM mode)...")
        pipeline.to("cuda")
    
    # Optional: Compile transformer for extra speed
    if USE_TORCH_COMPILE:
        print("🔥 Compiling transformer with torch.compile (this may take a minute)...")
        pipeline.transformer = torch.compile(
            pipeline.transformer, 
            mode="reduce-overhead",
            fullgraph=False  # More compatible
        )
    
    load_time = time.time() - load_start
    print(f"\n✅ Model loaded in {load_time:.1f}s")
    
    allocated, reserved = get_vram_usage()
    print(f"💾 VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Verify we're using VRAM efficiently
    if allocated < 15 and not USE_MODEL_OFFLOAD:
        print("⚠️  Warning: VRAM usage seems low - text encoder might not be loaded")
    
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


# =============================================================================
# INFERENCE - Streamlined for speed
# =============================================================================

def run_edit(
    pipeline, 
    base_img: Image.Image, 
    ref_img: Image.Image, 
    step_config: dict, 
    step_num: int,
    generator: torch.Generator
) -> Tuple[Image.Image, float]:
    """
    Run a single edit step.
    Reuses the generator for consistency.
    """
    
    steps = step_config['steps']
    name = step_config['name']
    prompt = step_config['prompt'].strip()
    negative = get_negative_prompt(step_config)
    
    print(f"\n🎨 Step {step_num}: {name} ({steps} steps)")
    
    # Reset generator seed for this step
    generator.manual_seed(SEED + step_num)
    
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
            generator=generator,
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    allocated, _ = get_vram_usage()
    print(f"   ⏱️  {elapsed:.2f}s | VRAM: {allocated:.2f}GB")
    
    return result, elapsed


def warmup(pipeline, generator: torch.Generator) -> None:
    """Quick warmup to compile CUDA kernels"""
    if SKIP_WARMUP:
        print("\n⏭️  Skipping warmup")
        return
        
    print("\n🔥 Warmup run...")
    
    dummy_base = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), color='gray')
    
    with torch.inference_mode():
        _ = pipeline(
            image=[dummy_base, dummy_ref],
            prompt="warmup",
            num_inference_steps=2,  # Minimal steps for warmup
            true_cfg_scale=1.0,
            guidance_scale=1.0,
            generator=generator,
        )
    
    print("✅ Warmup complete")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(pipeline, preloaded_refs: Dict[str, Image.Image]) -> None:
    """Run the full pipeline with preloaded references"""
    
    print_banner("🎨 WEDDING PIPELINE V15: MAXIMUM SPEED")
    
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
    
    # Create reusable generator
    generator = torch.Generator(device="cuda")
    
    # Warmup
    warmup(pipeline, generator)
    
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        # Get preloaded reference
        ref_img = preloaded_refs.get(step["name"])
        
        if ref_img is None:
            print(f"⚠️  Missing reference for {step['name']} - skipping")
            continue
        
        result, elapsed = run_edit(pipeline, current_image, ref_img, step, i, generator)
        
        # Save intermediate result
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"   💾 Saved: step_{i}_{step['name']}.png")
        
        step_times.append({"name": step["name"], "steps": step["steps"], "time": elapsed})
        current_image = result
    
    total_time = time.time() - pipeline_start
    
    # Save final result
    final_path = os.path.join(OUTPUT_DIR, "FINAL_RESULT.png")
    current_image.save(final_path)
    
    # Print summary
    print_banner("✅ PIPELINE COMPLETE")
    
    print("📊 TIMING BREAKDOWN:")
    print("-" * 50)
    for i, s in enumerate(step_times, 1):
        print(f"   {i}. {s['name']:<12} {s['steps']} steps → {s['time']:.2f}s")
    print("-" * 50)
    
    inference_total = sum(s['time'] for s in step_times)
    avg_per_layer = inference_total / len(step_times) if step_times else 0
    
    print(f"   Inference total: {inference_total:.2f}s")
    print(f"   Pipeline total:  {total_time:.2f}s")
    print(f"   Avg per layer:   {avg_per_layer:.2f}s")
    
    # Cost estimate
    cost = (total_time / 3600) * 0.34
    print(f"\n💰 Cost (RTX 4090 @ $0.34/hr): ${cost:.4f}")
    print(f"🏁 Final result: {final_path}")
    
    # Compare to v14
    v14_estimate = len(step_times) * 10  # ~10s per step with v14
    speedup = v14_estimate / inference_total if inference_total > 0 else 0
    print(f"\n📈 Estimated speedup vs v14: {speedup:.1f}x faster")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"WEDDING DECOR V15 - MAXIMUM SPEED\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Mode: {'Model Offload' if USE_MODEL_OFFLOAD else 'Full VRAM'}\n")
        f.write(f"Torch Compile: {USE_TORCH_COMPILE}\n\n")
        f.write("TIMING:\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['time']:.2f}s\n")
        f.write(f"\nTotal inference: {inference_total:.2f}s\n")
        f.write(f"Total pipeline: {total_time:.2f}s\n")
        f.write(f"Avg per layer: {avg_per_layer:.2f}s\n")
        f.write(f"Cost: ${cost:.4f}\n")
        f.write(f"Speedup vs v14: ~{speedup:.1f}x\n")

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
    print_banner("🚀 WEDDING DECOR V15 - MAXIMUM SPEED")
    nuke_gpu_memory()
    print(get_vram_usage())
    
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # Preload all reference images first (minimal memory, fast)
    preloaded_refs = preload_reference_images(PIPELINE_STEPS, INPUT_DIR)
    
    # Load pipeline (this is where VRAM gets used)
    pipeline = load_pipeline()
    
    # Run the full pipeline
    run_pipeline(pipeline, preloaded_refs)
    
    print(f"\n✨ Done at {datetime.now().strftime('%H:%M:%S')}")
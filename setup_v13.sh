#!/bin/bash
# =============================================================================
# WEDDING DECOR PIPELINE V13 - RTX 4090 NUNCHAKU INT4 LIGHTNING SETUP
# =============================================================================
#
# TARGET: RTX 4090 (24GB VRAM) - ~5 seconds per edit, ~35-40s full pipeline
# COST: $0.34/hr on RunPod = ~$0.004 per pipeline run (80% cheaper than PRO 6000)
#
# This script installs EVERYTHING from scratch:
# 1. System dependencies
# 2. Python packages
# 3. Correct Nunchaku wheel (NOT the PyPI statistics package!)
# 4. All model files (Nunchaku INT4 Lightning + Qwen base)
# 5. Creates inference script
#
# USAGE:
#   chmod +x setup_4090.sh
#   ./setup_4090.sh
#
# DISK SPACE REQUIRED: ~50GB
# TIME TO SETUP: ~10-15 minutes (depending on download speed)
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# =============================================================================
# HEADER
# =============================================================================

clear
print_header "WEDDING DECOR PIPELINE V13 - RTX 4090 SETUP"

echo "Target GPU: NVIDIA RTX 4090 (24GB)"
echo "Quantization: INT4 with Lightning 8-step"
echo "Expected speed: ~5 seconds per edit"
echo "Expected pipeline time: ~35-40 seconds (7 layers)"
echo ""
echo "This script will install everything from scratch."
echo ""

# =============================================================================
# STEP 0: Verify GPU
# =============================================================================

print_header "STEP 0: Verifying GPU"

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)

echo "Detected GPU: $GPU_NAME"
echo "GPU Memory: $GPU_MEMORY"

if [[ "$GPU_NAME" == *"5090"* ]] || [[ "$GPU_NAME" == *"5080"* ]] || [[ "$GPU_NAME" == *"5070"* ]]; then
    print_error "RTX 50-series (Blackwell) detected!"
    print_error "This script is for RTX 4090 (INT4). Blackwell needs FP4 models."
    print_error "Please use an RTX 4090 or earlier GPU."
    exit 1
fi

if [[ "$GPU_NAME" == *"4090"* ]]; then
    print_success "RTX 4090 detected - Perfect!"
elif [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"A6000"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
    print_warning "$GPU_NAME detected - Should work with INT4"
else
    print_warning "GPU: $GPU_NAME - Compatibility unknown, proceeding anyway"
fi

# =============================================================================
# STEP 1: System Dependencies
# =============================================================================

print_header "STEP 1: Installing System Dependencies"

apt-get update -qq
apt-get install -y -qq git wget curl tree htop

print_success "System dependencies installed"

# =============================================================================
# STEP 2: Python Environment Check
# =============================================================================

print_header "STEP 2: Checking Python Environment"

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.11" ]] && [[ "$PYTHON_VERSION" != "3.12" ]]; then
    print_warning "Python $PYTHON_VERSION detected. Nunchaku works best with 3.11 or 3.12"
fi

# Check PyTorch
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo "PyTorch version: $TORCH_VERSION"

CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "not installed")
echo "CUDA version: $CUDA_VERSION"

# =============================================================================
# STEP 3: Install Python Dependencies
# =============================================================================

print_header "STEP 3: Installing Python Dependencies"

pip install --upgrade pip --quiet

echo "Installing core packages..."
pip install torch torchvision torchaudio --quiet 2>/dev/null || print_warning "PyTorch already installed"

echo "Installing diffusers and transformers..."
pip install "transformers>=4.45.0" --quiet
pip install "diffusers>=0.32.0" --quiet
pip install accelerate --quiet
pip install safetensors --quiet
pip install pillow --quiet
pip install huggingface_hub --quiet
pip install hf_transfer --quiet
pip install sentencepiece --quiet
pip install protobuf --quiet
pip install einops --quiet
pip install peft --quiet

print_success "Python dependencies installed"

# =============================================================================
# STEP 4: Install Correct Nunchaku (FROM GITHUB, NOT PYPI!)
# =============================================================================

print_header "STEP 4: Installing Nunchaku (MIT-HAN-LAB version)"

# CRITICAL: First uninstall the WRONG nunchaku if present
echo "Removing any incorrect nunchaku installation..."
pip uninstall nunchaku -y 2>/dev/null || true

# Determine correct wheel
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_MAJOR_MINOR=$(echo $TORCH_VERSION | cut -d. -f1-2)

echo "Python: $PYTHON_VERSION, PyTorch: $TORCH_VERSION"

# Map Python version to cp tag
if [[ "$PYTHON_VERSION" == "3.10" ]]; then
    CP_TAG="cp310"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    CP_TAG="cp311"
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
    CP_TAG="cp312"
elif [[ "$PYTHON_VERSION" == "3.13" ]]; then
    CP_TAG="cp313"
else
    print_error "Unsupported Python version: $PYTHON_VERSION"
    exit 1
fi

# Try to install from GitHub releases
WHEEL_BASE="https://github.com/nunchaku-tech/nunchaku/releases/download"
NUNCHAKU_VERSION="v1.1.0"

# Try different torch versions in order of preference
TORCH_VERSIONS_TO_TRY=("$TORCH_MAJOR_MINOR" "2.7" "2.8" "2.6" "2.11")

INSTALLED=false
for TRY_TORCH in "${TORCH_VERSIONS_TO_TRY[@]}"; do
    WHEEL_URL="${WHEEL_BASE}/${NUNCHAKU_VERSION}/nunchaku-1.1.0+torch${TRY_TORCH}-${CP_TAG}-${CP_TAG}-linux_x86_64.whl"
    echo "Trying: torch${TRY_TORCH} wheel..."
    
    if pip install "$WHEEL_URL" 2>/dev/null; then
        print_success "Nunchaku installed with torch${TRY_TORCH} wheel!"
        INSTALLED=true
        break
    fi
done

if [ "$INSTALLED" = false ]; then
    print_warning "Could not install from GitHub, trying HuggingFace..."
    
    HF_WHEEL_BASE="https://huggingface.co/nunchaku-tech/nunchaku/resolve/main"
    
    for TRY_TORCH in "${TORCH_VERSIONS_TO_TRY[@]}"; do
        WHEEL_URL="${HF_WHEEL_BASE}/nunchaku-1.1.0+torch${TRY_TORCH}-${CP_TAG}-${CP_TAG}-linux_x86_64.whl"
        echo "Trying HuggingFace: torch${TRY_TORCH} wheel..."
        
        if pip install "$WHEEL_URL" 2>/dev/null; then
            print_success "Nunchaku installed from HuggingFace!"
            INSTALLED=true
            break
        fi
    done
fi

if [ "$INSTALLED" = false ]; then
    print_error "Failed to install Nunchaku wheel!"
    print_error "Please check https://github.com/nunchaku-tech/nunchaku/releases for available wheels"
    print_error "Your environment: Python $PYTHON_VERSION, PyTorch $TORCH_VERSION"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying Nunchaku installation..."
python3 << 'VERIFY_EOF'
try:
    from nunchaku import NunchakuQwenImageTransformer2DModel
    print("✅ Nunchaku imported successfully!")
    print("   NunchakuQwenImageTransformer2DModel: Available")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

try:
    from nunchaku.utils import get_precision, get_gpu_memory
    precision = get_precision()
    gpu_mem = get_gpu_memory()
    print(f"   Detected precision: {precision}")
    print(f"   GPU memory: {gpu_mem:.1f}GB")
except ImportError:
    print("   (nunchaku.utils not available - will use manual detection)")
VERIFY_EOF

# =============================================================================
# STEP 5: Create Workspace
# =============================================================================

print_header "STEP 5: Creating Workspace"

WORKSPACE="/workspace/wedding_decor"
mkdir -p $WORKSPACE
mkdir -p $WORKSPACE/images/output/v13_int4_lightning
mkdir -p $WORKSPACE/models

# Create subdirectories for reference images if they don't exist
mkdir -p $WORKSPACE/images/chairs
mkdir -p $WORKSPACE/images/tablecloths
mkdir -p $WORKSPACE/images/plates
mkdir -p $WORKSPACE/images/napkins
mkdir -p $WORKSPACE/images/cutlery
mkdir -p $WORKSPACE/images/glassware
mkdir -p $WORKSPACE/images/centerpieces

cd $WORKSPACE

print_success "Workspace created at $WORKSPACE"

# =============================================================================
# STEP 6: Download Models
# =============================================================================

print_header "STEP 6: Downloading Models (~40GB total)"

echo "This will take several minutes depending on your internet speed..."
echo ""

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

python3 << 'DOWNLOAD_EOF'
import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

cache_dir = "/workspace/wedding_decor/models"
os.makedirs(cache_dir, exist_ok=True)

print("=" * 60)
print("DOWNLOADING NUNCHAKU INT4 LIGHTNING MODELS")
print("=" * 60)

nunchaku_dir = f"{cache_dir}/nunchaku-qwen-image-edit-2509"
os.makedirs(nunchaku_dir, exist_ok=True)
os.makedirs(f"{nunchaku_dir}/lightning-251115", exist_ok=True)

# Models to download (INT4 for RTX 4090)
models = [
    # Lightning 8-step (main model for production)
    ("lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors", "INT4 r128 Lightning 8-step"),
    # Lightning 4-step (faster, slightly lower quality)
    ("lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors", "INT4 r128 Lightning 4-step"),
    # Base model (fallback)
    ("svdq-int4_r128-qwen-image-edit-2509.safetensors", "INT4 r128 Base"),
]

for model_file, description in models:
    print(f"\n📥 Downloading {description}...")
    try:
        hf_hub_download(
            repo_id="nunchaku-tech/nunchaku-qwen-image-edit-2509",
            filename=model_file,
            local_dir=nunchaku_dir,
            local_dir_use_symlinks=False
        )
        print(f"   ✅ Downloaded!")
    except Exception as e:
        print(f"   ⚠️  Could not download: {e}")

print("\n" + "=" * 60)
print("DOWNLOADING QWEN BASE MODEL COMPONENTS")
print("=" * 60)

# Download Qwen-Image-Edit-2509 (full model including text encoder)
print("\n📥 Downloading Qwen-Image-Edit-2509 (full model)...")
print("   This is the largest download (~40GB), please wait...")

qwen_dir = f"{cache_dir}/Qwen-Image-Edit-2509"
try:
    snapshot_download(
        "Qwen/Qwen-Image-Edit-2509",
        local_dir=qwen_dir,
        ignore_patterns=["*.gguf", "*.onnx"]  # Skip GGUF/ONNX files
    )
    print("   ✅ Qwen-Image-Edit-2509 downloaded!")
except Exception as e:
    print(f"   ⚠️  Error downloading full model: {e}")
    print("   Trying to download essential components only...")
    
    # Fall back to downloading just config files
    snapshot_download(
        "Qwen/Qwen-Image-Edit-2509",
        local_dir=qwen_dir,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.onnx"]
    )
    
    # Download Qwen-Image for text encoder and VAE
    print("\n📥 Downloading Qwen-Image (text encoder & VAE)...")
    qwen_image_dir = f"{cache_dir}/Qwen-Image"
    snapshot_download(
        "Qwen/Qwen-Image",
        local_dir=qwen_image_dir,
        allow_patterns=["text_encoder/*", "vae/*", "tokenizer/*", "*.json"]
    )
    
    # Create symlinks
    import os
    te_src = f"{qwen_image_dir}/text_encoder"
    te_dst = f"{qwen_dir}/text_encoder"
    vae_src = f"{qwen_image_dir}/vae"
    vae_dst = f"{qwen_dir}/vae"
    
    if os.path.exists(te_src) and not os.path.islink(te_dst):
        if os.path.isdir(te_dst):
            import shutil
            shutil.rmtree(te_dst)
        os.symlink(te_src, te_dst)
        print("   ✅ Linked text_encoder")
    
    if os.path.exists(vae_src) and not os.path.islink(vae_dst):
        if os.path.isdir(vae_dst):
            import shutil
            shutil.rmtree(vae_dst)
        os.symlink(vae_src, vae_dst)
        print("   ✅ Linked VAE")

print("\n" + "=" * 60)
print("✅ ALL MODELS DOWNLOADED!")
print("=" * 60)

# Show disk usage
import subprocess
result = subprocess.run(['du', '-sh', cache_dir], capture_output=True, text=True)
print(f"\n📊 Total disk usage: {result.stdout.strip()}")

# List downloaded Nunchaku models
print("\n📦 Nunchaku models available:")
lightning_dir = f"{nunchaku_dir}/lightning-251115"
if os.path.exists(lightning_dir):
    for f in os.listdir(lightning_dir):
        if f.endswith('.safetensors'):
            size = os.path.getsize(f"{lightning_dir}/{f}") / (1024**3)
            print(f"   - {f} ({size:.1f}GB)")

base_models = [f for f in os.listdir(nunchaku_dir) if f.endswith('.safetensors')]
for f in base_models:
    size = os.path.getsize(f"{nunchaku_dir}/{f}") / (1024**3)
    print(f"   - {f} ({size:.1f}GB)")
DOWNLOAD_EOF

print_success "Models downloaded"

# =============================================================================
# STEP 7: Create Inference Script
# =============================================================================

print_header "STEP 7: Creating Inference Script"

cat > $WORKSPACE/inference_v13_int4.py << 'INFERENCE_SCRIPT'
"""
===============================================================================
WEDDING DECOR VISUALIZATION - PIPELINE V13 (RTX 4090 INT4 LIGHTNING)
===============================================================================

Optimized for: NVIDIA RTX 4090 (24GB VRAM)
Quantization: INT4 with SVDQuant (Nunchaku)
Inference: Lightning 8-step (~5 seconds per edit)

Expected performance:
- Single edit: ~5 seconds
- 7-layer pipeline: ~35-40 seconds
- Cost on RunPod: ~$0.004 per pipeline run

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
OUTPUT_DIR = "/workspace/wedding_decor/images/output/v13_int4_lightning"
BASE_IMAGE = "base_image_table.png"
MODEL_DIR = "/workspace/wedding_decor/models"

# Image dimensions
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# Model configuration
NUNCHAKU_RANK = 128          # 128 = best quality, 32 = faster
USE_LIGHTNING = True         # Use Lightning for 8-step inference
LIGHTNING_STEPS = 8          # 4 or 8 (8 recommended for quality)

# Inference configuration
TRUE_CFG_SCALE = 1.0         # CFG scale for Lightning models
GUIDANCE_SCALE = 1.0         # Guidance scale
SEED = 42                    # Random seed for reproducibility

# VRAM mode: "full" for RTX 4090 24GB, "offload" if OOM
VRAM_MODE = "full"

# =============================================================================
# PIPELINE STEPS - Optimized order for best results
# =============================================================================

PIPELINE_STEPS = [
    # Structural elements first (6 steps each)
    {
        "name": "chairs",
        "steps": LIGHTNING_STEPS,
        "ref_image": "chairs/clear_chiavari.png",
        "prompt": "Replace all chairs with elegant gold chiavari chairs with white cushions matching the reference. 8 chairs evenly spaced around the round table with white tablecloth."
    },
    {
        "name": "tablecloth",
        "steps": LIGHTNING_STEPS,
        "ref_image": "tablecloths/satin_red.png",
        "prompt": "The round table now has a luxurious deep red satin tablecloth with elegant draping matching the reference. 8 gold chiavari chairs with white cushions surround the table."
    },
    # Place settings (6-8 steps each)
    {
        "name": "plates",
        "steps": LIGHTNING_STEPS,
        "ref_image": "plates/white_with_gold_rim.png",
        "prompt": "Add 8 white dinner plates with gold rim matching the reference. One plate at each place setting on the red tablecloth. Gold chiavari chairs around table."
    },
    {
        "name": "napkins",
        "steps": LIGHTNING_STEPS,
        "ref_image": "napkins/satin_pink.png",
        "prompt": "Add pink satin napkins folded in elegant fan shapes on each plate, matching the reference. 8 place settings with plates on red tablecloth. Gold chiavari chairs."
    },
    # Fine details (8 steps each)
    {
        "name": "cutlery",
        "steps": LIGHTNING_STEPS,
        "ref_image": "cutlery/gold_luxe.png",
        "prompt": "Add gold cutlery beside each plate - fork on left, knife and spoon on right, matching the reference. Complete place settings with plates and pink napkins on red tablecloth. Gold chiavari chairs."
    },
    {
        "name": "glassware",
        "steps": LIGHTNING_STEPS,
        "ref_image": "glassware/crystal_wine_glass.png",
        "prompt": "Add crystal wine glasses at each place setting above the knife, matching the reference. Realistic glass transparency. Complete settings with plates, napkins, cutlery on red tablecloth. Gold chiavari chairs."
    },
    {
        "name": "centerpiece",
        "steps": LIGHTNING_STEPS,
        "ref_image": "centerpieces/pink_flowral_with_gold_stand.png",
        "prompt": "Add a stunning pink rose centerpiece on gold stand to the center of the table, matching the reference. Spherical arrangement of fresh roses. All 8 place settings surround it with plates, napkins, cutlery, glasses on red tablecloth. Gold chiavari chairs."
    },
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    """Print a formatted banner"""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def get_precision_manual():
    """Manually detect GPU precision (fallback if nunchaku.utils unavailable)"""
    compute_cap = torch.cuda.get_device_capability()
    if compute_cap[0] >= 10:  # Blackwell (sm_100+)
        return "fp4"
    else:
        return "int4"


def get_gpu_memory_manual():
    """Get GPU memory in GB (fallback)"""
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def resize_to_fixed(img, width=FIXED_WIDTH, height=FIXED_HEIGHT):
    """Resize image to fixed dimensions"""
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img, size=REF_SIZE):
    """Resize reference image"""
    return img.resize((size, size), Image.LANCZOS)


def get_vram_usage():
    """Get current VRAM usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0, 0


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    """Load the Nunchaku INT4 Lightning pipeline"""
    
    print_banner("LOADING NUNCHAKU INT4 LIGHTNING MODEL")
    
    # Clear any existing memory
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
    print(f"💾 VRAM Mode: {VRAM_MODE}")
    print()
    
    # Verify we're using INT4 (not FP4)
    if precision != "int4":
        print(f"⚠️  Warning: Detected {precision}, but this script is optimized for INT4")
        print(f"   Your GPU may be Blackwell (RTX 50-series) which needs FP4 models")
        print(f"   Proceeding anyway, but you may encounter issues...")
        print()
    
    # Lightning scheduler configuration
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
    
    # Find the model file
    nunchaku_dir = f"{MODEL_DIR}/nunchaku-qwen-image-edit-2509"
    
    if USE_LIGHTNING:
        model_filename = f"svdq-{precision}_r{NUNCHAKU_RANK}-qwen-image-edit-2509-lightning-{LIGHTNING_STEPS}steps-251115.safetensors"
        local_path = f"{nunchaku_dir}/lightning-251115/{model_filename}"
        hf_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/{model_filename}"
    else:
        model_filename = f"svdq-{precision}_r{NUNCHAKU_RANK}-qwen-image-edit-2509.safetensors"
        local_path = f"{nunchaku_dir}/{model_filename}"
        hf_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/{model_filename}"
    
    # Load from local or HuggingFace
    if os.path.exists(local_path):
        model_path = local_path
        print(f"📦 Loading local model: {model_filename}")
    else:
        model_path = hf_path
        print(f"📦 Loading from HuggingFace: {hf_path}")
    
    load_start = time.time()
    
    # Load the quantized transformer
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
    
    # Load the full pipeline
    qwen_base = f"{MODEL_DIR}/Qwen-Image-Edit-2509"
    if not os.path.exists(qwen_base):
        qwen_base = "Qwen/Qwen-Image-Edit-2509"
        print(f"   Using HuggingFace base: {qwen_base}")
    else:
        print(f"   Using local base: {qwen_base}")
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        qwen_base,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    
    load_time = time.time() - load_start
    print(f"\n✅ Model loaded in {load_time:.1f}s")
    
    # Configure VRAM mode
    if VRAM_MODE == "full":
        print("📍 Mode: Full VRAM (no offloading)")
        pipeline.to("cuda")
    elif VRAM_MODE == "offload":
        print("📍 Mode: Model CPU Offload")
        pipeline.enable_model_cpu_offload()
    elif VRAM_MODE == "aggressive_offload":
        print("📍 Mode: Aggressive Per-Layer Offload")
        transformer.set_offload(True, use_pin_memory=True, num_blocks_on_gpu=4)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()
    
    # Disable progress bars for cleaner output
    pipeline.set_progress_bar_config(disable=True)
    
    allocated, reserved = get_vram_usage()
    print(f"💾 VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return pipeline


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_edit(pipeline, base_img, ref_img, step_config, step_num):
    """Run a single edit step"""
    
    steps = step_config['steps']
    name = step_config['name']
    prompt = step_config['prompt']
    
    print(f"\n🎨 Step {step_num}: {name} ({steps} steps)")
    print(f"   Prompt: {prompt[:60]}...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, bad anatomy",
            num_inference_steps=steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED + step_num),
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    result = output.images[0]
    
    # Ensure output is correct size
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    allocated, _ = get_vram_usage()
    print(f"   ⏱️  Done in {elapsed:.2f}s | VRAM: {allocated:.2f}GB")
    
    return result, elapsed


def warmup(pipeline):
    """Run a warmup inference to optimize CUDA kernels"""
    
    print("\n🔥 Warmup run (optimizing CUDA kernels)...")
    
    dummy_base = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), color='white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), color='gray')
    
    with torch.inference_mode():
        _ = pipeline(
            image=[dummy_base, dummy_ref],
            prompt="warmup inference",
            num_inference_steps=4,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
        )
    
    clear_memory()
    print("✅ Warmup complete")


def run_pipeline(pipeline):
    """Run the full wedding decor pipeline"""
    
    print_banner("🎨 WEDDING PIPELINE V13: INT4 LIGHTNING")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print configuration
    total_steps = sum(s['steps'] for s in PIPELINE_STEPS)
    print(f"📋 Pipeline: {len(PIPELINE_STEPS)} layers, {total_steps} total steps")
    print(f"📐 Output size: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"📐 Reference size: {REF_SIZE}x{REF_SIZE}")
    print(f"🎯 Lightning steps: {LIGHTNING_STEPS}")
    print()
    
    # Load base image
    base_path = os.path.join(INPUT_DIR, BASE_IMAGE)
    if not os.path.exists(base_path):
        print(f"❌ Base image not found: {base_path}")
        print(f"   Please place your base table image at: {base_path}")
        return
    
    current_image = resize_to_fixed(Image.open(base_path).convert("RGB"))
    current_image.save(os.path.join(OUTPUT_DIR, "step_0_original.png"))
    print(f"📷 Loaded base image: {BASE_IMAGE}")
    
    # Warmup
    warmup(pipeline)
    
    # Run pipeline
    step_times = []
    pipeline_start = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        ref_path = os.path.join(INPUT_DIR, step["ref_image"])
        
        if not os.path.exists(ref_path):
            print(f"⚠️  Reference image not found: {step['ref_image']} - skipping")
            continue
        
        ref_img = resize_reference(Image.open(ref_path).convert("RGB"))
        
        result, elapsed = run_edit(pipeline, current_image, ref_img, step, i)
        
        # Save intermediate result
        output_path = os.path.join(OUTPUT_DIR, f"step_{i}_{step['name']}.png")
        result.save(output_path)
        print(f"   💾 Saved: {output_path}")
        
        step_times.append({
            "name": step["name"],
            "steps": step["steps"],
            "time": elapsed
        })
        
        current_image = result
    
    # Save final result
    total_time = time.time() - pipeline_start
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
    print(f"   Inference time: {inference_total:.2f}s")
    print(f"   Total time:     {total_time:.2f}s")
    print(f"   Avg per layer:  {inference_total/len(step_times):.2f}s")
    
    print(f"\n🏁 Final result: {final_path}")
    
    # Cost calculation (RTX 4090 @ $0.34/hr on RunPod)
    cost_4090 = (total_time / 3600) * 0.34
    cost_pro6000 = (total_time / 3600) * 1.84
    savings = (1 - cost_4090/cost_pro6000) * 100
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"   RTX 4090 cost:     ${cost_4090:.4f}")
    print(f"   PRO 6000 cost:     ${cost_pro6000:.4f}")
    print(f"   Savings:           {savings:.1f}%")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("WEDDING DECOR PIPELINE V13 - INT4 LIGHTNING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Precision: INT4\n")
        f.write(f"Rank: {NUNCHAKU_RANK}\n")
        f.write(f"Lightning steps: {LIGHTNING_STEPS}\n")
        f.write(f"VRAM mode: {VRAM_MODE}\n\n")
        f.write("TIMING:\n")
        f.write("-" * 40 + "\n")
        for i, s in enumerate(step_times, 1):
            f.write(f"{i}. {s['name']}: {s['steps']} steps, {s['time']:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Inference total: {inference_total:.2f}s\n")
        f.write(f"Pipeline total: {total_time:.2f}s\n\n")
        f.write(f"COST:\n")
        f.write(f"RTX 4090: ${cost_4090:.4f}\n")
        f.write(f"PRO 6000: ${cost_pro6000:.4f}\n")
        f.write(f"Savings: {savings:.1f}%\n")
    
    print(f"\n📄 Report saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_banner("🚀 WEDDING DECOR V13 - INT4 LIGHTNING")
    
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Run pipeline
    run_pipeline(pipeline)
    
    print(f"\n✨ Complete at {datetime.now().strftime('%H:%M:%S')}")
INFERENCE_SCRIPT

print_success "Inference script created"

# =============================================================================
# STEP 8: Create Sample Reference Images (placeholders)
# =============================================================================

print_header "STEP 8: Checking Reference Images"

if [ ! -f "$WORKSPACE/images/base_image_table.png" ]; then
    print_warning "Base image not found at $WORKSPACE/images/base_image_table.png"
    print_info "Please upload your base table image to: $WORKSPACE/images/base_image_table.png"
fi

echo ""
echo "Reference images needed in $WORKSPACE/images/:"
echo "  - base_image_table.png (base table image)"
echo "  - chairs/clear_chiavari.png"
echo "  - tablecloths/satin_red.png"
echo "  - plates/white_with_gold_rim.png"
echo "  - napkins/satin_pink.png"
echo "  - cutlery/gold_luxe.png"
echo "  - glassware/crystal_wine_glass.png"
echo "  - centerpieces/pink_flowral_with_gold_stand.png"

# =============================================================================
# STEP 9: Final Verification
# =============================================================================

print_header "STEP 9: Final Verification"

python3 << 'FINAL_VERIFY_EOF'
import torch
import sys

print("=" * 50)
print("SYSTEM VERIFICATION")
print("=" * 50)

# Python
print(f"\n🐍 Python: {sys.version.split()[0]}")

# PyTorch
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🎮 CUDA: {torch.version.cuda}")
print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

# Diffusers
import diffusers
print(f"🎨 Diffusers: {diffusers.__version__}")

# Nunchaku
try:
    from nunchaku import NunchakuQwenImageTransformer2DModel
    print("⚡ Nunchaku: ✅ Installed correctly")
except ImportError as e:
    print(f"⚡ Nunchaku: ❌ {e}")
    sys.exit(1)

# Check models
import os
model_dir = "/workspace/wedding_decor/models"

# Nunchaku models
nunchaku_dir = f"{model_dir}/nunchaku-qwen-image-edit-2509"
lightning_dir = f"{nunchaku_dir}/lightning-251115"

print("\n📦 Nunchaku INT4 Lightning models:")
if os.path.exists(lightning_dir):
    models = [f for f in os.listdir(lightning_dir) if 'int4' in f and f.endswith('.safetensors')]
    if models:
        for m in models:
            size = os.path.getsize(f"{lightning_dir}/{m}") / (1024**3)
            print(f"   ✅ {m} ({size:.1f}GB)")
    else:
        print("   ⚠️  No INT4 Lightning models found")
else:
    print("   ⚠️  Lightning directory not found")

# Qwen base model
qwen_dir = f"{model_dir}/Qwen-Image-Edit-2509"
if os.path.exists(qwen_dir):
    te_path = f"{qwen_dir}/text_encoder"
    vae_path = f"{qwen_dir}/vae"
    
    if os.path.exists(te_path):
        print(f"   ✅ Text encoder: Present")
    else:
        print(f"   ⚠️  Text encoder: Missing")
    
    if os.path.exists(vae_path):
        print(f"   ✅ VAE: Present")
    else:
        print(f"   ⚠️  VAE: Missing")
else:
    print(f"   ⚠️  Qwen-Image-Edit-2509 not found")

print("\n" + "=" * 50)
print("✅ VERIFICATION COMPLETE")
print("=" * 50)
FINAL_VERIFY_EOF

# =============================================================================
# COMPLETE
# =============================================================================

print_header "🎉 SETUP COMPLETE!"

echo ""
echo "Workspace: $WORKSPACE"
echo ""
echo "To run the pipeline:"
echo "  cd $WORKSPACE"
echo "  python inference_v13_int4.py"
echo ""
echo "Configuration (edit inference_v13_int4.py):"
echo "  LIGHTNING_STEPS = 8      # 4 for faster, 8 for quality"
echo "  NUNCHAKU_RANK = 128      # 128 for quality, 32 for speed"
echo "  VRAM_MODE = 'full'       # 'offload' if OOM"
echo ""
echo "Expected performance:"
echo "  - Single edit: ~5 seconds"
echo "  - 7-layer pipeline: ~35-40 seconds"
echo "  - Cost per run: ~\$0.004 (80% cheaper than PRO 6000)"
echo ""
echo "Make sure to upload your reference images to:"
echo "  $WORKSPACE/images/"
echo ""

# Show disk usage
echo "📊 Disk usage:"
du -sh $WORKSPACE/models/* 2>/dev/null || true
echo ""
du -sh $WORKSPACE 2>/dev/null || true
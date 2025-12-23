#!/bin/bash
# =============================================================================
# WEDDING DECOR VISUALIZATION - RUNPOD SETUP SCRIPT
# =============================================================================
# This script sets up everything you need to run the inference pipeline.
# Run with: bash setup.sh
#
# What it does:
# 1. Installs system tools
# 2. Configures git with your credentials
# 3. Installs Python dependencies
# 4. Pre-downloads the Qwen model (~60GB) so inference is fast
# 5. Pre-downloads the Lightning LoRA
# 6. Clones/updates your repo
# =============================================================================

set -e  # Stop on errors

echo "========================================="
echo "   WEDDING DECOR - RUNPOD SETUP          "
echo "========================================="
echo "Started at: $(date)"
echo ""

# =============================================================================
# 1. SYSTEM TOOLS
# =============================================================================
echo "📦 [1/7] Installing system tools..."
apt-get update -y -qq
apt-get install -y -qq tree nano git wget htop nvtop
echo "   ✅ System tools installed"

# =============================================================================
# 2. GIT CONFIGURATION
# =============================================================================
echo ""
echo "🔧 [2/7] Configuring Git..."
git config --global user.email "Tahir.muhammad@alumni.utoronto.ca"
git config --global user.name "Tahir001"
git config --global credential.helper store
echo "   ✅ Git configured for Tahir001"

# =============================================================================
# 3. ENVIRONMENT VARIABLES
# =============================================================================
echo ""
echo "🌍 [3/7] Setting environment variables..."

# Hugging Face cache - store models in /workspace so they persist
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $HF_HUB_CACHE
mkdir -p $TRANSFORMERS_CACHE

# Add to bashrc for persistence
echo 'export HF_HOME="/workspace/.cache/huggingface"' >> ~/.bashrc
echo 'export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"' >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

echo "   ✅ Environment configured (HF_HOME=$HF_HOME)"

# =============================================================================
# 4. PYTHON DEPENDENCIES
# =============================================================================
echo ""
echo "🐍 [4/7] Installing Python dependencies..."

# Upgrade pip first
pip install --upgrade pip -q

# Fast Hugging Face downloads
pip install hf_transfer --upgrade -q

# PyTorch with CUDA (check what's already installed)
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "   PyTorch with CUDA already installed"
else
    echo "   Installing PyTorch with CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
fi

# Diffusers (latest from git for Qwen support)
echo "   Installing diffusers (latest)..."
pip install git+https://github.com/huggingface/diffusers.git -q

# Other dependencies
echo "   Installing other dependencies..."
pip install transformers>=4.51.3 accelerate sentencepiece protobuf pillow peft -q

echo "   ✅ Python dependencies installed"

# =============================================================================
# 5. PRE-DOWNLOAD QWEN MODEL (~60GB)
# =============================================================================
echo ""
echo "📥 [5/7] Pre-downloading Qwen-Image-Edit-2509 model..."
echo "   This is ~60GB and takes 5-10 minutes on first run."
echo "   Subsequent runs will use the cached version."
echo ""

python << 'EOF'
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download

print("   Downloading Qwen/Qwen-Image-Edit-2509...")
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2509",
    local_dir=None,  # Uses HF cache
    resume_download=True,
    max_workers=8
)
print("   ✅ Qwen model downloaded/cached")
EOF

# =============================================================================
# 6. PRE-DOWNLOAD LIGHTNING LORA
# =============================================================================
echo ""
echo "📥 [6/7] Pre-downloading Lightning LoRA..."

python << 'EOF'
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download

print("   Downloading lightx2v/Qwen-Image-Lightning...")
snapshot_download(
    repo_id="lightx2v/Qwen-Image-Lightning",
    local_dir=None,
    resume_download=True,
    max_workers=4
)
print("   ✅ Lightning LoRA downloaded/cached")
EOF

# =============================================================================
# 7. CLONE/UPDATE REPOSITORY
# =============================================================================
echo ""
echo "📂 [7/7] Setting up repository..."

cd /workspace

if [ -d "wedding_decor" ]; then
    echo "   Repository exists, pulling latest..."
    cd wedding_decor
    git pull || echo "   (pull failed, continuing with existing files)"
else
    echo "   Cloning repository..."
    git clone https://github.com/Tahir001/wedding_decor.git
    cd wedding_decor
fi

echo "   ✅ Repository ready at /workspace/wedding_decor"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "========================================="
echo "           SETUP COMPLETE ✅             "
echo "========================================="
echo ""
echo "Finished at: $(date)"
echo ""
echo "📍 Working directory: /workspace/wedding_decor"
echo "📦 Models cached at:  /workspace/.cache/huggingface"
echo ""
echo "🚀 TO RUN INFERENCE:"
echo "   cd /workspace/wedding_decor"
echo "   python inference_v11.py"
echo ""
echo "📊 GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "========================================="
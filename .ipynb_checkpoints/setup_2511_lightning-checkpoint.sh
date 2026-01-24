#!/bin/bash
# =============================================================================
# WEDDING DECOR VISUALIZATION - RUNPOD SETUP (V12C - 2511)
# =============================================================================
# Downloads ONLY what's needed for the 2511 pipeline:
#   - Qwen/Qwen-Image-Edit-2511 (base model)
#   - lightx2v/Qwen-Image-Edit-2511-Lightning (4-step LoRA)
#
# Run with: bash setup.sh
# =============================================================================

set -e

echo "========================================="
echo "   WEDDING DECOR - SETUP (2511)          "
echo "========================================="
echo "Started: $(date)"
echo ""

# =============================================================================
# 1. SYSTEM TOOLS
# =============================================================================
echo "📦 [1/6] System tools..."
apt-get update -y -qq
apt-get install -y -qq tree nano git wget htop nvtop
echo "   ✅ Done"

# =============================================================================
# 2. GIT CONFIG
# =============================================================================
echo ""
echo "🔧 [2/6] Git config..."
git config --global user.email "Tahir.muhammad@alumni.utoronto.ca"
git config --global user.name "Tahir001"
git config --global credential.helper store
echo "   ✅ Done"

# =============================================================================
# 3. ENVIRONMENT
# =============================================================================
echo ""
echo "🌍 [3/6] Environment..."

export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE

cat >> ~/.bashrc << 'EOF'
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF

echo "   ✅ HF_HOME=$HF_HOME"

# =============================================================================
# 4. PYTHON DEPENDENCIES
# =============================================================================
echo ""
echo "🐍 [4/6] Python dependencies..."

pip install --upgrade pip -q
pip install hf_transfer --upgrade -q

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "   PyTorch OK"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
fi

pip install git+https://github.com/huggingface/diffusers.git -q
pip install transformers>=4.51.3 accelerate sentencepiece protobuf pillow peft -q

echo "   ✅ Done"

# =============================================================================
# 5. DOWNLOAD MODELS (2511 ONLY)
# =============================================================================
echo ""
echo "📥 [5/6] Downloading Qwen-Image-Edit-2511 + Lightning LoRA..."
echo "   This takes ~10 min on first run, cached after."
echo ""

python << 'EOF'
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download

# 2511 Base Model
print("   [1/2] Qwen/Qwen-Image-Edit-2511...")
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2511",
    resume_download=True,
    max_workers=8
)
print("   ✅ Base model cached")

# 2511 Lightning LoRA (4-step)
print("   [2/2] lightx2v/Qwen-Image-Edit-2511-Lightning...")
snapshot_download(
    repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
    resume_download=True,
    max_workers=4
)
print("   ✅ Lightning LoRA cached")
EOF

# =============================================================================
# 6. REPOSITORY
# =============================================================================
echo ""
echo "📂 [6/6] Repository..."

cd /workspace

if [ -d "wedding_decor" ]; then
    cd wedding_decor
    git pull || echo "   (pull skipped)"
else
    git clone https://github.com/Tahir001/wedding_decor.git
    cd wedding_decor
fi

echo "   ✅ /workspace/wedding_decor"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "========================================="
echo "           SETUP COMPLETE ✅             "
echo "========================================="
echo ""
echo "Finished: $(date)"
echo ""
echo "📍 Code:   /workspace/wedding_decor"
echo "📦 Models: /workspace/.cache/huggingface"
echo ""
echo "🚀 RUN:"
echo "   cd /workspace/wedding_decor"
echo "   python inference_main_v12c.py"
echo ""
echo "🖥️  GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================="
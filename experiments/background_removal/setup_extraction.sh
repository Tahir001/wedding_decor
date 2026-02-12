#!/bin/bash
# =============================================================================
# WEDDING DECOR — EXTRACTION PIPELINE SETUP (RunPod / GPU Server)
# =============================================================================
# Models: GroundingDINO (~1GB) + SAM2 (~1.5GB) + FastSAM (~150MB)
# Viz:    supervision (Roboflow) + matplotlib for debug outputs
# NO diffusion models. Setup in ~3-5 min.
# =============================================================================

set -e
echo "========================================="
echo "   WEDDING DECOR — EXTRACTION SETUP      "
echo "========================================="

# System tools
apt-get update -y -qq 2>/dev/null
apt-get install -y -qq tree nano git wget htop 2>/dev/null

# Git config
git config --global user.email "Tahir.muhammad@alumni.utoronto.ca"
git config --global user.name "Tahir001"
git config --global credential.helper store

# HF environment
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
grep -q "HF_HOME" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc << 'EOF'
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF

# Python deps
pip install --upgrade pip -q
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

pip install -q \
    "transformers>=4.44.0" accelerate sentencepiece protobuf \
    pillow scipy numpy opencv-python-headless \
    "supervision>=0.22.0" matplotlib hf_transfer

# SAM2 native
pip install -q git+https://github.com/facebookresearch/sam2.git 2>/dev/null && \
    echo "✅ SAM2 native" || echo "⚠️  SAM2 → transformers fallback"

# FastSAM
pip install -q git+https://github.com/CASIA-IVA-Lab/FastSAM.git 2>/dev/null && \
    echo "✅ FastSAM package" || echo "⚠️  FastSAM unavailable"

FASTSAM_W="/workspace/.cache/FastSAM-x.pt"
[ ! -f "$FASTSAM_W" ] && wget -q -O "$FASTSAM_W" \
    "https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM-x.pt" 2>/dev/null
ln -sf "$FASTSAM_W" /workspace/FastSAM-x.pt 2>/dev/null || true

# Download HF models
python << 'PYEOF'
import os; os.environ['HF_HOME']='/workspace/.cache/huggingface'; os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
from huggingface_hub import snapshot_download
print("Downloading GroundingDINO..."); snapshot_download("IDEA-Research/grounding-dino-base", resume_download=True, max_workers=8)
print("Downloading SAM2..."); snapshot_download("facebook/sam2-hiera-large", resume_download=True, max_workers=8)
print("✅ Models cached")
PYEOF

# Repository
cd /workspace
[ -d "wedding_decor" ] && { cd wedding_decor; git pull || true; } || \
    { git clone https://github.com/Tahir001/wedding_decor.git; cd wedding_decor; }
mkdir -p experiments/background_removal/outputs

# Verify
python << 'PYEOF'
import torch, transformers
print(f"PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
try: import supervision as sv; print(f"supervision {sv.__version__} ✅")
except: print("supervision ❌")
try: from sam2.build_sam import build_sam2_hf; print("SAM2: native ✅")
except: print("SAM2: transformers fallback")
try: from fastsam import FastSAM; print("FastSAM: ✅")
except: print("FastSAM: ❌")
PYEOF

echo ""
echo "========================================="
echo "       SETUP COMPLETE ✅                 "
echo "========================================="
echo ""
echo "cd /workspace/wedding_decor/experiments/background_removal"
echo ""
echo "# Extract with debug (SAM2):"
echo "python 02_extract_items.py --batch --input-dir ./cutlery --item-type cutlery --output-dir ./outputs/cutlery --segmentor sam2 --debug"
echo ""
echo "# Extract with debug (FastSAM - faster):"
echo "python 02_extract_items.py --batch --input-dir ./cutlery --item-type cutlery --output-dir ./outputs/cutlery_fast --segmentor fastsam --debug"
echo ""
echo "# Debug outputs: *_1_detections.png, *_2_masks_grid.png, *_3_overlay.png, *_4_rgba_full.png"
echo "========================================="
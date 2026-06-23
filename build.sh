#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export HF_MODEL_REPO="${HF_MODEL_REPO:-Sahil35077/mosquitoscope-convnextv2-tiny}"
export FORCE_CPU=1
echo "Downloading model checkpoint during build..."
python -c "from webapp.download_assets import ensure_model; p=ensure_model(); print('Model ready:', p, p.stat().st_size)"

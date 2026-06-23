#!/usr/bin/env bash
set -e

export HF_MODEL_REPO="${HF_MODEL_REPO:-Sahil35077/mosquitoscope-convnextv2-tiny}"
export FORCE_CPU="${FORCE_CPU:-1}"
export LOW_MEMORY="${LOW_MEMORY:-1}"

echo "Ensuring model checkpoint is present..."
python -c "from webapp.download_assets import ensure_model; p=ensure_model(); print('Model ready:', p)"

exec gunicorn wsgi:app \
  --bind "0.0.0.0:${PORT:-10000}" \
  --workers 1 \
  --threads 1 \
  --timeout 300 \
  --graceful-timeout 300 \
  --keep-alive 5 \
  --max-requests 100 \
  --max-requests-jitter 20

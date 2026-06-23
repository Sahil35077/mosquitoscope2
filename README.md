# MosquitoScope Web App

Flask demo for 41-class mosquito species classification. Deployed on [Render](https://render.com).

**Model:** [huggingface.co/Sahil35077/mosquitoscope-convnextv2-tiny](https://huggingface.co/Sahil35077/mosquitoscope-convnextv2-tiny)

## Render settings

| Setting | Value |
|---------|--------|
| Build Command | `./build.sh` |
| Start Command | `./start.sh` |
| Health Check | `/health` |

## Environment variables

| Variable | Value |
|----------|--------|
| `HF_MODEL_REPO` | `Sahil35077/mosquitoscope-convnextv2-tiny` |
| `FORCE_CPU` | `1` |
| `LOW_MEMORY` | `1` |
| `DISABLE_TTA` | `1` |
| `TORCH_NUM_THREADS` | `1` |

## Local run

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m webapp.app
```

Open http://localhost:7860

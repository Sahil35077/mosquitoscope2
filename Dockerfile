# Root Dockerfile for Render (repo root). Builds the Flask app in webapp/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_CPU=1 \
    LOW_MEMORY=1 \
    DISABLE_TTA=1 \
    TORCH_NUM_THREADS=1 \
    MAX_UPLOAD_MB=8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt build.sh start.sh wsgi.py runtime.txt ./
COPY webapp/ ./webapp/

RUN chmod +x build.sh start.sh \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

ARG HF_MODEL_REPO=Sahil35077/mosquitoscope-convnextv2-tiny
ENV HF_MODEL_REPO=${HF_MODEL_REPO}
RUN ./build.sh

ENV PORT=10000
EXPOSE 10000

ENTRYPOINT ["./start.sh"]

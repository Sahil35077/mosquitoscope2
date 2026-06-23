#!/usr/bin/env bash
# Render free tier: 1 worker, 1 thread, recycle worker to limit memory growth
exec gunicorn wsgi:app \
  --bind "0.0.0.0:${PORT:-10000}" \
  --workers 1 \
  --threads 1 \
  --timeout 300 \
  --graceful-timeout 300 \
  --keep-alive 5 \
  --max-requests 100 \
  --max-requests-jitter 20

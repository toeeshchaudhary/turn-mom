#!/usr/bin/env bash
# Serve the local TEACHER (70B) that generates synthetic labels. Runs on the
# GH200 on port 8001 (student serving uses 8000, so both can coexist if needed —
# but 70B + 24B won't both fit; label first, then train, then serve the student).
#
# fp8 keeps a 70B comfortably inside 95GB HBM with room for long-context labeling.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-72B-Instruct}"   # or meta-llama/Llama-3.3-70B-Instruct

vllm serve "$MODEL" \
  --served-model-name teacher \
  --quantization fp8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8001

#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-72B-Instruct}"
vllm serve "$MODEL" \
  --served-model-name teacher \
  --quantization fp8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8001

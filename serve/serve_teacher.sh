#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.3-70B-Instruct}"
vllm serve "$MODEL" \
  --served-model-name teacher \
  --quantization fp8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8001

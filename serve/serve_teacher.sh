#!/usr/bin/env bash
set -euo pipefail
# Teacher (synthetic labeler). Default is a PRE-QUANTIZED FP8 checkpoint -> ~70GB download,
# fits the GH200's ~95GB HBM, no online quantization needed.
MODEL="${1:-nvidia/Llama-3.3-70B-Instruct-FP8}"
# QUANT: leave EMPTY for pre-quantized checkpoints (FP8/AWQ/modelopt) so vLLM auto-detects
# the format from the model config. Set QUANT=fp8 only to online-quantize a bf16 checkpoint.
QUANT="${QUANT:-}"
ARGS=(
  --served-model-name teacher
  --max-model-len "${MAXLEN:-6144}"   # >8192 overflows KV at high concurrency -> preemption thrash
  --gpu-memory-utilization 0.92
  --kv-cache-dtype fp8        # halve KV/token so 32-64 concurrent reqs fit without eviction
  --enable-prefix-caching     # the ~2k-token system prompt is identical on every call -> cache it
  --max-num-seqs 128          # continuous batching cap (actual limited by KV capacity)
  --port 8001
)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")
exec vllm serve "$MODEL" "${ARGS[@]}"

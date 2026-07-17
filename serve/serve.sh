#!/usr/bin/env bash
set -euo pipefail
ADAPTER="${1:-out/mistral_naf_lora}"
MERGED="${2:-out/merged}"
python3 - "$ADAPTER" "$MERGED" <<'PY'
import sys
from unsloth import FastLanguageModel
adapter, merged = sys.argv[1], sys.argv[2]
m, t = FastLanguageModel.from_pretrained(adapter, max_seq_length=4096,
                                         load_in_4bit=False, dtype=None)
m.save_pretrained_merged(merged, t, save_method="merged_16bit")
print("merged ->", merged)
PY
vllm serve "$MERGED" \
  --served-model-name naf \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000

#!/usr/bin/env bash
# End-to-end data -> labeled SFT pipeline for ChadGPT.
# Run from repo root with the venv active. Idempotent: skips any phase whose
# output already exists (delete the file to force a rebuild).
#
#   bash run.sh              # full data->label->sft pipeline
#   PROBE=1 bash run.sh      # label only 200 (speed probe), then stop
#   TEACHER=Qwen/Qwen2.5-14B-Instruct MAXPC=3 NSCEN=8000 bash run.sh
#
# After it finishes, train:
#   python3 train/unsloth_train.py --data data/sft/train.jsonl --val data/sft/val.jsonl \
#     --model unsloth/Mistral-Small-24B-Instruct-2501 --bsz 8 --epochs 3
set -euo pipefail
cd "$(dirname "$0")"

TEACHER=${TEACHER:-Qwen/Qwen2.5-14B-Instruct}
MAXPC=${MAXPC:-3}
NSCEN=${NSCEN:-8000}
D=data/interim
mkdir -p "$D" data/sft out

leak_check() { local n; n=$(grep -c '{[A-Z_]\+}' "$1" || true); [ "$n" = "0" ] || { echo "ABORT: $n token leaks in $1"; exit 1; }; }

# ---- 5. clean (batched to temp files then merged — avoids xargs overwrite bug) ----
if [ ! -s "$D/bonzo_clean.jsonl" ] || [ ! -s "$D/calls_clean.jsonl" ]; then
  echo "== 5. clean =="
  rm -f "$D"/bz_part_*.jsonl "$D"/tr_part_*.jsonl
  find data/raw/bonzo_clean_redacted -name '*.json' -print0 \
    | xargs -0 -n 800 sh -c 'python3 scripts/clean_bonzo.py "$@" --out "$(mktemp '"$D"'/bz_part_XXXXXX.jsonl)"' _
  cat "$D"/bz_part_*.jsonl > "$D/bonzo_clean.jsonl"
  find data/raw/transcripts -name '*.txt' -print0 \
    | xargs -0 -n 800 sh -c 'python3 scripts/clean_transcript.py "$@" --out "$(mktemp '"$D"'/tr_part_XXXXXX.jsonl)"' _
  cat "$D"/tr_part_*.jsonl > "$D/calls_clean.jsonl"
  rm -f "$D"/bz_part_*.jsonl "$D"/tr_part_*.jsonl
  echo "  bonzo_clean=$(wc -l < "$D/bonzo_clean.jsonl")  calls_clean=$(wc -l < "$D/calls_clean.jsonl")"
else echo "== 5. clean  (skip, exists) =="; fi

# ---- 6. rehydrate redaction tokens ----
if [ ! -s "$D/hydrated.jsonl" ]; then
  echo "== 6. rehydrate =="
  python3 scripts/rehydrate.py "$D/bonzo_clean.jsonl" "$D/calls_clean.jsonl" --out "$D/hydrated.jsonl"
else echo "== 6. rehydrate  (skip, exists) =="; fi
leak_check "$D/hydrated.jsonl"

# ---- 7. build tasks + scenarios ----
if [ ! -s "$D/all_tasks.jsonl" ]; then
  echo "== 7. build tasks =="
  python3 scripts/build_label_tasks.py "$D/hydrated.jsonl" --out "$D/tasks.jsonl" --max-per-convo "$MAXPC"
  python3 scripts/gen_scenarios.py --out "$D/scenarios.jsonl" --n "$NSCEN"
  cat "$D/tasks.jsonl" "$D/scenarios.jsonl" > "$D/all_tasks.jsonl"
else echo "== 7. build tasks  (skip, exists) =="; fi
echo "  all_tasks=$(wc -l < "$D/all_tasks.jsonl")"

# ---- 8. label (offline batched vLLM) ----
echo "== 8. label (offline, $TEACHER) =="
if [ "${PROBE:-0}" = "1" ]; then
  python3 scripts/label_offline.py "$D/all_tasks.jsonl" --out "$D/_probe.jsonl" --model "$TEACHER" --limit 200
  echo "PROBE done -> $D/_probe.jsonl  (set PROBE=0 for full run)"; exit 0
fi
python3 scripts/label_offline.py "$D/all_tasks.jsonl" --out "$D/labeled.jsonl" --model "$TEACHER"

# ---- 9. audit + format to SFT ----
echo "== 9. audit + to_sft =="
python3 scripts/audit_gate.py "$D/labeled.jsonl" --out "$D/labeled.ok.jsonl" --rejects "$D/rejects.jsonl"
python3 scripts/to_sft.py "$D/labeled.ok.jsonl" --out-train data/sft/train.jsonl --out-val data/sft/val.jsonl --val-frac 0.03
leak_check data/sft/train.jsonl
echo "  train=$(wc -l < data/sft/train.jsonl)  val=$(wc -l < data/sft/val.jsonl)"
echo
echo "DONE. Now train:"
echo "  python3 train/unsloth_train.py --data data/sft/train.jsonl --val data/sft/val.jsonl --model unsloth/Mistral-Small-24B-Instruct-2501 --bsz 8 --epochs 3"

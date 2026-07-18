#!/usr/bin/env bash
# Build the MAOS-first SFT dataset in NeMo chat-messages format.
# Output: data/chat_train.jsonl + data/chat_val.jsonl  (OpenAI {"messages":[...]} JSONL,
# exactly what NeMo AutoModel's ChatDataset expects — see nemo/diffusion_gemma_sft.yaml).
#
# Streams (MAOS-first, per the data spec — we train toward the MAOS IDEAL, not recorded CSS):
#   A. MAOS behavior (synthetic, from NAF playbooks)   gen_maos_scenarios.py
#   A'. MAOS behavior (REAL edge cases mined from Bonzo) mine_bonzo.py
#   B. screening mechanics (deterministic oracle)       gen_scenarios.py
#   C. voice texture (real Bonzo, demoted)              build_label_tasks.py (low --max-per-convo)
# then: teacher label -> audit -> to chat-JSONL.
#
# Run from repo root, venv active, teacher available. Env knobs:
#   TEACHER=Qwen/Qwen2.5-32B-Instruct  BONZO_DUMP=data/raw/bonzo_new/conversations.jsonl  bash nemo/build_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=${TEACHER:-Qwen/Qwen2.5-32B-Instruct}
BONZO_DUMP=${BONZO_DUMP:-data/raw/bonzo_new/conversations.jsonl}
D=data/interim
mkdir -p "$D" data

# ---- Stream C source: cleaned + rehydrated real Bonzo (voice texture, demoted) ----
if [ ! -s "$D/hydrated.jsonl" ] && [ -d data/raw/bonzo_clean_redacted ]; then
  echo "== clean + rehydrate Bonzo =="
  find data/raw/bonzo_clean_redacted -name '*.json' -print0 \
    | xargs -0 -n 800 sh -c 'python3 scripts/clean_bonzo.py "$@" --out "$(mktemp '"$D"'/bz_XXXX.jsonl)"' _
  cat "$D"/bz_*.jsonl > "$D/bonzo_clean.jsonl"; rm -f "$D"/bz_*.jsonl
  python3 scripts/rehydrate.py "$D/bonzo_clean.jsonl" --out "$D/hydrated.jsonl"
fi

echo "== build task streams (MAOS-first) =="
python3 scripts/gen_maos_scenarios.py --out "$D/maos.jsonl" --per "${MAOS_PER:-150}"
[ -s "$BONZO_DUMP" ] && python3 scripts/mine_bonzo.py "$BONZO_DUMP" --out "$D/bonzo_mined.jsonl" --per "${MINE_PER:-300}" || : > "$D/bonzo_mined.jsonl"
python3 scripts/gen_scenarios.py --out "$D/screening.jsonl" --n "${SCREEN_N:-5000}"
if [ -s "$D/hydrated.jsonl" ]; then
  python3 scripts/build_label_tasks.py "$D/hydrated.jsonl" --out "$D/voice.jsonl" --max-per-convo "${VOICE_PC:-2}"
else : > "$D/voice.jsonl"; fi

cat "$D/maos.jsonl" "$D/bonzo_mined.jsonl" "$D/screening.jsonl" "$D/voice.jsonl" > "$D/all_tasks.jsonl"
echo "  all_tasks=$(wc -l < "$D/all_tasks.jsonl")  (maos=$(wc -l < "$D/maos.jsonl") mined=$(wc -l < "$D/bonzo_mined.jsonl") screening=$(wc -l < "$D/screening.jsonl") voice=$(wc -l < "$D/voice.jsonl"))"

echo "== label (teacher: $TEACHER) =="
python3 scripts/label_offline.py "$D/all_tasks.jsonl" --out "$D/labeled.jsonl" --model "$TEACHER"

echo "== audit + to chat-JSONL =="
python3 scripts/audit_gate.py "$D/labeled.jsonl" --out "$D/labeled.ok.jsonl" --rejects "$D/rejects.jsonl"
python3 scripts/to_sft.py "$D/labeled.ok.jsonl" --out-train data/chat_train.jsonl --out-val data/chat_val.jsonl --val-frac 0.03

n=$(grep -c '{[A-Z_]\+}' data/chat_train.jsonl || true)
[ "$n" = "0" ] || { echo "ABORT: $n redaction-token leaks in data/chat_train.jsonl"; exit 1; }
echo "DONE. train=$(wc -l < data/chat_train.jsonl)  val=$(wc -l < data/chat_val.jsonl)  -> data/chat_train.jsonl (NeMo chat-messages)"

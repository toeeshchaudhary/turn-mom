# ChadGPT — 3-Suggestion CSS Agent (v1 rebuild)

Fine-tune a model that, given a **CONTEXT block**, outputs exactly **3 suggested SMS replies** a
Customer Service Specialist (CSS) at New American Funding can pick from — the `RecommendationResponse`
product. This is a clean restart (the old single-reply MVP is archived at
`~/turn_mom_archive_*`).

## The output we're training toward

```json
{"recommendations":[
  {"suggested_message":"Hi John, following up on our call — here are the refinance options that fit. Happy to walk you through them.","confidence":"high"},
  {"suggested_message":"Quick one, John — what's your current balance and rate so I can run exact numbers?","confidence":"high"},
  {"suggested_message":"One thing that could help: a few months of on-time payments could bump your rate tier. Want me to explain?","confidence":"medium"}
]}
```

Input to the model = the full CSS system prompt (`prompts/css_system_prompt.txt`) as `system` +
a rendered CONTEXT block as `user`.

## Decisions

- **Student model:** `unsloth/Mistral-Small-24B-Instruct-2501` — bf16 LoRA via **Unsloth**. No thinking-mode,
  no VL-tokenizer, clean JSON output; fits easily on the GH200. (Deliberately *not* Qwen3.5 — that arch
  cost us days on the MVP.)
- **Teacher model (synthetic labeler):** local **`meta-llama/Llama-3.3-70B-Instruct`** on the GH200
  via vLLM fp8. Free at scale, keeps PII on-box. (Gated on HF — set `HF_TOKEN` before serving.)

## The "everything sounds like a call" fix

The old data problem was two things at once, and the fix addresses both:

1. **Register.** Call transcripts sound like phone calls. **Bonzo is real SMS** (21k Agent↔Client threads) —
   so it's the *voice source*. Transcripts are used only for **flow/stage coverage**, and the teacher
   **rewrites them into text register** during labeling (`build_label_tasks` → teacher normalizes the gold reply).
2. **Format.** Raw data is flat conversations with no stages, no CONTEXT blocks, no 3-suggestion labels.
   We **manufacture** those labels with the teacher (Stream A) plus a deterministic screening oracle (Stream B).

## Pipeline

```
                      R2: Bonzo/*.json (SMS)          R2: */Transcripts/*.txt (calls)
                            │ clean_bonzo.py                  │ clean_transcript.py
                            ▼                                 ▼
                     {file,source,turns:[{role,text}]}  ── same shape ──┐
                            │                                            │
              Stream A ─────┴──── build_label_tasks.py ──► tasks (history + real gold reply)
                                                                 │
  Stream B ── gen_scenarios.py (screening oracle) ──► scenario tasks (deterministic CONTEXT)
                                                                 │
                            label_with_teacher.py (local 72B) ───┤  infers CONTEXT + writes 3 suggestions
                                                                 ▼        (one anchored on the real reply)
                            audit_gate.py ──► drop guardrail violations
                                                                 ▼
                            to_sft.py ──► data/sft/{train,val}.jsonl  (chat messages)
                                                                 ▼
                     train/unsloth_train.py (Mistral-Small-24B bf16 LoRA)
                                                                 ▼
                     serve/serve.sh (merge + vLLM :8000, model "naf")
                                                                 ▼
                     eval/run_eval.py (schema + guardrail scoring over all stages)
```

---

## 0. rclone remote (once)

```bash
rclone config create r2 s3 provider=Cloudflare \
  access_key_id=$R2_KEY secret_access_key=$R2_SECRET \
  endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com --non-interactive
# NOTE: the R2 keys were pasted in chat — rotate them in the Cloudflare dashboard.
```

## 1. Local smoke test (this repo, no GPU)

Prove the data pipeline on a handful of files before touching the box. `--dry-run` fabricates stub labels.

```bash
# a couple of samples already live in _samples/
python3 scripts/clean_bonzo.py      _samples/bonzo_clean_redacted/*.json --out data/interim/bz.jsonl
python3 scripts/clean_transcript.py _samples/transcripts/*.txt           --out data/interim/tr.jsonl
python3 scripts/build_label_tasks.py data/interim/bz.jsonl data/interim/tr.jsonl --out data/interim/tasks.jsonl
python3 scripts/gen_scenarios.py --out data/interim/scen.jsonl --n 200
cat data/interim/tasks.jsonl data/interim/scen.jsonl > data/interim/all.jsonl
python3 scripts/label_with_teacher.py data/interim/all.jsonl --out data/interim/lab.jsonl --dry-run
python3 scripts/audit_gate.py data/interim/lab.jsonl --out data/interim/lab.ok.jsonl --rejects data/interim/rej.jsonl
python3 scripts/to_sft.py data/interim/lab.ok.jsonl --out-train data/sft/train.jsonl --out-val data/sft/val.jsonl
```

**Done when:** `to_sft` prints train/val counts and a record is `[system,user,assistant]` with the
assistant a valid `{"recommendations":[...x3]}`.

## 2. On the GPU box (root@GPU, /var/turn-mom)

### 2a. Pull data + build cleaned convos
```bash
bash scripts/01_pull_data.sh data/raw
find data/raw/bonzo_clean_redacted -name '*.json' -print0 \
  | xargs -0 python3 scripts/clean_bonzo.py --out data/interim/bonzo_clean.jsonl
find data/raw/transcripts -name '*.txt' -print0 \
  | xargs -0 python3 scripts/clean_transcript.py --out data/interim/calls_clean.jsonl

# CRITICAL: rehydrate redaction tokens -> realistic surrogates BEFORE anything sees them.
# Without this the model learns to emit "Hey {NAME}!" literally (the old-model bug).
python3 scripts/rehydrate.py data/interim/bonzo_clean.jsonl data/interim/calls_clean.jsonl \
  --out data/interim/hydrated.jsonl
```

### 2b. Build label tasks (both streams)
```bash
python3 scripts/build_label_tasks.py data/interim/hydrated.jsonl \
  --out data/interim/tasks.jsonl --max-per-convo 6
python3 scripts/gen_scenarios.py --out data/interim/scenarios.jsonl --n 8000
cat data/interim/tasks.jsonl data/interim/scenarios.jsonl > data/interim/all_tasks.jsonl
```

### 2c. Serve the teacher, then label
```bash
HF_TOKEN=<your_hf_token> bash serve/serve_teacher.sh        # meta-llama/Llama-3.3-70B-Instruct, port 8001
# in another shell:
TEACHER_BASE_URL=http://localhost:8001/v1 TEACHER_MODEL=teacher \
  python3 scripts/label_with_teacher.py data/interim/all_tasks.jsonl \
  --out data/interim/labeled.jsonl
# tip: run on a small --limit 200 first, eyeball data/interim/labeled.jsonl for voice quality.
```

### 2d. Audit + format
```bash
python3 scripts/audit_gate.py data/interim/labeled.jsonl \
  --out data/interim/labeled.ok.jsonl --rejects data/interim/rejects.jsonl
python3 scripts/to_sft.py data/interim/labeled.ok.jsonl \
  --out-train data/sft/train.jsonl --out-val data/sft/val.jsonl --val-frac 0.03
```
Skim `data/interim/rejects.jsonl` — a high reject rate means the teacher prompt needs tightening,
not that the gate is wrong.

### 2e. Train (stop the teacher first to free HBM)
```bash
python3 train/unsloth_train.py \
  --data data/sft/train.jsonl --val data/sft/val.jsonl \
  --model unsloth/Mistral-Small-24B-Instruct-2501 --bsz 8 --epochs 3
```

### 2f. Serve + eval
```bash
bash serve/serve.sh out/mistral_naf_lora out/merged      # vLLM :8000, model "naf"
# another shell:
python3 eval/run_eval.py --base http://localhost:8000/v1 --model naf
```
**Done when:** every stage case passes schema+guardrails AND the three suggestions read like a real rep
(uneven, varied openers), not AI-symmetric.

## Order of operations on the single GH200

One GPU, so serialize the two big models: **serve teacher → label everything → kill teacher → train student
→ merge → serve student → eval.** They don't co-reside.

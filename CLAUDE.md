# CLAUDE.md — ChadGPT / turn_mom (v1 rebuild)

Fine-tune a **3-suggestion `RecommendationResponse` CSS agent** for New American Funding (NAF):
given a CONTEXT block, output exactly 3 pick-able SMS replies + confidence. **Full restart** — the
old single-reply MVP is archived at `~/turn_mom_archive_<ts>/`. Read `README.md` for the runbook.

## What this is (and isn't)
- **Target:** `system`(css prompt) + `user`(CONTEXT block) → `assistant` `{"recommendations":[{suggested_message,confidence}x3]}`.
- **Not** a single free-text reply (that was the MVP). The product is always 3 suggestions with stage logic
  (qualifying / confirming / eligible / ineligible + edge cases greeting/casual/logistics/offtopic).
- The authoritative behavior spec is `prompts/css_system_prompt.txt` (voice rules + stage rules + guardrails).

## Models
- **Student:** `unsloth/Mistral-Small-24B-Instruct-2501`, bf16 LoRA via Unsloth. Chosen over Qwen3.5 to avoid
  thinking-mode / VL-tokenizer / hybrid-arch kernel pain from the MVP.
- **Teacher (synthetic labeler):** local `Qwen2.5-72B-Instruct` (or Llama-3.3-70B) on the GH200, vLLM fp8, :8001.

## Data (R2 bucket `chadgpt-data`)
- **`Bonzo/` (21,307 json)** — real Agent↔Client **SMS**. `Bonzo/Clean Redacted/` has PII tokens
  (`{NAME_GIVEN}` …). **This is the voice source** (already texting register).
- **`{March,April,May} Transcripts/` (~4,800 txt)** — noisy ASR **call** transcripts (CSS↔customer).
  Used only for flow/stage coverage; teacher rewrites them into text register. This is the source of the
  old "everything sounds like a call" complaint — never treat transcripts as voice ground truth.
- Ignore for now: `Call Recordings`, `*Recording`, `pipeline-output`, `reports`.

## Pipeline (scripts/)
`clean_bonzo.py` / `clean_transcript.py` → `{file,source,turns}` · `build_label_tasks.py` (Stream A: real
convos → per-agent-turn tasks w/ gold reply) · `gen_scenarios.py` (Stream B: deterministic screening oracle
→ stage-coverage CONTEXT blocks) · `label_with_teacher.py` (teacher infers CONTEXT + writes 3 suggestions,
one anchored on the real reply; `--dry-run` stubs it for local testing) · `audit_gate.py` (deterministic
guardrail drop) · `to_sft.py` (→ chat messages train/val). Train: `train/unsloth_train.py`. Serve:
`serve/serve.sh` (student), `serve/serve_teacher.sh`. Eval: `eval/run_eval.py` over `eval/cases.jsonl`.

## Screening oracle (the 4 keys, in order)
`property_use, bankruptcy_past_3yr, foreclosure_past_2yr, late_payments_past_12mo`. Ineligible if any of the
last three is "yes" (reasons: `bankruptcy_within_3yr` / `foreclosure_within_2yr` / `late_mortgage_within_12mo`);
`property_use` is informational, never disqualifying. Kept in `gen_scenarios.py` — keep it in sync with the prompt.

## Machines
- **Local** (`~/Documents/turn_mom`): build/test scripts on 1–3 samples only (`--dry-run`, `_samples/`). Never bulk-download 21k files here.
- **GH200 box** (`/var/turn-mom`, root@GPU, 45.76.248.215): ~95GB HBM, aarch64/Grace, CUDA 13. All real pull/label/train/serve/eval runs. One GPU → serialize teacher then student (they don't co-reside).

## Gotchas
- **CPU-only torch on aarch64** = training on CPU (35s/it). `pip install torch --index-url https://download.pytorch.org/whl/cu128`, verify `torch.cuda.is_available()`.
- **One GH200**: serve teacher → label everything → **kill teacher** → train → merge → serve student. 72B + 24B do not co-reside.
- **Teacher JSON drift**: `label_with_teacher.py` parses the outermost `{...}` and requests `response_format=json_object`. Always dry-run a `--limit 200` and eyeball voice before a full label run.
- **Assistant-only loss** (`train_on_responses_only`, markers `[INST]`/`[/INST]`) — don't train on the long fixed system prompt.
- **R2 creds were pasted in chat — rotate them.**

## Conventions
- Redaction placeholders: curly tokens `{NAME_GIVEN} {PHONE_NUMBER} {MONEY} {ORGANIZATION} ...` (already in Clean Redacted).
- `source` tag on every cleaned convo (`bonzo` | `transcript`) — voice sampling prefers `bonzo`.
- CONTEXT block fields (exact order): Stage, Next question key, Answers collected, Ineligibility reason, Client's latest message, Is first message.

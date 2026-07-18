# ChadGPT — DiffusionGemma full-SFT (NeMo AutoModel) runbook

Clone → build the MAOS-first dataset → full-SFT DiffusionGemma-26B → export → inference.

## Which config (hardware)
Full SFT of `google/diffusiongemma-26B-A4B-it` (26B MoE) needs `ep_size: 8` → an 8-GPU node, and 26B
full FT (fp32 master + AdamW states) can't fit fewer GPUs. **We have ONE GH200**, so:
- **→ use `nemo/diffusion_gemma_lora_1gpu.yaml`** (LoRA, `ep_size: 1`, bf16 base). Fits ~1 GH200
  (the prior team ran the non-diffusion Gemma-26B LoRA at ~73 GB / 94 GB — this mirrors it).
- `nemo/diffusion_gemma_sft.yaml` (full SFT) is kept for **if you ever get an 8-GPU node.**
- Proven fallback if DiffusionGemma is troublesome: our **Mistral-24B LoRA** (`train/unsloth_train.py`).

## 0. Install
```bash
pip install -U "transformers>=5.11.0" nemo-automodel torchdata
# finetune.py + the recipe classes live in the Automodel repo — clone it next to this one:
git clone https://github.com/NVIDIA-NeMo/Automodel.git
# (or use NVIDIA's NeMo container, which ships both)
```

## 1. Data — get the sources
```bash
# Bonzo (redacted, voice source) from R2:
bash scripts/01_pull_data.sh data/raw            # -> data/raw/bonzo_clean_redacted + transcripts
# the big Bonzo dump for edge-case mining (35k threads) -> data/raw/bonzo_new/conversations.jsonl
```

## 2. Build the MAOS-first dataset  → NeMo chat-JSONL
The behavior we train toward is the **MAOS ideal**, not recorded CSS. Streams: MAOS synthetic (NAF
playbooks) + real MAOS edge-cases mined from Bonzo + screening oracle + demoted Bonzo voice. See
`Musings/ChadGPT - v1 DATA SPEC (MAOS-first).md`.
```bash
# teacher generates the ideal 3-suggestion labels; strong teacher matters (emotion needs warmth)
TEACHER=Qwen/Qwen2.5-32B-Instruct \
BONZO_DUMP=data/raw/bonzo_new/conversations.jsonl \
  bash nemo/build_dataset.sh
# -> data/chat_train.jsonl + data/chat_val.jsonl  (OpenAI {"messages":[...]} — ChatDataset format)
```
> The teacher runs locally (vLLM) — serve it first, or point `label_offline.py` at any OpenAI endpoint.
> **Redaction note:** `mine_bonzo.py` regex-redaction is high-recall, not compliance-grade — run a full
> redaction pass (prior `redact_bonzo` / NER) before shipping a customer-facing model.

## The two-model plan (same `data/chat_train.jsonl` feeds both)
You're training **DiffusionGemma first, then Gemma-4** — the dataset is identical, only the tooling
differs. Train, eval (the MAOS cases in `eval/cases.jsonl`), compare.
| Model | Tool | Command |
|---|---|---|
| **DiffusionGemma-26B-a4b** (1st, experimental) | NeMo | `nemo/diffusion_gemma_lora_1gpu.yaml` (below) |
| **Gemma-4-26B-a4b** (2nd, prior team's proven model) | Unsloth | `train/unsloth_train.py --model unsloth/gemma-4-26b-a4b-it` |
| Mistral-24B (fallback, ours) | Unsloth | `train/unsloth_train.py --model unsloth/Mistral-Small-24B-Instruct-2501` |

## 3. Train — LoRA on 1 GH200
```bash
torchrun --standalone --nproc-per-node=1 \
    Automodel/examples/dllm_sft/finetune.py \
    -c nemo/diffusion_gemma_lora_1gpu.yaml
# adapter -> dllm_checkpoints/chadgpt_diffusion_gemma_lora/
```
**Then Gemma-4** (after testing DiffusionGemma), same dataset, via Unsloth:
```bash
python3 train/unsloth_train.py --data data/chat_train.jsonl --val data/chat_val.jsonl \
    --model unsloth/gemma-4-26b-a4b-it --bsz 2 --grad-accum 8   # -> out/gemma_naf_lora
```
(auto-detects Gemma → FastModel, `<start_of_turn>` masking, vision layers off. Box-validate first steps;
if OOM, QLoRA / drop seq_length.)

### On epochs — don't hard-set a number
`--epochs` is a **ceiling (default 3)**; the trainer evals every `--eval-steps` (100), keeps the **best
checkpoint by eval_loss**, and **early-stops** (`--patience` 3). That's the real overfitting guard on a
26B — watch the eval-loss curve, not the epoch count (the prior Gemma run flattened by ~epoch 2). For
**DiffusionGemma** (NeMo has no auto early-stop): `val_every_steps: 100` + `ckpt_every_steps: 200` — let
it run, then **probe the checkpoints** with `eval/run_eval.py` and keep the best. Set its `max_steps`
to ~2-3× `ceil(train_examples / 8)` for your actual dataset size.

Config notes (DiffusionGemma): `torch_dtype: bfloat16` (fp32 26B won't fit one GPU), `ep_size: 1`, `seq_length: 2048`
(long system prompt), `canvas_length: 256` (response region — our JSON fits), `mask_history: true`
(assistant-only loss), LoRA `dim 16 / alpha 32`. **If it OOMs:** shorten `seq_length` to 1536, or add
`load_in_4bit: true` under `model:` (QLoRA), or reduce `local_batch_size`. Watch `nvidia-smi` on the
first steps.

## 4. Export → HF model
LoRA run saves the **adapter** to `dllm_checkpoints/chadgpt_diffusion_gemma_lora/`. For inference, either
load base + adapter together, or **merge** the adapter into the base and save a consolidated
`transformers>=5.11`-loadable DiffusionGemma directory. NeMo AutoModel's merge/export command is
version-specific — check the `dllm` guide for your installed version (look for a `merge`/`export` util,
or set `save_consolidated: true`). Confirm the merged dir loads before promising a serving demo.

## 5. Inference (next phase)
DiffusionGemma is a **block-diffusion** LM (iterative denoising, not left-to-right), so serving differs
from a normal autoregressive model — use the Automodel `dllm` generation path / NeMo's diffusion decode,
not plain vLLM autoregressive serving (verify vLLM diffusion-LM support before assuming it works).

## Honest open items (verify on the box, don't assume)
- **8-GPU node** is mandatory for full SFT — confirm you have one.
- **`google/diffusiongemma-26B-A4B-it`** must be gated-accessible on HF + downloadable (~49 GB+).
- **transformers 5.11** is very new — pin it; it's required to even load the checkpoint.
- **Export + diffusion inference** paths are the least-documented parts of the recipe — budget time to
  validate them against the installed Automodel version before promising an inference demo.
- DiffusionGemma full-SFT for a short-JSON reco task is **experimental** — the proven fallback is our
  Mistral-24B LoRA path, still in this repo (`train/unsloth_train.py` + `serve/serve.sh`).

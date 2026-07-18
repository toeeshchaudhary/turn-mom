# ChadGPT — DiffusionGemma full-SFT (NeMo AutoModel) runbook

Clone → build the MAOS-first dataset → full-SFT DiffusionGemma-26B → export → inference.

## ⚠️ Hardware requirement (read first)
Full SFT of **`google/diffusiongemma-26B-A4B-it`** (26B MoE) uses **expert parallelism `ep_size: 8`
→ an 8-GPU node.** A single GH200 (~98 GB) **cannot** do full SFT of a 26B model (weights +
fp32 master + AdamW states ≈ hundreds of GB even sharded). Options:
- **8×GPU node** → use `nemo/diffusion_gemma_sft.yaml` as-is (this runbook).
- **1 GPU** → switch to the LoRA recipe (`diffusion_gemma_lora.yaml` in the Automodel repo) and set
  `ep_size: 1`, or fall back to our proven **Mistral-24B LoRA** path (`train/unsloth_train.py`).

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

## 3. Train (full SFT, 8 GPUs)
```bash
# copy our config + data where the Automodel launcher can see them, then:
torchrun --standalone --nproc-per-node=8 \
    Automodel/examples/dllm_sft/finetune.py \
    -c nemo/diffusion_gemma_sft.yaml
# checkpoints -> dllm_checkpoints/chadgpt_diffusion_gemma_sft/  (safetensors, consolidated)
```
Config notes (`nemo/diffusion_gemma_sft.yaml`): `seq_length: 2048` (our long CSS+MAOS system prompt
needs the room), `canvas_length: 256` (response region — our JSON fits), `mask_history: true` (loss on
the assistant turn only), `max_steps: 2000` (tune to ~2–3 passes over your set), `save_consolidated: true`
(HF-loadable export).

## 4. Export → HF model
With `model_save_format: safetensors` + `save_consolidated: true` + `enable_hf_state_dict_adapter: true`,
the final consolidated checkpoint in `dllm_checkpoints/.../` is loadable by `transformers>=5.11` as
DiffusionGemma. Point inference/serving at that directory. (NeMo AutoModel export/convert specifics are
version-dependent — confirm against the Automodel `dllm` guide for your installed version.)

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

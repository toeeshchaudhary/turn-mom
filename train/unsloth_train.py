"""Unsloth bf16-LoRA SFT on the MAOS RecommendationResponse task.

Trains the student to emit the 3-suggestion JSON given (system prompt + CONTEXT block),
assistant-only loss. Handles two model families from the SAME chat-messages dataset:

  * Mistral-Small-24B  (default)      -> FastLanguageModel, [INST]/[/INST] masking
  * Gemma-4-26B-a4b    (--model ...)  -> FastModel, <start_of_turn> masking, vision layers OFF
    (the exact model + recipe the prior team proved: r16/a32, bf16 no-quant, ~73/94GB on a GH200)

DiffusionGemma is NOT trained here — it uses NeMo (nemo/diffusion_gemma_lora_1gpu.yaml).

Usage:
  python3 unsloth_train.py --data data/chat_train.jsonl --val data/chat_val.jsonl \
      --model unsloth/Mistral-Small-24B-Instruct-2501 --bsz 4 --epochs 1
  python3 unsloth_train.py --data data/chat_train.jsonl --val data/chat_val.jsonl \
      --model unsloth/gemma-4-26b-a4b-it --bsz 2 --grad-accum 8 --epochs 2   # Gemma-4 path
"""
import argparse

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--model", default="unsloth/Mistral-Small-24B-Instruct-2501")
    ap.add_argument("--out", default=None)
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    is_gemma = "gemma" in args.model.lower()
    out = args.out or (f"out/{'gemma' if is_gemma else 'mistral'}_naf_lora")

    if is_gemma:
        # Gemma-4 is multimodal MoE -> FastModel, keep vision layers frozen (text-only data)
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(
            model_name=args.model, max_seq_length=args.maxlen, load_in_4bit=False, dtype=None)
        model = FastModel.get_peft_model(
            model, r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
            target_modules=TARGET_MODULES,
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth", random_state=3407)
        instruction_part, response_part = "<start_of_turn>user\n", "<start_of_turn>model\n"
    else:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.maxlen, load_in_4bit=False, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
            target_modules=TARGET_MODULES,
            use_gradient_checkpointing="unsloth", random_state=3407)
        instruction_part, response_part = "[INST]", "[/INST]"

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = load_dataset("json", data_files=args.data, split="train").map(fmt)
    val_ds = load_dataset("json", data_files=args.val, split="train").map(fmt) if args.val else None

    cfg = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        bf16=True,
        max_length=args.maxlen,
        dataset_text_field="text",
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tok,
                         train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[train] adapter saved -> {out}")


if __name__ == "__main__":
    main()

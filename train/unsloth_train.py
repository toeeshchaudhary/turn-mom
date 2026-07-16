#!/usr/bin/env python3
"""Unsloth bf16-LoRA SFT of Mistral-Small-24B on the RecommendationResponse task.

Trains the student to emit the 3-suggestion JSON given (system prompt + CONTEXT
block). Uses assistant-only loss so the model learns to PRODUCE the JSON, not to
reproduce the (very long, fixed) system prompt.

Runs on the GH200 (bf16, ~24B in 16-bit ~= 48GB weights + LoRA + activations,
comfortable in 95GB HBM). No thinking-mode / VL-tokenizer headaches (that's the
whole reason we picked Mistral over Qwen3.5).

Usage (on the box):
  python3 unsloth_train.py --data ../data/sft/train.jsonl --val ../data/sft/val.jsonl \
      --model unsloth/Mistral-Small-24B-Instruct-2501 --bsz 8 --epochs 3
"""
import argparse
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--model", default="unsloth/Mistral-Small-24B-Instruct-2501")
    ap.add_argument("--out", default="out/mistral_naf_lora")
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.maxlen,
        load_in_4bit=False,          # bf16 LoRA, not QLoRA — GH200 has the HBM
        dtype=None,                  # auto -> bf16 on Hopper
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0,
    )

    def fmt(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = load_dataset("json", data_files=args.data, split="train").map(fmt)
    val_ds = (load_dataset("json", data_files=args.val, split="train").map(fmt)
              if args.val else None)

    cfg = SFTConfig(
        output_dir=args.out,
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

    # assistant-only loss: mask everything up to and including [/INST]
    trainer = train_on_responses_only(
        trainer, instruction_part="[INST]", response_part="[/INST]")

    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[train] adapter saved -> {args.out}")


if __name__ == "__main__":
    main()

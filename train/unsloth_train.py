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
    ap.add_argument("--epochs", type=float, default=3.0, help="CEILING; early stopping decides the real number")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3, help="evals w/o eval_loss improvement before stopping")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback
    from unsloth.chat_templates import train_on_responses_only
    is_gemma = "gemma" in args.model.lower()
    out = args.out or (f"out/{'gemma' if is_gemma else 'mistral'}_naf_lora")
    if is_gemma:
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
    has_val = val_ds is not None
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
        weight_decay=0.01,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps if has_val else None,
        save_strategy="steps" if has_val else "epoch",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)] if has_val else []
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=train_ds,
                         eval_dataset=val_ds, args=cfg, callbacks=callbacks)
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[train] adapter saved -> {out}")
if __name__ == "__main__":
    main()

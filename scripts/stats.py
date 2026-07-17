"""Print training + dataset stats for the ChadGPT run. No deps beyond stdlib.
Reads whatever artifacts exist and degrades gracefully if some are missing:
  out/mistral_naf_lora/adapter_config.json         (LoRA config)
  out/mistral_naf_lora/adapter_model.safetensors   (adapter param count, from header)
  out/mistral_naf_lora/checkpoint-*/trainer_state.json  (loss history, runtime)
  data/interim/all_tasks.jsonl, labeled.jsonl, labeled.ok.jsonl, rejects.jsonl
  data/sft/train.jsonl, val.jsonl, train_small.jsonl
  data/interim/bonzo_clean.jsonl, calls_clean.jsonl
Usage:  python3 scripts/stats.py            (run from repo root)
        python3 scripts/stats.py --root .   (or point at another checkout)
"""
import argparse, glob, json, os, struct
SPARK = "▁▂▃▄▅▆▇█"
def wc(path):
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n
def fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)
def sparkline(vals, width=60):
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / rng * 7))] for v in vals)
def safetensors_params(path):
    try:
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        total = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = meta.get("shape", [])
            p = 1
            for s in shape:
                p *= s
            total += p
        return total
    except Exception:
        return None
def latest_trainer_state(root):
    cands = glob.glob(os.path.join(root, "out/mistral_naf_lora/checkpoint-*/trainer_state.json"))
    cands += [os.path.join(root, "out/mistral_naf_lora/trainer_state.json")]
    cands = [c for c in cands if os.path.exists(c)]
    if not cands:
        return None
    cands.sort(key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]) if "checkpoint-" in p else 0)
    return json.load(open(cands[-1]))
def line(label, value):
    print(f"  {label:<20}: {value}")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    R = args.root
    print("═" * 60)
    print("  ChadGPT — Training & Dataset Stats")
    print("═" * 60)
    print("\nMODEL")
    cfg_p = os.path.join(R, "out/mistral_naf_lora/adapter_config.json")
    if os.path.exists(cfg_p):
        cfg = json.load(open(cfg_p))
        line("base model", cfg.get("base_model_name_or_path", "?"))
        line("LoRA r / alpha", f"{cfg.get('r')} / {cfg.get('lora_alpha')}")
        tm = cfg.get("target_modules", [])
        line("target modules", f"{len(tm)}  ({', '.join(tm)})" if isinstance(tm, list) else tm)
        line("dropout", cfg.get("lora_dropout"))
    else:
        line("adapter_config", "not found (train first)")
    ad = safetensors_params(os.path.join(R, "out/mistral_naf_lora/adapter_model.safetensors"))
    if ad:
        line("adapter params", f"{ad/1e6:.1f}M trainable (LoRA)")
    print("\nTRAINING")
    st = latest_trainer_state(R)
    if st:
        hist = st.get("log_history", [])
        losses = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
        finals = [h for h in hist if "train_runtime" in h]
        lrs = [h["learning_rate"] for h in hist if "learning_rate" in h]
        line("epochs", round(st.get("epoch", 0), 2))
        line("global steps", fmt(st.get("global_step", "?")))
        if losses:
            line("first / last loss", f"{losses[0][1]:.4f}  →  {losses[-1][1]:.4f}")
        if evals:
            line("eval loss", f"{evals[-1]:.4f}")
        if finals:
            fr = finals[-1]
            rt = fr.get("train_runtime", 0)
            line("train runtime", f"{int(rt//3600)}h {int(rt%3600//60)}m {int(rt%60)}s")
            line("throughput", f"{fr.get('train_samples_per_second','?')} samples/s, "
                              f"{fr.get('train_steps_per_second','?')} steps/s")
        if lrs:
            line("peak LR", f"{max(lrs):.2e}")
        if losses:
            print("\n  loss curve (train):")
            print("   ", sparkline([l for _, l in losses]))
            print(f"    {losses[0][1]:.2f}{'':<52}{losses[-1][1]:.2f}")
    else:
        line("trainer_state", "not found (no checkpoint yet)")
    print("\nDATASET  (pipeline funnel)")
    counts = [
        ("bonzo convos", "data/interim/bonzo_clean.jsonl"),
        ("transcript convos", "data/interim/calls_clean.jsonl"),
        ("label tasks", "data/interim/all_tasks.jsonl"),
        ("teacher-labeled", "data/interim/labeled.jsonl"),
        ("audit passed", "data/interim/labeled.ok.jsonl"),
        ("audit rejected", "data/interim/rejects.jsonl"),
        ("SFT train (full)", "data/sft/train.jsonl"),
        ("SFT train (used)", "data/sft/train_small.jsonl"),
        ("SFT val", "data/sft/val.jsonl"),
    ]
    labeled = wc(os.path.join(R, "data/interim/labeled.jsonl"))
    passed = wc(os.path.join(R, "data/interim/labeled.ok.jsonl"))
    for label, path in counts:
        n = wc(os.path.join(R, path))
        extra = ""
        if label == "audit passed" and labeled and passed:
            extra = f"   ({passed/labeled*100:.0f}% pass rate)"
        line(label, (fmt(n) + extra) if n is not None else "—")
    print("\n" + "═" * 60)
if __name__ == "__main__":
    main()

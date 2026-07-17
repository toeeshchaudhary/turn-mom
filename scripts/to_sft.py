"""Audited labels -> chat `messages` SFT file for Unsloth/TRL.
Each training record:
  system    = the full CSS system prompt (prompts/css_system_prompt.txt)
  user      = the rendered CONTEXT block
  assistant = the RecommendationResponse JSON (exactly what we want the model to emit)
We render the assistant target as compact, deterministic JSON so the student
learns a stable output shape:
  {"recommendations":[{"suggested_message":"...","confidence":"high"}, x3]}
Also shuffles and does a train/val split.
Usage:
  python3 to_sft.py labeled.audited.jsonl --out-train data/sft/train.jsonl \
      --out-val data/sft/val.jsonl --val-frac 0.03
"""
import argparse, json, os, random, sys
SYS_PROMPT = open(os.path.join(os.path.dirname(__file__), "..", "prompts",
                                "css_system_prompt.txt"), encoding="utf-8").read()
CTX_FIELDS = ["Stage", "Next question key", "Answers collected", "Ineligibility reason",
              "Agent name", "Client name", "Client's latest message", "Is first message"]
def ctx_block(ctx):
    lines = ["--- CURRENT CONTEXT ---"]
    for k in CTX_FIELDS:
        lines.append(f"{k}: {ctx.get(k, 'N/A')}")
    lines.append("---")
    return "\n".join(lines)
def assistant_json(recs):
    clean = [{"suggested_message": r["suggested_message"].strip(),
              "confidence": r["confidence"]} for r in recs]
    return json.dumps({"recommendations": clean}, ensure_ascii=False)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labeled")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = []
    with open(args.labeled, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            user = ctx_block(rec["context"])
            if rec.get("directive"):   # MAOS mode example: append the same directive the orchestrator sends
                user = user + "\n\n" + rec["directive"]
            rows.append({"messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant_json(rec["recommendations"])},
            ]})
    random.Random(args.seed).shuffle(rows)
    nval = max(1, int(len(rows) * args.val_frac)) if rows else 0
    val, train = rows[:nval], rows[nval:]
    for path, data in [(args.out_train, train), (args.out_val, val)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as o:
            for r in data:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[to_sft] train={len(train)} val={len(val)}", file=sys.stderr)
if __name__ == "__main__":
    main()

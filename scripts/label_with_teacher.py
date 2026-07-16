#!/usr/bin/env python3
"""Label tasks with the teacher LLM -> RecommendationResponse training examples.

Handles both streams:
  Stream A (real convos): task has {history, gold_reply}. The teacher infers the
    CONTEXT block from history AND returns 3 suggestions, one of which is a lightly
    cleaned, text-register version of the real rep reply (anchors genuine voice;
    also rewrites noisy ASR transcript replies into clean SMS text).
  Stream B (scenarios): task has {context}. The teacher writes 3 suggestions for
    the given, deterministic CONTEXT block.

Output record (one per task):
  {context:{...six fields...}, recommendations:[{suggested_message, confidence}x3],
   meta:{source, file, kind, case}}

Use --dry-run to fabricate deterministic stub labels (no teacher needed) so you
can smoke-test the plumbing locally before spinning up the GPU teacher.

Usage:
  TEACHER_BASE_URL=http://localhost:8001/v1 TEACHER_MODEL=teacher \
    python3 label_with_teacher.py tasks.jsonl --out labeled.jsonl
  python3 label_with_teacher.py tasks.jsonl --out labeled.jsonl --dry-run   # local test
"""
import argparse, json, os, re, sys

SYS_PROMPT = open(os.path.join(os.path.dirname(__file__), "..", "prompts",
                                "css_system_prompt.txt"), encoding="utf-8").read()

CTX_FIELDS = ["Stage", "Next question key", "Answers collected",
              "Ineligibility reason", "Client's latest message", "Is first message"]


def render_history(history):
    lines = []
    for t in history:
        who = "CLIENT" if t["role"] == "client" else "REP"
        lines.append(f"{who}: {t['text']}")
    return "\n".join(lines)


def render_ctx_block(ctx):
    lines = ["--- CURRENT CONTEXT ---"]
    for k in CTX_FIELDS:
        lines.append(f"{k}: {ctx.get(k, 'N/A')}")
    lines.append("---")
    return "\n".join(lines)


def stream_a_user(task):
    return (
        f"--- CONVERSATION SO FAR ---\n{render_history(task['history'])}\n\n"
        f"--- WHAT THE REP ACTUALLY SENT NEXT ---\n{task['gold_reply']}\n\n"
        "TASK:\n"
        "1. Infer the CONTEXT block for the rep's next turn (fields: Stage, "
        "Next question key, Answers collected, Ineligibility reason, "
        "Client's latest message, Is first message).\n"
        "2. Produce exactly 3 suggested messages following ALL your voice and "
        "stage rules. Make exactly ONE of the three a lightly-cleaned, natural "
        "text-message version of what the rep actually sent — keep their meaning "
        "and voice, fix typos/ASR errors, strip any phone-call phrasing. The other "
        "two must be genuine, meaningfully different alternatives.\n"
        'Return ONLY JSON: {"context": {<the six fields>}, "recommendations": '
        '[{"suggested_message": str, "confidence": "high|medium|low"}, x3]}'
    )


def stream_b_user(ctx):
    return (
        render_ctx_block(ctx) + "\n\n"
        "TASK: produce exactly 3 suggested messages following ALL your voice and "
        "stage rules for this context.\n"
        'Return ONLY JSON: {"recommendations": [{"suggested_message": str, '
        '"confidence": "high|medium|low"}, x3]}'
    )


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    # grab the outermost {...}
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("no json object in teacher output")
    return json.loads(text[s:e + 1])


def dry_label(task):
    """Deterministic stub so the pipeline is testable with no teacher."""
    if task.get("kind") == "scenario":
        ctx = task["context"]
        base = ctx["Client's latest message"]
    else:
        ctx = {f: "N/A" for f in CTX_FIELDS}
        ctx["Stage"] = "qualifying"
        ctx["Client's latest message"] = task["history"][-1]["text"][:60]
        ctx["Is first message"] = "yes" if task.get("is_first") else "no"
        base = task["gold_reply"]
    recs = [
        {"suggested_message": base[:160], "confidence": "medium"},
        {"suggested_message": "so what's the best number to reach you at?", "confidence": "low"},
        {"suggested_message": "happy to keep this moving whenever you are", "confidence": "low"},
    ]
    return {"context": ctx, "recommendations": recs}


def label(task, dry):
    if dry:
        out = dry_label(task)
    else:
        from teacher_client import chat
        if task.get("kind") == "scenario":
            content = chat([{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": stream_b_user(task["context"])}])
            parsed = parse_json(content)
            out = {"context": task["context"], "recommendations": parsed["recommendations"]}
        else:
            content = chat([{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": stream_a_user(task)}])
            out = parse_json(content)
    out["meta"] = {
        "source": task.get("source", "scenario"),
        "file": task.get("file"),
        "kind": task.get("kind", "convo"),
        "case": task.get("case"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    n = fail = 0
    with open(args.tasks, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            try:
                out.write(json.dumps(label(task, args.dry_run)) + "\n")
                n += 1
            except Exception as e:
                fail += 1
                print(f"[fail #{i}] {e}", file=sys.stderr)
    print(f"[label] wrote {n} labeled, {fail} failed -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

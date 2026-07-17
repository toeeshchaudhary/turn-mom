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
CTX_FIELDS = ["Stage", "Next question key", "Answers collected", "Ineligibility reason",
              "Agent name", "Client name", "Client's latest message", "Is first message"]
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
NO_TOKENS = ("NEVER output redaction placeholders like {NAME}, {NAME_GIVEN}, or "
             "{PHONE_NUMBER}. Use the real Agent name / Client name given below, or "
             "a natural first name — never a bracketed token.")
NO_BACKCHANNEL = (
    "This is a TEXT message, not a phone call. Do NOT open messages with call-style "
    "acknowledgement fillers ('Got it', 'Alright', 'Okay so', 'Great', 'Sounds good', "
    "'Perfect', 'Glad to hear', 'Awesome', 'I understand', 'Gotcha'). At MOST one of the "
    "three suggestions may briefly acknowledge in passing; the other two must get straight "
    "to the point with no filler. Vary the openers — they must not all start the same way, "
    "and none should read like a rep verbally nodding along on a call.")
def stream_a_user(task):
    return (
        f"Agent name: {task.get('agent_name','Alex')}\n"
        f"Client name: {task.get('client_name','there')}\n\n"
        f"--- CONVERSATION SO FAR ---\n{render_history(task['history'])}\n\n"
        f"--- WHAT THE REP ACTUALLY SENT NEXT ---\n{task['gold_reply']}\n\n"
        "TASK:\n"
        "1. Infer the CONTEXT block for the rep's next turn (fields: Stage, "
        "Next question key, Answers collected, Ineligibility reason, Agent name, "
        "Client name, Client's latest message, Is first message).\n"
        "2. Produce exactly 3 suggested messages following ALL your voice and "
        "stage rules. Make exactly ONE of the three a lightly-cleaned, natural "
        "text-message version of what the rep actually sent — keep their meaning "
        "and voice, fix typos/ASR errors, strip any phone-call phrasing. The other "
        "two must be genuine, meaningfully different alternatives.\n"
        f"{NO_TOKENS}\n{NO_BACKCHANNEL}\n"
        'Return ONLY JSON: {"context": {<the eight fields>}, "recommendations": '
        '[{"suggested_message": str, "confidence": "high|medium|low"}, x3]}'
    )
def stream_b_user(ctx):
    return (
        render_ctx_block(ctx) + "\n\n"
        "TASK: produce exactly 3 suggested messages following ALL your voice and "
        "stage rules for this context.\n"
        f"{NO_TOKENS}\n{NO_BACKCHANNEL}\n"
        'Return ONLY JSON: {"recommendations": [{"suggested_message": str, '
        '"confidence": "high|medium|low"}, x3]}'
    )
def stream_maos_user(task):
    parts = [render_ctx_block(task["context"]), "", task["directive"]]
    if task.get("anchor"):
        parts.append(
            '\nAn experienced NAF rep would reply something like: "' + task["anchor"] + '". '
            "Make exactly ONE of your 3 suggestions a natural text-message version of that "
            "(keep the meaning, text register). The other two are genuine alternatives that "
            "follow the SITUATION above.")
    parts.append("\nProduce exactly 3 suggested messages that follow the SITUATION directive above "
                 "(it overrides the normal stage behavior).")
    parts.append(f"{NO_TOKENS}\n{NO_BACKCHANNEL}")
    parts.append('Return ONLY JSON: {"recommendations": [{"suggested_message": str, '
                 '"confidence": "high|medium|low"}, x3]}')
    return "\n".join(parts)
def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("no json object in teacher output")
    return json.loads(text[s:e + 1])
def dry_label(task):
    if task.get("kind") == "maos":
        anchor = task.get("anchor") or "whenever you're ready, I'm here to help"
        recs = [{"suggested_message": anchor[:160], "confidence": "medium"},
                {"suggested_message": "no rush at all on my end", "confidence": "low"},
                {"suggested_message": "just let me know how you'd like to go", "confidence": "low"}]
        out = {"context": task["context"], "recommendations": recs,
               "directive": task["directive"], "mode": task["mode"]}
        return out
    if task.get("kind") == "scenario":
        ctx = task["context"]
        base = ctx["Client's latest message"]
    else:
        ctx = {f: "N/A" for f in CTX_FIELDS}
        ctx["Stage"] = "qualifying"
        ctx["Agent name"] = task.get("agent_name", "Alex")
        ctx["Client name"] = task.get("client_name", "there")
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
        if task.get("kind") == "maos":
            content = chat([{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": stream_maos_user(task)}])
            out = {"context": task["context"], "recommendations": parse_json(content)["recommendations"],
                   "directive": task["directive"], "mode": task["mode"]}
        elif task.get("kind") == "scenario":
            content = chat([{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": stream_b_user(task["context"])}])
            parsed = parse_json(content)
            out = {"context": task["context"], "recommendations": parsed["recommendations"]}
        else:
            content = chat([{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": stream_a_user(task)}])
            out = parse_json(content)
            out.setdefault("context", {})
            out["context"]["Agent name"] = task.get("agent_name", "Alex")
            out["context"]["Client name"] = task.get("client_name", "there")
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
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent teacher requests (vLLM batches these)")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    tasks = []
    with open(args.tasks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
            if args.limit and len(tasks) >= args.limit:
                break
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = fail = done = 0
    total = len(tasks)
    with open(args.out, "w", encoding="utf-8") as out,         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(label, t, args.dry_run): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            done += 1
            try:
                out.write(json.dumps(fut.result()) + "\n")
                out.flush()
                n += 1
            except Exception as e:
                fail += 1
                print(f"[fail #{futs[fut]}] {e}", file=sys.stderr)
            if done % 200 == 0:
                print(f"[label] {done}/{total} ({n} ok, {fail} fail)", file=sys.stderr)
    print(f"[label] wrote {n} labeled, {fail} failed -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

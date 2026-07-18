import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_with_teacher import (SYS_PROMPT, stream_a_user, stream_b_user, stream_maos_user, parse_json)
MAX_HISTORY_TURNS = 30
def build_messages(task):
    kind = task.get("kind")
    if kind == "maos":
        user = stream_maos_user(task)
    elif kind == "scenario":
        user = stream_b_user(task["context"])
    else:
        h = task.get("history", [])
        if len(h) > MAX_HISTORY_TURNS:
            task = {**task, "history": h[-MAX_HISTORY_TURNS:]}
        user = stream_a_user(task)
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]
def postprocess(task, text):
    kind = task.get("kind")
    if kind == "maos":
        out = {"context": task["context"], "recommendations": parse_json(text)["recommendations"],
               "directive": task["directive"], "mode": task["mode"]}
    elif kind == "scenario":
        out = {"context": task["context"], "recommendations": parse_json(text)["recommendations"]}
    else:
        out = parse_json(text)
        out.setdefault("context", {})
        out["context"]["Agent name"] = task.get("agent_name", "Alex")
        out["context"]["Client name"] = task.get("client_name", "there")
    out["meta"] = {"source": task.get("source", "scenario"), "file": task.get("file"),
                   "kind": task.get("kind", "convo"), "case": task.get("case")}
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--maxlen", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tasks = []
    with open(args.tasks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
            if args.limit and len(tasks) >= args.limit:
                break
    print(f"[label_offline] {len(tasks)} tasks; loading {args.model} ...", file=sys.stderr)
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.maxlen,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    convs = [build_messages(t) for t in tasks]
    tok = llm.get_tokenizer()
    sys_len = len(tok(SYS_PROMPT).input_ids)
    budget = args.maxlen - args.max_tokens - 256   
    keep_tasks, keep_convs, dropped = [], [], 0
    for t, c in zip(tasks, convs):
        if sys_len + len(tok(c[1]["content"]).input_ids) <= budget:
            keep_tasks.append(t)
            keep_convs.append(c)
        else:
            dropped += 1
    print(f"[label_offline] dropped {dropped} over-length prompts; "
          f"labeling {len(keep_tasks)}", file=sys.stderr)
    outs = llm.chat(keep_convs, sp)   
    n = fail = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for task, o in zip(keep_tasks, outs):
            try:
                out.write(json.dumps(postprocess(task, o.outputs[0].text)) + "\n")
                n += 1
            except Exception as e:
                fail += 1
                if fail <= 10:
                    print(f"[parse-fail] {e}", file=sys.stderr)
    print(f"[label_offline] wrote {n} labeled, {fail} parse-fails -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

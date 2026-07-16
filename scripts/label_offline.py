#!/usr/bin/env python3
"""Offline BATCHED teacher labeling with vLLM (no HTTP server, no --workers).

This replaces the serve_teacher.sh + label_with_teacher.py (HTTP) path for bulk
runs. It loads the teacher once and hands vLLM ALL prompts at once via llm.chat(),
so vLLM does optimal continuous batching with CUDA graphs — far faster than firing
concurrent HTTP requests at a served model (which stalled on JIT + scheduling on
the GH200). Same prompts, same output format; reuses the builders from
label_with_teacher.py.

Usage (on the box — this OWNS the GPU while it runs; no separate teacher server):
  python3 scripts/label_offline.py data/interim/all_tasks.jsonl \
      --out data/interim/labeled.jsonl --model Qwen/Qwen2.5-14B-Instruct
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_with_teacher import (SYS_PROMPT, stream_a_user, stream_b_user, parse_json)


def build_messages(task):
    user = stream_b_user(task["context"]) if task.get("kind") == "scenario" else stream_a_user(task)
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]


def postprocess(task, text):
    if task.get("kind") == "scenario":
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
    ap.add_argument("--maxlen", type=int, default=8192)
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
    outs = llm.chat(convs, sp)   # vLLM batches all of these internally

    n = fail = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for task, o in zip(tasks, outs):
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

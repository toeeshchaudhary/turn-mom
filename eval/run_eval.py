#!/usr/bin/env python3
"""Send eval CONTEXT blocks to the served student and score schema + guardrails.

For each case: builds [system=CSS prompt, user=CONTEXT block], calls the vLLM
OpenAI endpoint, then reuses the deterministic audit checks to score the reply.
Prints per-case PASS/FAIL and the 3 suggestions so you can eyeball voice quality.

Usage (on the box, after serve.sh is up):
  python3 run_eval.py --base http://localhost:8000/v1 --model naf
"""
import argparse, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from audit_gate import check          # reuse the exact guardrail checks
from to_sft import ctx_block, SYS_PROMPT


def call(base, model, ctx):
    body = json.dumps({
        "model": model, "temperature": 0.7, "max_tokens": 500,
        "messages": [{"role": "system", "content": SYS_PROMPT},
                     {"role": "user", "content": ctx_block(ctx)}],
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="naf")
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.jsonl"))
    args = ap.parse_args()

    passed = total = 0
    for line in open(args.cases, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        total += 1
        raw = call(args.base, args.model, case["context"])
        try:
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            rec = {"context": case["context"], "recommendations": parsed["recommendations"]}
            reason = check(rec)
        except Exception as e:
            reason = f"unparseable JSON: {e}"
        ok = reason is None
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}"
              + ("" if ok else f"  <- {reason}"))
        if ok:
            for r in rec["recommendations"]:
                print(f"    ({r['confidence']}) {r['suggested_message']}")
        else:
            print("    RAW:", raw[:300])
    print(f"\n{passed}/{total} cases passed")


if __name__ == "__main__":
    main()

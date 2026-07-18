import argparse, json, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from audit_gate import check
from to_sft import ctx_block, SYS_PROMPT
from gen_maos_scenarios import D as MODE_DIRECTIVES   
def call(base, model, ctx, directive=""):
    user = ctx_block(ctx) + (("\n\n" + directive) if directive else "")
    body = json.dumps({
        "model": model, "temperature": 0.7, "max_tokens": 500,
        "messages": [{"role": "system", "content": SYS_PROMPT},
                     {"role": "user", "content": user}],
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
        mode = case.get("mode")
        directive = MODE_DIRECTIVES.get(mode, "") if mode else ""
        raw = call(args.base, args.model, case["context"], directive)
        try:
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            rec = {"context": case["context"], "recommendations": parsed["recommendations"], "mode": mode}
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

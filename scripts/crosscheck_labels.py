#!/usr/bin/env python3
"""Crosscheck a labeled dataset against the CSS requirements.

Two layers per record:
  1. DETERMINISTIC  -- reuse audit_gate.check() (exactly-3, confidence values,
     no rate/APR/$, no {TOKENS}, <=3 sentences, no call-filler openers, stage
     guardrails, empathy/stop/logistics mode rules). Cheap, no model.
  2. TEACHER JUDGE  -- send CONTEXT + the 3 suggestions to the local teacher
     (nvidia/Llama-3.3-70B-Instruct-FP8, vLLM :8001) prompted with the SAME
     system spec used to label. It grades voice / stage logic / emotion-first /
     guardrail compliance and returns structured JSON.

Input : one JSON record per line ({context, recommendations, meta[, mode]}).
Output: same records + a "_crosscheck" field {det, judge, ok}.
        --rejects gets only the records that FAIL either layer.

    # local smoke test, no GPU needed
    python scripts/crosscheck_labels.py ~/Downloads/labeled.remaining.jsonl \
        --out /tmp/checked.jsonl --rejects /tmp/rejects.jsonl --limit 3 --dry-run

    # real run against the teacher on the GH200
    TEACHER_BASE_URL=http://localhost:8001/v1 TEACHER_MODEL=teacher \
    python scripts/crosscheck_labels.py data/labeled.remaining.jsonl \
        --out data/labeled.checked.jsonl --rejects data/labeled.rejects.jsonl \
        --workers 16
"""
import argparse, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_gate import check as det_check  # deterministic guardrail layer

_PROMPT_FILE = os.environ.get("TEACHER_PROMPT", os.path.join(
    os.path.dirname(__file__), "..", "prompts", "css_maos_system_prompt.txt"))
SYS_PROMPT = open(_PROMPT_FILE, encoding="utf-8").read()

CTX_FIELDS = ["Stage", "Next question key", "Answers collected", "Ineligibility reason",
              "Agent name", "Client name", "Client's latest message", "Is first message"]


def render_ctx_block(ctx):
    lines = ["--- CONTEXT ---"]
    for k in CTX_FIELDS:
        lines.append(f"{k}: {ctx.get(k, 'N/A')}")
    lines.append("---")
    return "\n".join(lines)


def judge_user(rec):
    ctx = rec.get("context", {}) or {}
    recs = rec.get("recommendations", []) or []
    parts = [render_ctx_block(ctx)]
    if rec.get("directive"):
        parts.append("\nSITUATION: " + str(rec["directive"]))
    parts.append("\n--- 3 DRAFT SUGGESTIONS TO AUDIT ---")
    for i, r in enumerate(recs, 1):
        conf = (r or {}).get("confidence", "?")
        msg = (r or {}).get("suggested_message", "")
        parts.append(f"{i}. ({conf}) {msg}")
    parts.append(
        "\nTASK: You are a strict QA reviewer. A junior rep drafted the 3 SMS "
        "suggestions above for this context. Judge whether they FOLLOW ALL of your "
        "own rules from the system spec: emotion-first, TEXT voice (no call-center "
        "filler openers), stage logic (right question key / never blur screening "
        "windows / never screen during emotional/casual/logistics/stop moments), and "
        "the guardrails (no rate/APR/payment, no guarantees, no competitors, ineligible "
        "framed as 'not yet'). Be picky about voice and about screening when the client "
        "is emotional.\n"
        'Return ONLY JSON: {"verdict":"pass"|"fail", '
        '"violations":[{"which":"1|2|3|all","rule":<short>,"detail":<short>}], '
        '"summary":<one short sentence>}'
    )
    return "\n".join(parts)


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("no json object in judge output")
    return json.loads(text[s:e + 1])


def dry_judge(rec):
    # cheap stub: mirror the deterministic result so the plumbing is testable offline
    det = det_check(rec)
    if det:
        return {"verdict": "fail",
                "violations": [{"which": "all", "rule": "deterministic", "detail": det}],
                "summary": "stub judge echoing deterministic reason"}
    return {"verdict": "pass", "violations": [], "summary": "stub: looks fine"}


def crosscheck(rec, dry):
    det = det_check(rec)  # None == pass
    try:
        if dry:
            judge = dry_judge(rec)
        else:
            from teacher_client import chat
            content = chat([{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": judge_user(rec)}],
                           temperature=0.0, max_tokens=400)
            judge = parse_json(content)
    except Exception as e:
        judge = {"verdict": "error", "violations": [],
                 "summary": f"{type(e).__name__}: {e}"}
    ok = (det is None) and (judge.get("verdict") == "pass")
    out = dict(rec)
    out["_crosscheck"] = {"det": det, "judge": judge, "ok": ok}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labeled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rejects", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true",
                    help="stub the teacher (offline); deterministic layer still runs")
    args = ap.parse_args()

    recs = []
    with open(args.labeled, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
            if args.limit and len(recs) >= args.limit:
                break
    total = len(recs)

    print("=" * 60, file=sys.stderr)
    print("  crosscheck_labels", file=sys.stderr)
    print(f"  records : {total}   spec: {os.path.basename(_PROMPT_FILE)}", file=sys.stderr)
    if args.dry_run:
        print("  teacher : DRY-RUN (stub)", file=sys.stderr)
    else:
        print(f"  teacher : {os.environ.get('TEACHER_MODEL', 'teacher')} @ "
              f"{os.environ.get('TEACHER_BASE_URL', 'http://localhost:8001/v1')}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    n = fail = err = done = 0
    reasons = Counter()
    start = time.time()
    with open(args.out, "w", encoding="utf-8") as good, \
         open(args.rejects, "w", encoding="utf-8") as rej, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(crosscheck, r, args.dry_run): i for i, r in enumerate(recs)}
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            good.write(json.dumps(res) + "\n")
            cc = res["_crosscheck"]
            v = cc["judge"].get("verdict")
            if v == "error":
                err += 1
            if not cc["ok"]:
                fail += 1
                rej.write(json.dumps(res) + "\n")
                if cc["det"]:
                    reasons["det:" + cc["det"][:40]] += 1
                for viol in cc["judge"].get("violations", [])[:1]:
                    reasons["judge:" + str(viol.get("rule"))[:40]] += 1
            else:
                n += 1
            if done % 25 == 0 or done == total:
                el = time.time() - start
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                print(f"\r\033[K[{done}/{total}] ok {n} fail {fail} err {err} | "
                      f"{rate:4.1f}/s | ETA {int(eta)}s", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(f"[crosscheck] pass={n} fail={fail} (err={err}) of {total} "
          f"-> {args.out}; rejects -> {args.rejects}", file=sys.stderr)
    if reasons:
        print("  top reject reasons:", file=sys.stderr)
        for r, c in reasons.most_common(15):
            print(f"    {c:5d}  {r}", file=sys.stderr)


if __name__ == "__main__":
    main()

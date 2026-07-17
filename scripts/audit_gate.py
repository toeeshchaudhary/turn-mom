"""Deterministic guardrail gate over teacher-labeled records.
Drops any example that violates the hard product rules so we never train on a
bad label. Rejects are written to --rejects with a reason for inspection.
Checks:
  - exactly 3 recommendations, each a non-empty string
  - confidence in {high, medium, low}
  - <= 3 sentences and <= 400 chars per message
  - no rates/APR/percentages/dollar-amount payments
  - the 3 messages don't all start with the same word (anti-AI-symmetry)
  - ineligible stage: no "failed/rejected/denied/disqualified"
  - greeting (is_first + greeting-ish latest): each message names New American Funding
Usage:
  python3 audit_gate.py labeled.jsonl --out labeled.audited.jsonl --rejects rejects.jsonl
"""
import argparse, json, re, sys
RATE_RE = re.compile(r"\b\d+(\.\d+)?\s?%|\bAPR\b|\$\s?\d", re.I)
TOKEN_RE = re.compile(r"\{[A-Z][A-Z_]*\}")   
FORBIDDEN_INELIGIBLE = re.compile(r"\b(failed|rejected|denied|disqualified)\b", re.I)
GREETING_RE = re.compile(r"^\s*(hey|hi|hello)\b", re.I)
def sentences(s):
    return [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
def check(rec):
    recs = rec.get("recommendations")
    if not isinstance(recs, list) or len(recs) != 3:
        return f"not exactly 3 recommendations ({len(recs) if isinstance(recs, list) else 'n/a'})"
    msgs = []
    for r in recs:
        m = (r or {}).get("suggested_message", "")
        c = (r or {}).get("confidence", "")
        if not isinstance(m, str) or not m.strip():
            return "empty suggested_message"
        if c not in ("high", "medium", "low"):
            return f"bad confidence '{c}'"
        if len(m) > 400:
            return "message too long"
        if len(sentences(m)) > 3:
            return "more than 3 sentences"
        if RATE_RE.search(m):
            return "mentions rate/APR/$ amount"
        if TOKEN_RE.search(m):
            return "contains a redaction placeholder token (e.g. {NAME})"
        msgs.append(m.strip())
    firsts = {m.split()[0].lower().strip(".,!?") for m in msgs if m.split()}
    if len(firsts) == 1:
        return "all 3 start with the same word"
    ctx = rec.get("context", {})
    stage = (ctx.get("Stage") or "").lower()
    if stage == "ineligible":
        for m in msgs:
            if FORBIDDEN_INELIGIBLE.search(m):
                return "ineligible message uses forbidden word"
    latest = ctx.get("Client's latest message", "")
    is_first = str(ctx.get("Is first message", "no")).lower() == "yes"
    if is_first and GREETING_RE.match(latest):
        for m in msgs:
            if "new american funding" not in m.lower():
                return "greeting message missing 'New American Funding'"
    return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labeled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rejects", required=True)
    args = ap.parse_args()
    ok = bad = 0
    with open(args.labeled, encoding="utf-8") as f,         open(args.out, "w", encoding="utf-8") as good,         open(args.rejects, "w", encoding="utf-8") as rej:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            reason = check(rec)
            if reason:
                rec["_reject_reason"] = reason
                rej.write(json.dumps(rec) + "\n")
                bad += 1
            else:
                good.write(json.dumps(rec) + "\n")
                ok += 1
    print(f"[audit] pass={ok} reject={bad} -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

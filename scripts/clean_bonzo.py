"""Clean Bonzo SMS JSON -> {file, source, turns:[{role,text}]}.
Bonzo threads are real Agent<->Client text messages. We normalize roles
(Agent->agent, Client->client), drop empty/automated lines, and collapse
consecutive same-speaker runs into one turn (people fire several texts in a row).
Input : one Bonzo .json per conversation (raw OR Clean Redacted). The redacted
        variant just has {NAME_GIVEN}-style tokens in place of PII; the logic is
        identical and idempotent either way.
Output: JSONL, one {file, source, turns} object per conversation.
Usage:
  python3 clean_bonzo.py a.json b.json ... --out bonzo_clean.jsonl
  find <dir> -name '*.json' -print0 | xargs -0 python3 clean_bonzo.py --out bonzo_clean.jsonl
"""
import argparse, json, os, re, sys
DROP_RE = re.compile(
    r"^\s*(unsubscribe|reply stop|msg\s*&?\s*data rates|this is an automated)",
    re.I,
)
SIG_RE = re.compile(r"^\s*(thanks[,!]*\s*)?[-–—]\s*\w+\s*$", re.I)
def norm_role(role):
    r = (role or "").strip().lower()
    if r == "agent":
        return "agent"
    if r == "client":
        return "client"
    return None  
def clean_text(t):
    t = (t or "").replace("\r", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    return t
def load_turns(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    hist = data.get("chat_history") or data.get("messages") or []
    turns = []
    for m in hist:
        role = norm_role(m.get("role"))
        text = clean_text(m.get("text", ""))
        if not role or not text:
            continue
        if DROP_RE.search(text) or SIG_RE.match(text):
            continue
        if turns and turns[-1]["role"] == role:  
            turns[-1]["text"] += " " + text
        else:
            turns.append({"role": role, "text": text})
    return turns
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-turns", type=int, default=2)
    args = ap.parse_args()
    kept = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for p in args.files:
            try:
                turns = load_turns(p)
            except Exception as e:
                print(f"[skip] {p}: {e}", file=sys.stderr)
                skipped += 1
                continue
            if len(turns) < args.min_turns:
                skipped += 1
                continue
            out.write(json.dumps({
                "file": os.path.basename(p),
                "source": "bonzo",
                "turns": turns,
            }) + "\n")
            kept += 1
    print(f"[clean_bonzo] kept={kept} skipped={skipped} -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

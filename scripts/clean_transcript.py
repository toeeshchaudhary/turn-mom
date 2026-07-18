import argparse, json, os, re, sys
LINE_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s*([A-Za-z ]+?):\s*(.*)$")
BOILER_RE = re.compile(
    r"(thank you for calling|quality assurance|may be monitored|please wait|"
    r"press \d|for english|joined call|left call)",
    re.I,
)
BACKCHANNEL = {"yeah", "yep", "uh huh", "mm hmm", "okay", "ok", "right", "gotcha", "sure"}
def norm_role(who):
    w = who.strip().lower()
    if w in ("internal", "agent"):
        return "agent"
    if w in ("customer", "client", "external"):
        return "client"
    return None  
def parse(path):
    turns = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            role = norm_role(m.group(1))
            text = re.sub(r"\s+", " ", m.group(2)).strip()
            if not role or not text:
                continue
            if BOILER_RE.search(text):
                continue
            if text.lower() in BACKCHANNEL:
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
    ap.add_argument("--min-turns", type=int, default=4)
    args = ap.parse_args()
    kept = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for p in args.files:
            try:
                turns = parse(p)
            except Exception as e:
                print(f"[skip] {p}: {e}", file=sys.stderr)
                skipped += 1
                continue
            if len(turns) < args.min_turns:
                skipped += 1
                continue
            out.write(json.dumps({
                "file": os.path.basename(p),
                "source": "transcript",
                "turns": turns,
            }) + "\n")
            kept += 1
    print(f"[clean_transcript] kept={kept} skipped={skipped} -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

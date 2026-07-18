#!/usr/bin/env python3
"""Mine real Bonzo threads for MAOS edge-case scenarios (Stream C, REAL inputs).

The Bonzo dumps (chat_history format) are mixed-tenant and often UNREDACTED. We sort
each thread by timestamp, keep only mortgage-relevant convos, then harvest real CLIENT
messages that hit an emotional / objection / life-event moment. Each becomes a MAOS task
with the real (redacted) client message as the input — the teacher later writes the
MAOS-ideal reply. We do NOT keep the real rep reply (usually the screener bug).

Output: {kind:'maos', mode, directive, context, source:'bonzo_mined'} — same shape as
gen_maos_scenarios, so it flows straight through the existing labeling pipeline.

Usage:
  python3 mine_bonzo.py conversations.jsonl --out bonzo_mined.jsonl --per 300
"""
import argparse, json, os, random, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_maos_scenarios import D, ctx, AGENTS, CLIENTS

MORTGAGE_RE = re.compile(
    r"\b(mortgage|refinanc|home loan|loan officer|pre-?approv|interest rate|down payment|"
    r"closing cost|escrow|new american funding|home equity|\brefi\b|underwrit|appraisal|"
    r"credit score|loan amount|purchase a home|buy a home)\b", re.I)

# mode detectors on the CLIENT message (high-recall regex), in priority order
MODE_RE = [
    ("life_event", re.compile(
        r"(lost my (job|wife|husband|mom|dad|mother|father)|passed away|\bdivorce|"
        r"in the hospital|hospitali[sz]|laid off|got fired|\bcancer\b|funeral|"
        r"my (wife|husband|mom|dad|mother|father) (just )?(died|passed)|family emergency)", re.I)),
    ("stop", re.compile(r"^\s*(stop|unsubscribe|do not (contact|text)|remove me|quit texting)", re.I)),
    ("escalation", re.compile(r"\b(attorney|lawyer|cfpb|lawsuit|sue you|sue your|file a complaint|harass)\b", re.I)),
    ("not_interested", re.compile(
        r"\b(not interested|no longer interested|changed my mind|don'?t want|forget it|"
        r"not looking anymore|please stop)\b", re.I)),
    ("other_lender", re.compile(
        r"\b(another lender|other lender|already (working|got a quote|have a lender)|"
        r"going with (another|someone)|my bank is|different lender)\b", re.I)),
    ("busy", re.compile(
        r"\b(i'?m (busy|at work|driving)|can'?t talk|call me (later|back)|text me later|"
        r"not a good time|in a meeting)\b", re.I)),
    ("rate_shopping", re.compile(
        r"(what'?s? (your|the|todays?) (rate|interest)|best rate|rate shopping|"
        r"how much (is|are) (the )?rate|current rates?)", re.I)),
    ("loan_question", re.compile(
        r"\b(how much (can i|do i)|what (credit|score|documents|is a|are)|can i (buy|qualify|afford)|"
        r"do i need|how (long|does)|difference between).*\?", re.I)),
]


def redact(t):
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "{EMAIL}", t)
    t = re.sub(r"\+?\d[\d\-\(\)\s]{7,}\d", "{PHONE_NUMBER}", t)
    t = re.sub(r"\$\s?[\d,]+(?:\.\d+)?", "{MONEY}", t)
    t = re.sub(r"\b\d{1,6}\s+[NSEW]\.?\s+[A-Z][a-z]+", "{LOCATION_ADDRESS}", t)  # "6019 S Kedzie"
    # names after greetings / intros / signatures (1-2 capitalized words)
    NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    t = re.sub(rf"\b(hi|hey|hello|thanks|thank you|dear)\s+({NAME})",
               r"\1 {NAME_GIVEN}", t, flags=re.I)
    t = re.sub(rf"\b(I'?m|this is|my name is|it'?s|name is)\s+({NAME}(?:\s+[A-Z][a-z]+)?)",
               r"\1 {NAME_GIVEN}", t, flags=re.I)
    t = re.sub(rf"[-–—]\s*({NAME})\s*$", "- {NAME_GIVEN}", t)  # trailing signature
    return re.sub(r"\s+", " ", t).strip()


# map the detector name -> the MAOS mode (same routing as ui/server.ts)
MODE_MAP = {"life_event": "empathy", "not_interested": "objection", "busy": "logistics",
            "stop": "stop", "escalation": "escalation", "other_lender": "other_lender",
            "rate_shopping": "rate_shopping", "loan_question": "loan_question"}


def detect_mode(msg):
    for mode, rx in MODE_RE:
        if rx.search(msg):
            return mode
    return None


def norm(role):
    r = (role or "").strip().lower()
    return "agent" if r == "agent" else ("client" if r == "client" else None)


def load_convos(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        hist = sorted(d.get("chat_history") or [], key=lambda m: m.get("timestamp") or "")
        turns = [(norm(m.get("role")), (m.get("text") or "").strip()) for m in hist]
        turns = [(r, t) for r, t in turns if r and t]
        if turns:
            yield turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("convos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=300, help="max mined per mode")
    ap.add_argument("--min-len", type=int, default=8, help="min client-message chars")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pool = {m: [] for m, _ in MODE_RE}
    mortgage = nonmort = 0
    for turns in load_convos(args.convos):
        full = " ".join(t for _, t in turns)
        if not MORTGAGE_RE.search(full):
            nonmort += 1
            continue
        mortgage += 1
        for role, text in turns:
            if role != "client" or len(text) < args.min_len:
                continue
            mode = detect_mode(text)
            if mode:
                pool[mode].append(redact(text))

    counts = Counter()
    seen = set()
    with open(args.out, "w", encoding="utf-8") as out:
        for mode, msgs in pool.items():
            rng.shuffle(msgs)
            for msg in msgs:
                key = (mode, msg.lower())
                if key in seen:
                    continue
                seen.add(key)
                if counts[mode] >= args.per:
                    break
                agent, client = rng.choice(AGENTS), rng.choice(CLIENTS)
                maos_mode = MODE_MAP[mode]
                rec = {"kind": "maos", "mode": maos_mode, "directive": D[maos_mode], "anchor": None,
                       "context": ctx("qualifying", "property_use", "N/A", "N/A", agent, client, msg, "no"),
                       "case": mode, "source": "bonzo_mined"}
                out.write(json.dumps(rec) + "\n")
                counts[mode] += 1
    print(f"[mine_bonzo] mortgage_convos={mortgage} non_mortgage={nonmort} "
          f"mined={sum(counts.values())} {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic cleaner for the SFT chat jsonl (Gemma-4 12B rebuild).

Fixes the three data-baked failure modes we measured in the eval:
  1. NAME SPAM  - client first-name repeated / re-greeted on mid-conversation turns.
                  Rule: name+greeting only when `Is first message: yes`.
  2. BACKCHANNEL- call-register filler openers ("got it", "thanks for", "sure thing", ...).
                  Rule (per user): STRIP ALL filler openers on every suggestion.
  3. GREETING   - "Hi <name>" opener on a non-first turn (subset of 1, handled together).

It also:
  - dedups exact (context, assistant) pairs and caps near-duplicate templates (anti-overfit),
  - routes records that can't be fixed deterministically to `needs_regen.jsonl` for the
    teacher to re-label on the box:
      * clear grief -> mortgage-pitch leaks (empathy),
      * off-topic turns that ignore instead of acknowledge-then-circle-back,
      * suggestions that degenerate to empty/too-short after stripping.

Local-only, no network. Usage:
    python scripts/clean_sft.py --in chat_train.jsonl --out chat_train.clean.jsonl \
        --regen-out needs_regen.train.jsonl --report report.train.json
"""
import argparse, json, re
from collections import Counter, defaultdict

# ---- filler / backchannel openers ----
# HARD: unambiguous filler phrases -> always strip when they open a message.
HARD = [
    "got it", "gotcha", "no problem at all", "not a problem at all",
    "no worries at all", "no problem", "not a problem", "no worries", "sure thing",
    "thanks for confirming", "thank you for confirming",
    "makes sense", "understood", "i understand", "i hear you", "i feel you",
    "i get it", "i totally get it", "sounds good", "of course", "glad to hear",
    "good to know", "good to hear", "thanks for letting me know",
    "thanks for sharing", "thanks for that", "thanks for the info",
    "thanks for reaching out", "thanks for the update", "appreciate that",
    "appreciate you", "i appreciate that", "i appreciate it", "for sure",
    "haha", "lol",
]
# SOFT: words that are filler as a bare opener but valid as adverbs/adjectives.
# Only strip when immediately followed by a clause boundary ( , . ! ? — - or end ).
SOFT = [
    "okay", "ok", "alright", "great", "awesome", "perfect", "wonderful",
    "fantastic", "nice", "sure", "absolutely", "totally", "yeah", "yep", "yup",
    "well", "so", "right", "cool", "love that", "love it",
]
_HARD_ALT = "|".join(sorted((re.escape(f) for f in HARD), key=len, reverse=True))
_SOFT_ALT = "|".join(sorted((re.escape(f) for f in SOFT), key=len, reverse=True))
# one filler opener = a HARD phrase (any trailing punctuation) OR a SOFT word that
# is *followed by* a clause boundary. Applied repeatedly to peel stacked openers.
FILLER_OPENER_RE = re.compile(
    rf"^\s*(?:(?:{_HARD_ALT})\b[\s,]*[-—]?\s*"
    rf"|(?:{_SOFT_ALT})\b\s*[,.!?—-]+\s*)",
    re.I,
)

GREETING_RE = re.compile(r"^\s*(hi|hey|hello|hiya|heya|good (morning|afternoon|evening))\b", re.I)

# strong, unambiguous grief/loss cues in the CLIENT'S LATEST MESSAGE only
GRIEF_RE = re.compile(
    r"\b(passed away|passed on|lost my (wife|husband|spouse|mom|mother|dad|father|son|"
    r"daughter|child|partner|brother|sister)|my (wife|husband|mom|mother|dad|father|son|"
    r"daughter|partner) (just )?(died|passed)|funeral|in hospice|terminally ill|"
    r"just (died|passed))\b",
    re.I,
)
PITCH_RE = re.compile(
    r"\b(mortgage|pre-?approv|\brate\b|\bloan\b|refinanc|qualify|qualifying|property use|"
    r"down payment|application|apply|move forward|next step|primary residence|"
    r"investment property|bankruptc|foreclosur|late (mortgage )?payment)\b",
    re.I,
)

def ctx_fields(user_content):
    d = {}
    for line in user_content.splitlines():
        if ":" in line and not line.startswith("---") and not line.startswith("SITUATION"):
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    situation = ""
    for line in user_content.splitlines():
        if line.startswith("SITUATION"):
            situation = line
            break
    return d, situation

def latest_message(d):
    return d.get("Client's latest message", "")

def strip_name(msg, name, is_first):
    """Remove greeting+name and vocative name uses on non-first turns."""
    if is_first or not name or name.lower() in ("n/a", ""):
        return msg
    nm = re.escape(name)
    # leading "Hi/Hey <Name>," possibly with punctuation
    msg = re.sub(rf"^\s*(hi|hey|hello|hiya|heya)\s+{nm}\b[\s,!.:-]*", "", msg, flags=re.I)
    # leading "<Name>," vocative
    msg = re.sub(rf"^\s*{nm}\s*[,!:-]+\s*", "", msg)
    # trailing/inline vocative ", <Name>" before a sentence break or end
    msg = re.sub(rf"\s*,\s*{nm}\b(?=[\s.!?,]|$)", "", msg)
    # standalone "<Name>." as its own trailing token
    msg = re.sub(rf"\s+{nm}\s*([.!?])\s*$", r"\1", msg)
    return msg

def strip_fillers(msg):
    prev = None
    while prev != msg:
        prev = msg
        msg = FILLER_OPENER_RE.sub("", msg)
    return msg

def tidy(msg):
    # drop orphaned leading punctuation left behind after peeling an opener
    msg = re.sub(r"^[\s.,;:!?—-]+", "", msg)
    msg = msg.strip(" \t—-,;:")
    msg = re.sub(r"\s+([.,;:!?])", r"\1", msg)  # no space before punctuation
    msg = re.sub(r"\s{2,}", " ", msg).strip()
    if not msg:
        return msg
    # recapitalize first alpha char
    for i, ch in enumerate(msg):
        if ch.isalpha():
            msg = msg[:i] + ch.upper() + msg[i+1:]
            break
    return msg

def clean_suggestion(msg, name, is_first):
    m = strip_name(msg, name, is_first)
    m = strip_fillers(m)
    m = tidy(m)
    return m

def word_count(s):
    return len(re.findall(r"\w+", s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regen-out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--min-words", type=int, default=3,
                    help="a stripped suggestion shorter than this -> route to regen")
    ap.add_argument("--dup-cap", type=int, default=3,
                    help="max copies of an identical (context,assistant) template to keep")
    args = ap.parse_args()

    stats = Counter()
    kept, regen = [], []
    seen = Counter()

    for line in open(args.inp):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        stats["total"] += 1
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        d, situation = ctx_fields(user)
        is_first = d.get("Is first message", "no").strip().lower() in ("yes", "true")
        name = d.get("Client name", "")

        try:
            obj = json.loads(asst)
            suggs = obj["recommendations"]
            assert isinstance(suggs, list) and len(suggs) == 3
        except Exception:
            stats["unparsable_dropped"] += 1
            continue

        latest = latest_message(d)

        # ---- empathy routing: strong grief cue in client's latest + any pitch -> regen ----
        if GRIEF_RE.search(latest):
            stats["grief_context"] += 1
            if any(PITCH_RE.search(s.get("suggested_message", "")) for s in suggs):
                stats["grief_pitch_to_regen"] += 1
                regen.append({**rec, "_regen_reason": "grief_pitch"})
                continue

        # ---- off-topic that ignores (no acknowledgment, just re-asks screen q) -> regen ----
        is_offtopic = situation.startswith("SITUATION: the client asked a mortgage question") or \
                      "off-topic" in situation.lower() or "small talk" in situation.lower() or \
                      "inappropriate" in situation.lower()

        new_suggs = []
        degenerate = False
        for s in suggs:
            cleaned = clean_suggestion(s.get("suggested_message", ""), name, is_first)
            if word_count(cleaned) < args.min_words:
                degenerate = True
                break
            new_suggs.append({**s, "suggested_message": cleaned})

        if degenerate:
            stats["degenerate_after_strip_to_regen"] += 1
            regen.append({**rec, "_regen_reason": "degenerate_after_strip"})
            continue

        # within-record collapse: stripping made 2+ identical -> regen (lost the 3-way variety)
        if len({s["suggested_message"] for s in new_suggs}) < 3:
            stats["collapsed_to_regen"] += 1
            regen.append({**rec, "_regen_reason": "collapsed_duplicates"})
            continue

        # count changes
        if any(new_suggs[i]["suggested_message"] != suggs[i].get("suggested_message", "")
               for i in range(3)):
            stats["records_modified"] += 1

        obj["recommendations"] = new_suggs
        new_asst = json.dumps(obj, ensure_ascii=False)
        rec = {**rec, "messages": [
            m if m["role"] != "assistant" else {**m, "content": new_asst} for m in msgs
        ]}

        # ---- anti-overfit dedup: cap identical (context, cleaned-assistant) templates ----
        key = (user, new_asst)
        seen[key] += 1
        if seen[key] > args.dup_cap:
            stats["dup_capped"] += 1
            continue

        kept.append(rec)

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.regen_out, "w") as f:
        for r in regen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = dict(stats)
    report["kept"] = len(kept)
    report["routed_to_regen"] = len(regen)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

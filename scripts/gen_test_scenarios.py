"""Exhaustive, high-quality scenario enumeration for model testing.

Unlike gen_scenarios.py (random sampling for the training set), this walks the FULL
screening-oracle state space deterministically: both buyer types, every stage, every
next-question step, every prior-answer combination, every disqualifier, plus all edge
cases — then crosses with agent/client names to yield 1k+ unique scenarios.

Reuses the oracle constants from gen_scenarios so it stays in sync with the prompt.
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_scenarios as G
from gen_scenarios import (KEYS, FIRST_TIME_KEYS, FIRST_TIME_SKIP, INELIGIBILITY_REASONS,
                           ctx, AGENT_NAMES, CLIENT_NAMES)

# ---- richer message variety than the training generator (quality) ----
PROPERTY_USES = ["live in it myself", "rent it out", "use as a second home",
                 "buy for my daughter", "vacation place", "help my parents move in"]
FIRST_TIME_PROPERTY_USES = ["live in it myself", "rent it out",
                            "buy for my daughter", "move in with my partner"]

MORTGAGE_OPENERS = [
    "i want to pull some equity out of my house",
    "looking to refinance, rates dropped right",
    "can you help me lower my monthly payment",
    "thinking about refinancing before rates climb again",
    "my current mortgage is killing me, any options",
    "wondering if a cash-out refi makes sense for me",
]
FIRST_TIME_OPENERS = [
    "thinking about buying my first place",
    "looking to buy my first home",
    "i'm a first-time buyer, where do i start",
    "never owned a home before, want to buy one",
    "my wife and i are ready to stop renting",
]

# fragments the client says when ANSWERING a screening question (all non-disqualifying
# — used to advance the qualifying flow without flipping the stage to ineligible)
OK_FRAGMENTS = {
    "property_use": ["gonna live there", "it'd be a rental", "second home for us",
                     "for my son actually", "just for us to live in"],
    "bankruptcy_past_3yr": ["no nothing like that", "nope never", "no bankruptcies here",
                            "never filed"],
    "foreclosure_past_2yr": ["no", "nope all good", "never lost a house",
                             "no foreclosures"],
    "late_payments_past_12mo": ["always on time", "no missed payments",
                                "never late", "all paid on time"],
}
# fragments that DISQUALIFY (used for ineligible scenarios)
BAD_FRAGMENTS = {
    "bankruptcy_past_3yr": ["yeah we filed a couple years back",
                            "we did go through a bankruptcy last year"],
    "foreclosure_past_2yr": ["we did lose a house recently unfortunately",
                             "yeah we had a foreclosure about a year ago"],
    "late_payments_past_12mo": ["maybe one late last winter",
                                "we missed a couple payments this year"],
}
CONFIRM_FRAGMENTS = ["yep that's everything", "that's about it", "yeah that's all correct",
                     "sounds right to me", "yep all set"]
ELIGIBLE_AFFIRM = ["yep that all looks right", "correct, all good",
                   "yes that's accurate", "that's right"]

GREETINGS = ["Hey", "Hi there", "Hello", "hey", "Good morning", "howdy"]
CASUAL = ["how's your day going", "man it's so hot out today",
          "just got back from my kid's game", "long week huh", "happy friday"]
LOGISTICS = ["can we talk tomorrow instead", "im driving right now",
             "give me like 10 min", "call me after 5 if that works",
             "in a meeting, text is better"]
OFFTOPIC = ["who are you voting for", "what do you think about this weather bill",
            "can you give me legal advice on my divorce", "got any stock tips",
            "what's your favorite football team"]


def _uses(first_time):
    return FIRST_TIME_PROPERTY_USES if first_time else PROPERTY_USES


def _eligible_answers(first_time, property_use):
    """A fully-collected, non-disqualifying answer set."""
    a = {"property_use": property_use, "bankruptcy_past_3yr": "no"}
    if first_time:
        a["foreclosure_past_2yr"] = "n/a"
        a["late_payments_past_12mo"] = "n/a"
    else:
        a["foreclosure_past_2yr"] = "no"
        a["late_payments_past_12mo"] = "no"
    return a


def _base_scenarios():
    """Every distinct scenario BEFORE crossing with names."""
    out = []

    def add(case, stage, next_key, answers, reason, latest, is_first):
        out.append({"case": case, "stage": stage, "next_key": next_key,
                    "answers": answers, "reason": reason, "latest": latest,
                    "is_first": is_first})

    for first_time in (False, True):
        akeys = FIRST_TIME_KEYS if first_time else KEYS
        uses = _uses(first_time)
        tag = "_first_time" if first_time else ""

        # ---- qualifying: rep is about to ask akeys[step] ----
        for step in range(len(akeys)):
            next_key = akeys[step]
            if step == 0:
                openers = FIRST_TIME_OPENERS if first_time else MORTGAGE_OPENERS
                for op in openers:
                    ans = {sk: "n/a" for sk in (FIRST_TIME_SKIP if first_time else [])}
                    add("qualifying" + tag, "qualifying", next_key, ans, "N/A", op, "no")
            else:
                prev = akeys[step - 1]
                # prior keys answered with canonical non-disqualifying values,
                # property_use varied across its options
                for use in uses:
                    ans = {}
                    for k in akeys[:step]:
                        ans[k] = use if k == "property_use" else "no"
                    if first_time:
                        for sk in FIRST_TIME_SKIP:
                            ans[sk] = "n/a"
                    for frag in OK_FRAGMENTS[prev]:
                        add("qualifying" + tag, "qualifying", next_key, dict(ans),
                            "N/A", frag, "no")

        # ---- confirming: everything collected, about to confirm ----
        for use in uses:
            ans = _eligible_answers(first_time, use)
            for frag in CONFIRM_FRAGMENTS:
                add("confirming" + tag, "confirming", "N/A", ans, "N/A", frag, "no")

        # ---- eligible ----
        for use in uses:
            ans = _eligible_answers(first_time, use)
            for frag in ELIGIBLE_AFFIRM:
                add("eligible" + tag, "eligible", "N/A", ans, "N/A", frag, "no")

        # ---- ineligible: one disqualifier per scenario ----
        disqualifiers = (["bankruptcy_past_3yr"] if first_time
                         else ["bankruptcy_past_3yr", "foreclosure_past_2yr",
                               "late_payments_past_12mo"])
        for bad in disqualifiers:
            for use in uses:
                ans = _eligible_answers(first_time, use)
                ans[bad] = "yes"
                for frag in BAD_FRAGMENTS[bad]:
                    add("ineligible" + tag, "ineligible", "N/A", dict(ans),
                        INELIGIBILITY_REASONS[bad], frag, "no")

    # ---- edge cases (buyer-type-agnostic) ----
    for g in GREETINGS:
        add("greeting", "qualifying", KEYS[0], {}, "N/A", g, "yes")
    for m in CASUAL:
        add("casual", "qualifying", "N/A", {}, "N/A", m, "no")
    for m in LOGISTICS:
        add("logistics", "qualifying", "N/A", {}, "N/A", m, "no")
    for m in OFFTOPIC:
        add("offtopic", "qualifying", "N/A", {}, "N/A", m, "no")

    return out


def build_scenarios(name_variants=8, seed=0):
    """Full enumeration crossed with names → list of records (gen_scenarios shape)."""
    bases = _base_scenarios()
    recs, seen = [], set()
    n_agents, n_clients = len(AGENT_NAMES), len(CLIENT_NAMES)
    for b in bases:
        for v in range(name_variants):
            agent = AGENT_NAMES[v % n_agents]
            client = CLIENT_NAMES[(v * 3 + 1) % n_clients]  # decorrelate from agent
            rec = ctx(b["stage"], b["next_key"], b["answers"], b["reason"],
                      b["latest"], b["is_first"], agent, client)
            key = (b["case"], b["stage"], b["next_key"],
                   rec["context"]["Answers collected"], b["reason"],
                   b["latest"], agent, client)
            if key in seen:
                continue
            seen.add(key)
            rec["case"] = b["case"]
            rec["kind"] = "scenario"
            recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser(description="Enumerate every test scenario -> jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name-variants", type=int, default=8,
                    help="how many agent/client name pairings per base scenario")
    args = ap.parse_args()
    recs = build_scenarios(args.name_variants)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    counts = Counter(r["case"] for r in recs)
    print(f"[gen_test_scenarios] wrote {len(recs)} scenarios -> {args.out}", file=sys.stderr)
    for c, n in sorted(counts.items()):
        print(f"    {c:26s} {n}", file=sys.stderr)


if __name__ == "__main__":
    main()

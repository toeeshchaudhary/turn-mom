"""Stream B: deterministic screening oracle -> stage-coverage label tasks.
Raw conversations rarely walk the clean qualifying->confirming->eligible/ineligible
flow with all four screening keys. This generator manufactures CONTEXT blocks that
cover that flow exhaustively, plus the edge cases (greeting/casual/logistics/offtopic).
The CONTEXT is fully deterministic here; the teacher only writes the 3 suggestions.
Screening checklist (order matters):
  property_use, bankruptcy_past_3yr, foreclosure_past_2yr, late_payments_past_12mo
Eligibility oracle:
  ineligible if bankruptcy_past_3yr OR foreclosure_past_2yr OR late_payments_past_12mo is "yes".
  otherwise eligible. (property_use is informational, never disqualifying.)
Output records:  {kind:'scenario', case, context:{...}, client_latest, is_first}
where context has the fields the CSS system prompt's CONTEXT block expects.
Usage:
  python3 gen_scenarios.py --out scenarios.jsonl --n 4000
"""
import argparse, itertools, json, random, sys
KEYS = ["property_use", "bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"]
PROPERTY_USES = ["live in it myself", "rent it out", "use as a second home", "buy for my daughter"]
YESNO = ["yes", "no"]
MORTGAGE_OPENERS = [
    "i want to pull some equity out of my house",
    "looking to refinance, rates dropped right",
    "thinking about buying my first place",
    "can you help me lower my monthly payment",
]
ANSWER_FRAGMENTS = {
    "property_use": ["gonna live there", "it'd be a rental", "second home for us", "for my son actually"],
    "bankruptcy_past_3yr": ["no nothing like that", "yeah we filed a couple years back", "nope never"],
    "foreclosure_past_2yr": ["no", "we did lose a house recently unfortunately", "nope all good"],
    "late_payments_past_12mo": ["always on time", "maybe one late last winter", "no missed payments"],
}
GREETINGS = ["Hey", "Hi there", "Hello", "hey"]
CASUAL = ["how's your day going", "man it's so hot out today", "just got back from my kid's game"]
LOGISTICS = ["can we talk tomorrow instead", "im driving right now", "give me like 10 min"]
OFFTOPIC = ["who are you voting for", "what do you think about this weather bill", "can you give me legal advice on my divorce"]
INELIGIBILITY_REASONS = {
    "bankruptcy_past_3yr": "bankruptcy_within_3yr",
    "foreclosure_past_2yr": "foreclosure_within_2yr",
    "late_payments_past_12mo": "late_mortgage_within_12mo",
}
def answers_summary(answers):
    if not answers:
        return "N/A"
    parts = []
    for k in KEYS:
        if k in answers:
            parts.append(f"{k}={answers[k]}")
    return ", ".join(parts)
AGENT_NAMES = ["Alex", "Jayme", "Marcus", "Sarah", "Priya", "Diego", "Nina", "Omar"]
CLIENT_NAMES = ["John", "Maria", "Tyler", "Grace", "Luis", "Kayla", "Rosa", "Ethan"]
def ctx(stage, next_key="N/A", answers=None, reason="N/A", client_latest="",
        is_first="no", agent="Alex", client="there"):
    return {
        "context": {
            "Stage": stage,
            "Next question key": next_key,
            "Answers collected": answers_summary(answers or {}),
            "Ineligibility reason": reason,
            "Agent name": agent,
            "Client name": client,
            "Client's latest message": client_latest,
            "Is first message": is_first,
        }
    }
def gen(rng):
    case = rng.choices(
        ["qualifying", "confirming", "eligible", "ineligible",
         "greeting", "casual", "logistics", "offtopic"],
        weights=[34, 10, 10, 14, 8, 8, 8, 8], k=1,
    )[0]
    if case == "qualifying":
        k = rng.randint(0, 3)                       
        answered = {key: rng.choice(YESNO) if key != "property_use"
                    else rng.choice(PROPERTY_USES) for key in KEYS[:k]}
        next_key = KEYS[k]
        latest = rng.choice(MORTGAGE_OPENERS) if k == 0 else rng.choice(ANSWER_FRAGMENTS[KEYS[k-1]])
        rec = ctx("qualifying", next_key, answered, "N/A", latest, "no")
        rec["case"] = "qualifying"
    elif case == "confirming":
        answers = {"property_use": rng.choice(PROPERTY_USES),
                   "bankruptcy_past_3yr": "no", "foreclosure_past_2yr": "no",
                   "late_payments_past_12mo": "no"}
        rec = ctx("confirming", "N/A", answers, "N/A",
                  rng.choice(ANSWER_FRAGMENTS["late_payments_past_12mo"]), "no")
        rec["case"] = "confirming"
    elif case == "eligible":
        answers = {"property_use": rng.choice(PROPERTY_USES),
                   "bankruptcy_past_3yr": "no", "foreclosure_past_2yr": "no",
                   "late_payments_past_12mo": "no"}
        rec = ctx("eligible", "N/A", answers, "N/A", "yep that all looks right", "no")
        rec["case"] = "eligible"
    elif case == "ineligible":
        bad = rng.choice(["bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"])
        answers = {"property_use": rng.choice(PROPERTY_USES),
                   "bankruptcy_past_3yr": "no", "foreclosure_past_2yr": "no",
                   "late_payments_past_12mo": "no"}
        answers[bad] = "yes"
        rec = ctx("ineligible", "N/A", answers, INELIGIBILITY_REASONS[bad],
                  "yeah that's all correct", "no")
        rec["case"] = "ineligible"
    elif case == "greeting":
        rec = ctx("qualifying", KEYS[0], {}, "N/A", rng.choice(GREETINGS), "yes")
        rec["case"] = "greeting"
    elif case == "casual":
        rec = ctx("qualifying", "N/A", {}, "N/A", rng.choice(CASUAL), "no")
        rec["case"] = "casual"
    elif case == "logistics":
        rec = ctx("qualifying", "N/A", {}, "N/A", rng.choice(LOGISTICS), "no")
        rec["case"] = "logistics"
    else:  
        rec = ctx("qualifying", "N/A", {}, "N/A", rng.choice(OFFTOPIC), "no")
        rec["case"] = "offtopic"
    rec["context"]["Agent name"] = rng.choice(AGENT_NAMES)
    rec["context"]["Client name"] = rng.choice(CLIENT_NAMES)
    rec["kind"] = "scenario"
    return rec
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as out:
        for _ in range(args.n):
            out.write(json.dumps(gen(rng)) + "\n")
    print(f"[gen_scenarios] wrote {args.n} scenarios -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

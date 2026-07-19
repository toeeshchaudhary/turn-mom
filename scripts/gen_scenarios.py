import argparse, itertools, json, random, sys
KEYS = ["property_use", "bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"]
# For a first-time buyer (never owned a home / never had a mortgage), foreclosure and
# late-mortgage-payment screening do not apply — you can't foreclose or miss mortgage
# payments on a home you never had. Only property_use + bankruptcy are relevant.
FIRST_TIME_KEYS = ["property_use", "bankruptcy_past_3yr"]
FIRST_TIME_SKIP = ["foreclosure_past_2yr", "late_payments_past_12mo"]
PROPERTY_USES = ["live in it myself", "rent it out", "use as a second home", "buy for my daughter"]
# a first-time buyer can't be buying a "second home" (that implies owning a first one)
FIRST_TIME_PROPERTY_USES = ["live in it myself", "rent it out", "buy for my daughter"]
YESNO = ["yes", "no"]
# openers that imply the client ALREADY owns a home (has a mortgage) -> full screening
MORTGAGE_OPENERS = [
    "i want to pull some equity out of my house",
    "looking to refinance, rates dropped right",
    "can you help me lower my monthly payment",
]
# openers that signal a FIRST-TIME buyer -> skip foreclosure + late-mortgage
FIRST_TIME_OPENERS = [
    "thinking about buying my first place",
    "looking to buy my first home",
    "i'm a first-time buyer, where do i start",
    "never owned a home before, want to buy one",
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
def base_answers(rng, first_time):
    """Collected-answers baseline. First-time buyers skip foreclosure + late-mortgage (n/a)."""
    uses = FIRST_TIME_PROPERTY_USES if first_time else PROPERTY_USES
    a = {"property_use": rng.choice(uses), "bankruptcy_past_3yr": "no"}
    if first_time:
        a["foreclosure_past_2yr"] = "n/a"
        a["late_payments_past_12mo"] = "n/a"
    else:
        a["foreclosure_past_2yr"] = "no"
        a["late_payments_past_12mo"] = "no"
    return a
def gen(rng):
    case = rng.choices(
        ["qualifying", "confirming", "eligible", "ineligible",
         "greeting", "casual", "logistics", "offtopic"],
        weights=[34, 10, 10, 14, 8, 8, 8, 8], k=1,
    )[0]
    # ~30% of screening scenarios are first-time buyers (skip foreclosure + late-mortgage)
    first_time = rng.random() < 0.30
    akeys = FIRST_TIME_KEYS if first_time else KEYS
    if case == "qualifying":
        k = rng.randint(0, len(akeys) - 1)
        uses = FIRST_TIME_PROPERTY_USES if first_time else PROPERTY_USES
        answered = {key: rng.choice(YESNO) if key != "property_use"
                    else rng.choice(uses) for key in akeys[:k]}
        if first_time:                                  # skipped keys shown as not-applicable
            for sk in FIRST_TIME_SKIP:
                answered[sk] = "n/a"
        next_key = akeys[k]
        if k == 0:
            latest = rng.choice(FIRST_TIME_OPENERS if first_time else MORTGAGE_OPENERS)
        else:
            latest = rng.choice(ANSWER_FRAGMENTS[akeys[k - 1]])
        rec = ctx("qualifying", next_key, answered, "N/A", latest, "no")
        rec["case"] = "qualifying_first_time" if first_time else "qualifying"
    elif case == "confirming":
        answers = base_answers(rng, first_time)
        rec = ctx("confirming", "N/A", answers, "N/A",
                  rng.choice(ANSWER_FRAGMENTS[akeys[-1]]), "no")
        rec["case"] = "confirming"
    elif case == "eligible":
        answers = base_answers(rng, first_time)
        rec = ctx("eligible", "N/A", answers, "N/A", "yep that all looks right", "no")
        rec["case"] = "eligible"
    elif case == "ineligible":
        # first-time buyers can only be disqualified by bankruptcy (no foreclosure/late-mortgage)
        bad = ("bankruptcy_past_3yr" if first_time
               else rng.choice(["bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"]))
        answers = base_answers(rng, first_time)
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

import argparse, json, random, sys
CTX_FIELDS = ["Stage", "Next question key", "Answers collected", "Ineligibility reason",
              "Agent name", "Client name", "Client's latest message", "Is first message"]
AGENTS = ["Sarah", "Marcus", "Priya", "Diego", "Nina", "Omar", "Kartikay", "KK", "Alex"]
CLIENTS = ["Mike", "John", "Maria", "Tyler", "Grace", "Luis", "Anas", "Rosa", "Ethan", "Dana"]
D = {
 "empathy": ("SITUATION: the client just shared something emotionally heavy (grief, illness, hospital, "
   "job loss, divorce). Respond as a caring human, NOT a salesperson. Acknowledge their situation "
   "genuinely and briefly. Do NOT ask ANY qualifying or screening question. Do NOT pitch, sell, or "
   "mention loans, applications, or property use. Give them space and let them lead — you may gently "
   "offer to pick things back up whenever they're ready. All 3 suggestions short, warm, human."),
 "not_now": ("SITUATION: based on the answers, the client is not in a position to be approved right now. "
   "Do NOT say 'declined', 'rejected', 'denied', or name the specific reason/timeframe. Shift from seller "
   "to EDUCATOR: frame it gently as a timing issue ('not quite there yet' / 'not yet'), stay warm, and "
   "invite them to reach back out when the time is right."),
 "post_handoff": ("SITUATION: you have ALREADY told the client a loan officer will reach out. Do NOT repeat "
   "that or re-pitch anything. Reply briefly like a real person — acknowledge what they said, reassure if "
   "needed, and stop selling. If they're chasing an update, offer to nudge the loan officer. Keep the 3 "
   "suggestions short and genuinely different."),
 "objection": ("SITUATION: the client is losing interest or pushing back. Do NOT keep qualifying. Respect it, "
   "acknowledge it, and either softly ask what changed or leave the door open gracefully with a low-pressure "
   "next step. Never argue or hard-sell. No screening questions."),
 "logistics": ("SITUATION: the client is busy or wants to connect later. Accommodate warmly and honor their "
   "timing. Do NOT push a qualifying question or force a call. Confirm you'll follow up when it works for them."),
 "other_lender": ("SITUATION: the client mentions another lender. Do NOT criticize the competitor. Offer a "
   "no-pressure 'clean second look / compare apples to apples', and be gracious if they decline. No screening questions."),
 "rate_shopping": ("SITUATION: the client is asking about rates. Do NOT quote any specific rate, APR, or number. "
   "Acknowledge, then find out if it's a purchase or refinance and what matters most (lower payment vs lower "
   "cash to close), pointing toward real numbers as the next step."),
 "loan_question": ("SITUATION: the client asked a mortgage question. Answer it briefly and accurately, then "
   "reframe to what really matters, then offer an easy next step. Do NOT quote specific rates or APRs. Warm, "
   "clear advisor tone — not a sales pitch."),
 "stop": ("SITUATION: the client asked to stop or opt out. Produce 3 short, clean acknowledgements that fully "
   "respect it with ZERO pushback — no questions, no pitch, no re-engagement. E.g. 'Understood — thanks for letting me know.'"),
 "escalation": ("SITUATION: the client raised a legal, compliance, or complaint issue (attorney, CFPB, lawsuit). "
   "Do NOT sell or qualify. Respond calmly and professionally, take it seriously, and let them know the right "
   "person will follow up. Keep it brief."),
 "inappropriate": ("SITUATION: the client's message is inappropriate. Redirect cleanly and professionally back "
   "to how you can help with their mortgage. Do NOT engage the content, do NOT be preachy, awkward, or use "
   "euphemisms. Brief and unbothered."),
}
SEEDS = {
 "empathy": [(m, None) for m in [
   "I just lost my wife", "my husband passed away last month", "I just lost my job",
   "I'm in the hospital right now, please don't text me", "my mom is having surgery tomorrow",
   "we're going through a divorce right now", "my dad just passed away", "I got laid off yesterday",
   "my daughter is really sick and in the hospital", "I'm dealing with a family emergency right now"]],
 "logistics": [
   ("I'm at work right now", "No problem at all—text works. I'll keep it simple. Are you mainly looking to buy soon, or just getting a game plan together?"),
   ("can't talk right now", None), ("give me a call on Monday", None),
   ("I'm driving, text me later", None), ("busy right now, hit me up this evening", None)],
 "objection": [
   ("not interested anymore", "Understood, and I respect that. If anything changes down the road, I'm here—no pressure at all."),
   ("I changed my mind", "No worries at all—just curious, what made you change your mind? Either way I'll leave you be."),
   ("not ready yet", "No problem. Should I check back in a few weeks, or is this more of a later-this-year thing?"),
   ("just looking for now", "Totally fine—that's actually the best time to get clear. Thinking in the next few months, or just planning ahead?"),
   ("stop bothering me with this", None), ("I don't think I want to do this", None)],
 "other_lender": [
   ("I'm already working with another lender", "Totally understand—a lot of people compare before making a move. Want me to give you a clean second look so you can compare apples to apples?"),
   ("I already got a quote somewhere else", "Makes sense to shop it. Happy to be a second set of eyes if you want to compare—no pressure."),
   ("my bank is handling my loan", None)],
 "rate_shopping": [
   ("what are your rates?", "Happy to help with that. Rates depend on a few things, so I don't want to quote you a fake number. Is this for a purchase or a refinance?"),
   ("just rate shopping, what can you do?", "Smart to compare. Rather than a random number, is this purchase or refi—and are you focused more on lowest payment or lowest cash to close?"),
   ("give me your best rate", None), ("what's todays interest rate?", None)],
 "loan_question": [
   ("how much house can I afford?", "That depends on your income, monthly debts, credit, and down payment—best to run real numbers than guess. Want me to help estimate a comfortable payment range?"),
   ("what credit score do I need?", "It depends on the loan program and the full strength of your file. A lot of people assume their score is too low when they still have options. Want help seeing what yours could qualify for?"),
   ("can I buy with bad credit?", "Possibly—depends how low it is and what else is in your file. Bad credit doesn't always mean no. Want me to see whether your profile looks workable?"),
   ("how do I get pre-approved?", "Usually it's completing an application, authorizing credit, and sharing basic income and asset docs. Sounds like a lot but most of it moves fast. Want me to outline the simplest first steps?"),
   ("what's the difference between FHA and conventional?", "They differ on down payment, mortgage insurance, and credit flexibility—neither is universally better, it depends on your numbers. Want me to compare which fits your situation?"),
   ("how long does the mortgage process take?", "It varies with documentation, the property, and underwriting, but a clean file upfront keeps it smooth. Want me to walk you through the major steps?"),
   ("do I need to talk to my wife first?", "Absolutely—that makes sense. Want me to send a simple recap you can share, then reconnect once you both review it?"),
   ("can I buy with no money down?", "In some cases yes, depending on eligibility and the program. Want me to check whether a low or no-down option might apply to you?"),
   ("are you a real person or a bot?", "Real person here—I'm the one reviewing your request and helping point you the right way."),
   ("what documents do I need?", "Usually ID, income docs, and asset statements—the exact list varies but it feels manageable once you know it. Want me to outline the common ones?")],
 "stop": [(m, "Understood—thanks for letting me know.") for m in
   ["stop", "unsubscribe", "do not contact me again", "stop texting me", "remove me from your list"]],
 "escalation": [(m, None) for m in
   ["I'm going to report you to the CFPB", "my attorney will be in touch", "this is harassment, I'll sue",
    "I want to file a formal complaint", "I'm reporting this to my lawyer"]],
 "inappropriate": [(m, None) for m in
   ["are you single?", "send me a pic", "you sound hot", "what are you wearing"]],
 "post_handoff": [(m, None) for m in
   ["ok", "sounds good", "thanks", "great", "it's been 2 days and the loan officer never reached out",
    "he still hasn't called me", "when will they contact me?"]],
}
def ctx(stage, next_key, answers, reason, agent, client, msg, is_first):
    return {"Stage": stage, "Next question key": next_key, "Answers collected": answers,
            "Ineligibility reason": reason, "Agent name": agent, "Client name": client,
            "Client's latest message": msg, "Is first message": is_first}
def make(mode, msg, anchor, rng):
    agent, client = rng.choice(AGENTS), rng.choice(CLIENTS)
    if mode == "not_now":
        bad = rng.choice(["bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"])
        reason = {"bankruptcy_past_3yr": "bankruptcy_within_3yr", "foreclosure_past_2yr": "foreclosure_within_2yr",
                  "late_payments_past_12mo": "late_mortgage_within_12mo"}[bad]
        answers = f"property_use=primary residence, {bad}=yes"
        c = ctx("ineligible", "N/A", answers, reason, agent, client, msg, "no")
    elif mode == "post_handoff":
        c = ctx("eligible", "N/A", "property_use=primary residence, bankruptcy_past_3yr=no, "
                "foreclosure_past_2yr=no, late_payments_past_12mo=no", "N/A", agent, client, msg, "no")
    else:
        c = ctx("qualifying", "property_use", "N/A", "N/A", agent, client, msg, "no")
    return {"kind": "maos", "mode": mode, "directive": D[mode], "anchor": anchor, "context": c, "case": mode}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=60, help="target records per mode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for mode, seeds in SEEDS.items():
            for _ in range(args.per):
                msg, anchor = rng.choice(seeds)
                out.write(json.dumps(make(mode, msg, anchor, rng)) + "\n")
                n += 1
    print(f"[gen_maos_scenarios] wrote {n} records ({len(SEEDS)} modes) -> {args.out}", file=sys.stderr)
if __name__ == "__main__":
    main()

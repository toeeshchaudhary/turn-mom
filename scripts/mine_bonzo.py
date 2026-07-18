import argparse, json, os, random, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_maos_scenarios import D, ctx, AGENTS, CLIENTS
MORTGAGE_RE = re.compile(
    r"\b(mortgage|refinanc|home loan|loan officer|pre-?approv|interest rate|down payment|"
    r"closing cost|escrow|new american funding|home equity|\brefi\b|underwrit|appraisal|"
    r"credit score|loan amount|purchase a home|buy a home)\b", re.I)
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
NAMES = frozenset("""james john robert michael david richard joseph thomas charles christopher daniel
matthew anthony donald steven paul andrew joshua kenneth kevin brian george edward ronald timothy jason
jeffrey ryan jacob gary nicholas eric jonathan stephen larry justin scott brandon benjamin samuel gregory
alexander patrick jack dennis jerry tyler aaron jose henry adam nathan zachary carlos kyle noah ethan jeremy
mary patricia jennifer linda elizabeth barbara susan jessica sarah karen nancy lisa margaret betty sandra
ashley kimberly emily donna michelle carol amanda dorothy melissa deborah stephanie rebecca laura sharon
cynthia kathleen amy angela shirley anna brenda pamela nicole samantha katherine christine emma catherine
debra rachel carolyn janet maria heather diane julie joyce victoria kelly christina joan evelyn olivia
lauren judith megan cheryl andrea hannah jacqueline gloria teresa sara janice marie julia kathryn frances
alexis rosa kayla dustin francisco selena justin jenna kenna anas priya diego nina omar marcus luis grace
tyler""".split())
SURR = ["Jordan", "Casey", "Taylor", "Morgan", "Riley", "Avery", "Quinn", "Reese",
        "Devon", "Skylar", "Rowan", "Sage", "Harper", "Emerson"]
def _sur(s):
    return SURR[abs(hash(s.strip().lower())) % len(SURR)]
def redact(t):
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "someone@example.com", t)
    t = re.sub(r"\+?\d[\d\-\(\)\s]{7,}\d", "(555) 010-4321", t)
    t = re.sub(r"\$\s?[\d,]+(?:\.\d+)?", "$300k", t)
    t = re.sub(r"\b\d{1,6}\s+[NSEW]\.?\s+[A-Z][a-z]+", "a property nearby", t)  
    NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    t = re.sub(rf"\b(hi|hey|hello|thanks|thank you|dear)\s+({NAME})",
               lambda m: f"{m.group(1)} {_sur(m.group(2))}", t, flags=re.I)
    t = re.sub(rf"\b(I'?m|this is|my name is|it'?s|name is)\s+({NAME}(?:\s+[A-Z][a-z]+)?)",
               lambda m: f"{m.group(1)} {_sur(m.group(2))}", t, flags=re.I)
    t = re.sub(rf"[-–—]\s*({NAME})\s*$", lambda m: f"- {_sur(m.group(1))}", t)  
    t = re.sub(r"\b[A-Z][a-z]+\b",
               lambda m: _sur(m.group(0)) if m.group(0).lower() in NAMES else m.group(0), t)
    return re.sub(r"\s{2,}", " ", t).strip()
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

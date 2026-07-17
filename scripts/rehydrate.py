"""Rehydrate redaction placeholders -> realistic surrogate values.
CRITICAL: the redacted data contains literal curly tokens ({NAME_GIVEN},
{PHONE_NUMBER}, ...). If those reach training, the student learns to EMIT them
("Hey {NAME}!") — the exact bug we saw in the old model. This step swaps every
token for a realistic surrogate BEFORE the teacher ever sees the text, so all
downstream text (history, gold reply, suggestions) is natural.
Deterministic per conversation (seeded by filename) so a rerun is stable and the
same conversation reads consistently. We also detect and attach a per-convo
`agent_name` and `client_name` surrogate (from the "this is X with New American
Funding" / "Hi X" patterns) so they can be threaded into the CONTEXT block.
Input : cleaned convo JSONL ({file, source, turns}).
Output: same, with tokens replaced + {agent_name, client_name} added.
Usage:
  python3 rehydrate.py bonzo_clean.jsonl --out bonzo_hydrated.jsonl
"""
import argparse, hashlib, json, re, sys
FIRST_NAMES = ["Marcus", "Sarah", "Diego", "Priya", "Jordan", "Alicia", "Tyler",
               "Nina", "Andre", "Grace", "Omar", "Hannah", "Luis", "Kayla",
               "Devin", "Rosa", "Ethan", "Maya", "Carlos", "Brianna"]
ORGS = ["Walmart", "Target", "the county", "Amazon", "the hospital", "FedEx"]
STATES = ["Texas", "Ohio", "Florida", "Georgia", "Arizona", "Nevada"]
CITIES = ["Dallas", "Columbus", "Tampa", "Mesa", "Reno", "Athens"]
MONEY_RE = re.compile(r"\{MONEY\}")
PHONE_RE = re.compile(r"\{PHONE_NUMBER\}")
NAME_RE = re.compile(r"\{NAME(?:_GIVEN|_SURNAME|_FAMILY)?\}")
GENERIC_RE = re.compile(r"\{[A-Z][A-Z_]*\}")
INTRO_RE = re.compile(r"this is\s+\{NAME(?:_GIVEN)?\}\s+with", re.I)
GREET_RE = re.compile(r"^\s*(hi|hey|hello)[,!\s]+\{NAME(?:_GIVEN)?\}", re.I)
class Surrogates:
    def __init__(self, seed):
        self.r = __import__("random").Random(seed)
        self.name_pool = FIRST_NAMES[:]
        self.r.shuffle(self.name_pool)
        self.i = 0
    def name(self):
        n = self.name_pool[self.i % len(self.name_pool)]
        self.i += 1
        return n
    def phone(self):
        return f"({self.r.randint(200,989)}) {self.r.randint(200,989)}-{self.r.randint(1000,9999)}"
    def money(self):
        return self.r.choice(["$285k", "$320,000", "$412k", "$150,000", "$95k", "$540,000"])
    def generic(self, tok):
        t = tok.strip("{}")
        if "ORG" in t:
            return self.r.choice(ORGS)
        if "STATE" in t:
            return self.r.choice(STATES)
        if "CITY" in t or "LOCATION" in t or "ADDRESS" in t:
            return self.r.choice(CITIES)
        if "EMAIL" in t:
            return f"{self.name().lower()}@example.com"
        if "DATE" in t:
            return "next Tuesday"
        return ""  
def rehydrate_convo(rec):
    seed = int(hashlib.md5(rec["file"].encode()).hexdigest()[:8], 16)
    s = Surrogates(seed)
    agent_name = s.name()
    client_name = s.name()
    def sub_text(text):
        text = INTRO_RE.sub(lambda m: m.group(0).replace(
            re.search(r"\{NAME(?:_GIVEN)?\}", m.group(0)).group(0), agent_name), text)
        text = GREET_RE.sub(lambda m: m.group(0).replace(
            re.search(r"\{NAME(?:_GIVEN)?\}", m.group(0)).group(0), client_name), text)
        text = NAME_RE.sub(lambda m: s.name(), text)
        text = PHONE_RE.sub(lambda m: s.phone(), text)
        text = MONEY_RE.sub(lambda m: s.money(), text)
        text = GENERIC_RE.sub(lambda m: s.generic(m.group(0)), text)
        return re.sub(r"\s{2,}", " ", text).strip()
    for t in rec["turns"]:
        t["text"] = sub_text(t["text"])
    rec["agent_name"] = agent_name
    rec["client_name"] = client_name
    return rec
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = leaks = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in args.files:
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = rehydrate_convo(json.loads(line))
                blob = json.dumps(rec)
                if "{" in blob and re.search(r"\{[A-Z][A-Z_]*\}", blob):
                    leaks += 1
                out.write(blob + "\n")
                n += 1
    print(f"[rehydrate] wrote {n} convos, {leaks} with surviving tokens -> {args.out}",
          file=sys.stderr)
if __name__ == "__main__":
    main()

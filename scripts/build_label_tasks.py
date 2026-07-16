#!/usr/bin/env python3
"""Stream A: turn cleaned conversations into per-agent-turn LABEL TASKS.

For every agent turn that has at least one preceding client turn, emit a task:
  {file, source, turn_idx, is_first, history:[{role,text}...], gold_reply:str}

`history` = everything before the agent turn. `gold_reply` = the real agent
message (this anchors the student on genuine rep voice). Downstream, the teacher
LLM infers the CONTEXT block from `history` and writes 2 alternative suggestions
around a text-normalized `gold_reply`.

We cap agent turns per conversation (--max-per-convo) so a few long threads
don't dominate, and skip agent turns that are pure logistics noise if too short.

Usage:
  python3 build_label_tasks.py bonzo_clean.jsonl calls_clean.jsonl \
      --out tasks.jsonl --max-per-convo 6
"""
import argparse, json, random, sys


def tasks_from_convo(rec, max_per):
    turns = rec["turns"]
    cands = []
    for i, t in enumerate(turns):
        if t["role"] != "agent":
            continue
        if i == 0:  # need a client message before an agent reply
            continue
        history = turns[:i]
        # must have at least one client turn in history (a "latest message")
        if not any(h["role"] == "client" for h in history):
            continue
        cands.append({
            "file": rec["file"],
            "source": rec["source"],
            "turn_idx": i,
            "is_first": (i == 1),  # agent's first reply follows the opening client msg
            "history": history,
            "gold_reply": t["text"],
            "agent_name": rec.get("agent_name", "Alex"),
            "client_name": rec.get("client_name", "there"),
        })
    if len(cands) > max_per:
        # keep the first (greeting-ish) + a random spread of the rest
        head = cands[:1]
        rest = random.sample(cands[1:], max_per - 1)
        cands = head + rest
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="cleaned convo JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-convo", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in args.files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    for task in tasks_from_convo(rec, args.max_per_convo):
                        out.write(json.dumps(task) + "\n")
                        n += 1
    print(f"[build_label_tasks] wrote {n} tasks -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

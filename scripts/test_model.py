import argparse, csv, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_with_teacher import (stream_a_user, stream_b_user, stream_maos_user,
                                parse_json, render_ctx_block)

DEF_PROMPT = os.path.join(os.path.dirname(__file__), "..", "prompts", "css_system_prompt.txt")


def chat(base, model, messages, temperature, max_tokens, timeout, retries=4):
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base}/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"call failed after {retries} tries: {last}")


def user_message(task):
    kind = task.get("kind", "scenario")
    if kind == "maos":
        return stream_maos_user(task)
    if kind == "scenario":
        return stream_b_user(task["context"])
    return stream_a_user(task)


def load_tasks(args):
    """Tasks come from --scenarios file if given, else the full exhaustive enumeration."""
    if args.scenarios:
        with open(args.scenarios, encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f if line.strip()]
    else:
        import gen_test_scenarios
        tasks = gen_test_scenarios.build_scenarios(args.name_variants)
    if args.limit:
        tasks = tasks[:args.limit]
    return tasks


def flat_ctx(task):
    ctx = task.get("context", {})
    return {
        "kind": task.get("kind", ""),
        "case": task.get("case", ""),
        "stage": ctx.get("Stage", ""),
        "next_key": ctx.get("Next question key", ""),
        "answers_collected": ctx.get("Answers collected", ""),
        "ineligibility_reason": ctx.get("Ineligibility reason", ""),
        "client_latest": ctx.get("Client's latest message", ""),
        "is_first": ctx.get("Is first message", ""),
        "agent": ctx.get("Agent name", ""),
        "client": ctx.get("Client name", ""),
    }


def run_one(idx, task, sys_prompt, args):
    row = {"idx": idx, **flat_ctx(task), "ok": 0, "n_suggestions": 0,
           "suggestion_1": "", "confidence_1": "", "suggestion_2": "", "confidence_2": "",
           "suggestion_3": "", "confidence_3": "", "raw_response": "", "error": ""}
    try:
        content = chat(args.base_url, args.model,
                       [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_message(task)}],
                       args.temperature, args.max_tokens, args.timeout)
        row["raw_response"] = content
        recs = parse_json(content).get("recommendations", [])
        row["n_suggestions"] = len(recs)
        row["ok"] = 1
        for i, r in enumerate(recs[:3], 1):
            row[f"suggestion_{i}"] = r.get("suggested_message", "")
            row[f"confidence_{i}"] = r.get("confidence", "")
    except Exception as e:
        row["error"] = str(e)
    return row


FIELDS = ["idx", "kind", "case", "stage", "next_key", "answers_collected",
          "ineligibility_reason", "client_latest", "is_first", "agent", "client",
          "ok", "n_suggestions", "suggestion_1", "confidence_1", "suggestion_2",
          "confidence_2", "suggestion_3", "confidence_3", "raw_response", "error"]


def main():
    ap = argparse.ArgumentParser(description="Query the served CSS model over many scenarios -> CSV")
    ap.add_argument("--out", default="model_test.csv")
    ap.add_argument("--scenarios", help="jsonl of tasks to run (default: full exhaustive enumeration)")
    ap.add_argument("--name-variants", type=int, default=8,
                    help="agent/client name pairings per base scenario (scales total count)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of scenarios (0 = all)")
    ap.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "naf"))
    ap.add_argument("--system-prompt", default=os.environ.get("SYSTEM_PROMPT", DEF_PROMPT))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    sys_prompt = open(args.system_prompt, encoding="utf-8").read()
    tasks = load_tasks(args)
    print(f"[test_model] {len(tasks)} tasks -> {args.model} @ {args.base_url} "
          f"(concurrency={args.concurrency})", file=sys.stderr)

    rows, done, ok = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, i, t, sys_prompt, args): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            ok += row["ok"]
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  parsed_ok={ok}", file=sys.stderr)

    rows.sort(key=lambda r: r["idx"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"[test_model] wrote {len(rows)} rows ({ok} parsed ok) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

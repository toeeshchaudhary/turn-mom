import json, os, time, urllib.request, urllib.error
BASE = os.environ.get("TEACHER_BASE_URL", "http://localhost:8001/v1")
MODEL = os.environ.get("TEACHER_MODEL", "teacher")
KEY = os.environ.get("TEACHER_API_KEY", "dummy")
TIMEOUT = int(os.environ.get("TEACHER_TIMEOUT", "600"))
# Guided JSON decoding (response_format=json_object) forces a per-request grammar that can
# CPU-bottleneck vLLM and tank throughput. Off by default — the prompt already demands JSON
# and parse_json() extracts the outer {...}. Re-enable with TEACHER_JSON=1 if outputs drift.
JSON_MODE = os.environ.get("TEACHER_JSON", "0") == "1"
def chat(messages, temperature=0.7, max_tokens=700, json_mode=None, retries=4):
    if json_mode is None:
        json_mode = JSON_MODE
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE}/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"teacher call failed after {retries} tries: {last}")

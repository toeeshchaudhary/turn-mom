import json, os, time, urllib.request, urllib.error
BASE = os.environ.get("TEACHER_BASE_URL", "http://localhost:8001/v1")
MODEL = os.environ.get("TEACHER_MODEL", "teacher")
KEY = os.environ.get("TEACHER_API_KEY", "dummy")
def chat(messages, temperature=0.7, max_tokens=700, json_mode=True, retries=4):
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
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"teacher call failed after {retries} tries: {last}")

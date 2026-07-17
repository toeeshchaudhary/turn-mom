# ChadGPT UI

Minimal Bun + React + Tailwind + shadcn front-end for the CSS reply assistant.

- **Assist mode** (default): set the CONTEXT (stage, next question, answers, agent/client names), type the client's latest message, get **3 suggested replies** with confidence; click **Use** to send one, or **Regenerate** with a revision note.
- **Chat mode** (toggle top-right): plain single-reply chat.

The Bun server (`server.ts`) reads `../prompts/css_system_prompt.txt`, builds the CONTEXT block, and proxies to your vLLM endpoint — so the system prompt never leaves the server and the browser has no CORS issues.

## Install
```bash
cd ui
bun install
```

## Run (demo — one process serves UI + API)
Point it at the running vLLM (`naf` model on :8000), then:
```bash
VLLM_URL=http://localhost:8000/v1 MODEL=naf bun run start
# -> builds the front-end and serves everything on http://localhost:8787
```
Open http://localhost:8787.

## Run (dev — hot reload)
Two terminals:
```bash
VLLM_URL=http://localhost:8000/v1 MODEL=naf bun run server   # API on :8787
bun run dev                                                   # Vite on :5173 (proxies /api -> :8787)
```

## Env
| var | default | meaning |
|---|---|---|
| `VLLM_URL` | `http://localhost:8000/v1` | your vLLM OpenAI endpoint |
| `MODEL` | `naf` | served model name |
| `PORT` | `8787` | UI/API server port |

If the UI runs on your laptop but vLLM is on the GPU box, tunnel first:
`ssh -L 8000:localhost:8000 root@<box>` then use the default `VLLM_URL`.

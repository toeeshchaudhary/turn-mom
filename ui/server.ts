import { readFileSync, existsSync } from "fs";
import { join, extname } from "path";

const PORT = Number(process.env.PORT ?? 8787);
const VLLM_URL = process.env.VLLM_URL ?? "http://localhost:8000/v1";
const MODEL = process.env.MODEL ?? "naf";
const PROMPT_PATH = process.env.PROMPT_PATH ?? "../prompts/css_system_prompt.txt";
const DIST = join(import.meta.dir, "dist");

const SYSTEM_PROMPT = readFileSync(join(import.meta.dir, PROMPT_PATH), "utf8");

const CTX_FIELDS = [
  ["stage", "Stage"],
  ["nextKey", "Next question key"],
  ["answers", "Answers collected"],
  ["ineligibilityReason", "Ineligibility reason"],
  ["agentName", "Agent name"],
  ["clientName", "Client name"],
  ["clientMessage", "Client's latest message"],
  ["isFirst", "Is first message"],
] as const;

function buildContextBlock(ctx: Record<string, string>, revise?: string) {
  const lines = ["--- CURRENT CONTEXT ---"];
  for (const [key, label] of CTX_FIELDS) {
    lines.push(`${label}: ${ctx[key]?.trim() || "N/A"}`);
  }
  lines.push("---");
  if (revise?.trim()) {
    lines.push(
      `\nThe rep did not like the previous suggestions. Feedback: "${revise.trim()}". ` +
        `Produce 3 new suggestions that address this while following all your rules.`
    );
  }
  return lines.join("\n");
}

function extractJson(text: string) {
  const s = text.indexOf("{");
  const e = text.lastIndexOf("}");
  if (s === -1 || e === -1) throw new Error("no JSON in model output");
  return JSON.parse(text.slice(s, e + 1));
}

async function suggest(body: any) {
  const user = buildContextBlock(body.context ?? {}, body.revise);
  const res = await fetch(`${VLLM_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer dummy" },
    body: JSON.stringify({
      model: MODEL,
      temperature: body.revise ? 0.9 : 0.7,
      max_tokens: 400,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`vLLM ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const raw = data.choices?.[0]?.message?.content ?? "";
  const parsed = extractJson(raw);
  return { recommendations: parsed.recommendations ?? [], contextBlock: user, raw };
}

const MIME: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".ico": "image/x-icon",
};

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/api/suggest" && req.method === "POST") {
      try {
        const out = await suggest(await req.json());
        return Response.json(out);
      } catch (e: any) {
        return Response.json({ error: String(e?.message ?? e) }, { status: 500 });
      }
    }

    if (url.pathname === "/api/health") {
      return Response.json({ ok: true, model: MODEL, vllm: VLLM_URL });
    }

    let path = url.pathname === "/" ? "/index.html" : url.pathname;
    let file = join(DIST, path);
    if (!existsSync(file)) file = join(DIST, "index.html");
    if (existsSync(file)) {
      return new Response(readFileSync(file), {
        headers: { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" },
      });
    }
    return new Response("build the front-end first: bun run build", { status: 404 });
  },
});

console.log(`ChadGPT UI server on http://localhost:${PORT}  ->  vLLM ${VLLM_URL} (model ${MODEL})`);

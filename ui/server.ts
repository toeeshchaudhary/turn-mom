import { readFileSync, existsSync } from "fs";
import { join, extname } from "path";

const PORT = Number(process.env.PORT ?? 8787);
const VLLM_URL = process.env.VLLM_URL ?? "http://localhost:8000/v1";
const MODEL = process.env.MODEL ?? "naf";
const PROMPT_PATH = process.env.PROMPT_PATH ?? "../prompts/css_system_prompt.txt";
const DIST = join(import.meta.dir, "dist");

const SYSTEM_PROMPT = readFileSync(join(import.meta.dir, PROMPT_PATH), "utf8");

const KEYS = ["property_use", "bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"];
const DISQUALIFY: Record<string, string> = {
  bankruptcy_past_3yr: "bankruptcy_within_3yr",
  foreclosure_past_2yr: "foreclosure_within_2yr",
  late_payments_past_12mo: "late_mortgage_within_12mo",
};

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
  for (const [key, label] of CTX_FIELDS) lines.push(`${label}: ${ctx[key]?.toString().trim() || "N/A"}`);
  lines.push("---");
  if (revise?.trim())
    lines.push(
      `\nThe rep did not like the previous suggestions. Feedback: "${revise.trim()}". ` +
        `Produce 3 new suggestions that address this while following all your rules.`
    );
  return lines.join("\n");
}

function extractJson(text: string) {
  const s = text.indexOf("{");
  const e = text.lastIndexOf("}");
  if (s === -1 || e === -1) throw new Error("no JSON in model output");
  return JSON.parse(text.slice(s, e + 1));
}

async function callModel(messages: any[], temperature: number, max_tokens: number) {
  const res = await fetch(`${VLLM_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer dummy" },
    body: JSON.stringify({ model: MODEL, temperature, max_tokens, messages }),
  });
  if (!res.ok) throw new Error(`vLLM ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}

async function generate(ctx: Record<string, string>, revise?: string) {
  const user = buildContextBlock(ctx, revise);
  const raw = await callModel(
    [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: user },
    ],
    revise ? 0.9 : 0.7,
    400
  );
  return { recommendations: extractJson(raw).recommendations ?? [], contextBlock: user };
}

// ---- orchestrator: deterministic screening engine + LLM answer detector ----

const CLASSIFY_SYSTEM =
  "You extract mortgage-screening answers. Given a QUESTION KEY and the client's latest message, " +
  "decide whether the message DIRECTLY answers that key, and extract the value.\n" +
  "Keys and expected values:\n" +
  "- property_use: how they will use the property (primary residence | second home | investment/rental)\n" +
  "- bankruptcy_past_3yr: yes | no\n" +
  "- foreclosure_past_2yr: yes | no\n" +
  "- late_payments_past_12mo: yes | no\n" +
  "- confirm: whether they confirm the recap is correct (yes | no)\n" +
  "If the message is a greeting, a question back, small talk, off-topic, or does not answer THIS key, " +
  'set answered=false. Return ONLY JSON: {"answered": true|false, "value": "..."}';

async function classify(key: string, message: string): Promise<{ answered: boolean; value: string }> {
  try {
    const raw = await callModel(
      [
        { role: "system", content: CLASSIFY_SYSTEM },
        { role: "user", content: `Question key: ${key}\nClient message: "${message}"` },
      ],
      0,
      60
    );
    const j = extractJson(raw);
    return { answered: !!j.answered, value: String(j.value ?? "") };
  } catch {
    return { answered: false, value: "" };
  }
}

function normalize(key: string, value: string) {
  if (key === "property_use") return value.trim() || "unclear";
  return /\b(yes|yeah|yep|had|did|filed|there was|we did)\b/i.test(value) ? "yes" : "no";
}

function firstUnanswered(answers: Record<string, string>) {
  return KEYS.find((k) => !(k in answers)) ?? null;
}

function oracleReason(answers: Record<string, string>) {
  for (const k of ["bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo"])
    if (answers[k] === "yes") return DISQUALIFY[k];
  return null;
}

function answersSummary(answers: Record<string, string>) {
  const parts = KEYS.filter((k) => k in answers).map((k) => `${k}=${answers[k]}`);
  return parts.length ? parts.join(", ") : "N/A";
}

async function turn(body: any) {
  const state = { answers: {}, stage: "qualifying", confirmed: false, ...(body.state ?? {}) };
  const answers: Record<string, string> = { ...state.answers };
  let stage: string = state.stage;
  let confirmed: boolean = state.confirmed;
  const msg: string = (body.clientMessage ?? "").trim();
  const isFirst = !!body.isFirst;

  if (msg && !isFirst) {
    if (stage === "qualifying") {
      const key = firstUnanswered(answers);
      if (key) {
        const c = await classify(key, msg);
        if (c.answered) answers[key] = normalize(key, c.value);
      }
    } else if (stage === "confirming") {
      const c = await classify("confirm", msg);
      if (c.answered && /\b(yes|correct|right|yep|yeah|looks good|good)\b/i.test(c.value)) confirmed = true;
    }
  }

  const nextKey = firstUnanswered(answers);
  if (nextKey) stage = "qualifying";
  else if (!confirmed) stage = "confirming";
  else stage = oracleReason(answers) ? "ineligible" : "eligible";

  const ctx: Record<string, string> = {
    stage,
    nextKey: nextKey ?? "N/A",
    answers: answersSummary(answers),
    ineligibilityReason: stage === "ineligible" ? oracleReason(answers) ?? "N/A" : "N/A",
    agentName: body.agentName || "Sarah",
    clientName: body.clientName || "there",
    clientMessage: msg,
    isFirst: isFirst ? "yes" : "no",
  };

  const gen = await generate(ctx);
  return { recommendations: gen.recommendations, context: ctx, state: { answers, stage, confirmed } };
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

    if (url.pathname === "/api/turn" && req.method === "POST") {
      try {
        return Response.json(await turn(await req.json()));
      } catch (e: any) {
        return Response.json({ error: String(e?.message ?? e) }, { status: 500 });
      }
    }

    if (url.pathname === "/api/suggest" && req.method === "POST") {
      try {
        const body = await req.json();
        const gen = await generate(body.context ?? {}, body.revise);
        return Response.json({ recommendations: gen.recommendations, context: body.context });
      } catch (e: any) {
        return Response.json({ error: String(e?.message ?? e) }, { status: 500 });
      }
    }

    if (url.pathname === "/api/health") return Response.json({ ok: true, model: MODEL, vllm: VLLM_URL });

    let path = url.pathname === "/" ? "/index.html" : url.pathname;
    let file = join(DIST, path);
    if (!existsSync(file)) file = join(DIST, "index.html");
    if (existsSync(file))
      return new Response(readFileSync(file), {
        headers: { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" },
      });
    return new Response("build the front-end first: bun run build", { status: 404 });
  },
});

console.log(`ChadGPT UI server on http://localhost:${PORT}  ->  vLLM ${VLLM_URL} (model ${MODEL})`);

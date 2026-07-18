import { readFileSync, existsSync } from "fs";
import { join, extname } from "path";

const PORT = Number(process.env.PORT ?? 8787);
const VLLM_URL = process.env.VLLM_URL ?? "http://localhost:8000/v1";
const MODEL = process.env.MODEL ?? "naf";
const PROMPT_PATH = process.env.PROMPT_PATH ?? "../prompts/css_maos_system_prompt.txt";
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

function buildContextBlock(ctx: Record<string, string>) {
  const lines = ["--- CURRENT CONTEXT ---"];
  for (const [key, label] of CTX_FIELDS) lines.push(`${label}: ${ctx[key]?.toString().trim() || "N/A"}`);
  lines.push("---");
  return lines.join("\n");
}

function renderHistory(history: { role: string; text: string }[]) {
  return history.map((m) => `${m.role === "client" ? "CLIENT" : "REP"}: ${m.text}`).join("\n");
}

// Regenerate = full chat context + the rep's ACTUAL revision request, not a blind re-roll.
function buildRegenPrompt(ctx: Record<string, string>, revise: string, history?: { role: string; text: string }[]) {
  const hist = history?.length ? `--- CONVERSATION SO FAR ---\n${renderHistory(history)}\n\n` : "";
  return (
    hist +
    buildContextBlock(ctx) +
    `\n\nThe rep rejected the previous suggestions and asked for this specific change:\n` +
    `"${revise.trim()}"\n\n` +
    `Produce 3 NEW suggested messages that (1) fit this exact conversation, (2) directly do what ` +
    `the rep asked, and (3) still follow every voice, stage, and guardrail rule.`
  );
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

async function generate(
  ctx: Record<string, string>,
  revise?: string,
  history?: { role: string; text: string }[],
  directive?: string
) {
  let user = revise?.trim() ? buildRegenPrompt(ctx, revise, history) : buildContextBlock(ctx);
  if (directive && !revise?.trim()) user = user + "\n\n" + directive;
  const raw = await callModel(
    [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: user },
    ],
    revise ? 0.85 : 0.7,
    400
  );
  return { recommendations: extractJson(raw).recommendations ?? [], contextBlock: user };
}

// ---- orchestrator: MAOS situation router + deterministic screening engine ----

const CLASSIFY_SYSTEM =
  "You are the situation router for a mortgage TEXT assistant (New American Funding). Read the client's " +
  "latest message and the current screening question (if any) and classify it. Return ONLY JSON:\n" +
  '{"intent":"...","emotion":"...","is_answer":true|false,"value":"..."}\n\n' +
  "intent (pick exactly ONE):\n" +
  "- answer: directly answers the current screening question\n" +
  "- greeting: hi/hello opening with no substance\n" +
  "- loan_question: asks a mortgage/process/program/rate/credit question\n" +
  "- life_event: shares a hardship or heavy life event (death, illness, hospital, job loss, divorce)\n" +
  "- not_interested: not interested, changed mind, wants to stop pursuing\n" +
  "- busy: busy, at work, can't talk now, call me later\n" +
  "- other_lender: already working with / quoted by another lender\n" +
  "- rate_shopping: asking for the rate / rate shopping\n" +
  "- stop: asks to stop, unsubscribe, do not contact\n" +
  "- escalation: legal threat, attorney, CFPB, complaint, lawsuit\n" +
  "- inappropriate: sexual, abusive, or clearly inappropriate content\n" +
  "- offtopic: unrelated to mortgage (politics, random)\n" +
  "- casual: friendly small talk (how's your day, weather)\n" +
  "- other: mortgage-related but none of the above\n\n" +
  "emotion (the client's emotional state right now): distressed | overloaded | neutral | ready\n" +
  "is_answer / value: ONLY if intent is 'answer' — whether it answers the current question and the extracted value; else false/\"\".\n" +
  "Be decisive. Any hardship/grief/illness message is intent 'life_event' with emotion 'distressed'.";

type Situation = { intent: string; emotion: string; is_answer: boolean; value: string };

async function classify(message: string, currentQuestion: string): Promise<Situation> {
  try {
    const raw = await callModel(
      [
        { role: "system", content: CLASSIFY_SYSTEM },
        { role: "user", content: `Current screening question: ${currentQuestion || "none"}\nClient message: "${message}"` },
      ],
      0,
      80
    );
    const j = extractJson(raw);
    return {
      intent: String(j.intent ?? "other"),
      emotion: String(j.emotion ?? "neutral"),
      is_answer: !!j.is_answer,
      value: String(j.value ?? ""),
    };
  } catch {
    return { intent: "other", emotion: "neutral", is_answer: false, value: "" };
  }
}

// MAOS-grounded behavior directives per routed mode. Appended to the CONTEXT block.
// (qualify / greeting / casual / offtopic are handled by the base system prompt — no directive.)
const MODE_DIRECTIVES: Record<string, string> = {
  empathy:
    "SITUATION: the client just shared something emotionally heavy (grief, illness, hospital, job loss, divorce). " +
    "Respond as a caring human, NOT a salesperson. Acknowledge their situation genuinely and briefly. Do NOT ask ANY " +
    "qualifying or screening question. Do NOT pitch, sell, or mention loans, applications, or property use. Give them " +
    "space and let them lead — you may gently offer to pick things back up whenever they're ready. All 3 suggestions short, warm, human.",
  not_now:
    "SITUATION: based on the answers, the client is not in a position to be approved right now. Do NOT say 'declined', " +
    "'rejected', 'denied', or name the specific reason/timeframe. Shift from seller to EDUCATOR: frame it gently as a " +
    "timing issue ('not quite there yet' / 'not yet'), stay warm, and invite them to reach back out when the time is right.",
  post_handoff:
    "SITUATION: you have ALREADY told the client a loan officer will reach out. Do NOT repeat that or re-pitch anything. " +
    "Reply briefly like a real person — acknowledge what they said, reassure if needed, and stop selling. If they're chasing " +
    "an update, offer to nudge the loan officer. Keep the 3 suggestions short and genuinely different.",
  objection:
    "SITUATION: the client is losing interest or pushing back. Do NOT keep qualifying. Respect it, acknowledge it, and " +
    "either softly ask what changed or leave the door open gracefully with a low-pressure next step. Never argue or hard-sell. No screening questions.",
  logistics:
    "SITUATION: the client is busy or wants to connect later. Accommodate warmly and honor their timing. Do NOT push a " +
    "qualifying question or force a call. Confirm you'll follow up when it works for them.",
  other_lender:
    "SITUATION: the client mentions another lender. Do NOT criticize the competitor. Offer a no-pressure 'clean second look / " +
    "compare apples to apples', and be gracious if they decline. No screening questions.",
  rate_shopping:
    "SITUATION: the client is asking about rates. Do NOT quote any specific rate, APR, or number. Acknowledge, then find out " +
    "if it's a purchase or refinance and what matters most (lower payment vs lower cash to close), pointing toward real numbers as the next step.",
  loan_question:
    "SITUATION: the client asked a mortgage question. Answer it briefly and accurately, then reframe to what really matters, " +
    "then offer an easy next step. Do NOT quote specific rates or APRs. Warm, clear advisor tone — not a sales pitch.",
  stop:
    "SITUATION: the client asked to stop or opt out. Produce 3 short, clean acknowledgements that fully respect it with ZERO " +
    "pushback — no questions, no pitch, no re-engagement. E.g. 'Understood — thanks for letting me know.'",
  escalation:
    "SITUATION: the client raised a legal, compliance, or complaint issue (attorney, CFPB, lawsuit). Do NOT sell or qualify. " +
    "Respond calmly and professionally, take it seriously, and let them know the right person will follow up. Keep it brief.",
  inappropriate:
    "SITUATION: the client's message is inappropriate. Redirect cleanly and professionally back to how you can help with their " +
    "mortgage. Do NOT engage the content, do NOT be preachy, awkward, or use euphemisms. Brief and unbothered.",
};

function routeMode(intent: string, emotion: string): string {
  if (intent === "stop") return "stop";
  if (intent === "escalation") return "escalation";
  if (intent === "life_event" || emotion === "distressed") return "empathy";
  if (intent === "not_interested") return "objection";
  if (intent === "busy") return "logistics";
  if (intent === "other_lender") return "other_lender";
  if (intent === "rate_shopping") return "rate_shopping";
  if (intent === "loan_question") return "loan_question";
  if (intent === "inappropriate") return "inappropriate";
  return "qualify"; // answer / greeting / casual / offtopic / other -> base system prompt
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

function stageOf(answers: Record<string, string>, confirmed: boolean) {
  if (firstUnanswered(answers)) return "qualifying";
  if (!confirmed) return "confirming";
  return oracleReason(answers) ? "ineligible" : "eligible";
}

async function turn(body: any) {
  const state = { answers: {}, stage: "qualifying", confirmed: false, ...(body.state ?? {}) };
  const answers: Record<string, string> = { ...state.answers };
  let confirmed: boolean = state.confirmed;
  const msg: string = (body.clientMessage ?? "").trim();
  const isFirst = !!body.isFirst;

  // state BEFORE this turn (drives answer-detection + post-handoff routing)
  const preKey = firstUnanswered(answers);
  const preStage = stageOf(answers, confirmed);
  const currentQuestion =
    preStage === "qualifying" ? preKey ?? "" : preStage === "confirming" ? "confirm: does the recap look right?" : "";

  // one classifier call: intent + emotion + (if answering) the value
  const sit: Situation = msg
    ? await classify(msg, currentQuestion)
    : { intent: isFirst ? "greeting" : "other", emotion: "neutral", is_answer: false, value: "" };

  let mode = isFirst && sit.intent === "greeting" ? "qualify" : routeMode(sit.intent, sit.emotion);

  // advance the screening ONLY when we're actually qualifying and they answered
  if (mode === "qualify" && msg && !isFirst && sit.is_answer) {
    if (preStage === "qualifying" && preKey) answers[preKey] = normalize(preKey, sit.value);
    else if (preStage === "confirming" && /\b(yes|correct|right|yep|yeah|good)\b/i.test(sit.value)) confirmed = true;
  }

  const nextKey = firstUnanswered(answers);
  const stage = stageOf(answers, confirmed);

  // MAOS overrides on the qualify path: don't-sell states take precedence
  if (mode === "qualify") {
    if (preStage === "eligible") mode = "post_handoff";
    else if (stage === "ineligible") mode = "not_now";
  }

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

  const gen = await generate(ctx, undefined, undefined, MODE_DIRECTIVES[mode] ?? "");
  return {
    recommendations: gen.recommendations,
    situation: { intent: sit.intent, emotion: sit.emotion, mode },
    context: ctx,
    contextBlock: gen.contextBlock,
    state: { answers, stage, confirmed },
  };
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
        const gen = await generate(body.context ?? {}, body.revise, body.history);
        return Response.json({ recommendations: gen.recommendations, context: body.context, contextBlock: gen.contextBlock });
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

import { useEffect, useRef, useState } from "react";
import { Send, RefreshCw, Sparkles, User, Headset, Loader2, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Field, Select } from "@/components/ui/field";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

type Rec = { suggested_message: string; confidence: "high" | "medium" | "low" };
type Msg = { role: "client" | "rep"; text: string };
type Ctx = {
  stage: string;
  nextKey: string;
  answers: string;
  ineligibilityReason: string;
  agentName: string;
  clientName: string;
  isFirst: string;
};

const STAGES = ["qualifying", "confirming", "eligible", "ineligible"];
const KEYS = ["property_use", "bankruptcy_past_3yr", "foreclosure_past_2yr", "late_payments_past_12mo", "N/A"];
const DEFAULT_CTX: Ctx = {
  stage: "qualifying",
  nextKey: "property_use",
  answers: "N/A",
  ineligibilityReason: "N/A",
  agentName: "Sarah",
  clientName: "Mike",
  isFirst: "no",
};
const CONF: Record<string, "default" | "secondary" | "outline"> = {
  high: "default",
  medium: "secondary",
  low: "outline",
};
const EXAMPLES = [
  "yeah i wanna refinance",
  "how's your day going?",
  "can we talk tomorrow instead?",
  "i want to pull equity out of my house",
];

export default function App() {
  const [assist, setAssist] = useState(true);
  const [ctx, setCtx] = useState<Ctx>(DEFAULT_CTX);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [recs, setRecs] = useState<Rec[] | null>(null);
  const [revise, setRevise] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const lastClient = useRef("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, recs, loading]);

  async function fetchRecs(clientMessage: string, reviseNote?: string): Promise<Rec[]> {
    const res = await fetch("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context: { ...ctx, clientMessage }, revise: reviseNote }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error ?? `HTTP ${res.status}`);
    return data.recommendations as Rec[];
  }

  async function run(clientMessage: string) {
    if (!clientMessage.trim() || loading) return;
    setError("");
    setMessages((m) => [...m, { role: "client", text: clientMessage }]);
    setInput("");
    lastClient.current = clientMessage;
    setRecs(null);
    setLoading(true);
    try {
      const r = await fetchRecs(clientMessage);
      if (assist) setRecs(r);
      else setMessages((m) => [...m, { role: "rep", text: r[0]?.suggested_message ?? "…" }]);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  async function regenerate(note?: string) {
    if (!lastClient.current || loading) return;
    setError("");
    setLoading(true);
    try {
      setRecs(await fetchRecs(lastClient.current, note));
      setRevise("");
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  function useSuggestion(r: Rec) {
    setMessages((m) => [...m, { role: "rep", text: r.suggested_message }]);
    setRecs(null);
  }

  const outgoing = (role: Msg["role"]) => (assist ? role === "rep" : role === "client");

  return (
    <div className="mx-auto flex h-screen max-w-6xl flex-col">
      <header className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-md border bg-card">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-none">ChadGPT</h1>
            <p className="mt-0.5 text-xs text-muted-foreground">CSS reply assistant · New American Funding</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-md border bg-card px-2.5 py-1.5">
            <MessageSquare className={cn("h-3.5 w-3.5", !assist ? "text-foreground" : "text-muted-foreground")} />
            <Switch checked={assist} onCheckedChange={setAssist} />
            <span className={cn("text-xs font-medium", assist ? "text-foreground" : "text-muted-foreground")}>
              Assist
            </span>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <div
        className={cn(
          "grid min-h-0 flex-1 gap-4 p-4",
          assist ? "grid-cols-1 md:grid-cols-[300px_1fr]" : "grid-cols-1"
        )}
      >
        {assist && (
          <Card className="hidden overflow-y-auto shadow-sm md:block">
            <CardContent className="grid gap-3 p-4">
              <div className="flex items-center gap-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Context</p>
                <span className="text-xs text-muted-foreground">— what the model sees</span>
              </div>
              <Field label="Stage">
                <Select value={ctx.stage} onChange={(e) => setCtx({ ...ctx, stage: e.target.value })}>
                  {STAGES.map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </Select>
              </Field>
              <Field label="Next question key">
                <Select value={ctx.nextKey} onChange={(e) => setCtx({ ...ctx, nextKey: e.target.value })}>
                  {KEYS.map((k) => (
                    <option key={k}>{k}</option>
                  ))}
                </Select>
              </Field>
              <Field label="Answers collected">
                <Input value={ctx.answers} onChange={(e) => setCtx({ ...ctx, answers: e.target.value })} />
              </Field>
              <Field label="Ineligibility reason">
                <Input
                  value={ctx.ineligibilityReason}
                  onChange={(e) => setCtx({ ...ctx, ineligibilityReason: e.target.value })}
                />
              </Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Agent">
                  <Input value={ctx.agentName} onChange={(e) => setCtx({ ...ctx, agentName: e.target.value })} />
                </Field>
                <Field label="Client">
                  <Input value={ctx.clientName} onChange={(e) => setCtx({ ...ctx, clientName: e.target.value })} />
                </Field>
              </div>
              <Field label="Is first message">
                <Select value={ctx.isFirst} onChange={(e) => setCtx({ ...ctx, isFirst: e.target.value })}>
                  <option>no</option>
                  <option>yes</option>
                </Select>
              </Field>
            </CardContent>
          </Card>
        )}

        <div className={cn("flex min-h-0 flex-col gap-3", !assist && "mx-auto w-full max-w-2xl")}>
          <Card className="flex min-h-0 flex-1 flex-col overflow-hidden shadow-sm">
            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              {messages.length === 0 && !loading && (
                <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full border bg-card">
                    {assist ? <Sparkles className="h-5 w-5" /> : <MessageSquare className="h-5 w-5" />}
                  </div>
                  <div>
                    <p className="text-sm font-medium">
                      {assist ? "Get 3 suggested replies" : "Chat with the assistant"}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {assist
                        ? "Type the client's message — the model suggests three, you pick one."
                        : "Type a message and get a single reply."}
                    </p>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {EXAMPLES.map((ex) => (
                      <button
                        key={ex}
                        onClick={() => run(ex)}
                        className="rounded-full border bg-card px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((m, i) => (
                <div key={i} className={cn("flex items-end gap-2", outgoing(m.role) && "flex-row-reverse")}>
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-card">
                    {m.role === "client" ? <User className="h-3.5 w-3.5" /> : <Headset className="h-3.5 w-3.5" />}
                  </div>
                  <div
                    className={cn(
                      "max-w-[78%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed",
                      outgoing(m.role)
                        ? "rounded-br-sm bg-primary text-primary-foreground"
                        : "rounded-bl-sm bg-muted text-foreground"
                    )}
                  >
                    {m.text}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> generating…
                </div>
              )}

              {assist && recs && (
                <div className="space-y-2 pt-1">
                  <p className="px-1 text-xs font-medium text-muted-foreground">
                    {recs.length} suggestions — pick one to send
                  </p>
                  {recs.map((r, i) => (
                    <button
                      key={i}
                      onClick={() => useSuggestion(r)}
                      className="group flex w-full items-start justify-between gap-3 rounded-lg border bg-card p-3 text-left transition-colors hover:border-foreground/30 hover:bg-accent"
                    >
                      <p className="text-sm leading-relaxed">{r.suggested_message}</p>
                      <div className="flex shrink-0 flex-col items-end gap-2">
                        <Badge variant={CONF[r.confidence] ?? "outline"}>{r.confidence}</Badge>
                        <span className="text-xs text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">
                          Use →
                        </span>
                      </div>
                    </button>
                  ))}
                  <div className="flex gap-2 pt-1">
                    <Input
                      placeholder="ask for a revision (e.g. more professional)…"
                      value={revise}
                      onChange={(e) => setRevise(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && regenerate(revise || undefined)}
                    />
                    <Button variant="outline" size="sm" onClick={() => regenerate(revise || undefined)}>
                      <RefreshCw className="h-3.5 w-3.5" /> Regenerate
                    </Button>
                  </div>
                </div>
              )}

              {error && (
                <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {error}
                </div>
              )}
              <div ref={bottom} />
            </div>

            <div className="flex items-end gap-2 border-t p-3">
              <Textarea
                rows={1}
                className="max-h-32 resize-none"
                placeholder={assist ? "Client's latest message…" : "Message…"}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    run(input);
                  }
                }}
              />
              <Button onClick={() => run(input)} disabled={loading || !input.trim()}>
                {assist ? <Sparkles className="h-4 w-4" /> : <Send className="h-4 w-4" />}
                {assist ? "Suggest" : "Send"}
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { Send, RefreshCw, Sparkles, User, Headset, Loader2, MessageSquare, RotateCcw, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Field } from "@/components/ui/field";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

type Rec = { suggested_message: string; confidence: "high" | "medium" | "low" };
type Msg = { role: "client" | "rep"; text: string };
type State = { answers: Record<string, string>; stage: string; confirmed: boolean };

const INIT: State = { answers: {}, stage: "qualifying", confirmed: false };
const KEYS: [string, string][] = [
  ["property_use", "Property use"],
  ["bankruptcy_past_3yr", "Bankruptcy · 3yr"],
  ["foreclosure_past_2yr", "Foreclosure · 2yr"],
  ["late_payments_past_12mo", "Late pmts · 12mo"],
];
const STAGE_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  qualifying: "secondary",
  confirming: "outline",
  eligible: "default",
  ineligible: "destructive",
};
const CONF: Record<string, "default" | "secondary" | "outline"> = { high: "default", medium: "secondary", low: "outline" };
const EXAMPLES = ["hey", "yeah i wanna refinance", "how's your day going?", "can we talk tomorrow?"];

export default function App() {
  const [assist, setAssist] = useState(true);
  const [agentName, setAgentName] = useState("Sarah");
  const [clientName, setClientName] = useState("Mike");
  const [state, setState] = useState<State>(INIT);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [recs, setRecs] = useState<Rec[] | null>(null);
  const [lastCtx, setLastCtx] = useState<Record<string, string> | null>(null);
  const [revise, setRevise] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [messages, recs, loading]);

  async function run(clientMessage: string) {
    if (!clientMessage.trim() || loading) return;
    const isFirst = messages.length === 0;
    setError("");
    setMessages((m) => [...m, { role: "client", text: clientMessage }]);
    setInput("");
    setRecs(null);
    setLoading(true);
    try {
      const res = await fetch("/api/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, clientMessage, isFirst, agentName, clientName }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error ?? `HTTP ${res.status}`);
      setState(data.state);
      setLastCtx(data.context);
      if (assist) setRecs(data.recommendations);
      else setMessages((m) => [...m, { role: "rep", text: data.recommendations[0]?.suggested_message ?? "…" }]);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  async function regenerate(note?: string) {
    if (!lastCtx || loading) return;
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context: lastCtx, revise: note }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error ?? `HTTP ${res.status}`);
      setRecs(data.recommendations);
      setRevise("");
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setMessages([]);
    setState(INIT);
    setRecs(null);
    setLastCtx(null);
    setInput("");
    setError("");
  }

  const outgoing = (role: Msg["role"]) => (assist ? role === "rep" : role === "client");
  const nextKey = KEYS.find(([k]) => !(k in state.answers))?.[0] ?? null;

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
            <span className={cn("text-xs font-medium", assist ? "text-foreground" : "text-muted-foreground")}>Assist</span>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <div className={cn("grid min-h-0 flex-1 gap-4 p-4", assist ? "grid-cols-1 md:grid-cols-[300px_1fr]" : "grid-cols-1")}>
        {assist && (
          <Card className="hidden overflow-y-auto p-4 shadow-sm md:block">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Screening state</p>
              <Button variant="ghost" size="sm" onClick={reset} className="h-6 px-2 text-xs">
                <RotateCcw className="h-3 w-3" /> Reset
              </Button>
            </div>

            <div className="mt-3 flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Stage</span>
              <Badge variant={STAGE_VARIANT[state.stage] ?? "secondary"}>{state.stage}</Badge>
            </div>

            <div className="mt-3 grid gap-1.5">
              <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Answers collected</span>
              {KEYS.map(([k, label]) => {
                const val = state.answers[k];
                const isNext = k === nextKey;
                return (
                  <div
                    key={k}
                    className={cn(
                      "flex items-center justify-between rounded-md border px-2 py-1.5 text-xs",
                      val ? "bg-card" : isNext ? "border-foreground/30 bg-accent" : "opacity-60"
                    )}
                  >
                    <span className="flex items-center gap-1.5">
                      {val ? <Check className="h-3 w-3 text-foreground" /> : <span className="h-3 w-3" />}
                      {label}
                    </span>
                    <span className={cn("font-medium", !val && "text-muted-foreground")}>
                      {val ?? (isNext ? "asking…" : "—")}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Field label="Agent">
                <Input value={agentName} onChange={(e) => setAgentName(e.target.value)} />
              </Field>
              <Field label="Client">
                <Input value={clientName} onChange={(e) => setClientName(e.target.value)} />
              </Field>
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
              The screening engine tracks answers and advances the stage automatically — the model only writes the reply.
            </p>
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
                    <p className="text-sm font-medium">{assist ? "Get 3 suggested replies" : "Chat with the assistant"}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {assist
                        ? "Type the client's message — the engine tracks screening, the model suggests three."
                        : "Play the client — the assistant screens you and collects your answers."}
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
                      outgoing(m.role) ? "rounded-br-sm bg-primary text-primary-foreground" : "rounded-bl-sm bg-muted"
                    )}
                  >
                    {m.text}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> thinking…
                </div>
              )}

              {assist && recs && (
                <div className="space-y-2 pt-1">
                  <p className="px-1 text-xs font-medium text-muted-foreground">{recs.length} suggestions — pick one to send</p>
                  {recs.map((r, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setMessages((m) => [...m, { role: "rep", text: r.suggested_message }]);
                        setRecs(null);
                      }}
                      className="group flex w-full items-start justify-between gap-3 rounded-lg border bg-card p-3 text-left transition-colors hover:border-foreground/30 hover:bg-accent"
                    >
                      <p className="text-sm leading-relaxed">{r.suggested_message}</p>
                      <div className="flex shrink-0 flex-col items-end gap-2">
                        <Badge variant={CONF[r.confidence] ?? "outline"}>{r.confidence}</Badge>
                        <span className="text-xs text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">Use →</span>
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
                <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{error}</div>
              )}
              <div ref={bottom} />
            </div>

            <div className="flex items-end gap-2 border-t p-3">
              {!assist && (
                <Button variant="ghost" size="icon" onClick={reset} title="Reset conversation">
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
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

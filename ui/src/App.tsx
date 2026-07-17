import { useRef, useState } from "react";
import { Send, RefreshCw, Sparkles, User, Headset, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Field, Select } from "@/components/ui/field";
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

const confColor: Record<string, string> = { high: "default", medium: "secondary", low: "muted" };

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

  async function getSuggestions() {
    if (!input.trim() || loading) return;
    const clientMessage = input.trim();
    setError("");
    setMessages((m) => [...m, { role: "client", text: clientMessage }]);
    setInput("");
    lastClient.current = clientMessage;
    setRecs(null);
    setLoading(true);
    try {
      setRecs(await fetchRecs(clientMessage));
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

  async function sendChat() {
    if (!input.trim() || loading) return;
    const clientMessage = input.trim();
    setError("");
    setMessages((m) => [...m, { role: "client", text: clientMessage }]);
    setInput("");
    setLoading(true);
    try {
      const r = await fetchRecs(clientMessage);
      setMessages((m) => [...m, { role: "rep", text: r[0]?.suggested_message ?? "…" }]);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  const send = assist ? getSuggestions : sendChat;

  return (
    <div className="mx-auto flex h-screen max-w-6xl flex-col gap-4 p-4">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/15">
            <Sparkles className="h-4 w-4 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-none">ChadGPT</h1>
            <p className="text-xs text-muted-foreground">CSS reply assistant · New American Funding</p>
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className={cn(!assist && "text-foreground")}>Chat</span>
          <Switch checked={assist} onCheckedChange={setAssist} />
          <span className={cn(assist && "text-foreground")}>Assist (3 suggestions)</span>
        </label>
      </header>

      <div className="grid flex-1 grid-cols-1 gap-4 overflow-hidden md:grid-cols-[280px_1fr]">
        {assist && (
          <Card className="overflow-y-auto">
            <CardContent className="grid gap-3 pt-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Context</p>
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

        <div className="flex min-h-0 flex-col gap-3">
          <Card className="flex-1 overflow-y-auto">
            <CardContent className="flex flex-col gap-3 pt-4">
              {messages.length === 0 && (
                <p className="m-auto text-sm text-muted-foreground">
                  {assist
                    ? "Type a client message → get 3 suggested replies to pick from."
                    : "Chat mode — type a message and get one reply."}
                </p>
              )}
              {messages.map((m, i) => (
                <div key={i} className={cn("flex gap-2", m.role === "rep" && "flex-row-reverse")}>
                  <div
                    className={cn(
                      "flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
                      m.role === "client" ? "bg-secondary" : "bg-primary/20"
                    )}
                  >
                    {m.role === "client" ? <User className="h-3.5 w-3.5" /> : <Headset className="h-3.5 w-3.5 text-primary" />}
                  </div>
                  <div
                    className={cn(
                      "max-w-[75%] rounded-lg px-3 py-2 text-sm",
                      m.role === "client" ? "bg-secondary" : "bg-primary/15"
                    )}
                  >
                    {m.text}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> generating…
                </div>
              )}

              {assist && recs && (
                <div className="grid gap-2">
                  <p className="text-xs font-medium text-muted-foreground">Pick a reply to send:</p>
                  {recs.map((r, i) => (
                    <Card key={i} className="border-primary/25 transition-colors hover:border-primary/60">
                      <CardContent className="flex items-start justify-between gap-3 p-3">
                        <p className="text-sm">{r.suggested_message}</p>
                        <div className="flex shrink-0 flex-col items-end gap-2">
                          <Badge variant={(confColor[r.confidence] as any) ?? "muted"}>{r.confidence}</Badge>
                          <Button size="sm" onClick={() => useSuggestion(r)}>
                            Use
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                  <div className="flex gap-2">
                    <Input
                      placeholder="ask for a revision (e.g. more professional)…"
                      value={revise}
                      onChange={(e) => setRevise(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && regenerate(revise)}
                    />
                    <Button variant="outline" size="sm" onClick={() => regenerate(revise || undefined)}>
                      <RefreshCw className="h-3.5 w-3.5" /> Regenerate
                    </Button>
                  </div>
                </div>
              )}

              {error && <p className="text-xs text-red-400">⚠ {error}</p>}
            </CardContent>
          </Card>

          <div className="flex gap-2">
            <Textarea
              rows={1}
              placeholder={assist ? "Client's latest message…" : "Message…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <Button onClick={send} disabled={loading || !input.trim()}>
              {assist ? <Sparkles className="h-4 w-4" /> : <Send className="h-4 w-4" />}
              {assist ? "Suggest" : "Send"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send, Loader2, Sparkles, Zap, Check, XCircle, Tag, AtSign } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { askAssistant, confirmAssistantAction, searchTaggables, type TaggedRef } from "@/lib/actions/chatbot";
import { cn } from "@/lib/utils";

interface PendingAction {
  name: string;
  args: Record<string, any>;
  label: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  text?: string;
  tags?: string[];
  pendingAction?: PendingAction;
  actionState?: "pending" | "confirmed" | "cancelled";
}

interface Taggable { type: string; id: string; label: string; sub: string }

// Strip any markdown the model might still emit, so bubbles never show literal
// **asterisks**, # headings, or - bullets.
const cleanText = (t?: string) =>
  (t || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/(^|\s)\*(?!\s)(.*?)\*/g, "$1$2")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*[-*]\s+/gm, "• ");

const SUGGESTED_QUESTIONS = [
  "What's our sales pipeline look like?",
  "How's this month's finance forecast?",
  "Type @ and pick a lead — then ask how to pitch them",
];

export function ChatbotWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", text: "Hi! I'm your BAS assistant. Ask about sales, finance, inventory, HR — or type @ to tag a lead/customer and ask how to pitch or handle them." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState<number | null>(null);

  // @-mention state
  const [tagged, setTagged] = useState<Taggable[]>([]);
  const [mentionQuery, setMentionQuery] = useState<string | null>(null);
  const [mentionResults, setMentionResults] = useState<Taggable[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, open]);

  // Search as the user types "@name"
  useEffect(() => {
    if (mentionQuery === null) return;
    let active = true;
    const t = setTimeout(async () => {
      const res = await searchTaggables(mentionQuery);
      if (active && res.success) setMentionResults((res.results as Taggable[]) || []);
    }, 150);
    return () => { active = false; clearTimeout(t); };
  }, [mentionQuery]);

  function onInputChange(v: string) {
    setInput(v);
    const m = v.match(/@([\p{L}\w]*)$/u);
    setMentionQuery(m ? m[1] : null);
    if (!m) setMentionResults([]);
  }

  function pickMention(t: Taggable) {
    setInput((prev) => prev.replace(/@[\p{L}\w]*$/u, ""));
    setTagged((prev) => (prev.some((x) => x.id === t.id) ? prev : [...prev, t]));
    setMentionQuery(null);
    setMentionResults([]);
  }

  function removeTag(id: string) {
    setTagged((prev) => prev.filter((t) => t.id !== id));
  }

  async function send(text: string) {
    if ((!text.trim() && tagged.length === 0) || loading) return;
    const sentTags = tagged;
    setMessages((prev) => [...prev, { role: "user", text: text.trim() || "(about the tagged record)", tags: sentTags.map((t) => t.label) }]);
    setInput("");
    setTagged([]);
    setMentionQuery(null);
    setLoading(true);

    const refs: TaggedRef[] = sentTags.map((t) => ({ type: t.type as "lead" | "contact", id: t.id }));
    const res = await askAssistant(text, refs);
    if (res.success && res.kind === "text") {
      setMessages((prev) => [...prev, { role: "assistant", text: res.reply }]);
    } else if (res.success && res.kind === "confirm") {
      const pendingAction: PendingAction = { ...res.action, label: res.label };
      setMessages((prev) => [...prev, { role: "assistant", pendingAction, actionState: "pending" }]);
    } else {
      setMessages((prev) => [...prev, { role: "assistant", text: `⚠️ ${(res as any).error}` }]);
    }
    setLoading(false);
  }

  async function handleConfirm(index: number, action: PendingAction) {
    setConfirming(index);
    const res = await confirmAssistantAction(action.name, action.args);
    setMessages((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], actionState: "confirmed" };
      return [...next, { role: "assistant", text: res.success ? `✅ ${res.message}` : `⚠️ ${res.error}` }];
    });
    setConfirming(null);
  }

  function handleCancel(index: number) {
    setMessages((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], actionState: "cancelled" };
      return [...next, { role: "assistant", text: "Cancelled — no changes made." }];
    });
  }

  return (
    <>
      <Button
        onClick={() => setOpen((o) => !o)}
        className="fixed bottom-6 right-6 z-[60] h-14 w-14 rounded-full shadow-2xl shadow-primary/30 bg-gradient-to-br from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 p-0"
        aria-label="Open AI assistant"
      >
        {open ? <X className="w-6 h-6 text-white" /> : <MessageCircle className="w-6 h-6 text-white" />}
      </Button>

      {open && (
        <div className="fixed bottom-24 right-6 z-[60] w-[380px] max-w-[calc(100vw-3rem)] h-[520px] max-h-[70vh] bg-card border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-200">
          <div className="px-4 py-3 border-b border-border bg-gradient-to-r from-indigo-600 to-violet-600 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-white" />
            <span className="font-bold text-sm text-white">BAS Assistant</span>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m, i) => (
              <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                {m.pendingAction ? (
                  <div className="max-w-[90%] rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 space-y-2">
                    <div className="flex items-center gap-1.5 text-xs font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wide">
                      <Zap className="w-3.5 h-3.5" /> Confirm action
                    </div>
                    <p className="text-sm text-foreground">{m.pendingAction.label}</p>
                    {m.actionState === "pending" && (
                      <div className="flex gap-2 pt-1">
                        <Button size="sm" className="h-7 text-xs bg-emerald-600 hover:bg-emerald-700" disabled={confirming === i} onClick={() => handleConfirm(i, m.pendingAction!)}>
                          {confirming === i ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Check className="w-3 h-3 mr-1" />}
                          Confirm
                        </Button>
                        <Button size="sm" variant="outline" className="h-7 text-xs" disabled={confirming === i} onClick={() => handleCancel(i)}>
                          <XCircle className="w-3 h-3 mr-1" /> Cancel
                        </Button>
                      </div>
                    )}
                    {m.actionState === "confirmed" && <p className="text-[11px] text-emerald-600 font-semibold">Confirmed ✓</p>}
                    {m.actionState === "cancelled" && <p className="text-[11px] text-muted-foreground font-semibold">Cancelled</p>}
                  </div>
                ) : (
                  <div className={cn("max-w-[85%] rounded-xl px-3 py-2 text-sm leading-relaxed", m.role === "user" ? "bg-primary text-primary-foreground selection:bg-white selection:text-primary" : "bg-muted text-foreground")}>
                    {m.tags && m.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-1.5">
                        {m.tags.map((t, k) => (
                          <span key={k} className="inline-flex items-center gap-1 text-[10px] font-semibold bg-white/20 rounded px-1.5 py-0.5">
                            <Tag className="w-2.5 h-2.5" /> {t}
                          </span>
                        ))}
                      </div>
                    )}
                    {m.role === "assistant" ? cleanText(m.text) : m.text}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-muted rounded-xl px-3 py-2 text-sm flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Thinking...
                </div>
              </div>
            )}
          </div>

          {messages.length <= 1 && (
            <div className="px-3 pb-2 flex flex-wrap gap-1.5">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button key={q} onClick={() => onInputChange(q.startsWith("Type @") ? "@" : q)} className="text-[11px] px-2.5 py-1.5 rounded-full bg-muted hover:bg-muted/70 text-muted-foreground border border-border/50 transition-colors">
                  {q}
                </button>
              ))}
            </div>
          )}

          <div className="relative">
            {/* @-mention dropdown */}
            {mentionQuery !== null && mentionResults.length > 0 && (
              <div className="absolute bottom-full left-3 right-3 mb-1 max-h-48 overflow-y-auto bg-popover border border-border rounded-xl shadow-2xl z-10">
                <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-muted-foreground border-b border-border/50 flex items-center gap-1">
                  <AtSign className="w-3 h-3" /> Tag a record
                </div>
                {mentionResults.map((r) => (
                  <button key={r.id} onClick={() => pickMention(r)} className="w-full text-left px-3 py-2 hover:bg-muted flex flex-col">
                    <span className="text-sm font-medium text-foreground">{r.label}</span>
                    <span className="text-[11px] text-muted-foreground">{r.sub}</span>
                  </button>
                ))}
              </div>
            )}

            {/* Tagged chips */}
            {tagged.length > 0 && (
              <div className="px-3 pt-2 flex flex-wrap gap-1.5">
                {tagged.map((t) => (
                  <span key={t.id} className="inline-flex items-center gap-1 text-[11px] font-semibold bg-primary/10 text-primary rounded-full pl-2 pr-1 py-0.5 border border-primary/20">
                    <Tag className="w-3 h-3" /> {t.label}
                    <button onClick={() => removeTag(t.id)} className="hover:bg-primary/20 rounded-full p-0.5"><X className="w-3 h-3" /></button>
                  </span>
                ))}
              </div>
            )}

            <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="p-3 border-t border-border flex gap-2">
              <Input
                value={input}
                onChange={(e) => onInputChange(e.target.value)}
                placeholder="Ask, or type @ to tag a lead/customer..."
                className="h-9 text-sm"
                disabled={loading}
              />
              <Button type="submit" size="icon" className="h-9 w-9 shrink-0" disabled={loading || (!input.trim() && tagged.length === 0)}>
                <Send className="w-4 h-4" />
              </Button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}

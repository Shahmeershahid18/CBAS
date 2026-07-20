import { getAiInsights } from "@/lib/actions/ai-insights";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles, Target, UserMinus, MessageSquareHeart } from "lucide-react";

interface Segment { label: string; value: number; className: string }

function DistributionBar({ segments, scored, total, emptyLabel }: {
    segments: Segment[]; scored: number; total: number; emptyLabel: string;
}) {
    const sum = segments.reduce((s, x) => s + x.value, 0);
    return (
        <div className="space-y-2">
            <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
                {sum === 0
                    ? <div className="h-full w-full bg-muted" />
                    : segments.map((s, i) => (
                        <div key={i} className={s.className} style={{ width: `${(s.value / sum) * 100}%` }} title={`${s.label}: ${s.value}`} />
                    ))}
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1">
                {segments.map((s, i) => (
                    <span key={i} className="inline-flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                        <span className={`h-2 w-2 rounded-full ${s.className}`} /> {s.label} <b className="text-foreground">{s.value}</b>
                    </span>
                ))}
            </div>
            <p className="text-[11px] text-muted-foreground/70">
                {scored > 0 ? `${scored} of ${total} scored` : emptyLabel}
            </p>
        </div>
    );
}

export async function AiInsightsWidget() {
    const res = await getAiInsights();
    if (!res.success || !res.data) return null;
    const { leadScoring, churn, sentiment, engine } = res.data;

    const anyLeadModel = engine.models?.lead_scoring?.ready;

    return (
        <Card className="border-border/60 bg-card rounded-3xl shadow-sm overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 border-b border-border/50">
                <CardTitle className="flex items-center gap-2.5 text-base font-black tracking-tight">
                    <span className="bg-gradient-to-br from-indigo-500/15 to-teal-500/15 text-indigo-500 p-2 rounded-xl border border-indigo-500/20">
                        <Sparkles className="h-4 w-4" />
                    </span>
                    AI Insights
                </CardTitle>
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider ${
                    engine.online ? "bg-green-500/10 text-green-600 dark:text-green-400 ring-1 ring-green-500/20"
                        : "bg-red-500/10 text-red-600 dark:text-red-400 ring-1 ring-red-500/20"
                }`}>
                    <span className={`h-2 w-2 rounded-full ${engine.online ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                    {engine.online ? "Engine online" : "Engine offline"}
                </span>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-5">
                {/* Lead scoring */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                        <Target className="h-4 w-4 text-indigo-500" /> Lead Scoring
                    </div>
                    <DistributionBar
                        scored={leadScoring.scored} total={leadScoring.total}
                        emptyLabel={anyLeadModel ? "No leads scored yet — open a lead and click Score." : "Start the AI Engine to score leads."}
                        segments={[
                            { label: "Hot", value: leadScoring.hot, className: "bg-red-500" },
                            { label: "Warm", value: leadScoring.warm, className: "bg-amber-500" },
                            { label: "Cold", value: leadScoring.cold, className: "bg-slate-400" },
                        ]}
                    />
                </div>
                {/* Churn */}
                <div className="space-y-3 md:border-l md:border-border/50 md:pl-6">
                    <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                        <UserMinus className="h-4 w-4 text-rose-500" /> Churn Risk
                    </div>
                    <DistributionBar
                        scored={churn.scored} total={churn.total}
                        emptyLabel="No contacts scored yet — use Predict Churn Risk."
                        segments={[
                            { label: "High", value: churn.high, className: "bg-red-500" },
                            { label: "Medium", value: churn.medium, className: "bg-amber-500" },
                            { label: "Low", value: churn.low, className: "bg-green-500" },
                        ]}
                    />
                </div>
                {/* Sentiment */}
                <div className="space-y-3 md:border-l md:border-border/50 md:pl-6">
                    <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                        <MessageSquareHeart className="h-4 w-4 text-teal-500" /> Note Sentiment
                    </div>
                    <DistributionBar
                        scored={sentiment.total} total={sentiment.total}
                        emptyLabel="No analysed notes yet — log a note on a lead."
                        segments={[
                            { label: "Positive", value: sentiment.positive, className: "bg-green-500" },
                            { label: "Neutral", value: sentiment.neutral, className: "bg-slate-400" },
                            { label: "Negative", value: sentiment.negative, className: "bg-red-500" },
                        ]}
                    />
                </div>
            </CardContent>
        </Card>
    );
}

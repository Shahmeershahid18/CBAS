/**
 * Bulk AI scoring — scores EVERY lead and predicts churn for EVERY contact in
 * the database, writing the results back onto the records (same fields the
 * in-app "Score" / "Predict Churn Risk" buttons use). Handy for populating the
 * dashboard AI Insights widget or a demo in one shot.
 *
 * Requires the AI Engine to be running (AI_ENGINE_URL, default 127.0.0.1:8088)
 * with the lead-scoring and CRM churn models trained.
 *
 * Run:
 *   npx tsx scripts/bulk-ai-score.ts            # score leads + contacts
 *   npx tsx scripts/bulk-ai-score.ts leads      # only leads
 *   npx tsx scripts/bulk-ai-score.ts churn      # only contacts (churn)
 *   npx tsx scripts/bulk-ai-score.ts --new      # skip already-scored records
 */
import "dotenv/config";
import { PrismaClient } from "../src/generated/prisma/client/client";
import { PrismaPg } from "@prisma/adapter-pg";
import pg from "pg";

const AI = process.env.AI_ENGINE_URL || "http://127.0.0.1:8088";
const KEY = process.env.AI_ENGINE_API_KEY || "";
const CONCURRENCY = 5; // parallel requests to the engine

const pool = new pg.Pool({ connectionString: `${process.env.DATABASE_URL}` });
const prisma = new PrismaClient({ adapter: new PrismaPg(pool as any) });

const args = process.argv.slice(2);
const NEW_ONLY = args.includes("--new");
const onlyChurn = args.includes("churn") && !args.includes("leads");
const onlyLeads = args.includes("leads") && !args.includes("churn");

async function post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${AI}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(KEY ? { "x-ai-engine-key": KEY } : {}) },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${path} -> ${res.status}: ${(await res.text().catch(() => "")).slice(0, 120)}`);
    return res.json() as Promise<T>;
}

/** Run `fn` over `items` with bounded concurrency, showing a progress counter. */
async function runPool<T>(label: string, items: T[], fn: (item: T) => Promise<void>) {
    let done = 0, ok = 0, fail = 0;
    let i = 0;
    async function worker() {
        while (i < items.length) {
            const item = items[i++];
            try { await fn(item); ok++; } catch { fail++; }
            done++;
            if (done % 5 === 0 || done === items.length) {
                process.stdout.write(`\r  ${label}: ${done}/${items.length} (ok ${ok}, failed ${fail})   `);
            }
        }
    }
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, items.length) }, worker));
    process.stdout.write("\n");
    return { ok, fail };
}

// ---- Lead scoring ----------------------------------------------------------
async function scoreLeads() {
    const where = NEW_ONLY ? { aiScore: null } : {};
    const leads = await prisma.lead.findMany({
        where,
        select: { id: true, source: true, service: true, quotation: true, email: true, phone: true,
                  _count: { select: { activities: true } } },
    });
    console.log(`\nScoring ${leads.length} lead(s)...`);
    if (!leads.length) return;

    await runPool("leads", leads, async (lead) => {
        const r = await post<{ score: number; band: string; reason: string }>("/score-lead-crm", {
            source: lead.source || undefined,
            service: lead.service || undefined,
            quotation: lead.quotation ?? 0,
            num_activities: lead._count.activities,
            has_email: lead.email ? 1 : 0,
            has_phone: lead.phone ? 1 : 0,
        });
        await prisma.lead.update({
            where: { id: lead.id },
            data: { aiScore: r.score, aiScoreBand: r.band, aiScoreReason: r.reason, aiScoredAt: new Date() },
        });
    });
}

// ---- Contact churn ---------------------------------------------------------
const POLARITY: Record<string, number> = { positive: 1, negative: -1, neutral: 0 };

async function predictChurn() {
    const where = NEW_ONLY ? { churnScore: null } : {};
    const contacts = await prisma.contact.findMany({
        where,
        select: {
            id: true, createdAt: true,
            deal: { select: { stage: true } },
            activities: { select: { sentiment: true } },
            orders: { select: { total: true, items: { select: { product: { select: { category: true } } } } } },
        },
    });
    console.log(`\nPredicting churn for ${contacts.length} contact(s)...`);
    if (!contacts.length) return;

    await runPool("contacts", contacts, async (c) => {
        const acts = c.activities;
        const numActivities = acts.length;
        const numNegative = acts.filter((a) => a.sentiment === "negative").length;
        const avgSentiment = numActivities
            ? acts.reduce((s, a) => s + (POLARITY[a.sentiment ?? "neutral"] ?? 0), 0) / numActivities
            : 0;
        const cats = new Set<string>();
        c.orders.forEach((o) => o.items.forEach((it) => it.product.category && cats.add(it.product.category)));
        const hasOpenDeal = c.deal && !["WON", "LOST"].includes(c.deal.stage) ? 1 : 0;

        const r = await post<{ risk_score: number; risk_band: string; reason: string }>("/predict-churn-crm", {
            tenure_days: Math.max(0, Math.round((Date.now() - new Date(c.createdAt).getTime()) / 86400000)),
            num_orders: c.orders.length,
            total_spend: Math.round(c.orders.reduce((s, o) => s + (o.total || 0), 0)),
            num_activities: numActivities,
            num_negative_notes: numNegative,
            avg_sentiment: Number(avgSentiment.toFixed(3)),
            num_categories: cats.size,
            has_open_deal: hasOpenDeal,
        });
        await prisma.contact.update({
            where: { id: c.id },
            data: { churnScore: r.risk_score, churnBand: r.risk_band, churnReason: r.reason, churnScoredAt: new Date() },
        });
    });
}

async function main() {
    // Fail fast if the engine is down.
    try {
        const h = await fetch(`${AI}/health`).then((r) => r.json());
        console.log(`AI Engine at ${AI}: ${h.status} (models: ${Object.keys(h.models || {}).join(", ")})`);
    } catch {
        throw new Error(`Cannot reach the AI Engine at ${AI}. Start it: cd ai-engine && uvicorn app.main:app --port 8088`);
    }

    if (!onlyChurn) await scoreLeads();
    if (!onlyLeads) await predictChurn();
    console.log("\nDone. Refresh the dashboard — the AI Insights widget will reflect the new scores.");
}

main()
    .catch((e) => { console.error("\n" + (e.message || e)); process.exit(1); })
    .finally(() => prisma.$disconnect());

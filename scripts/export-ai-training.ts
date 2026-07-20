/**
 * Exports the workspace's CRM data into CSVs the AI Engine trainers consume:
 *   ai-engine/data/crm_orders.csv  — user_id,item_id,item_name,quantity  (recommender)
 *   ai-engine/data/crm_churn.csv   — engineered features + churn label     (churn model)
 *
 * Churn label = a customer is "lapsed" if their most recent activity is older
 * than LAPSE_DAYS (or they have never been engaged and are past that age). The
 * feature set deliberately EXCLUDES that recency, so the model must predict
 * lapse from engagement signals (orders, spend, sentiment, deal state) rather
 * than from the quantity that defines the label.
 *
 * Run:  npx tsx scripts/export-ai-training.ts
 */
import "dotenv/config";
import { PrismaClient } from "../src/generated/prisma/client/client";
import { PrismaPg } from "@prisma/adapter-pg";
import pg from "pg";
import fs from "fs";
import path from "path";

const pool = new pg.Pool({ connectionString: `${process.env.DATABASE_URL}` });
const prisma = new PrismaClient({ adapter: new PrismaPg(pool as any) });

const LAPSE_DAYS = 60;
const DATA_DIR = path.join(process.cwd(), "ai-engine", "data");

const csvCell = (v: any) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
const writeCsv = (file: string, header: string[], rows: any[][]) => {
    const out = [header.join(","), ...rows.map((r) => r.map(csvCell).join(","))].join("\n");
    fs.writeFileSync(path.join(DATA_DIR, file), out);
    console.log(`  wrote ${file}  (${rows.length} rows)`);
};

const daysSince = (d: Date) => (Date.now() - new Date(d).getTime()) / (24 * 3600 * 1000);

async function main() {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    const owner = await prisma.user.findFirst({ where: { email: "demo@digixcrm.com" } });
    if (!owner) throw new Error("Demo user not found.");
    const workspaceId = (owner as any).activeWorkspaceId as string;

    console.log("Exporting CRM training data...");

    // --- Orders CSV (recommender) -----------------------------------------
    const items = await prisma.orderItem.findMany({
        where: { order: { workspaceId } },
        select: { quantity: true, order: { select: { contactId: true } }, product: { select: { id: true, name: true } } },
    });
    const orderRows = items
        .filter((it) => it.order?.contactId)
        .map((it) => [it.order!.contactId, it.product.id, it.product.name, it.quantity]);
    writeCsv("crm_orders.csv", ["user_id", "item_id", "item_name", "quantity"], orderRows);

    // --- Churn CSV (churn model) ------------------------------------------
    const contacts = await prisma.contact.findMany({
        where: { workspaceId },
        select: {
            id: true, createdAt: true, dealId: true,
            deal: { select: { stage: true } },
            activities: { select: { createdAt: true, sentiment: true } },
            orders: { select: { total: true, items: { select: { product: { select: { category: true } } } } } },
        },
    });

    const polarity: Record<string, number> = { positive: 1, negative: -1, neutral: 0 };
    const churnRows = contacts.map((c) => {
        const tenureDays = Math.round(daysSince(c.createdAt));
        const acts = c.activities;
        const numActivities = acts.length;
        const lastActivityAge = numActivities
            ? Math.min(...acts.map((a) => daysSince(a.createdAt)))
            : tenureDays;
        const numNegative = acts.filter((a) => a.sentiment === "negative").length;
        const avgSentiment = numActivities
            ? acts.reduce((s, a) => s + (polarity[a.sentiment ?? "neutral"] ?? 0), 0) / numActivities
            : 0;
        const numOrders = c.orders.length;
        const totalSpend = c.orders.reduce((s, o) => s + (o.total || 0), 0);
        const categories = new Set<string>();
        c.orders.forEach((o) => o.items.forEach((it) => it.product.category && categories.add(it.product.category)));
        const hasOpenDeal = c.deal && !["WON", "LOST"].includes(c.deal.stage) ? 1 : 0;

        // Label: lapsed if no engagement within the window.
        const churned = lastActivityAge > LAPSE_DAYS ? 1 : 0;

        return [
            tenureDays, numOrders, Math.round(totalSpend), numActivities,
            numNegative, Number(avgSentiment.toFixed(3)), categories.size, hasOpenDeal, churned,
        ];
    });
    writeCsv("crm_churn.csv",
        ["tenure_days", "num_orders", "total_spend", "num_activities", "num_negative_notes", "avg_sentiment", "num_categories", "has_open_deal", "churned"],
        churnRows);

    const churnRate = (churnRows.filter((r) => r[8] === 1).length / churnRows.length * 100).toFixed(1);
    console.log(`Churn base rate: ${churnRate}%  (${churnRows.length} contacts)`);
}

main().catch((e) => { console.error(e); process.exit(1); }).finally(() => prisma.$disconnect());

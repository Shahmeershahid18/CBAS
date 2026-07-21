/**
 * Seeds an APP-NATIVE dataset for the AI models to train on: products, contacts
 * with a latent taste + engagement profile, orders (for recommendations), and
 * time-spread activities with sentiment (for churn). The behaviour is generated,
 * but it lives entirely in the CBAS schema, so the models train on real
 * products, customers, and engagement signals rather than external Kaggle data.
 *
 * Idempotent: it removes anything it previously seeded (products, orders, and
 * contacts tagged @ai-seed.demo) before recreating.
 *
 * Run:  npx tsx scripts/seed-ai-demo.ts
 */
import "dotenv/config";
import { PrismaClient } from "../src/generated/prisma/client/client";
import { PrismaPg } from "@prisma/adapter-pg";
import pg from "pg";
import { randomUUID } from "crypto";

const pool = new pg.Pool({ connectionString: `${process.env.DATABASE_URL}` });
const adapter = new PrismaPg(pool as any);
const prisma = new PrismaClient({ adapter });

// Digital services offered by a software house (the "products" a CRM customer buys).
const CATEGORIES: Record<string, string[]> = {
    "Web & Mobile": ["Website Development", "E-commerce Store", "iOS App Development", "Android App Development", "Progressive Web App", "API Integration", "Website Redesign", "Landing Page Design"],
    "Cloud & DevOps": ["Cloud Hosting", "Managed VPS", "Cloud Migration", "DevOps Setup", "Database Management", "Server Monitoring", "Backup & Recovery", "SSL & Domain Setup"],
    "Marketing & SEO": ["SEO Package", "Google Ads Management", "Social Media Management", "Content Writing", "Email Marketing Campaign", "Branding Package", "PPC Campaign", "Analytics Setup"],
    "Software & Support": ["CRM License", "Custom Software Development", "UI/UX Design", "QA & Testing", "Annual Maintenance Contract", "Technical Support Plan", "Software Audit", "Staff Augmentation"],
};

const FIRST = ["Alex", "Sam", "Jordan", "Casey", "Riley", "Taylor", "Morgan", "Jamie", "Avery", "Quinn", "Drew", "Reese", "Skyler", "Cameron", "Parker", "Rowan", "Sage", "Blake", "Hayden", "Emerson"];
const LAST = ["Carter", "Reed", "Bishop", "Hale", "Nolan", "Frost", "Vance", "Beck", "Cross", "Dane", "Ellis", "Flynn", "Grant", "Hart", "Iverson", "Jennings", "Knox", "Lowe", "Mercer", "Nash"];

const POS_NOTES = ["Great call, very happy with the service.", "Loved the new features, will expand usage.", "Renewed early, excellent support experience.", "Positive feedback on delivery speed."];
const NEG_NOTES = ["Frustrated with the latest issue, considering alternatives.", "Complained about slow support response.", "Unhappy with pricing changes.", "Reported a recurring bug, wants a fix."];
const NEU_NOTES = ["Checked in on the account status.", "Discussed the roadmap, no decision yet.", "Standard follow-up, nothing new.", "Requested documentation."];

const rand = (n: number) => Math.floor(Math.random() * n);
const pick = <T,>(a: T[]) => a[rand(a.length)];
const poisson = (lambda: number) => {
    const L = Math.exp(-lambda); let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
};
const daysAgo = (d: number) => new Date(Date.now() - d * 24 * 60 * 60 * 1000);

async function main() {
    console.log("Seeding app-native AI dataset...");
    const owner = await prisma.user.findFirst({ where: { email: "demo@digixcrm.com" } });
    if (!owner) throw new Error("Demo user not found. Run: npm run db:seed:demo first.");
    const workspaceId = (owner as any).activeWorkspaceId as string;
    if (!workspaceId) throw new Error("Demo user has no active workspace.");

    // --- Clean previous AI seed -------------------------------------------
    console.log("Clearing previous AI seed...");
    await prisma.orderItem.deleteMany({ where: { order: { workspaceId } } });
    await prisma.order.deleteMany({ where: { workspaceId } });
    const oldContacts = await prisma.contact.findMany({ where: { workspaceId, email: { endsWith: "@ai-seed.demo" } }, select: { id: true } });
    const oldIds = oldContacts.map((c) => c.id);
    if (oldIds.length) {
        await prisma.activity.deleteMany({ where: { contactId: { in: oldIds } } });
        await prisma.contact.deleteMany({ where: { id: { in: oldIds } } });
    }
    const oldLeads = await prisma.lead.findMany({ where: { workspaceId, email: { endsWith: "@ai-seed.demo" } }, select: { id: true } });
    const oldLeadIds = oldLeads.map((l) => l.id);
    if (oldLeadIds.length) {
        await prisma.activity.deleteMany({ where: { leadId: { in: oldLeadIds } } });
        await prisma.lead.deleteMany({ where: { id: { in: oldLeadIds } } });
    }
    await prisma.product.deleteMany({ where: { workspaceId } });

    // --- Products ----------------------------------------------------------
    const productRows: any[] = [];
    for (const [category, names] of Object.entries(CATEGORIES)) {
        names.forEach((name, i) => productRows.push({
            id: randomUUID(), name, category, workspaceId,
            sku: `${category.slice(0, 3).toUpperCase()}-${String(i + 1).padStart(3, "0")}`,
            unitPrice: 15 + rand(400), currentStock: 20 + rand(200),
        }));
    }
    await prisma.product.createMany({ data: productRows });
    const productsByCat: Record<string, any[]> = {};
    for (const p of productRows) (productsByCat[p.category] ??= []).push(p);
    const cats = Object.keys(CATEGORIES);
    console.log(`  ${productRows.length} products`);

    // --- Contacts with latent segment + engagement -------------------------
    const N_CONTACTS = 400;
    const contactRows: any[] = [];
    const profiles: { id: string; prefs: string[]; p: number }[] = [];
    for (let i = 0; i < N_CONTACTS; i++) {
        const id = randomUUID();
        const prefs = [...cats].sort(() => Math.random() - 0.5).slice(0, 2);
        const p = Math.random();                          // engagement propensity 0..1
        const tenure = 30 + rand(690);                    // account age (days)
        contactRows.push({
            id, firstName: pick(FIRST), lastName: pick(LAST),
            email: `${id.slice(0, 8)}@ai-seed.demo`,
            phone: `+1${200 + rand(799)}${1000000 + rand(8999999)}`,
            workspaceId, ownerId: owner.id, createdAt: daysAgo(tenure),
        });
        profiles.push({ id, prefs, p });
    }
    await prisma.contact.createMany({ data: contactRows });
    const tenureById = new Map(contactRows.map((c) => [c.id, c.createdAt as Date]));
    console.log(`  ${contactRows.length} contacts`);

    // Give EXISTING (non-seed) contacts a taste profile too, so every contact —
    // including the original demo contacts — has purchase history and gets
    // personalized recommendations rather than the popularity fallback.
    const existing = await prisma.contact.findMany({
        where: { workspaceId, NOT: { email: { endsWith: "@ai-seed.demo" } } },
        select: { id: true },
    });
    const orderProfiles = [
        ...profiles,
        ...existing.map((c) => ({ id: c.id, prefs: [...cats].sort(() => Math.random() - 0.5).slice(0, 2), p: Math.random() })),
    ];
    console.log(`  ${existing.length} existing contacts also given purchase history`);

    // --- Orders + items (recommendation signal) ----------------------------
    const orderRows: any[] = [];
    const itemRows: any[] = [];
    for (const { id: contactId, prefs, p } of orderProfiles) {
        const nOrders = Math.max(1, poisson(1 + p * 6));
        for (let o = 0; o < nOrders; o++) {
            const orderId = randomUUID();
            const nItems = 1 + rand(3);
            let total = 0;
            const chosen = new Set<string>();
            for (let k = 0; k < nItems; k++) {
                const cat = Math.random() < 0.75 ? pick(prefs) : pick(cats);
                const prod = pick(productsByCat[cat]);
                if (chosen.has(prod.id)) continue;
                chosen.add(prod.id);
                const qty = 1 + rand(3);
                total += prod.unitPrice * qty;
                itemRows.push({ id: randomUUID(), orderId, productId: prod.id, quantity: qty, unitPrice: prod.unitPrice });
            }
            orderRows.push({ id: orderId, contactId, workspaceId, total, createdAt: daysAgo(rand(365)) });
        }
    }
    // Batched inserts to avoid oversized statements.
    for (let i = 0; i < orderRows.length; i += 500) await prisma.order.createMany({ data: orderRows.slice(i, i + 500) });
    for (let i = 0; i < itemRows.length; i += 1000) await prisma.orderItem.createMany({ data: itemRows.slice(i, i + 1000) });
    console.log(`  ${orderRows.length} orders, ${itemRows.length} order items`);

    // --- Activities with recency + sentiment (churn signal) ----------------
    const actRows: any[] = [];
    for (const { id: contactId, p } of profiles) {
        const nAct = poisson(0.5 + p * 6);
        if (nAct === 0) continue;
        // Recency gap: engaged customers acted recently; disengaged long ago.
        const recentGap = Math.round((1 - p) * 150 + rand(20));
        const tenureStart = Math.round((Date.now() - (tenureById.get(contactId) as Date).getTime()) / (24 * 3600 * 1000));
        for (let a = 0; a < nAct; a++) {
            const gap = a === 0 ? recentGap : recentGap + rand(Math.max(1, tenureStart - recentGap));
            const r = Math.random();
            const sentiment = r < p ? "positive" : r < p + (1 - p) * 0.55 ? "negative" : "neutral";
            const notes = sentiment === "positive" ? pick(POS_NOTES) : sentiment === "negative" ? pick(NEG_NOTES) : pick(NEU_NOTES);
            actRows.push({
                id: randomUUID(), type: "NOTE", notes, sentiment, sentimentScore: 0.7 + Math.random() * 0.29,
                contactId, userId: owner.id, workspaceId, createdAt: daysAgo(gap),
            });
        }
    }
    for (let i = 0; i < actRows.length; i += 1000) await prisma.activity.createMany({ data: actRows.slice(i, i + 1000) });
    console.log(`  ${actRows.length} activities`);

    // --- Leads with latent CONVERSION signal (lead-scoring model) -----------
    // Source, service, quotation and engagement drive a latent conversion
    // propensity, so an app-native model can learn a real Hot/Warm/Cold spread.
    const LEAD_SOURCES: Record<string, number> = {
        "Referral": 0.75, "Website": 0.6, "API": 0.5, "Social Media": 0.4,
        "Trade Show": 0.3, "Cold Call": 0.2,
    };
    const SERVICES = Object.values(CATEGORIES).flat();
    const N_LEADS = 450;
    const leadRows: any[] = [];
    const leadProfiles: { id: string; p: number }[] = [];
    for (let i = 0; i < N_LEADS; i++) {
        const id = randomUUID();
        const source = pick(Object.keys(LEAD_SOURCES));
        const service = pick(SERVICES);
        const quotation = 500 + rand(30000);
        // Latent propensity: warm source + higher quote value + noise.
        const z = -1.4 + (LEAD_SOURCES[source] - 0.4) * 3 + (quotation / 30000) * 1.2 + (Math.random() - 0.5);
        const p = 1 / (1 + Math.exp(-z));
        const converted = Math.random() < p;
        const status = converted ? "CONVERTED"
            : pick(["NEW", "NEW", "CONTACTED", "CONTACTED", "QUALIFIED"]);
        leadRows.push({
            id, firstName: pick(FIRST), lastName: pick(LAST),
            email: `${id.slice(0, 8)}@ai-seed.demo`,
            phone: Math.random() < 0.85 ? `+1${200 + rand(799)}${1000000 + rand(8999999)}` : null,
            source, service, quotation, status,
            workspaceId, ownerId: owner.id, createdAt: daysAgo(rand(400)),
        });
        leadProfiles.push({ id, p });
    }
    for (let i = 0; i < leadRows.length; i += 500) await prisma.lead.createMany({ data: leadRows.slice(i, i + 500) });

    // Engaged leads (higher propensity) accumulate more activities.
    const leadActs: any[] = [];
    for (const { id: leadId, p } of leadProfiles) {
        const nAct = poisson(p * 4);
        for (let a = 0; a < nAct; a++) {
            leadActs.push({
                id: randomUUID(), type: pick(["CALL", "EMAIL", "MEETING", "NOTE"]),
                notes: "Follow-up logged.", leadId, userId: owner.id, workspaceId, createdAt: daysAgo(rand(120)),
            });
        }
    }
    for (let i = 0; i < leadActs.length; i += 1000) await prisma.activity.createMany({ data: leadActs.slice(i, i + 1000) });
    console.log(`  ${leadRows.length} leads (${leadRows.filter((l) => l.status === "CONVERTED").length} converted), ${leadActs.length} lead activities`);

    console.log("\nApp-native AI dataset seeded. Next: export + train (scripts/export-*.ts, training.*_crm).");
}

main().catch((e) => { console.error(e); process.exit(1); }).finally(() => prisma.$disconnect());

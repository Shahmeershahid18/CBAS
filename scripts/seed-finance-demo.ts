/**
 * Seeds demo data for the Finance + Orders dashboards (which are empty by
 * default): expenses, PAID invoices (revenue), and onboarded orders. Targets the
 * demo workspace (demo@digixcrm.com). Idempotent — clears its own previous seed.
 *
 * Matches how getFinanceOverview reads data:
 *   - expenses: by `date` within the last 6 months
 *   - revenue : invoices with status PAID, by `createdAt` within the last 6 months
 *   - orders  : deals with customStage = "ONBOARDED"
 *
 * Run:  npx tsx scripts/seed-finance-demo.ts
 */
import "dotenv/config";
import { PrismaClient } from "../src/generated/prisma/client/client";
import { PrismaPg } from "@prisma/adapter-pg";
import pg from "pg";
import { randomUUID } from "crypto";

const prisma = new PrismaClient({ adapter: new PrismaPg(new pg.Pool({ connectionString: `${process.env.DATABASE_URL}` }) as any) });

const rand = (n: number) => Math.floor(Math.random() * n);
const pick = <T,>(a: T[]) => a[rand(a.length)];
const daysAgo = (d: number) => new Date(Date.now() - d * 86400000);

const EXPENSE_CATEGORIES = ["Software", "Marketing", "Travel", "Facilities", "Hardware", "Salaries", "Utilities", "Consulting"];
const EXPENSE_TITLES: Record<string, string[]> = {
    Software: ["Cloud hosting (AWS)", "CRM licenses", "Design suite subscription", "Analytics platform"],
    Marketing: ["Google Ads campaign", "LinkedIn sponsored posts", "Trade show booth", "Content agency retainer"],
    Travel: ["Client visit — flights", "Conference travel", "Team offsite", "Hotel & lodging"],
    Facilities: ["Office rent", "Internet & phone", "Cleaning services", "Office supplies"],
    Hardware: ["Developer laptops", "Monitors & docks", "Server upgrade", "Networking gear"],
    Salaries: ["Contractor payout", "Payroll top-up", "Freelance design", "QA contractor"],
    Utilities: ["Electricity", "Water & gas", "Coworking day passes"],
    Consulting: ["Legal retainer", "Accounting services", "Security audit", "DevOps consultant"],
};

async function main() {
    console.log("Seeding Finance + Orders demo data...");
    const owner = await prisma.user.findFirst({ where: { email: "demo@digixcrm.com" } });
    if (!owner) throw new Error("Demo user not found. Run: npm run db:seed:demo first.");
    const workspaceId = (owner as any).activeWorkspaceId as string;
    if (!workspaceId) throw new Error("Demo user has no active workspace.");

    const orgs = await prisma.organization.findMany({ where: { workspaceId }, select: { id: true } });

    // --- Clear previous finance seed --------------------------------------
    await prisma.expense.deleteMany({ where: { workspaceId } });
    await prisma.invoice.deleteMany({ where: { workspaceId } });

    // --- Expenses (spread over the last ~6 months) ------------------------
    const expenseRows: any[] = [];
    for (let i = 0; i < 60; i++) {
        const category = pick(EXPENSE_CATEGORIES);
        const base = category === "Salaries" ? 3000 + rand(9000)
            : category === "Facilities" ? 800 + rand(4000)
                : 150 + rand(4000);
        expenseRows.push({
            id: randomUUID(),
            title: pick(EXPENSE_TITLES[category]),
            amount: base,
            category,
            date: daysAgo(rand(178)),
            workspaceId, createdById: owner.id,
        });
    }
    await prisma.expense.createMany({ data: expenseRows });
    const totalExp = expenseRows.reduce((s, e) => s + e.amount, 0);
    console.log(`  ${expenseRows.length} expenses ($${Math.round(totalExp).toLocaleString()})`);

    // --- Invoices (mostly PAID -> revenue) --------------------------------
    const invoiceRows: any[] = [];
    for (let i = 0; i < 45; i++) {
        const r = Math.random();
        const status = r < 0.75 ? "PAID" : r < 0.9 ? "PENDING" : "OVERDUE";
        const created = daysAgo(rand(178));
        invoiceRows.push({
            id: randomUUID(),
            invoiceNumber: `INV-${1000 + i}`,
            amount: 2000 + rand(38000),
            status,
            dueDate: new Date(created.getTime() + 30 * 86400000),
            notes: "Services rendered.",
            workspaceId, createdById: owner.id,
            organizationId: orgs.length ? pick(orgs).id : null,
            createdAt: created,
        });
    }
    await prisma.invoice.createMany({ data: invoiceRows });
    const revenue = invoiceRows.filter((i) => i.status === "PAID").reduce((s, i) => s + i.amount, 0);
    console.log(`  ${invoiceRows.length} invoices (${invoiceRows.filter(i => i.status === "PAID").length} PAID -> $${Math.round(revenue).toLocaleString()} revenue)`);

    // --- Orders (onboarded deals) -----------------------------------------
    // Promote a batch of the highest-value deals into the fulfillment/orders view.
    const deals = await prisma.deal.findMany({
        where: { workspaceId }, orderBy: { value: "desc" }, take: 18, select: { id: true },
    });
    if (deals.length) {
        await prisma.deal.updateMany({
            where: { id: { in: deals.map((d) => d.id) } },
            data: { stage: "WON", customStage: "ONBOARDED" },
        });
    }
    console.log(`  ${deals.length} deals marked as ONBOARDED orders`);

    // --- Closed-Lost (archived deals) -------------------------------------
    const lostDeals = await prisma.deal.findMany({
        where: { workspaceId, stage: "LOST" }, take: 14, select: { id: true },
    });
    if (lostDeals.length) {
        await prisma.deal.updateMany({
            where: { id: { in: lostDeals.map((d) => d.id) } },
            data: { customStage: "ARCHIVED" },
        });
    }
    console.log(`  ${lostDeals.length} deals archived (Closed-Lost)`);

    console.log("\nDone. Finance Hub, Expenses, Invoices, Orders, and Closed-Lost now have data for the demo account.");
}

main().catch((e) => { console.error(e); process.exit(1); }).finally(() => prisma.$disconnect());

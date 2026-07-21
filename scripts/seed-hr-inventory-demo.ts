/**
 * Seeds demo data for the HR + Inventory dashboards (empty by default):
 *   - EmployeeProfile (department/designation/salary) for workspace members
 *   - Attendance records over the last 30 days
 *   - Candidates for the recruitment / AI-screening view (some left unscored)
 *   - StockMovement history so the inventory forecast/reorder logic has demand
 *
 * Targets the demo workspace (demo@digixcrm.com). Idempotent.
 *
 * Run:  npx tsx scripts/seed-hr-inventory-demo.ts
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

const DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Design", "Operations", "Finance", "HR", "Support"];
const DESIGNATIONS = ["Junior Associate", "Associate", "Senior Associate", "Team Lead", "Manager", "Senior Manager"];

const CANDIDATES = [
    { name: "Ayesha Raza", role: "Senior Frontend Engineer (React, Next.js)", resume: "6 years building React/Next.js dashboards, TypeScript, led a design-system migration, strong on performance and accessibility." },
    { name: "Bilal Sheikh", role: "Backend Engineer (Node.js, PostgreSQL)", resume: "5 years Node.js + PostgreSQL, built multi-tenant SaaS APIs, Prisma, queue systems, and CI/CD pipelines." },
    { name: "Sara Khan", role: "Product Designer (Figma, UX Research)", resume: "4 years product design, Figma component systems, ran usability testing for B2B SaaS, built design systems for 3 teams." },
    { name: "Usman Tariq", role: "Data Scientist (Python, ML)", resume: "PhD-level ML, scikit-learn/XGBoost, churn and recommendation models, deployed models behind FastAPI." },
    { name: "Hina Farooq", role: "Sales Development Rep", resume: "3 years outbound SaaS sales, exceeded quota, HubSpot/CRM power user, strong discovery and objection handling." },
    { name: "Zainab Ali", role: "Product Designer (Figma, UX Research)", resume: "Junior designer, 1 year internship experience, learning Figma, eager to grow in UX research." },
    { name: "Omar Nadeem", role: "DevOps Engineer (AWS, Docker)", resume: "AWS, Docker, Kubernetes, Terraform, Nginx, set up observability and blue-green deploys for a fintech." },
    { name: "Fatima Yousaf", role: "Marketing Manager", resume: "7 years B2B marketing, SEO/PPC, content strategy, marketing automation, grew MQLs 3x YoY." },
];

async function main() {
    console.log("Seeding HR + Inventory demo data...");
    const owner = await prisma.user.findFirst({ where: { email: "demo@digixcrm.com" } });
    if (!owner) throw new Error("Demo user not found. Run: npm run db:seed:demo first.");
    const workspaceId = (owner as any).activeWorkspaceId as string;
    if (!workspaceId) throw new Error("Demo user has no active workspace.");

    const members = await prisma.workspaceMember.findMany({ where: { workspaceId }, select: { userId: true } });
    const memberIds = members.map((m) => m.userId);
    console.log(`  ${memberIds.length} workspace members`);

    // --- Clear previous seed ----------------------------------------------
    await prisma.attendance.deleteMany({ where: { userId: { in: memberIds } } });
    await prisma.candidate.deleteMany({ where: { workspaceId } });
    await prisma.stockMovement.deleteMany({ where: { workspaceId } });

    // --- EmployeeProfile (upsert per member) ------------------------------
    for (const uid of memberIds) {
        await prisma.employeeProfile.upsert({
            where: { userId: uid },
            update: {},
            create: {
                userId: uid, department: pick(DEPARTMENTS), designation: pick(DESIGNATIONS),
                salary: 35000 + rand(115000), joiningDate: daysAgo(60 + rand(1400)),
            },
        });
    }
    console.log(`  ${memberIds.length} employee profiles`);

    // --- Attendance (last 30 days, weekdays, mostly present) --------------
    const attRows: any[] = [];
    for (let d = 0; d < 30; d++) {
        const date = daysAgo(d);
        if (date.getDay() === 0 || date.getDay() === 6) continue; // skip weekends
        for (const uid of memberIds) {
            const r = Math.random();
            const status = r < 0.86 ? "PRESENT" : r < 0.91 ? "ABSENT" : r < 0.96 ? "HALF_DAY" : "ON_LEAVE";
            attRows.push({ id: randomUUID(), userId: uid, date, status });
        }
    }
    for (let i = 0; i < attRows.length; i += 1000) await prisma.attendance.createMany({ data: attRows.slice(i, i + 1000) });
    console.log(`  ${attRows.length} attendance records`);

    // --- Candidates (some unscored for the AI screening demo) -------------
    await prisma.candidate.createMany({
        data: CANDIDATES.map((c, i) => ({
            id: randomUUID(), name: c.name, email: `${c.name.toLowerCase().replace(/\s+/g, ".")}@applicant.demo`,
            appliedRole: c.role, resumeText: c.resume,
            status: pick(["APPLIED", "APPLIED", "SCREENED", "INTERVIEWING", "OFFERED"]) as any,
            workspaceId,
        })),
    });
    console.log(`  ${CANDIDATES.length} candidates`);

    // --- StockMovement: 7 weeks of weekly OUT demand + periodic IN restocks -
    const products = await prisma.product.findMany({ where: { workspaceId }, select: { id: true } });
    const stockRows: any[] = [];
    for (const p of products) {
        const baseDemand = 5 + rand(40);           // this product's typical weekly demand
        const trend = (Math.random() - 0.4) * 3;   // slight up/down trend
        for (let w = 6; w >= 0; w--) {
            const weekDate = daysAgo(w * 7 + rand(6));
            const out = Math.max(0, Math.round(baseDemand + trend * (6 - w) + (rand(10) - 5)));
            if (out > 0) stockRows.push({ id: randomUUID(), productId: p.id, type: "OUT", quantity: out, date: weekDate, workspaceId });
            if (w % 2 === 0) stockRows.push({ id: randomUUID(), productId: p.id, type: "IN", quantity: baseDemand * 2, date: daysAgo(w * 7 + 3), workspaceId });
        }
    }
    for (let i = 0; i < stockRows.length; i += 1000) await prisma.stockMovement.createMany({ data: stockRows.slice(i, i + 1000) });
    console.log(`  ${stockRows.length} stock movements across ${products.length} products`);

    console.log("\nDone. HR (Employees, Attendance, Recruitment) and Inventory now have data for the demo account.");
}

main().catch((e) => { console.error(e); process.exit(1); }).finally(() => prisma.$disconnect());

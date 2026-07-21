"use server";

import { prisma } from "@/lib/prisma";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { askGemini, type FunctionDeclaration } from "@/lib/ai/gemini";
import { getFinanceOverview, createExpense } from "./finance";
import { getInventoryOverview, recordStockMovement } from "./inventory";
import { getAttendanceOverview } from "./hr";

async function requireUser() {
    const session = await getServerSession(authOptions);
    if (!session?.user?.email) throw new Error("Unauthorized");

    const user = await prisma.user.findUnique({ where: { email: session.user.email } });
    if (!user) throw new Error("User not found");

    return user;
}

// --- Tool declarations -----------------------------------------------------
// Deliberately small and explicit: two tools, each mapping 1:1 onto an
// already-tested server action (createExpense, recordStockMovement). The
// model can only ever *propose* a call to one of these — see askAssistant()
// below, which never executes a tool itself. Execution happens exclusively
// in confirmAssistantAction(), and only after the user clicks Confirm.

const TOOLS: FunctionDeclaration[] = [
    {
        name: "log_expense",
        description: "Log a new business expense in the Finance module.",
        parameters: {
            type: "OBJECT",
            properties: {
                title: { type: "STRING", description: "Short description of the expense" },
                amount: { type: "NUMBER", description: "Amount in dollars" },
                category: { type: "STRING", description: "Category, e.g. Software, Marketing, Travel, Facilities, Hardware" },
            },
            required: ["title", "amount", "category"],
        },
    },
    {
        name: "receive_stock",
        description: "Record newly received stock for an existing inventory product (a stock-in movement).",
        parameters: {
            type: "OBJECT",
            properties: {
                productName: { type: "STRING", description: "Name (or close match) of the product" },
                quantity: { type: "NUMBER", description: "Quantity received" },
            },
            required: ["productName", "quantity"],
        },
    },
];

function describeAction(name: string, args: any): string {
    if (name === "log_expense") {
        return `Log expense: "${args.title}" — $${Number(args.amount).toFixed(2)} (${args.category})`;
    }
    if (name === "receive_stock") {
        return `Receive stock: +${args.quantity} units of "${args.productName}"`;
    }
    return `Run ${name}`;
}

/**
 * AI assistant for system navigation and data questions. Uses retrieval-
 * augmented generation (Path B in the AI module plan): aggregate workspace
 * metrics plus the specific CRM records most semantically relevant to the
 * question (via the local embedding model in @/lib/ai/retrieval) are grounded
 * into the prompt, so the model answers about real leads/contacts/deals rather
 * than guessing. It may also propose ONE of the two tools above — but only ever
 * proposes; the caller must call confirmAssistantAction() to actually execute it.
 */
// --- @-mention support -----------------------------------------------------

export interface TaggedRef {
    type: "lead" | "contact";
    id: string;
}

/**
 * Search leads/contacts by name for the chat's @-mention picker. The returned
 * `label` (the person's name) is shown ONLY in the user's own browser to pick a
 * record — it is never sent to the LLM. askAssistant() receives just {type,id}
 * and builds a PII-redacted context from it.
 */
export async function searchTaggables(query: string) {
    try {
        const user = await requireUser();
        const workspaceId = (user as any).activeWorkspaceId || null;
        const q = (query || "").trim();
        if (!q) return { success: true, results: [] as { type: string; id: string; label: string; sub: string }[] };

        const [leads, contacts] = await Promise.all([
            prisma.lead.findMany({
                where: { workspaceId, OR: [{ firstName: { contains: q, mode: "insensitive" } }, { lastName: { contains: q, mode: "insensitive" } }] },
                select: { id: true, firstName: true, lastName: true, service: true, status: true },
                take: 5,
            }),
            prisma.contact.findMany({
                where: { workspaceId, OR: [{ firstName: { contains: q, mode: "insensitive" } }, { lastName: { contains: q, mode: "insensitive" } }] },
                select: { id: true, firstName: true, lastName: true, organization: { select: { name: true } } },
                take: 5,
            }),
        ]);

        const results = [
            ...leads.map((l) => ({ type: "lead", id: l.id, label: `${l.firstName} ${l.lastName}`, sub: `Lead · ${l.status}` })),
            ...contacts.map((c) => ({ type: "contact", id: c.id, label: `${c.firstName} ${c.lastName}`, sub: `Contact${(c as any).organization?.name ? " · " + (c as any).organization.name : ""}` })),
        ];
        return { success: true, results };
    } catch (error: any) {
        return { success: false, error: error.message || "Search failed", results: [] };
    }
}

const tally = (xs: (string | null)[]) => {
    const m: Record<string, number> = {};
    for (const x of xs) if (x) m[x] = (m[x] || 0) + 1;
    return Object.entries(m).map(([k, v]) => `${k} x${v}`).join(", ");
};

/**
 * Builds a PII-REDACTED context block for a tagged record. Personal identifiers
 * (name, email, phone) and free-text notes/remarks (which may contain personal
 * data) are DELIBERATELY EXCLUDED. Only non-identifying business attributes are
 * included, which is enough to advise on how to pitch/handle the record.
 */
async function buildTaggedContext(ref: TaggedRef, workspaceId: string | null, index: number): Promise<string | null> {
    if (ref.type === "lead") {
        const lead = await prisma.lead.findFirst({
            where: { id: ref.id, workspaceId },
            include: { activities: { select: { type: true, sentiment: true } } },
        });
        if (!lead) return null;
        const acts = lead.activities;
        return `[Tagged record #${index + 1} — LEAD | personal identifiers withheld for privacy]
- Interest / service: ${lead.service || "unspecified"}
- Pipeline status: ${lead.status}
- Quoted value: $${lead.quotation ?? 0}
- Lead source: ${lead.source}
- AI lead score: ${lead.aiScore != null ? `${lead.aiScore}/100 (${lead.aiScoreBand})` : "not scored yet"}
- Engagement: ${acts.length} activities${acts.length ? ` (${tally(acts.map((a) => a.type))})` : ""}${acts.some((a) => a.sentiment) ? `; interaction sentiment: ${tally(acts.map((a) => a.sentiment))}` : ""}`;
    }

    const contact = await prisma.contact.findFirst({
        where: { id: ref.id, workspaceId },
        include: {
            activities: { select: { type: true, sentiment: true } },
            orders: { select: { total: true, items: { select: { product: { select: { category: true } } } } } },
            organization: { select: { name: true } },
        },
    });
    if (!contact) return null;
    const acts = contact.activities;
    const cats = new Set<string>();
    contact.orders.forEach((o) => o.items.forEach((it) => it.product.category && cats.add(it.product.category)));
    const spend = contact.orders.reduce((s, o) => s + (o.total || 0), 0);
    return `[Tagged record #${index + 1} — CUSTOMER | personal identifiers withheld for privacy]
- Organization: ${(contact as any).organization?.name || "n/a"}
- Churn risk: ${(contact as any).churnScore != null ? `${(contact as any).churnBand} (${(contact as any).churnScore}/100)` : "not scored yet"}
- Purchases: ${contact.orders.length} orders${cats.size ? ` across ${[...cats].join(", ")}` : ""}; total spend $${Math.round(spend)}
- Engagement: ${acts.length} activities${acts.some((a) => a.sentiment) ? `; sentiment: ${tally(acts.map((a) => a.sentiment))}` : ""}`;
}

export async function askAssistant(message: string, tagged: TaggedRef[] = []) {
    try {
        const user = await requireUser();
        const workspaceId = (user as any).activeWorkspaceId || null;

        const [finance, inventory, attendance, leadCount, dealCount, candidateCount] = await Promise.all([
            getFinanceOverview(),
            getInventoryOverview(),
            getAttendanceOverview(),
            prisma.lead.count({ where: { workspaceId } }),
            prisma.deal.count({ where: { workspaceId } }),
            (prisma as any).candidate.count({ where: { workspaceId } }),
        ]);

        const financeSummary = finance.success && finance.data
            ? `Total revenue (6mo): $${finance.data.totalRevenue.toFixed(2)}. Total expenses (6mo): $${finance.data.totalExpenses.toFixed(2)}. Forecasted next month's expenses: $${finance.data.forecastNextMonth.toFixed(2)} (trend: ${finance.data.trend}). Flagged expense anomalies: ${finance.data.anomalies.length}.`
            : "Finance data unavailable.";

        const inventorySummary = inventory.success && inventory.data
            ? `${inventory.data.totalProducts} products tracked: ${inventory.data.products.map((p: any) => p.name).join(", ") || "none"}. ${inventory.data.lowStockCount} need reordering. Total stock value: $${inventory.data.totalStockValue.toFixed(2)}.`
            : "Inventory data unavailable.";

        const attendanceSummary = attendance.success && attendance.data
            ? `Team size: ${attendance.data.totalMembers}. Present today: ${attendance.data.presentToday}. Absent today: ${attendance.data.absentToday}. Flagged attendance anomalies: ${attendance.data.anomalies.length}.`
            : "Attendance data unavailable.";

        // RAG: retrieve the specific CRM records most relevant to the question so
        // the assistant can answer about individual leads/contacts/deals, not just
        // the aggregate counts. Best-effort — never let retrieval break a reply.
        let retrieved = "";
        try {
            const { retrieveContext } = await import("@/lib/ai/retrieval");
            retrieved = await retrieveContext(workspaceId, message, 6);
        } catch (e) {
            console.error("RAG retrieval skipped:", e);
        }

        // Tagged records (@-mentions): build PII-redacted context blocks.
        let taggedBlock = "";
        if (tagged.length > 0) {
            const blocks = await Promise.all(tagged.slice(0, 5).map((r, i) => buildTaggedContext(r, workspaceId, i)));
            taggedBlock = blocks.filter(Boolean).join("\n\n");
        }

        const systemContext = `You are the built-in AI assistant for CoreAxis, a Business Automation System (BAS). Answer the user's question using ONLY the real workspace data below. If the question can't be answered from this data, say so plainly instead of guessing. Keep general answers to 2-4 sentences; for advice about a tagged record you may use up to 6 short sentences.

Write in plain conversational sentences. Do NOT use any markdown formatting — no **bold**, no asterisks, no headings, no bullet points.

If — and only if — the user is clearly asking you to record/log/add something (an expense, or received stock), call the matching tool instead of replying in text. Otherwise, always reply in plain text. Never call a tool for a question that's just asking for information.

PRIVACY: The tagged records below have had all personal identifiers (name, email, phone) and free-text notes removed on purpose. Never ask for or invent them. Refer to a tagged record as "this lead" / "this customer".

When advising on a tagged record, you MUST explicitly tailor the advice to its specific attributes — do not give generic sales tips:
- Adapt the strategy to the LEAD SOURCE: a Cold Call means low initial intent, so be consultative and earn trust before pitching; a Referral or Reference means warm trust you should leverage by mentioning the referrer's confidence and moving faster; an inbound Website/Landing Page or Olark Chat means the lead sought you out, so intent is higher and you can focus on closing; a Google/Organic source means research-driven, so lead with proof and comparisons.
- Weigh the PIPELINE STATUS (NEW = qualify first; QUALIFIED = pitch and handle objections; CONVERTED = shift to onboarding, retention, and upsell).
- Scale effort to the QUOTED VALUE and the AI SCORE (higher value/hotter score = prioritise and involve a senior rep).
- Use the ENGAGEMENT and SENTIMENT level (low/zero engagement = re-engage gently; negative sentiment = address concerns before selling).
Name the relevant attribute in your answer (e.g. "because this came from a cold call…") and recommend one concrete next best action, plus specific services to pitch when the interest is known. If a key attribute is unspecified, say the first step is to discover it.

Workspace snapshot:
- Sales/CRM: ${leadCount} leads, ${dealCount} deals in pipeline.
- Finance: ${financeSummary}
- Inventory: ${inventorySummary}
- HR/Attendance: ${attendanceSummary}
- Recruitment: ${candidateCount} candidates in the pipeline.
${taggedBlock ? `\nTagged record(s) the user is asking about:\n${taggedBlock}` : ""}${retrieved ? `\nOther relevant records (semantic search):\n${retrieved}` : ""}`;

        const result = await askGemini(systemContext, message, TOOLS);

        if (result.type === "functionCall") {
            return {
                success: true,
                kind: "confirm" as const,
                action: { name: result.name, args: result.args },
                label: describeAction(result.name, result.args),
            };
        }

        return { success: true, kind: "text" as const, reply: result.text };
    } catch (error: any) {
        return { success: false, error: error.message || "The assistant is unavailable right now." };
    }
}

/**
 * Executes a tool the assistant proposed — ONLY called after the user
 * explicitly clicks "Confirm" in the UI. Re-validates everything server-side;
 * never trusts the proposed args blindly.
 */
export async function confirmAssistantAction(name: string, args: any) {
    try {
        const user = await requireUser();
        const workspaceId = (user as any).activeWorkspaceId || null;

        if (name === "log_expense") {
            const res = await createExpense({
                title: String(args.title || "").slice(0, 200),
                amount: args.amount,
                category: String(args.category || "Other").slice(0, 100),
            });
            if (!res.success) throw new Error(res.error);
            return { success: true, message: `Logged expense: ${args.title} ($${Number(args.amount).toFixed(2)})` };
        }

        if (name === "receive_stock") {
            const productName = String(args.productName || "").toLowerCase();
            const products = await (prisma as any).product.findMany({ where: { workspaceId } });
            const matches = products.filter((p: any) => p.name.toLowerCase().includes(productName) || productName.includes(p.name.toLowerCase()));

            if (matches.length === 0) {
                throw new Error(`No product matching "${args.productName}" found in your inventory.`);
            }
            if (matches.length > 1) {
                throw new Error(`Multiple products match "${args.productName}": ${matches.map((m: any) => m.name).join(", ")}. Be more specific.`);
            }

            const res = await recordStockMovement(matches[0].id, "IN", Number(args.quantity), "Added via AI assistant");
            if (!res.success) throw new Error(res.error);
            const newStock = matches[0].currentStock + Number(args.quantity);
            return { success: true, message: `Received ${args.quantity} units of ${matches[0].name}. New stock level: ${newStock}.` };
        }

        throw new Error(`Unknown action: ${name}`);
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to execute action" };
    }
}

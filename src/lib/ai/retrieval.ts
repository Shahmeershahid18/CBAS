// Server-only Retrieval-Augmented Generation (RAG) layer for the AI assistant.
//
// Embeds the workspace's CRM records (leads, contacts, deals) with the local
// MiniLM model, then retrieves the few records most semantically similar to the
// user's question so the chatbot can answer about SPECIFIC records — not just
// aggregate counts. Embeddings are cached in-process keyed by record id +
// updatedAt, so only new/changed records are re-embedded after the first query.

import { prisma } from "@/lib/prisma";
import { embedText, cosineSimilarity } from "./embeddings";

interface CachedDoc {
    key: string;          // `${id}:${updatedAtISO}` — changes when the record changes
    text: string;
    vector: Float32Array;
}

// id -> cached embedding. Bounded implicitly by workspace size (FYP scale).
const cache = new Map<string, CachedDoc>();

// Cap how many records of each type we consider, so a huge workspace can't make
// a single chat turn embed thousands of rows.
const PER_TYPE_LIMIT = 100;

function leadDoc(l: any): string {
    return `Lead: ${l.firstName} ${l.lastName}. Status: ${l.status}. ` +
        `Service: ${l.service || "n/a"}. Quotation: $${l.quotation ?? 0}. ` +
        `Source: ${l.source}. Location: ${l.location || "n/a"}. ` +
        (l.email ? `Email: ${l.email}. ` : "") +
        (l.aiScore != null ? `AI lead score: ${l.aiScore}/100 (${l.aiScoreBand}). ` : "") +
        (l.remarks ? `Remarks: ${l.remarks}.` : "");
}

function contactDoc(c: any): string {
    return `Contact: ${c.firstName} ${c.lastName}. ` +
        (c.email ? `Email: ${c.email}. ` : "") +
        (c.phone ? `Phone: ${c.phone}. ` : "") +
        (c.organization?.name ? `Organization: ${c.organization.name}. ` : "") +
        (c.churnScore != null ? `Churn risk: ${c.churnBand} (${c.churnScore}/100).` : "");
}

function dealDoc(d: any): string {
    return `Deal: ${d.title}. Stage: ${d.stage}. Value: $${d.value ?? 0}. ` +
        (d.organization?.name ? `Organization: ${d.organization.name}.` : "");
}

async function embedDoc(id: string, updatedAt: Date, text: string): Promise<Float32Array> {
    const key = `${id}:${new Date(updatedAt).toISOString()}`;
    const hit = cache.get(id);
    if (hit && hit.key === key) return hit.vector;
    const vector = await embedText(text);
    cache.set(id, { key, text, vector });
    return vector;
}

/**
 * Returns the top-k CRM record snippets most relevant to `query` in the given
 * workspace, as a formatted string ready to drop into the LLM prompt. Empty
 * string if there is nothing to retrieve.
 */
export async function retrieveContext(
    workspaceId: string | null,
    query: string,
    k = 6
): Promise<string> {
    if (!workspaceId) return "";

    const [leads, contacts, deals] = await Promise.all([
        prisma.lead.findMany({
            where: { workspaceId },
            orderBy: { updatedAt: "desc" },
            take: PER_TYPE_LIMIT,
        }),
        prisma.contact.findMany({
            where: { workspaceId },
            orderBy: { updatedAt: "desc" },
            take: PER_TYPE_LIMIT,
            include: { organization: { select: { name: true } } },
        }),
        prisma.deal.findMany({
            where: { workspaceId },
            orderBy: { updatedAt: "desc" },
            take: PER_TYPE_LIMIT,
            include: { organization: { select: { name: true } } },
        }),
    ]);

    const docs: { id: string; updatedAt: Date; text: string }[] = [
        ...leads.map((l) => ({ id: `lead_${l.id}`, updatedAt: l.updatedAt, text: leadDoc(l) })),
        ...contacts.map((c) => ({ id: `contact_${c.id}`, updatedAt: c.updatedAt, text: contactDoc(c) })),
        ...deals.map((d) => ({ id: `deal_${d.id}`, updatedAt: d.updatedAt, text: dealDoc(d) })),
    ];

    if (docs.length === 0) return "";

    const [queryVec, docVecs] = await Promise.all([
        embedText(query),
        Promise.all(docs.map((d) => embedDoc(d.id, d.updatedAt, d.text))),
    ]);

    const scored = docs.map((d, i) => ({ text: d.text, score: cosineSimilarity(queryVec, docVecs[i]) }));
    scored.sort((a, b) => b.score - a.score);

    return scored
        .slice(0, k)
        .map((s, i) => `${i + 1}. ${s.text}`)
        .join("\n");
}

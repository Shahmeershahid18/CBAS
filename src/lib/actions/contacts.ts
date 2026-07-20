"use server";

import { prisma } from "@/lib/prisma";
import { revalidatePath } from "next/cache";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { createAuditLog } from "./audit";
import { getVerifiedRecord, getEffectiveRole } from "@/lib/permissions";
import { touchWorkspace } from "./workspaces";
import { z } from "zod";

const contactSchema = z.object({
    firstName: z.string().min(1, "First name is required"),
    lastName: z.string().min(1, "Last name is required"),
    email: z.string().email().optional().or(z.literal("")),
    phone: z.string().optional().or(z.literal("")),
    organizationId: z.string().optional().nullable(),
});

export async function createContact(rawData: any) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");

        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");
        const data = contactSchema.parse(rawData);

        const workspaceId = (user as any).activeWorkspaceId || null;

        const newContact = await prisma.contact.create({
            data: {
                firstName: data.firstName,
                lastName: data.lastName,
                email: data.email || null,
                phone: data.phone || null,
                organizationId: data.organizationId || null,
                ownerId: user.id,
                workspaceId
            }
        });

        await createAuditLog({
            action: "CREATE",
            entityType: "CONTACT",
            entityId: newContact.id,
            details: `Created contact: ${newContact.firstName} ${newContact.lastName}`
        });

        await touchWorkspace(workspaceId);

        revalidatePath("/dashboard/contacts");
        return { success: true, data: newContact };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to create contact" };
    }
}

export async function updateContact(id: string, rawData: any) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");

        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");

        const effectiveRole = await getEffectiveRole(user);
        const workspaceId = (user as any).activeWorkspaceId || null;
        await getVerifiedRecord(prisma.contact, id, user.id, effectiveRole, workspaceId);

        const data = contactSchema.partial().parse(rawData);

        const updatedContact = await prisma.contact.update({
            where: { id },
            data: {
                ...data,
                email: data.email || null,
                phone: data.phone || null,
            }
        });

        await createAuditLog({
            action: "UPDATE",
            entityType: "CONTACT",
            entityId: updatedContact.id,
            details: `Updated contact: ${updatedContact.firstName} ${updatedContact.lastName}`
        });

        await touchWorkspace(workspaceId);

        revalidatePath("/dashboard/contacts");
        return { success: true, data: updatedContact };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to update contact" };
    }
}

export async function deleteContact(id: string) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");

        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");

        const effectiveRole = await getEffectiveRole(user);
        const workspaceId = (user as any).activeWorkspaceId || null;
        await getVerifiedRecord(prisma.contact, id, user.id, effectiveRole, workspaceId);

        const deletedContact = await prisma.contact.delete({
            where: { id }
        });

        await createAuditLog({
            action: "DELETE",
            entityType: "CONTACT",
            entityId: id,
            details: `Deleted contact: ${deletedContact.firstName} ${deletedContact.lastName}`
        });

        await touchWorkspace(workspaceId);

        revalidatePath("/dashboard/contacts");
        return { success: true, data: deletedContact };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to delete contact" };
    }
}

export async function importBulkContacts(contacts: any[]) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");

        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");

        const workspaceId = (user as any).activeWorkspaceId || null;

        const formattedContacts = contacts.map(contact => ({
            firstName: contact.firstName || contact["First Name"] || "Unknown",
            lastName: contact.lastName || contact["Last Name"] || "Unknown",
            email: contact.email || contact.Email || null,
            phone: String(contact.phone || contact.Phone || "").trim(),
            organizationId: contact.organizationId || null,
            ownerId: user.id,
            workspaceId,
            createdAt: contact.createdAt ? new Date(contact.createdAt) : new Date(),
            updatedAt: new Date()
        }));

        const result = await prisma.contact.createMany({
            data: formattedContacts,
            skipDuplicates: true
        });

        await createAuditLog({
            action: "BULK_IMPORT",
            entityType: "CONTACT",
            details: `Imported ${result.count} contacts via CSV`
        });

        await touchWorkspace(workspaceId);

        revalidatePath("/dashboard/contacts");
        return { success: true, count: result.count };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to import contacts" };
    }
}

/**
 * AI churn prediction for a contact (treated as a customer). Maps the CRM
 * contact/deal signals onto the AI Engine churn model's features and writes the
 * risk score + band + rationale back onto the contact.
 *
 * Uses the CRM-native churn model (trained on CBAS's own data via
 * training.churn_crm) with engagement features engineered here — order history,
 * spend, activity volume, note sentiment, product breadth, and deal state. These
 * features match exactly what scripts/export-ai-training.ts produces, so
 * inference and training agree.
 */
export async function predictContactChurn(contactId: string) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");

        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");

        const effectiveRole = await getEffectiveRole(user);
        const workspaceId = (user as any).activeWorkspaceId || null;
        await getVerifiedRecord(prisma.contact, contactId, user.id, effectiveRole, workspaceId);

        const contact = await prisma.contact.findFirst({
            where: { id: contactId, workspaceId },
            include: {
                deal: { select: { stage: true } },
                activities: { select: { sentiment: true } },
                orders: { select: { total: true, items: { select: { product: { select: { category: true } } } } } },
            },
        });
        if (!contact) throw new Error("Contact not found");

        // Engineer the same features the trainer was given.
        const polarity: Record<string, number> = { positive: 1, negative: -1, neutral: 0 };
        const acts = contact.activities;
        const numActivities = acts.length;
        const numNegative = acts.filter((a) => a.sentiment === "negative").length;
        const avgSentiment = numActivities
            ? acts.reduce((s, a) => s + (polarity[a.sentiment ?? "neutral"] ?? 0), 0) / numActivities
            : 0;
        const categories = new Set<string>();
        contact.orders.forEach((o) => o.items.forEach((it) => it.product.category && categories.add(it.product.category)));
        const hasOpenDeal = contact.deal && !["WON", "LOST"].includes(contact.deal.stage) ? 1 : 0;

        const { predictChurnCrmRemote } = await import("@/lib/ai/engine");
        const result = await predictChurnCrmRemote({
            tenure_days: Math.max(0, Math.round((Date.now() - new Date(contact.createdAt).getTime()) / (1000 * 60 * 60 * 24))),
            num_orders: contact.orders.length,
            total_spend: Math.round(contact.orders.reduce((s, o) => s + (o.total || 0), 0)),
            num_activities: numActivities,
            num_negative_notes: numNegative,
            avg_sentiment: Number(avgSentiment.toFixed(3)),
            num_categories: categories.size,
            has_open_deal: hasOpenDeal,
        });

        const updated = await prisma.contact.update({
            where: { id: contact.id },
            data: {
                churnScore: result.risk_score,
                churnBand: result.risk_band,
                churnReason: result.reason,
                churnScoredAt: new Date(),
            },
        });

        await createAuditLog({
            action: "AI_CHURN",
            entityType: "CONTACT",
            entityId: contact.id,
            details: `Churn prediction for ${contact.firstName} ${contact.lastName}: ${result.risk_score}/100 (${result.risk_band} risk)`,
        });

        revalidatePath("/dashboard/contacts");
        return { success: true, data: { riskScore: result.risk_score, riskBand: result.risk_band, reason: result.reason, contact: updated } };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to predict churn" };
    }
}

/**
 * Product recommendations for a contact (customer). Calls the AI Engine
 * collaborative-filtering model and returns the top-N suggested items.
 *
 * The recommender is trained on the app's OWN order history (exported by
 * scripts/export-ai-training.ts), so it knows real contacts and products and
 * returns personalized picks; contacts with no orders get a popularity fallback.
 */
export async function getContactRecommendations(contactId: string, n = 5) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");

        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");

        const effectiveRole = await getEffectiveRole(user);
        const workspaceId = (user as any).activeWorkspaceId || null;
        await getVerifiedRecord(prisma.contact, contactId, user.id, effectiveRole, workspaceId);

        const { getRecommendationsRemote } = await import("@/lib/ai/engine");
        const result = await getRecommendationsRemote(contactId, n);

        return {
            success: true,
            data: {
                personalized: result.personalized,
                items: result.recommendations.map((r) => ({
                    id: r.item_id,
                    name: r.name,
                    category: r.category ?? null,
                    score: r.score ?? null,
                })),
            },
        };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to get recommendations" };
    }
}

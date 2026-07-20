"use server";

import { prisma } from "@/lib/prisma";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";

/**
 * Aggregates the outputs of the AI Engine models already stored on CRM records
 * (lead scores, churn bands, activity sentiment) into a single dashboard
 * summary, plus a live check of whether the Python AI Engine is reachable.
 */
export async function getAiInsights() {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.email) throw new Error("Unauthorized");
        const user = await prisma.user.findUnique({ where: { email: session.user.email } });
        if (!user) throw new Error("User not found");
        const workspaceId = (user as any).activeWorkspaceId || null;

        const [leadBands, totalLeads, churnBands, totalContacts, sentiments, engineHealth] =
            await Promise.all([
                prisma.lead.groupBy({ by: ["aiScoreBand"], where: { workspaceId }, _count: true }),
                prisma.lead.count({ where: { workspaceId } }),
                prisma.contact.groupBy({ by: ["churnBand"], where: { workspaceId }, _count: true }),
                prisma.contact.count({ where: { workspaceId } }),
                prisma.activity.groupBy({ by: ["sentiment"], where: { workspaceId, sentiment: { not: null } }, _count: true }),
                import("@/lib/ai/engine").then((m) => m.getEngineHealth()),
            ]);

        const bandCount = (rows: any[], key: string, band: string) =>
            rows.find((r) => r[key] === band)?._count ?? 0;

        const leadScored = leadBands.reduce((s, r) => s + (r.aiScoreBand ? r._count : 0), 0);
        const churnScored = churnBands.reduce((s, r) => s + (r.churnBand ? r._count : 0), 0);
        const sentTotal = sentiments.reduce((s, r) => s + r._count, 0);

        return {
            success: true,
            data: {
                leadScoring: {
                    scored: leadScored,
                    total: totalLeads,
                    hot: bandCount(leadBands, "aiScoreBand", "Hot"),
                    warm: bandCount(leadBands, "aiScoreBand", "Warm"),
                    cold: bandCount(leadBands, "aiScoreBand", "Cold"),
                },
                churn: {
                    scored: churnScored,
                    total: totalContacts,
                    high: bandCount(churnBands, "churnBand", "High"),
                    medium: bandCount(churnBands, "churnBand", "Medium"),
                    low: bandCount(churnBands, "churnBand", "Low"),
                },
                sentiment: {
                    total: sentTotal,
                    positive: bandCount(sentiments, "sentiment", "positive"),
                    neutral: bandCount(sentiments, "sentiment", "neutral"),
                    negative: bandCount(sentiments, "sentiment", "negative"),
                },
                engine: engineHealth,
            },
        };
    } catch (error: any) {
        return { success: false, error: error.message || "Failed to load AI insights" };
    }
}

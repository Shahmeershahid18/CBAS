// Server-only client for the CBAS AI Engine (the Python/FastAPI service in
// /ai-engine). The Next.js backend calls these REST endpoints and stores the
// results against CRM records — the models themselves live in the Python
// service, not in this app.

const BASE_URL = process.env.AI_ENGINE_URL || "http://127.0.0.1:8088";
const API_KEY = process.env.AI_ENGINE_API_KEY || "";
const TIMEOUT_MS = 8000;

export class AiEngineError extends Error {}

async function post<T>(path: string, body: unknown): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
        const res = await fetch(`${BASE_URL}${path}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...(API_KEY ? { "x-ai-engine-key": API_KEY } : {}),
            },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        if (!res.ok) {
            const detail = await res.text().catch(() => "");
            throw new AiEngineError(
                `AI Engine ${path} returned ${res.status}: ${detail.slice(0, 200)}`
            );
        }
        return (await res.json()) as T;
    } catch (err: any) {
        if (err instanceof AiEngineError) throw err;
        if (err?.name === "AbortError") {
            throw new AiEngineError(`AI Engine timed out after ${TIMEOUT_MS}ms. Is it running at ${BASE_URL}?`);
        }
        throw new AiEngineError(`Could not reach the AI Engine at ${BASE_URL}. Is it running? (${err?.message || err})`);
    } finally {
        clearTimeout(timer);
    }
}

async function get<T>(path: string): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
        const res = await fetch(`${BASE_URL}${path}`, {
            method: "GET",
            headers: { ...(API_KEY ? { "x-ai-engine-key": API_KEY } : {}) },
            signal: controller.signal,
        });
        if (!res.ok) {
            const detail = await res.text().catch(() => "");
            throw new AiEngineError(`AI Engine ${path} returned ${res.status}: ${detail.slice(0, 200)}`);
        }
        return (await res.json()) as T;
    } catch (err: any) {
        if (err instanceof AiEngineError) throw err;
        if (err?.name === "AbortError") {
            throw new AiEngineError(`AI Engine timed out after ${TIMEOUT_MS}ms. Is it running at ${BASE_URL}?`);
        }
        throw new AiEngineError(`Could not reach the AI Engine at ${BASE_URL}. Is it running? (${err?.message || err})`);
    } finally {
        clearTimeout(timer);
    }
}

// Feature payload accepted by the lead-scoring model. All optional — the model
// imputes anything the CRM doesn't know.
export interface LeadScoringFeatures {
    total_visits?: number;
    time_on_site?: number;
    page_views_per_visit?: number;
    lead_origin?: string;
    lead_source?: string;
    do_not_email?: string;
    last_activity?: string;
    occupation?: string;
    specialization?: string;
}

export interface LeadScoreResult {
    score: number;          // 0-100
    probability: number;    // 0-1
    band: string;           // Hot / Warm / Cold
    model: string;
    reason: string;
    key_factors: { feature: string; importance: number }[];
}

export function scoreLeadRemote(features: LeadScoringFeatures): Promise<LeadScoreResult> {
    return post<LeadScoreResult>("/score-lead", features);
}

// --- Churn prediction ------------------------------------------------------

// Feature payload accepted by the churn model. All optional — the model imputes
// anything the CRM doesn't know.
export interface ChurnFeatures {
    tenure?: number;              // months as a customer
    monthly_charges?: number;
    total_charges?: number;
    contract?: string;           // Month-to-month / One year / Two year
    payment_method?: string;
    internet_service?: string;
    paperless_billing?: string;
    tech_support?: string;
    online_security?: string;
    senior_citizen?: string;     // "0" | "1"
}

export interface ChurnResult {
    churn_probability: number;   // 0-1
    risk_score: number;          // 0-100
    risk_band: string;           // Low / Medium / High
    model: string;
    reason: string;
    key_factors: { feature: string; importance: number }[];
}

export function predictChurnRemote(features: ChurnFeatures): Promise<ChurnResult> {
    return post<ChurnResult>("/predict-churn", features);
}

// CRM-native churn — features engineered from the app's own data. Served by the
// model trained on CBAS data (vs. predictChurnRemote which uses the Telco benchmark).
export interface CrmChurnFeatures {
    tenure_days?: number;
    num_orders?: number;
    total_spend?: number;
    num_activities?: number;
    num_negative_notes?: number;
    avg_sentiment?: number;
    num_categories?: number;
    has_open_deal?: number;
}

export function predictChurnCrmRemote(features: CrmChurnFeatures): Promise<ChurnResult> {
    return post<ChurnResult>("/predict-churn-crm", features);
}

// --- Sentiment analysis ----------------------------------------------------

export interface SentimentResult {
    sentiment: string;                     // positive / negative / neutral
    confidence: number;                    // 0-1
    scores: Record<string, number>;        // per-class probability
}

export function analyzeSentimentRemote(text: string): Promise<SentimentResult> {
    return post<SentimentResult>("/analyze-sentiment", { text });
}

// CRM-note sentiment — model trained on business language (used for activity
// notes; the review-trained /analyze-sentiment mislabels CRM phrasing).
export function analyzeSentimentCrmRemote(text: string): Promise<SentimentResult> {
    return post<SentimentResult>("/analyze-sentiment-crm", { text });
}

// --- Recommendations -------------------------------------------------------

export interface RecommendedItem {
    item_id: number;
    name: string;
    category?: string | null;
    score?: number | null;      // null for the popularity fallback
}

export interface RecommendationsResult {
    customer_id: string;
    personalized: boolean;       // false = cold-start popularity fallback
    recommendations: RecommendedItem[];
}

export function getRecommendationsRemote(customerId: string, n = 5): Promise<RecommendationsResult> {
    return get<RecommendationsResult>(`/recommendations/${encodeURIComponent(customerId)}?n=${n}`);
}

// --- Engine health ---------------------------------------------------------

export interface EngineHealth {
    online: boolean;
    models: Record<string, { ready: boolean }>;
}

/** Ping the AI Engine. Never throws — returns {online:false} if unreachable. */
export async function getEngineHealth(): Promise<EngineHealth> {
    try {
        const res = await get<{ status: string; models: Record<string, { ready: boolean }> }>("/health");
        return { online: true, models: res.models || {} };
    } catch {
        return { online: false, models: {} };
    }
}

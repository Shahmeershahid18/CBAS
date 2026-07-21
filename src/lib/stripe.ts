import Stripe from "stripe";

// Lazily instantiate Stripe on first use. Instantiating at module load throws
// "Neither apiKey nor config.authenticator provided" when STRIPE_SECRET_KEY is
// unset, which breaks `next build` (it evaluates every route). With this proxy
// the key is only required at runtime, when a billing endpoint is actually hit.
let _stripe: Stripe | null = null;

function getStripe(): Stripe {
    if (!_stripe) {
        const key = process.env.STRIPE_SECRET_KEY;
        if (!key) throw new Error("STRIPE_SECRET_KEY is not configured.");
        _stripe = new Stripe(key, { apiVersion: "2026-02-25.clover", typescript: true });
    }
    return _stripe;
}

export const stripe = new Proxy({} as Stripe, {
    get(_target, prop) {
        const client = getStripe();
        const value = (client as any)[prop];
        return typeof value === "function" ? value.bind(client) : value;
    },
});

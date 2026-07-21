import { SquareClient, SquareEnvironment } from "square";

// Lazily instantiate the Square client (same reasoning as lib/stripe.ts): avoid
// constructing SDK clients at module load so a missing token never breaks the
// production build. The token is only needed at runtime.
let _square: SquareClient | null = null;

function getSquare(): SquareClient {
    if (!_square) {
        _square = new SquareClient({
            token: process.env.SQUARE_ACCESS_TOKEN || "",
            environment: process.env.NODE_ENV === "production"
                ? SquareEnvironment.Production
                : SquareEnvironment.Sandbox,
        });
    }
    return _square;
}

export const square = new Proxy({} as SquareClient, {
    get(_target, prop) {
        const client = getSquare();
        const value = (client as any)[prop];
        return typeof value === "function" ? value.bind(client) : value;
    },
});

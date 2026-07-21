// When the app is served under a sub-path (e.g. "/cbas"), Next.js auto-prefixes
// <Link>, router.push(), redirect(), and next/image — but NOT raw fetch(),
// EventSource, <a href>, <img src>, or window.location. Use withBasePath() for
// those so they resolve correctly both at the root (dev) and under /cbas (prod).

export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

/** Prefix an absolute app path (API route, public asset, redirect target) with the base path. */
export function withBasePath(path: string): string {
    if (!path || !path.startsWith("/")) return path;
    if (BASE_PATH && path.startsWith(BASE_PATH + "/")) return path; // already prefixed
    return `${BASE_PATH}${path}`;
}

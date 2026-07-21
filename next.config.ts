import type { NextConfig } from "next";

// Serve the app under a sub-path in production (e.g. "/cbas" on
// digixworkspace.com/cbas). Empty in local dev so it stays at the root.
// Set NEXT_PUBLIC_BASE_PATH="/cbas" in the server's .env before `npm run build`.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  images: {
    domains: ['localhost', 'github.com', 'res.cloudinary.com', 'digixworkspace.com'],
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
};


export default nextConfig;

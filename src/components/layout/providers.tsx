"use client";

import { SessionProvider } from "next-auth/react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { Toaster } from "sonner";

import { ConfirmDialogProvider } from "@/components/providers/confirm-dialog-provider";
import { AppPreloader } from "./app-preloader";
import { DeepLinkHandler } from "./deep-link-handler";

export function Providers({ children }: { children: React.ReactNode }) {
    // When the app is mounted under a sub-path, NextAuth's client must call
    // <basePath>/api/auth instead of /api/auth. Empty base path -> default.
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH;
    const authBasePath = basePath ? `${basePath}/api/auth` : undefined;
    return (
        <NextThemesProvider attribute="class" defaultTheme="system" enableSystem>
            <SessionProvider basePath={authBasePath}>
                <ConfirmDialogProvider>
                    <AppPreloader />
                    <DeepLinkHandler />
                    {children}
                    <Toaster position="top-center" richColors closeButton />
                </ConfirmDialogProvider>
            </SessionProvider>
        </NextThemesProvider>
    );
}



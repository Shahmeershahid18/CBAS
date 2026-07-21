"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export function DeepLinkHandler() {
    const router = useRouter();

    useEffect(() => {
        // Guard FIRST: on the web there is no Capacitor native runtime, so we must
        // never import @capacitor/app (that native-only chunk fails to load in the
        // browser and throws ChunkLoadError). Only load it on android/ios.
        const platform = (window as any).Capacitor?.getPlatform?.();
        if (platform !== "android" && platform !== "ios") return;

        let remove: (() => void) | undefined;

        (async () => {
            try {
                const { App } = await import("@capacitor/app");
                const handle = await App.addListener("appUrlOpen", (data: any) => {
                    const url = new URL(data.url);
                    // Handle Magic Link (Registration Wizard)
                    if (url.pathname.includes("/auth/register-flow")) {
                        const token = url.searchParams.get("token");
                        if (token) router.push(`/auth/register-flow?token=${token}`);
                    }
                    // Handle other deep-link features here.
                });
                remove = () => handle.remove();
            } catch (e) {
                console.error("Deep-link setup skipped:", e);
            }
        })();

        return () => { remove?.(); };
    }, [router]);

    return null; // This component doesn't render anything
}

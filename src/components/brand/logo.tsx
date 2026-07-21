"use client";

import { cn } from "@/lib/utils";
import { withBasePath } from "@/lib/base-path";

interface LogoProps {
    className?: string;
    size?: "sm" | "md" | "lg" | "xl";
    showText?: boolean;
}

export function Logo({ className, size = "md", showText = false }: LogoProps) {
    const sizeMap = {
        sm: "w-8 h-8",
        md: "w-10 h-10 md:w-11 md:h-11",
        lg: "w-14 h-14",
        xl: "w-16 h-16",
    };

    return (
        <div className={cn("inline-flex items-center gap-2.5 group", className)}>
            <div className={cn("relative flex items-center justify-center shrink-0 overflow-visible transition-transform group-hover:scale-105", sizeMap[size])}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                    src={withBasePath("/logo.png")}
                    alt="CBAS Logo"
                    className="w-full h-full object-contain drop-shadow-sm"
                />
            </div>

            {showText && (
                <span className="text-xl md:text-2xl font-extrabold tracking-tight text-foreground whitespace-nowrap">
                    CBAS
                </span>
            )}
        </div>
    );
}

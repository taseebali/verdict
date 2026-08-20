import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: "#6366f1",
        "accent-light": "#818cf8",
        ink: "#18181b",
        canvas: "#fafaf9",
        "canvas-100": "#fafafa",
        sidebar: "#111113",
      },
      fontFamily: {
        sans: ["-apple-system", "SF Pro Display", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["SF Mono", "JetBrains Mono", "monospace"],
      },
      boxShadow: {
        tile: "0 10px 26px -18px rgba(0,0,0,0.12)",
        "tile-accent": "0 12px 30px -16px rgba(99,102,241,0.25)",
      },
    },
  },
  plugins: [],
} satisfies Config;

import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: { DEFAULT: "#f6f3ec", raised: "#fbf9f4" },
        ink: { DEFAULT: "#1c1a17", muted: "#5d564c", faint: "#736b5f" },
        rule: "#e3dccf",
        verdict: "#a4262c",
        gain: "#2f6b3a",
      },
      fontFamily: {
        serif: ['"Iowan Old Style"', '"Palatino Linotype"', "Georgia", "serif"],
        sans: ["system-ui", "-apple-system", '"Segoe UI"', "Roboto", "sans-serif"],
      },
    },
  },
  plugins: [],
} satisfies Config;

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
      fontSize: {
        // 0.8rem = 12px on the 15px root: the smallest size the design allows.
        xs: ["0.8rem", { lineHeight: "1.1rem" }],
      },
    },
  },
  plugins: [],
} satisfies Config;

import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        parchment: "#f5f0e8",
        "parchment-dark": "#ede6d6",
        ink: "#1a1108",
        "ink-light": "#4a3f2f",
        burgundy: "#8b1a1a",
        "burgundy-dark": "#6b1414",
        "burgundy-light": "#c44",
        navy: "#2d4a6b",
        cream: "#fffdf9",
      },
      fontFamily: {
        serif: ["var(--font-playfair)", "Georgia", "serif"],
        body: ["var(--font-crimson)", "Georgia", "serif"],
        sans: ["var(--font-lato)", "system-ui", "sans-serif"],
      },
      boxShadow: {
        book: "0 2px 8px rgba(30,15,0,0.10), 0 0 0 1px rgba(30,15,0,0.06)",
        "book-hover":
          "0 8px 24px rgba(30,15,0,0.14), 0 0 0 1px rgba(30,15,0,0.08)",
        "book-cover": "3px 3px 10px rgba(0,0,0,0.25), inset -2px 0 4px rgba(0,0,0,0.1)",
      },
    },
  },
  plugins: [],
};

export default config;

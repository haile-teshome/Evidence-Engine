import { defineConfig } from "vitest/config";
import path from "path";
import react from "@vitejs/plugin-react";

// Test config is kept separate from vite.config.ts so the dev server and the
// production build are unaffected by anything here.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  test: {
    // jsdom for component and store tests; the pure-logic files do not care.
    environment: "jsdom",
    // jsdom gives an opaque origin by default, and on an opaque origin
    // localStorage is a stub without clear(). A real URL makes Storage behave
    // like a browser's, which is what the persistence code expects.
    environmentOptions: { jsdom: { url: "http://localhost:3000" } },
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    // Vitest workers each need their own jsdom; keep the pool small so a full
    // run stays under a couple of seconds on a laptop.
    pool: "threads",
    coverage: {
      provider: "v8",
      include: ["src/app/lib/**", "src/app/pages/**", "src/app/components/**"],
      exclude: ["**/__tests__/**", "src/app/components/ui/**"],
      reporter: ["text"],
    },
  },
});

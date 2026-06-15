import { defineConfig } from "vite";

// Plain Vite config: vanilla TypeScript, no framework plugins.
// The dev server runs on 5173 and `vite preview` on 4173 (see package.json).
export default defineConfig({
  server: {
    host: true,
    port: 5173,
  },
  preview: {
    host: true,
    port: 4173,
  },
});

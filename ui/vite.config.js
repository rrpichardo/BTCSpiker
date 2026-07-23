import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxy mirrors nginx.conf so `fetch("/api/...")` works identically
// in `npm run dev` and in the production nginx container.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // More specific rule first: /api/predictions -> materializer service.
      "/api/predictions": {
        target: "http://localhost:8090",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/predictions/, "/predictions"),
      },
      // Everything else under /api -> main api service.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});

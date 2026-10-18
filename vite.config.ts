import { defineConfig } from "vitest/config";

// Relative base keeps this working whether it is served from
// ethan-puyaubreau.github.io/nbody-webgpu/ or a custom domain at the root.
export default defineConfig({
  base: "./",
  build: {
    target: "esnext",
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "scripts/**/*.test.mjs"],
  },
});

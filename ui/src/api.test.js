import assert from "node:assert/strict";
import test from "node:test";

import { getJson } from "./api.js";

test("getJson aborts a hung request at its deadline", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = (_url, { signal }) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
    });

  await assert.rejects(getJson("/never", undefined, 5), /timed out after 5ms/);
});

test("getJson preserves caller cancellation", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = (_url, { signal }) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
    });
  const controller = new AbortController();
  const request = getJson("/cancelled", controller.signal, 1_000);

  controller.abort();

  await assert.rejects(request, { name: "AbortError" });
});

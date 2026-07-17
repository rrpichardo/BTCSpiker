// All URLs here are relative — nginx (prod) and the Vite dev proxy (local)
// both route /api/* to the right backend. Never hardcode a host or port here.

export const REQUEST_TIMEOUT_MS = 8_000;

export async function getJson(url, signal, timeoutMs = REQUEST_TIMEOUT_MS) {
  const deadlineController = new AbortController();
  let timedOut = false;
  const abortFromCaller = () => deadlineController.abort(signal.reason);

  if (signal?.aborted) {
    abortFromCaller();
  } else {
    signal?.addEventListener("abort", abortFromCaller, { once: true });
  }
  const deadlineId = setTimeout(() => {
    timedOut = true;
    deadlineController.abort();
  }, timeoutMs);

  try {
    const res = await fetch(url, { signal: deadlineController.signal });
    if (!res.ok) {
      throw new Error(`${url} -> HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    if (timedOut) {
      throw new Error(`${url} -> timed out after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(deadlineId);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

export function fetchRecentPredictions(signal) {
  return getJson("/api/predictions/recent?limit=500", signal);
}

export function fetchPredictionsHealth(signal) {
  return getJson("/api/predictions/health", signal);
}

export function fetchSettings(signal) {
  return getJson("/api/settings", signal);
}

export function fetchSystemStatus(signal) {
  return getJson("/api/system/status", signal);
}

// All URLs here are relative — nginx (prod) and the Vite dev proxy (local)
// both route /api/* to the right backend. Never hardcode a host or port here.

async function getJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new Error(`${url} -> HTTP ${res.status}`);
  }
  return res.json();
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

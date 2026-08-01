const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

export function formatTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? "Unavailable" : timeFormatter.format(date);
}

export function formatDateTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? "Unavailable" : dateTimeFormatter.format(date);
}

export function ageMs(value) {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return Math.max(0, Date.now() - date.getTime());
}

// Provenance (where the data came from) and freshness (whether polling is
// current) are independent facts — a feed can be "Live" (actively polling,
// getting fresh responses) while every row it's polling is replayed
// historical data, which is exactly what produced the 2026-04-06
// duplicated-fixture incident: the connection-status badge correctly read
// "Live" while the underlying market data was a 10-minute loop from days
// earlier. This never collapses the two into one label.
export function formatProvenance(ingestMode) {
  if (ingestMode === "live") return "Live market data";
  if (ingestMode === "replay") return "Replay data";
  return "Unknown data source";
}

export function formatAge(value) {
  const elapsed = ageMs(value);
  if (elapsed === null) return "Awaiting first response";
  const seconds = Math.floor(elapsed / 1000);
  if (seconds < 2) return "Updated just now";
  if (seconds < 60) return `Updated ${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  return `Updated ${minutes}m ago`;
}

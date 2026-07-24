// Pure data-shaping helpers for the Predictions timeline (price + spike
// prediction-vs-reality). Framework-free so they're testable with node --test
// (see timelineData.test.js).
//
// Every point is keyed on feature_ts (when the market feature was observed),
// never api_ts (when the API happened to score it) -- api_ts is diagnostic
// only and must never drive the x-axis.

export { outcomeBands as classBands } from "./performanceData.js";

const RANGE_SECONDS = {
  "1m": 60,
  "5m": 5 * 60,
  "15m": 15 * 60,
  "1h": 60 * 60,
  "6h": 6 * 60 * 60,
  "24h": 24 * 60 * 60,
};

export const RANGE_OPTIONS = Object.keys(RANGE_SECONDS);

// The timeline API's window is a half-open [from, to) interval (feature_ts <
// to, strictly). anchorIso is itself the newest feature_ts, so without this
// nudge the anchor's own point -- always the most recent, most interesting
// one -- would be excluded from every single request.
const ANCHOR_INCLUSION_PAD_MS = 1000;

// {from, to} ISO strings for `rangeKey`, anchored to `anchorIso` (the newest
// feature_ts known to the client -- never wall-clock now, so replay and
// seeded fixtures produce the same window as live data).
export function rangeToWindow(rangeKey, anchorIso) {
  const anchorMs = Date.parse(anchorIso);
  if (!Number.isFinite(anchorMs)) return null;
  const seconds = RANGE_SECONDS[rangeKey];
  if (!seconds) return null;
  return {
    from: new Date(anchorMs - seconds * 1000).toISOString(),
    to: new Date(anchorMs + ANCHOR_INCLUSION_PAD_MS).toISOString(),
  };
}

// A raw (non-aggregated) point carries a single "class"; a bucketed point
// carries "classes", a count per class. Both shapes reduce to the same
// five boolean flags the charts need.
function classFlags(point) {
  if (point.classes) {
    const c = point.classes;
    return {
      predicted: (c.correct_call ?? 0) + (c.false_alarm ?? 0) > 0,
      confirmedSpike: (c.correct_call ?? 0) + (c.missed_spike ?? 0) > 0,
      falseAlarm: (c.false_alarm ?? 0) > 0,
      missedSpike: (c.missed_spike ?? 0) > 0,
      pending: (c.pending ?? 0) > 0,
      unavailable: (c.unavailable ?? 0) > 0,
    };
  }
  const cls = point.class;
  return {
    predicted: cls === "correct_call" || cls === "false_alarm",
    confirmedSpike: cls === "correct_call" || cls === "missed_spike",
    falseAlarm: cls === "false_alarm",
    missedSpike: cls === "missed_spike",
    pending: cls === "pending",
    unavailable: cls === "unavailable",
  };
}

// Turns raw `timeline.points` (from GET /api/predictions/timeline) into
// timestamp-ordered chart points. Missing price/score become null (not
// dropped, not zero) so a chart can render a genuine gap rather than a
// misleading interpolation or a false zero.
export function buildTimelinePoints(points) {
  if (!Array.isArray(points)) return [];
  return points
    .map((point) => ({
      timestamp: Date.parse(point.feature_ts),
      price: typeof point.market_price === "number" ? point.market_price : null,
      score: typeof point.score === "number" ? point.score : null,
      tau: typeof point.tau === "number" ? point.tau : null,
      ...classFlags(point),
    }))
    .filter((point) => Number.isFinite(point.timestamp))
    .sort((left, right) => left.timestamp - right.timestamp);
}


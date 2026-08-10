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
// carries "classes", a count per class. Both shapes reduce to the same six
// boolean flags -- one per outcome state -- so a chart can band on any of
// them directly. Deliberately NOT the "predicted" (correct_call OR
// false_alarm) / "confirmedSpike" (correct_call OR missed_spike) ingredient
// flags this used to expose: those two overlap on every correct_call (both
// true over the same span), which is exactly the "two overlapping bands"
// problem the timeline chart used to have -- drawing the six real states
// directly keeps them mutually exclusive per point.
function classFlags(point) {
  if (point.classes) {
    const c = point.classes;
    return {
      correctCall: (c.correct_call ?? 0) > 0,
      falseAlarm: (c.false_alarm ?? 0) > 0,
      missedSpike: (c.missed_spike ?? 0) > 0,
      correctQuiet: (c.correct_quiet ?? 0) > 0,
      pending: (c.pending ?? 0) > 0,
      unavailable: (c.unavailable ?? 0) > 0,
    };
  }
  const cls = point.class;
  return {
    correctCall: cls === "correct_call",
    falseAlarm: cls === "false_alarm",
    missedSpike: cls === "missed_spike",
    correctQuiet: cls === "correct_quiet",
    pending: cls === "pending",
    unavailable: cls === "unavailable",
  };
}

// Turns raw `timeline.points` (from GET /api/predictions/timeline) into
// timestamp-ordered chart points. Missing price/score become null (not
// dropped, not zero) so a chart can render a genuine gap rather than a
// misleading interpolation or a false zero.
// Precision scales with the rendered price range, not a fixed $1000
// granularity — over a short window (a few minutes) BTC typically moves
// tens of dollars, so a fixed "$Xk" formatter collapses every auto-generated
// axis tick to the same string (all five read "$70k"). `domainSpan` is the
// max-min of the prices actually being plotted.
export function priceTickLabel(value, domainSpan) {
  if (domainSpan >= 5000) return `$${Math.round(value / 1000)}k`;
  if (domainSpan >= 500) return `$${(value / 1000).toFixed(1)}k`;
  if (domainSpan >= 50) return `$${Math.round(value)}`;
  return `$${value.toFixed(1)}`;
}

// Widens alongside priceTickLabel's precision so longer labels ("$69800.5")
// aren't clipped. Applied to BOTH stacked charts in TimelineCharts.jsx so
// they stay pixel-aligned even though only the price chart's labels vary.
export function priceTickAxisWidth(domainSpan) {
  if (domainSpan >= 5000) return 56;
  if (domainSpan >= 500) return 64;
  if (domainSpan >= 50) return 68;
  return 76;
}

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


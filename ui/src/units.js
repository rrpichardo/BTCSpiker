// Turns the model's raw feature values into quantities a non-specialist can
// read, and outcome class strings into plain-language verdicts. Pure and
// framework-free so it's testable with node --test (see units.test.js).
//
// The rule everywhere here: never invent precision the underlying number
// doesn't have, and never render a missing value as zero. A missing feature
// is "—", not "0.000%", because zero volatility is a real and very different
// claim from "we didn't record this".

const MISSING = "—";

function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

// vol_60s and log_return are both log-return quantities — dimensionless
// ratios that read naturally as percent-of-price. Raw they render as
// "5.0e-5", which is true but unreadable; as "0.005%" it's the same number
// in the unit a person actually thinks in.
export function formatPercent(value, digits = 3) {
  if (!isNum(value)) return MISSING;
  return `${(value * 100).toFixed(digits)}%`;
}

// Signed variant for log_return, where the direction is the whole point:
// "did the price just go up or down" is unreadable without the leading sign.
export function formatSignedPercent(value, digits = 3) {
  if (!isNum(value)) return MISSING;
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

// spread_bps already ships in basis points; it just needs sane precision and
// its unit spelled out rather than exponential notation.
export function formatBps(value, digits = 2) {
  if (!isNum(value)) return MISSING;
  return `${value.toFixed(digits)} bps`;
}

export function formatRate(value, digits = 1) {
  if (!isNum(value)) return MISSING;
  return value.toFixed(digits);
}

// Outcome classes come from materializer/timeline.py classify_row, which is
// the single source of truth for all six. Kept exhaustive on purpose: an
// unrecognized class falls through to the neutral "unknown" entry rather
// than rendering a raw snake_case string at the user.
const OUTCOMES = {
  correct_call: {
    label: "Called it",
    tone: "green",
    info: "The model alerted, and a volatility spike really did follow.",
  },
  false_alarm: {
    label: "False alarm",
    tone: "amber",
    info: "The model alerted, but no spike followed.",
  },
  missed_spike: {
    label: "Missed it",
    tone: "red",
    info: "A real spike happened and the model stayed quiet.",
  },
  correct_quiet: {
    label: "Quiet — correct",
    tone: "gray",
    info: "The model stayed quiet and the market stayed calm. This is the common case.",
  },
  pending: {
    label: "Waiting for result",
    tone: "gray",
    info: "Too recent to grade. A spike is defined over the next 60 seconds, so this prediction's answer doesn't exist yet.",
  },
  unavailable: {
    label: "Not gradeable",
    tone: "muted",
    info: "This one can't be honestly scored — the outcome never arrived, or the score was recorded after the answer was already known. It is excluded from the metrics rather than guessed at.",
  },
};

const UNKNOWN_OUTCOME = {
  label: MISSING,
  tone: "muted",
  info: "No outcome recorded for this prediction.",
};

export function outcomeMeta(outcome) {
  return OUTCOMES[outcome] ?? UNKNOWN_OUTCOME;
}

// The alert verdict: a score only means something relative to the threshold
// the model shipped with. `null` tau is NOT treated as zero — a model with
// no threshold cannot be said to be alerting or calm, and silently comparing
// against nothing would manufacture a verdict out of missing data.
export function alertVerdict(score, tau) {
  if (!isNum(score) || !isNum(tau)) return null;
  return score >= tau ? "alert" : "calm";
}

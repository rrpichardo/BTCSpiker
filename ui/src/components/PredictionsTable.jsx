import InfoIcon from "./InfoIcon.jsx";
import {
  formatBps,
  formatPercent,
  formatRate,
  formatSignedPercent,
  outcomeMeta,
} from "../units.js";

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

function fmtTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  return `${timeFormatter.format(d)}.${String(d.getMilliseconds()).padStart(3, "0")}`;
}

function fmtFixed(value, digits) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}

const ROW_LIMIT = 50;

// Column headers carry the plain-English name; the raw feature name lives in
// the InfoIcon so the technical reading is still one hover away rather than
// deleted. Cell `title` attributes keep the unrounded value reachable too --
// humanizing the display must not destroy the underlying number.
const FEATURE_COLUMNS = [
  {
    key: "vol_60s",
    label: "Recent swing",
    info: "vol_60s — how much the price has been bouncing around over the last minute, as a percentage of price. This is the quantity a 'spike' is defined on.",
    format: (value) => formatPercent(value, 3),
  },
  {
    key: "spread_bps",
    label: "Buy/sell gap",
    info: "spread_bps — the gap between the best buy and best sell price, in basis points (hundredths of a percent). Wider means it costs more to trade.",
    format: (value) => formatBps(value, 2),
  },
  {
    key: "log_return",
    label: "Last move",
    info: "log_return — how far the price moved on this tick versus the one before it, as a percentage. Positive is up.",
    format: (value) => formatSignedPercent(value, 4),
  },
  {
    key: "trade_intensity_60s",
    label: "Trades/sec",
    info: "trade_intensity_60s — how busy the market has been over the last minute, in trades per second.",
    format: (value) => formatRate(value, 1),
  },
];

export default function PredictionsTable({ predictions, unavailable = false }) {
  const rows = predictions.slice(0, ROW_LIMIT);
  const columnCount = FEATURE_COLUMNS.length + 4;

  return (
    <div className="table-scroll" tabIndex="0" aria-label="Recent predictions table">
      <table className="data-table">
        <caption className="sr-only">
          The 50 most recent volatility prediction events, newest first, each with
          the outcome it produced.
        </caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">
              Outcome{" "}
              <InfoIcon label="What actually happened after this prediction. Graded against realized volatility over the following 60 seconds." />
            </th>
            <th scope="col">
              Score{" "}
              <InfoIcon label="The model's confidence that a volatility spike is coming, from 0 to 1. It only means something compared with the alert threshold." />
            </th>
            <th scope="col">Model</th>
            {FEATURE_COLUMNS.map((column) => (
              <th scope="col" key={column.key}>
                {column.label} <InfoIcon label={column.info} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => {
            const meta = outcomeMeta(p.outcome);
            return (
              <tr key={p.event_id}>
                <td>{fmtTime(p.feature_ts || p.api_ts)}</td>
                <td>
                  <span className={`outcome-pill outcome-pill-${meta.tone}`}>
                    {meta.label}
                  </span>{" "}
                  <InfoIcon label={meta.info} />
                </td>
                <td>{fmtFixed(p.score, 3)}</td>
                <td className="cell-muted">
                  {p.model_variant} {p.model_version}
                </td>
                {FEATURE_COLUMNS.map((column) => (
                  <td
                    key={column.key}
                    title={
                      typeof p[column.key] === "number" ? String(p[column.key]) : undefined
                    }
                  >
                    {column.format(p[column.key])}
                  </td>
                ))}
              </tr>
            );
          })}
          {rows.length === 0 && (
            <tr>
              <td colSpan={columnCount} className="empty-cell">
                {unavailable
                  ? "No cached predictions are available."
                  : "Connected successfully; no predictions have been recorded yet."}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

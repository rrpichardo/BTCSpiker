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

function fmtScientific(value, digits) {
  return typeof value === "number" ? value.toExponential(digits) : "-";
}

const ROW_LIMIT = 50;

export default function PredictionsTable({ predictions, unavailable = false }) {
  const rows = predictions.slice(0, ROW_LIMIT);

  return (
    <div className="table-scroll" tabIndex="0" aria-label="Recent predictions table">
      <table className="data-table">
        <caption className="sr-only">
          The 50 most recent volatility prediction events, newest first.
        </caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">Score</th>
            <th scope="col">Variant</th>
            <th scope="col">Version</th>
            <th scope="col">vol_60s</th>
            <th scope="col">spread_bps</th>
            <th scope="col">log_return</th>
            <th scope="col">trade_intensity_60s</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.event_id}>
              <td>{fmtTime(p.feature_ts || p.api_ts)}</td>
              <td>{fmtFixed(p.score, 3)}</td>
              <td>{p.model_variant}</td>
              <td>{p.model_version}</td>
              <td>{fmtScientific(p.vol_60s, 6)}</td>
              <td>{fmtScientific(p.spread_bps, 6)}</td>
              <td>{fmtScientific(p.log_return, 6)}</td>
              <td>{fmtFixed(p.trade_intensity_60s, 2)}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={8} className="empty-cell">
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

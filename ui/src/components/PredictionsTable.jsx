function fmtTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleTimeString(undefined, { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function fmtFixed(value, digits) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}

function fmtScientific(value, digits) {
  return typeof value === "number" ? value.toExponential(digits) : "-";
}

const ROW_LIMIT = 50;

export default function PredictionsTable({ predictions }) {
  const rows = predictions.slice(0, ROW_LIMIT);

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Score</th>
            <th>Variant</th>
            <th>Version</th>
            <th>vol_60s</th>
            <th>spread_bps</th>
            <th>log_return</th>
            <th>trade_intensity_60s</th>
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
              <td>{fmtFixed(p.spread_bps, 2)}</td>
              <td>{fmtScientific(p.log_return, 6)}</td>
              <td>{fmtFixed(p.trade_intensity_60s, 2)}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={8} className="empty-cell">
                No predictions yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

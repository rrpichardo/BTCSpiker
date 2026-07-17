import { usePolling } from "../usePolling.js";
import { fetchRecentPredictions, fetchPredictionsHealth } from "../api.js";
import ScoreChart from "../components/ScoreChart.jsx";
import PredictionsTable from "../components/PredictionsTable.jsx";

const STALE_MS = 60 * 1000;

function isStale(health, healthError) {
  if (healthError) return true;
  if (!health) return false; // no data yet — not stale, just loading
  if (health.ok === false) return true;
  if (health.last_write_ts) {
    const age = Date.now() - new Date(health.last_write_ts).getTime();
    if (age > STALE_MS) return true;
  }
  return false;
}

export default function PredictionsPage() {
  const { data: predData, error: predError } = usePolling(fetchRecentPredictions, 2000);
  const { data: health, error: healthError } = usePolling(fetchPredictionsHealth, 5000);

  const predictions = predData?.predictions ?? [];
  const latest = predictions[0] ?? null;
  const stale = isStale(health, healthError);

  return (
    <div className="page">
      {stale && (
        <div className="stale-banner">
          Prediction feed is stale or degraded — the chart and table below may
          not reflect current market conditions.
          {healthError && ` (health check failed: ${healthError.message})`}
        </div>
      )}

      <div className="latest-score-panel">
        <div className="stat">
          <span className="stat-label">Latest score</span>
          <span className="stat-value">
            {latest ? latest.score.toFixed(3) : "-"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Model</span>
          <span className="stat-value stat-value-small">
            {latest ? `${latest.model_variant} / ${latest.model_version}` : "-"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Timestamp</span>
          <span className="stat-value stat-value-small">
            {latest ? (latest.feature_ts || latest.api_ts) : "-"}
          </span>
        </div>
      </div>

      <div className="panel">
        <h2>Score over time</h2>
        {predError && <p className="error-text">Failed to load predictions: {predError.message}</p>}
        <ScoreChart predictions={predictions} />
      </div>

      <div className="panel">
        <h2>Recent predictions</h2>
        <PredictionsTable predictions={predictions} />
      </div>
    </div>
  );
}

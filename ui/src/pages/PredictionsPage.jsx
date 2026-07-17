import { usePolling } from "../usePolling.js";
import { fetchRecentPredictions, fetchPredictionsHealth } from "../api.js";
import ScoreChart from "../components/ScoreChart.jsx";
import PredictionsTable from "../components/PredictionsTable.jsx";
import { ageMs, formatAge, formatDateTime } from "../format.js";

const STALE_MS = 60 * 1000;
const RESPONSE_STALE_MS = 10 * 1000;

function degradedReasons({ health, healthError, predError, predLastUpdated }) {
  const reasons = [];
  if (predError) reasons.push("the prediction query failed");
  if (healthError) reasons.push("the materializer health check failed");
  if (health?.ok === false) reasons.push("the materializer reports unhealthy");

  if (health?.last_write_ts) {
    const writeAge = ageMs(health.last_write_ts);
    if (writeAge === null) {
      reasons.push("the last-write timestamp is invalid");
    } else if (writeAge > STALE_MS) {
      reasons.push(`the prediction log has not advanced for ${Math.floor(writeAge / 1000)}s`);
    }
  }

  const responseAge = ageMs(predLastUpdated);
  if (responseAge !== null && responseAge > RESPONSE_STALE_MS) {
    reasons.push(`the last prediction response is ${Math.floor(responseAge / 1000)}s old`);
  }

  return reasons;
}

export default function PredictionsPage() {
  const {
    data: predData,
    error: predError,
    lastUpdated: predLastUpdated,
    isLoading: predictionsLoading,
    isRefreshing,
    isPaused,
  } = usePolling(fetchRecentPredictions, 2000);
  const { data: health, error: healthError } = usePolling(fetchPredictionsHealth, 5000);

  const predictions = predData?.predictions ?? [];
  const latest = predictions[0] ?? null;
  const reasons = degradedReasons({ health, healthError, predError, predLastUpdated });
  const stale = reasons.length > 0;
  const hasPredictionResponse = predData !== null;
  const statusLabel = isPaused ? "Paused" : stale ? "Degraded" : "Live";

  return (
    <section className="page" id="predictions-page" aria-labelledby="predictions-heading">
      <div className="page-header page-header-split">
        <div>
          <p className="eyebrow">Live scoring ledger</p>
          <h2 id="predictions-heading">Prediction feed</h2>
          <p className="page-header-note">
            2-second materializer polling · newest Bitcoin volatility signals first
          </p>
        </div>
        <div
          className={`feed-state ${stale ? "feed-state-degraded" : ""}`}
          aria-live="polite"
        >
          <span className="feed-state-dot" aria-hidden="true" />
          <span>{statusLabel}</span>
          <span className="feed-state-age">{formatAge(predLastUpdated)}</span>
          {isRefreshing && <span className="sr-only">Refreshing prediction feed</span>}
        </div>
      </div>

      {stale && (
        <div className="stale-banner" role="alert" aria-live="assertive">
          <strong>Prediction feed degraded.</strong> Cached values may not reflect current
          market conditions. Reasons: {reasons.join("; ")}.
        </div>
      )}

      <div className="latest-score-panel">
        <div className="stat stat-primary">
          <span className="stat-label">Spike score</span>
          <span className="stat-value">
            {latest && typeof latest.score === "number" ? latest.score.toFixed(3) : "—"}
          </span>
          <span className="stat-context">Latest model output · 0 to 1</span>
        </div>
        <div className="stat">
          <span className="stat-label">Model</span>
          <span className="stat-value stat-value-small">
            {latest ? `${latest.model_variant} / ${latest.model_version}` : "—"}
          </span>
          <span className="stat-context">Active scoring bundle</span>
        </div>
        <div className="stat">
          <span className="stat-label">Event time</span>
          <span className="stat-value stat-value-small">
            {latest ? formatDateTime(latest.feature_ts || latest.api_ts) : "—"}
          </span>
          <span className="stat-context">Original feature timestamp</span>
        </div>
        <div className="stat stat-compact">
          <span className="stat-label">Feed samples</span>
          <span className="stat-value stat-value-small">
            {hasPredictionResponse ? predictions.length.toLocaleString() : "—"}
          </span>
          <span className="stat-context">Up to 500 recent events</span>
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Signal trajectory</p>
            <h3 id="score-chart-heading">Score over time</h3>
          </div>
          <span className="panel-meta">0 = calm · 1 = elevated</span>
        </div>
        {predictionsLoading && !hasPredictionResponse && (
          <p className="state-message" role="status" aria-live="polite">
            Loading prediction history…
          </p>
        )}
        {!predictionsLoading && predError && !hasPredictionResponse && (
          <p className="state-message state-message-error" role="alert">
            Prediction history is unavailable. Check the materializer service and retry.
          </p>
        )}
        {hasPredictionResponse && predictions.length === 0 && (
          <p className="state-message">Connected successfully; no predictions have been recorded yet.</p>
        )}
        {predictions.length > 0 && <ScoreChart predictions={predictions} />}
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Event detail</p>
            <h3>Recent predictions</h3>
          </div>
          <span className="panel-meta">Showing newest 50</span>
        </div>
        {predictionsLoading && !hasPredictionResponse ? (
          <p className="state-message" role="status">Loading recent events…</p>
        ) : (
          <PredictionsTable predictions={predictions} unavailable={!hasPredictionResponse} />
        )}
      </div>
    </section>
  );
}

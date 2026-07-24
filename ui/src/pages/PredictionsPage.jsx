import { useCallback, useEffect, useState } from "react";
import { usePolling } from "../usePolling.js";
import { fetchRecentPredictions, fetchPredictionsHealth, fetchTimeline } from "../api.js";
import { buildTimelinePoints, rangeToWindow, RANGE_OPTIONS } from "../timelineData.js";
import TimelineCharts from "../components/TimelineCharts.jsx";
import PredictionsTable from "../components/PredictionsTable.jsx";
import { ageMs, formatAge, formatDateTime } from "../format.js";

const STALE_MS = 60 * 1000;
const RESPONSE_STALE_MS = 10 * 1000;

// 1m/5m/15m poll fast since the whole window is only minutes of data; 1h+
// ranges poll slower since a single new event barely moves the chart.
function pollMsForRange(range) {
  return range === "1h" || range === "6h" || range === "24h" ? 10_000 : 2_000;
}

function degradedReasons({ health, healthError, predError, predLastUpdated }) {
  const reasons = [];
  if (predError) reasons.push("the prediction query failed");
  if (healthError) reasons.push("the materializer health check failed");
  if (health?.ok === false) reasons.push("the materializer reports unhealthy");

  if (health?.ok === true) {
    if (typeof health.last_write_ts !== "string" || !health.last_write_ts.trim()) {
      reasons.push("the materializer has no durable-write timestamp");
    } else {
      const writeAge = ageMs(health.last_write_ts);
      if (writeAge === null) {
        reasons.push("the last-write timestamp is invalid");
      } else if (writeAge > STALE_MS) {
        reasons.push(`the prediction log has not advanced for ${Math.floor(writeAge / 1000)}s`);
      }
    }
  }

  const responseAge = ageMs(predLastUpdated);
  if (responseAge !== null && responseAge > RESPONSE_STALE_MS) {
    reasons.push(`the last prediction response is ${Math.floor(responseAge / 1000)}s old`);
  }

  return reasons;
}

export default function PredictionsPage() {
  const [range, setRange] = useState("15m");
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
  const bootstrapAnchor = latest?.feature_ts ?? latest?.api_ts ?? null;

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
            <h3>Price &amp; spike timeline</h3>
          </div>
          <div className="segmented-control">
            <div className="segmented-buttons" role="group" aria-label="Timeline range">
              {RANGE_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`segmented-btn ${range === option ? "segmented-btn-active" : ""}`}
                  aria-pressed={range === option}
                  onClick={() => setRange(option)}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        </div>
        <TimelinePanel key={range} range={range} bootstrapAnchor={bootstrapAnchor} />
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

// Keyed by `range` in the parent so switching ranges remounts this body and
// refetches immediately, rather than waiting out whatever interval was
// already in flight for the previous range.
function TimelinePanel({ range, bootstrapAnchor }) {
  const [anchor, setAnchor] = useState(bootstrapAnchor);

  const fetchFn = useCallback(
    (signal) => {
      const effectiveAnchor = anchor ?? bootstrapAnchor;
      const window_ = effectiveAnchor ? rangeToWindow(range, effectiveAnchor) : null;
      if (!window_) return Promise.resolve(null);
      return fetchTimeline(window_.from, window_.to, undefined, signal);
    },
    [range, anchor, bootstrapAnchor],
  );

  const { data, error, isLoading } = usePolling(fetchFn, pollMsForRange(range));

  // Anchor subsequent polls to the timeline's own newest feature_ts (not
  // wall-clock now), so the rolling window tracks data time and behaves
  // identically for live and replayed traffic.
  useEffect(() => {
    if (data?.available_to) setAnchor(data.available_to);
  }, [data]);

  const points = data?.points ? buildTimelinePoints(data.points) : [];
  const hasResponse = data !== null && data !== undefined;

  if (isLoading && !hasResponse) {
    return (
      <p className="state-message" role="status" aria-live="polite">
        Loading prediction history…
      </p>
    );
  }
  if (error && !hasResponse) {
    return (
      <p className="state-message state-message-error" role="alert">
        Timeline is unavailable. Check the materializer service and retry.
      </p>
    );
  }
  if (hasResponse && points.length === 0) {
    return <p className="state-message">Connected successfully; no predictions in this range yet.</p>;
  }
  if (!hasResponse) {
    return (
      <p className="state-message" role="status" aria-live="polite">
        Waiting for the first prediction to anchor the timeline…
      </p>
    );
  }

  return (
    <TimelineCharts
      points={points}
      from={data.from}
      to={data.to}
      complete={data.complete}
      availableFrom={data.available_from}
    />
  );
}

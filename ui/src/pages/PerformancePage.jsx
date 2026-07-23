import { useCallback, useMemo, useState } from "react";
import { usePolling } from "../usePolling.js";
import { fetchPerformance } from "../api.js";
import { buildEvalChartData, leadTimeLabel } from "../performanceData.js";
import EvalChart from "../components/EvalChart.jsx";
import MetricsTable from "../components/MetricsTable.jsx";
import InfoIcon from "../components/InfoIcon.jsx";
import { formatAge } from "../format.js";

const WINDOW_OPTIONS = [5, 30, 60];

const MODE_OPTIONS = [
  {
    id: "official",
    label: "Official definition",
    info: "A spike is what the model was trained on: volatility above the fixed training-time bar. Comparable to the training benchmarks.",
  },
  {
    id: "adaptive",
    label: "This window's top 15%",
    info: "A contextual activity view: the wildest 15% of THIS window counts as a spike. Always has something to grade, but it's a different exam — benchmarks don't apply.",
  },
];

export default function PerformancePage() {
  const [windowMinutes, setWindowMinutes] = useState(30);
  const [mode, setMode] = useState("official");

  return (
    // key={windowMinutes} remounts the polling body (and so its usePolling
    // effect) whenever the window selector changes, since usePolling only
    // restarts its effect on intervalMs changes, not fetchFn changes.
    <PerformanceBody
      key={windowMinutes}
      windowMinutes={windowMinutes}
      setWindowMinutes={setWindowMinutes}
      mode={mode}
      setMode={setMode}
    />
  );
}

function PerformanceBody({ windowMinutes, setWindowMinutes, mode, setMode }) {
  const fetchFn = useCallback(
    (signal) => fetchPerformance(windowMinutes, signal),
    [windowMinutes],
  );
  const { data, error, lastUpdated, isLoading, isRefreshing } = usePolling(fetchFn, 10000);

  const hasResponse = data !== null;
  const activeMode = data?.modes?.[mode];
  const windowInfo = data?.window;

  const chartData = useMemo(() => buildEvalChartData(data?.chart?.rows ?? []), [data]);
  const mlSeries = activeMode?.series?.find((s) => s.model_variant === "ml");
  const outcomeKey = mode === "official" ? "outcome_official" : "outcome_adaptive";

  const modeMeta =
    mode === "official"
      ? activeMode?.threshold_info
      : typeof activeMode?.percentile === "number"
        ? `top ${100 - activeMode.percentile}% by score in this window`
        : "";

  return (
    <section className="page" id="performance-page" aria-labelledby="performance-heading">
      <div className="page-header page-header-split">
        <div>
          <p className="eyebrow">Model evaluation</p>
          <h2 id="performance-heading">Model performance</h2>
          <p className="page-header-note">
            10-second polling · a prediction needs 60s of hindsight before it can be graded
          </p>
        </div>
        <p className="page-header-note page-header-freshness">
          {formatAge(lastUpdated)}
          {isRefreshing ? " · refreshing" : ""}
        </p>
      </div>

      <div className="perf-controls-row">
        <div className="segmented-control">
          <span className="segmented-label">
            Window
            <InfoIcon label="How far back we grade, in market time. Each prediction needs 60 seconds of hindsight before it can be graded." />
          </span>
          <div className="segmented-buttons" role="group" aria-label="Grading window">
            {WINDOW_OPTIONS.map((minutes) => (
              <button
                key={minutes}
                type="button"
                className={`segmented-btn ${windowMinutes === minutes ? "segmented-btn-active" : ""}`}
                aria-pressed={windowMinutes === minutes}
                onClick={() => setWindowMinutes(minutes)}
              >
                {minutes}m
              </button>
            ))}
          </div>
        </div>

        <div className="lead-time-stat">
          <span className="lead-time-label">
            Forecast lead: {leadTimeLabel(windowInfo?.median_lead_seconds)}
            <InfoIcon label="How far ahead of reality the model actually calls it. Near 60s is ideal; near zero means scores arrived when the answer was already known." />
          </span>
          {typeof windowInfo?.n_scored_late === "number" && windowInfo.n_scored_late > 0 && (
            <span className="lead-time-muted">({windowInfo.n_scored_late} scored late — excluded)</span>
          )}
        </div>
      </div>

      <div className="segmented-control">
        <div className="segmented-buttons" role="group" aria-label="Spike definition">
          {MODE_OPTIONS.map((option) => (
            <span className="segmented-item" key={option.id}>
              <button
                type="button"
                className={`segmented-btn ${mode === option.id ? "segmented-btn-active" : ""}`}
                aria-pressed={mode === option.id}
                onClick={() => setMode(option.id)}
              >
                {option.label}
              </button>
              <InfoIcon label={option.info} />
            </span>
          ))}
        </div>
      </div>

      {activeMode?.note?.triggered && (
        <div className="banner banner-amber" role="status">
          <span>{activeMode.note.detail}</span>
          <InfoIcon label="Why cautious wording? Short windows can't tell model decay apart from a calmer or wilder market. The project's own retraining rule uses a 7-day window." />
        </div>
      )}

      {isLoading && !hasResponse && (
        <p className="state-message" role="status" aria-live="polite">
          Loading performance data…
        </p>
      )}
      {error && (
        <p className="stale-banner" role="alert">
          {hasResponse
            ? "Performance refresh failed. Showing cached values."
            : "Performance data is unavailable. Check the API service and retry."}
        </p>
      )}

      {hasResponse && (
        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Signal trajectory</p>
              <h3 id="eval-chart-heading">Score over time</h3>
            </div>
            <span className="panel-meta">0 = calm · 1 = elevated</span>
          </div>
          {chartData.length === 0 ? (
            <p className="state-message">No graded predictions in this window yet.</p>
          ) : (
            <EvalChart chartData={chartData} outcomeKey={outcomeKey} tau={mlSeries?.tau} />
          )}
        </div>
      )}

      {hasResponse && (
        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Grading results</p>
              <h3>Metrics</h3>
            </div>
            {modeMeta && <span className="panel-meta">{modeMeta}</span>}
          </div>
          <MetricsTable
            mode={activeMode}
            modeKey={mode}
            reference={data?.reference}
            gradedN={windowInfo?.n_graded}
          />
        </div>
      )}
    </section>
  );
}

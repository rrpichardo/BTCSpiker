import { useCallback, useEffect, useState } from "react";
import { usePolling } from "../usePolling.js";
import { fetchTournamentRuns, fetchTournamentRun } from "../api.js";
import { formatAge } from "../format.js";
import { formatMetric } from "../performanceData.js";

// The tournament is an offline batch job, not a live stream: runs appear every
// few minutes at most, so this polls far more slowly than the trading pages.
const POLL_MS = 15_000;

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

function statusPill(status) {
  if (status === "FINISHED") return "pill-green";
  if (status === "RUNNING" || status === "SCHEDULED") return "pill-amber";
  if (status === "FAILED" || status === "KILLED") return "pill-red";
  return "pill-gray";
}

function RunDetail({ runId }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    setDetail(null);
    setError(null);
    fetchTournamentRun(runId, controller.signal)
      .then(setDetail)
      .catch((err) => {
        if (err.name !== "AbortError") setError(err);
      });
    return () => controller.abort();
  }, [runId]);

  if (error) {
    return (
      <p className="state-message" role="alert">
        {error.status === 503
          ? "The tournament store is unavailable right now — this isn’t about this specific run."
          : "Could not load this run’s detail. It may have been removed from the store."}
      </p>
    );
  }
  if (!detail) {
    return (
      <p className="state-message" role="status">
        Loading run detail…
      </p>
    );
  }

  const modelParams = detail.model_params;
  // model_params is normally a JSON object, but the backend deliberately
  // passes an unparseable value through as a raw string rather than dropping
  // it, so both shapes have to render.
  const paramEntries =
    modelParams && typeof modelParams === "object" ? Object.entries(modelParams) : [];

  return (
    <div className="panel">
      <h3 className="panel-heading">{detail.run_name || detail.run_id}</h3>
      <p className="panel-meta">
        {detail.stage ? `${detail.stage} stage` : "stage unknown"} ·{" "}
        {detail.model_family || "model unrecorded"} · ran for{" "}
        {formatDuration(detail.duration_seconds)}
      </p>

      <h4 className="panel-heading">Settings chosen for this run</h4>
      {paramEntries.length > 0 ? (
        <table className="data-table tournament-table">
          <thead>
            <tr>
              <th scope="col">Hyperparameter</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            {paramEntries.map(([key, value]) => (
              <tr key={key}>
                <th scope="row">{key}</th>
                <td>{typeof value === "number" ? String(value) : String(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="state-message">
          {modelParams
            ? String(modelParams)
            : "This run recorded no model hyperparameters (typical for the baseline stage)."}
        </p>
      )}

      <h4 className="panel-heading">Score on each time fold</h4>
      {detail.fold_pr_aucs?.length > 0 ? (
        <table className="data-table tournament-table">
          <thead>
            <tr>
              <th scope="col">Fold</th>
              <th scope="col">PR-AUC</th>
            </tr>
          </thead>
          <tbody>
            {detail.fold_pr_aucs.map((fold) => (
              <tr key={fold.fold}>
                <th scope="row">Fold {fold.fold}</th>
                <td>{formatMetric(fold.pr_auc, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="state-message">No per-fold scores were recorded.</p>
      )}

      <h4 className="panel-heading">Decision threshold &amp; features</h4>
      <table className="data-table tournament-table">
        <tbody>
          <tr>
            <th scope="row">tau (alert threshold)</th>
            <td>{detail.params?.tau ?? "—"}</td>
          </tr>
          <tr>
            <th scope="row">Feature set</th>
            <td>{detail.params?.feature_set_id ?? detail.tags?.feature_set_id ?? "—"}</td>
          </tr>
          <tr>
            <th scope="row">Features used</th>
            <td>{detail.params?.feature_cols ?? "—"}</td>
          </tr>
          <tr>
            <th scope="row">Deployable</th>
            <td>{detail.deployable ? "Yes" : "No"}</td>
          </tr>
        </tbody>
      </table>

      {detail.artifacts?.length > 0 && (
        <>
          <h4 className="panel-heading">Evidence recorded</h4>
          <p className="panel-meta">
            Files this run wrote alongside its scores: {detail.artifacts.join(", ")}
          </p>
        </>
      )}
    </div>
  );
}

export default function TournamentPage() {
  const [selected, setSelected] = useState(null);
  const fetcher = useCallback((signal) => fetchTournamentRuns(signal), []);
  const { data, error, lastUpdated, isLoading, isRefreshing } = usePolling(
    fetcher,
    POLL_MS,
  );

  const runs = data?.runs ?? [];
  const hasResponse = data !== null;
  // 503 is the backend's "the store isn't mounted / no tournament has run"
  // signal, which is a setup state rather than a failure.
  const notWiredUp = error?.status === 503;

  return (
    <section className="page" id="tournament-page" aria-labelledby="tournament-heading">
      <div className="page-header page-header-split">
        <div>
          <p className="eyebrow">Offline model search</p>
          <h2 id="tournament-heading">Tournament</h2>
          <p className="page-header-note">
            Every model tried, ranked by PR-AUC. Higher is better; compare against the
            spike rate, not against zero.
          </p>
        </div>
        <div className="system-summary" aria-live="polite">
          <span className="page-header-note">
            {formatAge(lastUpdated)}
            {isRefreshing ? " · checking" : ""}
          </span>
        </div>
      </div>

      {isLoading && !hasResponse && !error && (
        <p className="state-message" role="status" aria-live="polite">
          Loading tournament runs…
        </p>
      )}

      {notWiredUp && !hasResponse && (
        <p className="state-message" role="status">
          No tournament results yet. Runs appear here once{" "}
          <code>scripts/run_experiments.py</code> has completed a stage.
        </p>
      )}

      {notWiredUp && hasResponse && (
        <p className="stale-banner" role="alert">
          The tournament store became unavailable; the table below is cached and may be
          out of date.
        </p>
      )}

      {error && !notWiredUp && (
        <p className="stale-banner" role="alert">
          {hasResponse
            ? "The refresh failed; the table below is cached and may be out of date."
            : "Tournament results are unavailable. Confirm the materializer is reachable."}
        </p>
      )}

      {hasResponse && runs.length === 0 && !notWiredUp && (
        <p className="state-message">
          The tournament store is set up but holds no runs yet.
        </p>
      )}

      {runs.length > 0 && (
        <div className="panel">
          <h3 className="panel-heading">Leaderboard</h3>
          <p className="panel-meta">
            Select a run to see the settings it used and how it scored on each time fold.
          </p>
          <div className="metrics-table-wrap">
            <table className="data-table tournament-table">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Run</th>
                  <th scope="col">Stage</th>
                  <th scope="col">Model</th>
                  <th scope="col">PR-AUC</th>
                  <th scope="col">Folds won</th>
                  <th scope="col">Status</th>
                  <th scope="col">Deployable</th>
                  <th scope="col">Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run, index) => (
                  <tr
                    key={run.run_id}
                    onClick={() => setSelected(run.run_id)}
                    className={selected === run.run_id ? "row-selected" : undefined}
                  >
                    <td>{index + 1}</td>
                    <th scope="row">
                      <button
                        type="button"
                        className="linklike"
                        onClick={() => setSelected(run.run_id)}
                        aria-expanded={selected === run.run_id}
                      >
                        {run.run_name || run.run_id}
                      </button>
                    </th>
                    <td>{run.stage || "—"}</td>
                    <td>{run.model_family || "—"}</td>
                    <td>{formatMetric(run.aggregate_pr_auc, 4)}</td>
                    <td>{formatMetric(run.folds_won, 0)}</td>
                    <td>
                      <span className={`pill ${statusPill(run.status)}`}>{run.status}</span>
                    </td>
                    <td>{run.deployable ? "Yes" : "No"}</td>
                    <td>{formatDuration(run.duration_seconds)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {selected && <RunDetail runId={selected} />}
    </section>
  );
}

import { formatMetric } from "../performanceData.js";
import InfoIcon from "./InfoIcon.jsx";

// Metric rows shown for every series column. `refKey` names the field on
// reference.ml that feeds the two training-benchmark columns — most rows
// have no training-time equivalent in the payload, so they show "—" there.
// Plain-language name first, technical name kept in parentheses: the numbers
// are unchanged and still comparable to the training benchmarks, but the row
// should be readable by someone who has never seen "PR-AUC" before.
const METRIC_ROWS = [
  {
    key: "pr_auc",
    label: "Ranking quality (PR-AUC)",
    refKey: "pr_auc",
    info: "The headline metric. Higher means the moments the model was most confident about really were the spikes. Judged on ranking, so it doesn't depend on where the alert threshold sits.",
  },
  {
    key: "precision",
    label: "When it alerted, how often it was right (Precision@τ)",
    refKey: null,
    info: "Of all the times the model raised an alert, the share where a spike really followed.",
  },
  {
    key: "recall",
    label: "Of real spikes, how many it caught (Recall@τ)",
    refKey: null,
    info: "Of all the spikes that actually happened, the share the model alerted on.",
  },
  {
    key: "f1",
    label: "Balance of those two (F1@τ)",
    refKey: "f1",
    info: "A single score combining the two above. Useful because you can always make one look good by sacrificing the other.",
  },
];

export default function MetricsTable({ mode, modeKey, reference, gradedN }) {
  const series = mode?.series ?? [];
  const baseRate = mode?.base_rate;
  const mlAvailable = mode?.ml_available !== false;
  const refGreyed = modeKey === "adaptive";
  const refCellClass = refGreyed ? "metrics-col-greyed" : "";

  const zeroPositives = baseRate === 0;

  return (
    <div className="metrics-table-wrap">
      {!mlAvailable && (
        <p className="state-message">
          {mode?.threshold_info || "ML model produced no graded predictions in this window."}
        </p>
      )}
      {zeroPositives && (
        <p className="state-message">
          No real spikes occurred in this window ({formatMetric(gradedN, 0)} graded, spike rate 0%)
          — that's a fact about the market, not an error.
        </p>
      )}

      <div className="table-scroll">
        <table className="data-table metrics-table">
          <caption className="sr-only">
            Grading metrics per model series, with training-time benchmark columns.
          </caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              {series.map((s) => (
                <th scope="col" key={s.name}>
                  {s.name}
                </th>
              ))}
              <th scope="col" className={refCellClass}>
                Training (test)
              </th>
              <th scope="col" className={refCellClass}>
                Training (val)
              </th>
            </tr>
          </thead>
          <tbody>
            {METRIC_ROWS.map((row) => (
              <tr key={row.key}>
                <th scope="row">
                  {row.label} <InfoIcon label={row.info} />
                </th>
                {series.map((s) => (
                  <td key={s.name}>
                    {formatMetric(s[row.key])}
                    {row.key === "pr_auc" && s.pr_auc == null && s.pr_auc_reason && (
                      <InfoIcon label={s.pr_auc_reason} />
                    )}
                  </td>
                ))}
                <td className={refCellClass}>
                  {row.refKey ? formatMetric(reference?.ml?.[`${row.refKey}_test`]) : "—"}
                </td>
                <td className={refCellClass}>
                  {row.refKey ? formatMetric(reference?.ml?.[`${row.refKey}_val`]) : "—"}
                </td>
              </tr>
            ))}
            <tr>
              <th scope="row">
                Alerts: right / wrong{" "}
                <InfoIcon label="Of the predictions the model flagged as spikes, how many were followed by a real spike and how many weren't (true positives / false positives)." />
              </th>
              {series.map((s) => (
                <td key={s.name}>
                  {formatMetric(s.tp, 0)} / {formatMetric(s.fp, 0)}
                </td>
              ))}
              <td className={refCellClass}>—</td>
              <td className={refCellClass}>—</td>
            </tr>
            <tr>
              <th scope="row">
                Spikes missed / calm called right{" "}
                <InfoIcon label="Real spikes the model stayed quiet through, and calm stretches it correctly left alone (false negatives / true negatives)." />
              </th>
              {series.map((s) => (
                <td key={s.name}>
                  {formatMetric(s.fn, 0)} / {formatMetric(s.tn, 0)}
                </td>
              ))}
              <td className={refCellClass}>—</td>
              <td className={refCellClass}>—</td>
            </tr>
            <tr>
              <th scope="row">
                How often spikes actually happened{" "}
                <InfoIcon label="A single window-level fact about the market, not a measure of any model: the share of graded moments that were real spikes. A model has to beat this to be worth anything." />
              </th>
              <td colSpan={series.length || 1}>{formatMetric(baseRate)}</td>
              <td className={refCellClass}>—</td>
              <td className={refCellClass}>—</td>
            </tr>
            <tr>
              <th scope="row">Graded predictions (n)</th>
              {series.map((s) => (
                <td key={s.name}>{formatMetric(s.n, 0)}</td>
              ))}
              <td className={refCellClass}>—</td>
              <td className={refCellClass}>—</td>
            </tr>
          </tbody>
        </table>
      </div>

      {refGreyed && (
        <p className="metrics-footnote">Benchmarks apply to the official definition.</p>
      )}
      {reference?.baseline_reference && (
        <p className="metrics-footnote">
          Baseline reference: PR-AUC {formatMetric(reference.baseline_reference.pr_auc_test)} (test),{" "}
          {formatMetric(reference.baseline_reference.pr_auc_val)} (val).{" "}
          {reference.baseline_reference.scorer_note}
        </p>
      )}
    </div>
  );
}

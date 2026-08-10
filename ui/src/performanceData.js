// Pure data-shaping helpers for the Performance tab. Kept framework-free so
// they're testable with node --test (see performanceData.test.js).

// Turns raw `chart.rows` from the performance endpoint into event-time
// ordered points for EvalChart. Rows with an unparsable feature_ts are
// dropped rather than plotted at NaN.
export function buildEvalChartData(rows) {
  if (!Array.isArray(rows)) return [];
  return rows
    .map((row) => ({
      timestamp: new Date(row.feature_ts).getTime(),
      score: typeof row.score === "number" ? row.score : null,
      model_variant: row.model_variant ?? null,
      baseline_score: typeof row.baseline_score === "number" ? row.baseline_score : null,
      outcome_official: row.outcome_official ?? null,
      outcome_adaptive: row.outcome_adaptive ?? null,
    }))
    .filter((point) => Number.isFinite(point.timestamp))
    .sort((left, right) => left.timestamp - right.timestamp);
}

// Merges consecutive rows where chartData[i][key] === true into contiguous
// [x1, x2] timestamp spans, for ReferenceArea shading in EvalChart. A span
// covering a single row (no true neighbor) is otherwise invisible on the
// chart, so it's widened to a minimum half the median inter-row gap on
// each side.
export function outcomeBands(chartData, key) {
  if (!Array.isArray(chartData) || chartData.length === 0) return [];

  const gaps = [];
  for (let i = 1; i < chartData.length; i++) {
    const gap = chartData[i].timestamp - chartData[i - 1].timestamp;
    if (Number.isFinite(gap) && gap > 0) gaps.push(gap);
  }
  gaps.sort((a, b) => a - b);
  const medianGap = gaps.length > 0 ? gaps[Math.floor(gaps.length / 2)] : 1000;
  const minHalfWidth = medianGap / 2;

  const bands = [];
  let start = null;
  let end = null;

  for (const point of chartData) {
    if (point[key] === true) {
      if (start === null) start = point.timestamp;
      end = point.timestamp;
    } else if (start !== null) {
      bands.push(closeSpan(start, end, minHalfWidth));
      start = null;
      end = null;
    }
  }
  if (start !== null) bands.push(closeSpan(start, end, minHalfWidth));

  return bands;
}

function closeSpan(start, end, minHalfWidth) {
  if (start !== end) return { x1: start, x2: end };
  return { x1: start - minHalfWidth, x2: end + minHalfWidth };
}

export function formatMetric(value, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

// ML precision minus the baseline series' precision, in percentage points --
// null (not 0) when either series is missing or lacks a numeric precision,
// so the caller can render "—" instead of a misleading "+0.0pp".
export function precisionDeltaPp(mlSeries, baselineSeries) {
  if (typeof mlSeries?.precision !== "number" || typeof baselineSeries?.precision !== "number") {
    return null;
  }
  return (mlSeries.precision - baselineSeries.precision) * 100;
}

export function formatSignedPercentagePoints(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
  return `${sign}${Math.abs(value).toFixed(1)}pp`;
}

// Shared by the Grading lag and Label horizon stats — both are "N seconds",
// nothing more; only their surrounding label text differs.
export function formatSeconds(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  return `${Math.round(seconds)}s`;
}

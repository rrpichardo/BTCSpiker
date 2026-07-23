import assert from "node:assert/strict";
import test from "node:test";

import {
  buildEvalChartData,
  outcomeBands,
  formatMetric,
  leadTimeLabel,
} from "./performanceData.js";

test("buildEvalChartData sorts by feature_ts and drops invalid timestamps", () => {
  const points = buildEvalChartData([
    {
      feature_ts: "2026-07-16T19:00:03Z",
      score: 0.3,
      baseline_score: 0,
      outcome_official: false,
      outcome_adaptive: true,
    },
    {
      feature_ts: "not-a-date",
      score: 0.9,
      baseline_score: 1,
      outcome_official: true,
      outcome_adaptive: true,
    },
    {
      feature_ts: "2026-07-16T19:00:01Z",
      score: 0.1,
      baseline_score: 0,
      outcome_official: false,
      outcome_adaptive: false,
    },
  ]);

  assert.deepEqual(
    points.map((point) => point.timestamp),
    [Date.parse("2026-07-16T19:00:01Z"), Date.parse("2026-07-16T19:00:03Z")],
  );
  assert.deepEqual(
    points.map((point) => point.score),
    [0.1, 0.3],
  );
});

test("outcomeBands merges consecutive true rows into one span", () => {
  const chartData = [
    { timestamp: 0, hit: true },
    { timestamp: 1000, hit: true },
    { timestamp: 2000, hit: true },
    { timestamp: 3000, hit: false },
  ];
  assert.deepEqual(outcomeBands(chartData, "hit"), [{ x1: 0, x2: 2000 }]);
});

test("outcomeBands splits into separate spans across a false gap", () => {
  const chartData = [
    { timestamp: 0, hit: true },
    { timestamp: 1000, hit: true },
    { timestamp: 2000, hit: false },
    { timestamp: 3000, hit: true },
  ];
  assert.deepEqual(outcomeBands(chartData, "hit"), [
    { x1: 0, x2: 1000 },
    { x1: 2500, x2: 3500 },
  ]);
});

test("outcomeBands widens an isolated true row to a minimum visual width", () => {
  const chartData = [
    { timestamp: 0, hit: false },
    { timestamp: 1000, hit: true },
    { timestamp: 2000, hit: false },
    { timestamp: 3000, hit: false },
  ];
  // Gaps are all 1000ms, so median gap = 1000ms and half-width = 500ms.
  assert.deepEqual(outcomeBands(chartData, "hit"), [{ x1: 500, x2: 1500 }]);
});

test("outcomeBands returns an empty array for no data or no matches", () => {
  assert.deepEqual(outcomeBands([], "hit"), []);
  assert.deepEqual(
    outcomeBands([{ timestamp: 0, hit: false }], "hit"),
    [],
  );
});

test("formatMetric renders null and undefined as an em dash", () => {
  assert.equal(formatMetric(null), "—");
  assert.equal(formatMetric(undefined), "—");
  assert.equal(formatMetric(0.20999), "0.210");
  assert.equal(formatMetric(0.5, 1), "0.5");
  assert.equal(formatMetric(37, 0), "37");
});

test("leadTimeLabel formats seconds and falls back for null", () => {
  assert.equal(leadTimeLabel(57.4), "57s ahead");
  assert.equal(leadTimeLabel(0), "0s ahead");
  assert.equal(leadTimeLabel(null), "—");
  assert.equal(leadTimeLabel(undefined), "—");
});

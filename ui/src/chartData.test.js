import assert from "node:assert/strict";
import test from "node:test";

import { buildChartData } from "./chartData.js";

test("buildChartData orders live scores by api timestamp", () => {
  const points = buildChartData([
    {
      api_ts: "2026-07-16T19:00:03Z",
      feature_ts: "2026-07-16T18:00:00Z",
      score: 0.3,
    },
    {
      api_ts: "2026-07-16T19:00:01Z",
      feature_ts: "2026-07-16T20:00:00Z",
      score: 0.1,
    },
    {
      api_ts: "2026-07-16T19:00:02Z",
      feature_ts: "2026-07-16T17:00:00Z",
      score: 0.2,
    },
  ]);

  assert.deepEqual(
    points.map((point) => point.timestamp),
    [
      Date.parse("2026-07-16T19:00:01Z"),
      Date.parse("2026-07-16T19:00:02Z"),
      Date.parse("2026-07-16T19:00:03Z"),
    ],
  );
  assert.deepEqual(
    points.map((point) => point.score),
    [0.1, 0.2, 0.3],
  );
});

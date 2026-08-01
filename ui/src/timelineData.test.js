import assert from "node:assert/strict";
import test from "node:test";

import {
  buildTimelinePoints,
  rangeToWindow,
  classBands,
  priceTickLabel,
  priceTickAxisWidth,
} from "./timelineData.js";

test("buildTimelinePoints keys on feature_ts, ignoring api_ts entirely", () => {
  const points = buildTimelinePoints([
    { feature_ts: "2026-07-16T19:00:03Z", api_ts: "2026-07-16T10:00:00Z", score: 0.3, class: "pending" },
    { feature_ts: "2026-07-16T19:00:01Z", api_ts: "2026-07-16T23:00:00Z", score: 0.1, class: "pending" },
    { feature_ts: "2026-07-16T19:00:02Z", api_ts: "2026-07-16T05:00:00Z", score: 0.2, class: "pending" },
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

test("buildTimelinePoints treats missing price/score as null, not dropped or zero", () => {
  const points = buildTimelinePoints([
    { feature_ts: "2026-07-16T19:00:00Z", market_price: 65000.5, score: 0.4, class: "correct_call" },
    { feature_ts: "2026-07-16T19:00:01Z", class: "pending" }, // no price, no score
    { feature_ts: "2026-07-16T19:00:02Z", market_price: "bad", score: 0.1, class: "pending" },
  ]);

  assert.equal(points.length, 3);
  assert.deepEqual(
    points.map((p) => p.price),
    [65000.5, null, null],
  );
  assert.deepEqual(
    points.map((p) => p.score),
    [0.4, null, 0.1],
  );
});

test("buildTimelinePoints derives flags from a raw point's single class", () => {
  const [correctCall, falseAlarm, missedSpike, correctQuiet, pending, unavailable] = buildTimelinePoints([
    { feature_ts: "2026-07-16T19:00:00Z", class: "correct_call" },
    { feature_ts: "2026-07-16T19:00:01Z", class: "false_alarm" },
    { feature_ts: "2026-07-16T19:00:02Z", class: "missed_spike" },
    { feature_ts: "2026-07-16T19:00:03Z", class: "correct_quiet" },
    { feature_ts: "2026-07-16T19:00:04Z", class: "pending" },
    { feature_ts: "2026-07-16T19:00:05Z", class: "unavailable" },
  ]);

  assert.deepEqual(
    [correctCall.predicted, correctCall.confirmedSpike],
    [true, true],
  );
  assert.deepEqual(
    [falseAlarm.predicted, falseAlarm.confirmedSpike, falseAlarm.falseAlarm],
    [true, false, true],
  );
  assert.deepEqual(
    [missedSpike.predicted, missedSpike.confirmedSpike, missedSpike.missedSpike],
    [false, true, true],
  );
  assert.deepEqual(
    [correctQuiet.predicted, correctQuiet.confirmedSpike],
    [false, false],
  );
  assert.equal(pending.pending, true);
  assert.equal(unavailable.unavailable, true);
});

test("buildTimelinePoints derives flags from a bucketed point's class counts, keeping pairs distinct", () => {
  // A bucket with one missed_spike and one false_alarm must show BOTH, and
  // must never claim a correct_call it never had -- proves aggregation never
  // pairs one row's prediction with a different row's outcome.
  const [bucket] = buildTimelinePoints([
    {
      feature_ts: "2026-07-16T19:00:00Z",
      classes: { missed_spike: 1, false_alarm: 1 },
    },
  ]);

  assert.equal(bucket.missedSpike, true);
  assert.equal(bucket.falseAlarm, true);
  assert.equal(bucket.confirmedSpike, true); // missed_spike counts as a real spike
  assert.equal(bucket.predicted, true); // false_alarm counts as a predicted call
});

test("rangeToWindow anchors on the given timestamp, not wall-clock now", () => {
  const anchor = "2026-07-16T19:10:00Z";

  const oneMinute = rangeToWindow("1m", anchor);
  assert.equal(oneMinute.from, new Date(Date.parse(anchor) - 60_000).toISOString());

  const oneHour = rangeToWindow("1h", anchor);
  assert.equal(oneHour.from, new Date(Date.parse(anchor) - 3_600_000).toISOString());

  assert.equal(rangeToWindow("bogus", anchor), null);
  assert.equal(rangeToWindow("1m", "not-a-date"), null);
});

test("rangeToWindow pads `to` past the anchor so the newest point isn't excluded by the API's feature_ts < to bound", () => {
  const anchor = "2026-07-16T19:10:00Z";
  const anchorMs = Date.parse(anchor);

  const { to } = rangeToWindow("1m", anchor);

  assert.ok(Date.parse(to) > anchorMs, "to must be strictly after the anchor");
});

test("classBands (re-exported outcomeBands) merges consecutive flagged points into spans", () => {
  const points = buildTimelinePoints([
    { feature_ts: "2026-07-16T19:00:00Z", class: "correct_quiet" },
    { feature_ts: "2026-07-16T19:00:01Z", class: "correct_call" },
    { feature_ts: "2026-07-16T19:00:02Z", class: "false_alarm" },
    { feature_ts: "2026-07-16T19:00:03Z", class: "correct_quiet" },
  ]);

  const bands = classBands(points, "predicted");

  assert.deepEqual(bands, [
    { x1: Date.parse("2026-07-16T19:00:01Z"), x2: Date.parse("2026-07-16T19:00:02Z") },
  ]);
});

test("priceTickLabel produces five distinct labels for a tight real-world domain", () => {
  // The exact defect observed: five recharts auto-ticks over a 5-minute
  // window (BTC moving tens of dollars) all rounded to "$70k".
  const values = [69800, 69900, 70000, 70100, 70200];
  const domainSpan = Math.max(...values) - Math.min(...values); // 400

  const labels = values.map((v) => priceTickLabel(v, domainSpan));

  assert.equal(new Set(labels).size, 5, `expected 5 distinct labels, got ${labels}`);
});

test("priceTickLabel falls back to the coarse $Xk form over a wide domain", () => {
  const domainSpan = 8000; // a multi-hour range with a real four-figure swing
  assert.equal(priceTickLabel(69000, domainSpan), "$69k");
  assert.equal(priceTickLabel(70000, domainSpan), "$70k");
});

test("priceTickAxisWidth widens as priceTickLabel's precision increases", () => {
  const wide = priceTickAxisWidth(8000);
  const narrow = priceTickAxisWidth(50);
  assert.ok(narrow > wide, `expected a tighter domain to need more width (${narrow} > ${wide})`);
});

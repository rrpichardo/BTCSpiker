import test from "node:test";
import assert from "node:assert/strict";
import {
  alertVerdict,
  formatBps,
  formatPercent,
  formatRate,
  formatSignedPercent,
  outcomeMeta,
} from "./units.js";

test("formatPercent renders a log-return ratio as a readable percentage", () => {
  // 5e-5 is what the table used to print as "5.000000e-5".
  assert.equal(formatPercent(0.00005), "0.005%");
  assert.equal(formatPercent(0.01), "1.000%");
});

test("formatSignedPercent keeps the direction of the move", () => {
  assert.equal(formatSignedPercent(0.0001, 4), "+0.0100%");
  assert.equal(formatSignedPercent(-0.0001, 4), "-0.0100%");
  // Zero is genuinely no move; it must not claim an upward one.
  assert.equal(formatSignedPercent(0, 4), "0.0000%");
});

test("formatBps and formatRate spell out their units at sane precision", () => {
  assert.equal(formatBps(1.5), "1.50 bps");
  assert.equal(formatRate(10), "10.0");
});

test("a missing value never renders as zero", () => {
  // The whole point: absent data and a genuine zero are different claims.
  for (const format of [formatPercent, formatSignedPercent, formatBps, formatRate]) {
    assert.equal(format(null), "—");
    assert.equal(format(undefined), "—");
    assert.equal(format(NaN), "—");
    assert.equal(format(Infinity), "—");
    assert.equal(format("0.5"), "—");
  }
});

test("every outcome class from classify_row has a plain-language label", () => {
  // These six strings are exactly what materializer/timeline.py classify_row
  // can return; a new one there must not surface as raw snake_case here.
  const classes = [
    "correct_call",
    "false_alarm",
    "missed_spike",
    "correct_quiet",
    "pending",
    "unavailable",
  ];
  for (const cls of classes) {
    const meta = outcomeMeta(cls);
    assert.notEqual(meta.label, "—", `${cls} should have a label`);
    assert.ok(meta.info.length > 0, `${cls} should explain itself`);
    assert.match(meta.label, /^[A-Z]/, `${cls} label should read as prose`);
    assert.doesNotMatch(meta.label, /_/, `${cls} label should not be snake_case`);
  }
});

test("an unrecognized outcome degrades to neutral, not to a raw string", () => {
  const meta = outcomeMeta("some_future_class");
  assert.equal(meta.label, "—");
  assert.equal(meta.tone, "muted");
  assert.equal(outcomeMeta(null).label, "—");
  assert.equal(outcomeMeta(undefined).label, "—");
});

test("alertVerdict compares the score against the shipped threshold", () => {
  assert.equal(alertVerdict(0.9, 0.7), "alert");
  assert.equal(alertVerdict(0.5, 0.7), "calm");
  // At tau exactly: the model alerts (score >= tau), matching classify_row.
  assert.equal(alertVerdict(0.7, 0.7), "alert");
});

test("alertVerdict refuses to invent a verdict from missing data", () => {
  // A null tau must NOT be treated as 0, which would make every score alert.
  assert.equal(alertVerdict(0.9, null), null);
  assert.equal(alertVerdict(null, 0.7), null);
  assert.equal(alertVerdict(undefined, undefined), null);
  assert.equal(alertVerdict(NaN, 0.7), null);
});

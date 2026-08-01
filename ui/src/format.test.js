import assert from "node:assert/strict";
import test from "node:test";

import { formatProvenance } from "./format.js";

test("formatProvenance distinguishes live, replay, and unknown sources", () => {
  assert.equal(formatProvenance("live"), "Live market data");
  assert.equal(formatProvenance("replay"), "Replay data");
  assert.equal(formatProvenance(null), "Unknown data source");
  assert.equal(formatProvenance(undefined), "Unknown data source");
});

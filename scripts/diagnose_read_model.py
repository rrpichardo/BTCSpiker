"""
Diagnostics gate for the prediction read model (Phase 1.2 of
docs/goals/prediction-quality-goal.md's provenance plan).

Asserts the invariants that decide whether /predictions/performance is
measuring something real, versus a materialization artifact. Run it against
a live stack (`docker compose up -d`) any time you want to trust the
dashboard's numbers:

    python scripts/diagnose_read_model.py

Read-only: issues HTTP GETs against the running api/materializer/mlflow
services and read-only SQL SELECTs (via `docker compose exec`) against the
materializer's SQLite file. Never writes to the DB, never touches Kafka
offsets, never restarts a container.

Exit 0 if every invariant passes, 1 otherwise.
"""

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATERIALIZER_URL = "http://localhost:8090"
API_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5001"

# Variant B, the shipped feature set -- see handoff/docs/feature_spec.md
# ("Final model feature set (Variant B - 7 features)"). This is the
# order api/main.py's FEATURE_COLS_DEFAULT and the MLflow run's
# `feature_cols` param are both expected to match.
CANONICAL_FEATURE_COLS = [
    "log_return",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "trade_intensity_60s",
    "n_ticks_60s",
    "spread_mean_60s",
]

# Executed inside the materializer container (docker compose exec), which
# has this repo's materializer.py on PYTHONPATH (materializer/Dockerfile sets
# PYTHONPATH=/app/materializer) but not the rest of the repo -- so the query
# logic travels as a string rather than assuming scripts/ is mounted there.
# argv: from_ts, stream_id (JSON-encoded, since it can be null).
#
# No upper bound on feature_ts, matching materializer.py's PERFORMANCE_JOIN_SQL
# exactly (`WHERE feature_ts >= ? AND stream_id IS ?`, no `<= to_ts` --
# contrast TIMELINE_JOIN_SQL, which is bounded on both sides for a different
# endpoint). A bounded-above version would silently undercount relative to
# the live endpoint on a continuously-replaying stream, since new rows keep
# arriving past any snapshot's "newest" timestamp.
SQLITE_SCRIPT = r'''
import json, sys
from datetime import datetime

from_ts, stream_id_json = sys.argv[1], sys.argv[2]
stream_id = json.loads(stream_id_json)

import materializer
con = materializer._open_readonly(materializer.PREDICTIONS_DB_PATH)
cur = con.cursor()


def scalar(sql, params=()):
    return cur.execute(sql, params).fetchone()[0]


out = {}

# Duplicate-join check is table-wide: INSERT OR IGNORE dedupes on the whole
# table, not per-window, so that's the scope that actually tests the guarantee.
out["predictions_total"] = scalar("SELECT COUNT(*) FROM predictions")
out["predictions_distinct_feature_id"] = scalar(
    "SELECT COUNT(DISTINCT feature_id) FROM predictions"
)
out["outcomes_total"] = scalar("SELECT COUNT(*) FROM outcomes")
out["outcomes_distinct_feature_id"] = scalar(
    "SELECT COUNT(DISTINCT feature_id) FROM outcomes"
)

# Lineage accounting from the same cutoff the performance endpoint anchored
# on, but WITHOUT its stream_id filter -- this is what lets us verify the
# active-segment scoping actually excludes other lineages, rather than
# assuming the WHERE clause behaves. When stream_id is None (predictions
# empty, or the active segment IS the pre-migration legacy bucket -- see
# _current_stream_segment's docstring), "active" and "null" are the SAME
# bucket, not two: `stream_id IS ?` with a NULL param is identical to
# `stream_id IS NULL`. Reported separately anyway (main() accounts for the
# overlap) so the raw counts stay inspectable either way.
out["predictions_total_from_cutoff"] = scalar(
    "SELECT COUNT(*) FROM predictions WHERE feature_ts >= ?", (from_ts,)
)
out["predictions_active_stream"] = scalar(
    "SELECT COUNT(*) FROM predictions WHERE feature_ts >= ? AND stream_id IS ?",
    (from_ts, stream_id),
)
out["predictions_null_stream"] = scalar(
    "SELECT COUNT(*) FROM predictions WHERE feature_ts >= ? AND stream_id IS NULL",
    (from_ts,),
)
out["predictions_other_stream"] = scalar(
    "SELECT COUNT(*) FROM predictions WHERE feature_ts >= ? "
    "AND stream_id IS NOT NULL AND stream_id IS NOT ?",
    (from_ts, stream_id),
)
out["distinct_other_stream_ids"] = scalar(
    "SELECT COUNT(DISTINCT stream_id) FROM predictions WHERE feature_ts >= ? "
    "AND stream_id IS NOT NULL AND stream_id IS NOT ?",
    (from_ts, stream_id),
)
# Null-safe sort: ingest_mode is a nullable migrated column (old rows predate
# it), and plain sorted() raises TypeError comparing None to str the moment a
# window mixes a legacy NULL row with a post-migration one -- realistic
# whenever ticks.raw retains any backlog produced before ingest_mode-stamping
# was deployed (Kafka retention is documented at-least-once).
ingest_modes = [
    r[0]
    for r in cur.execute(
        "SELECT DISTINCT ingest_mode FROM predictions WHERE feature_ts >= ? AND stream_id IS ?",
        (from_ts, stream_id),
    ).fetchall()
]
out["distinct_ingest_modes"] = sorted(ingest_modes, key=lambda v: (v is None, v or ""))

# Lead-time distribution for graded rows in the active segment -- same
# api_ts < written_at rule as evaluation.py's _split_graded, so this
# reproduces the "graded" bucket independently rather than trusting the
# median the API already reports. typeof(...) mirrors _split_graded's
# isinstance(score, (int, float)) check: a matched-but-malformed-score row is
# excluded from real grading and must be excluded here too, or this count
# silently inflates past what n_graded actually is.
rows = cur.execute(
    """
    SELECT p.api_ts, o.written_at
    FROM predictions p JOIN outcomes o ON p.feature_id = o.feature_id
    WHERE p.feature_ts >= ? AND p.stream_id IS ? AND typeof(p.score) IN ('integer', 'real')
    """,
    (from_ts, stream_id),
).fetchall()


def parse(v):
    return datetime.fromisoformat(v.replace("Z", "+00:00"))


leads = []
for api_ts, written_at in rows:
    if api_ts is None or written_at is None:
        continue
    a, w = parse(api_ts), parse(written_at)
    if a < w:
        leads.append((w - a).total_seconds())
leads.sort()


def pct(p):
    if not leads:
        return None
    idx = min(len(leads) - 1, max(0, round(p / 100 * (len(leads) - 1))))
    return leads[idx]


out["lead_seconds"] = {
    "n": len(leads),
    "min": leads[0] if leads else None,
    "p1": pct(1),
    "p50": pct(50),
    "p99": pct(99),
    "max": leads[-1] if leads else None,
}

# Loop detection: a fixture replayed on a short loop (the incident this
# guards against -- a 10-minute smoke sample replayed 51x as if it were
# hours of market data) repeats the same price sequence verbatim every
# cycle. Prevalence alone doesn't catch this: a fixture with a couple of
# real spikes can pass a ">0" check while still being a tiny loop.
# Hash market_price in fixed 30s buckets (by feature_ts) within the active
# lineage; two non-adjacent buckets sharing a hash means the same tick
# sequence played out twice, which real, continuously-evolving market data
# essentially never does at this resolution.
import hashlib
from collections import defaultdict

price_rows = cur.execute(
    "SELECT feature_ts, market_price FROM predictions "
    "WHERE feature_ts >= ? AND stream_id IS ? AND market_price IS NOT NULL "
    "ORDER BY feature_ts",
    (from_ts, stream_id),
).fetchall()

BUCKET_SECONDS = 30
buckets = defaultdict(list)
t_start = None
for feature_ts, market_price in price_rows:
    ts = parse(feature_ts).timestamp()
    if t_start is None:
        t_start = ts
    bucket_idx = int((ts - t_start) // BUCKET_SECONDS)
    # Round to damp float noise between cycles without hiding a genuine repeat.
    buckets[bucket_idx].append(round(float(market_price), 2))

hash_to_buckets = defaultdict(list)
for bucket_idx, prices in buckets.items():
    if len(prices) < 5:
        continue  # too short to be a meaningful fingerprint
    digest = hashlib.sha256(json.dumps(prices).encode()).hexdigest()
    hash_to_buckets[digest].append(bucket_idx)


def _has_nonadjacent_repeat(idxs):
    # A single contiguous run of adjacent buckets (e.g. a flat/quiet spell
    # where price didn't move for a minute or two) is expected and not a
    # loop signal on its own -- only a pair that ISN'T adjacent means the
    # same tick sequence played out twice. Contiguous <=> max-min == len-1.
    idxs = sorted(idxs)
    return idxs[-1] - idxs[0] != len(idxs) - 1


repeated = {
    digest: sorted(idxs)
    for digest, idxs in hash_to_buckets.items()
    if len(idxs) > 1 and _has_nonadjacent_repeat(idxs)
}
out["loop_detection"] = {
    "n_buckets": len(buckets),
    "bucket_seconds": BUCKET_SECONDS,
    "repeated_bucket_groups": list(repeated.values()),
}

print(json.dumps(out))
'''


class DiagnosticsUnavailable(RuntimeError):
    pass


def _get_json(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DiagnosticsUnavailable(f"GET {url} failed: {exc}") from exc


def _sqlite_diagnostics(from_ts: str, stream_id: str | None) -> dict:
    cmd = [
        "docker", "compose", "exec", "-T", "materializer",
        "python3", "-c", SQLITE_SCRIPT, from_ts, json.dumps(stream_id),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise DiagnosticsUnavailable(
            f"docker compose exec materializer sqlite query failed: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _running_services() -> set[str]:
    result = subprocess.run(
        ["docker", "compose", "ps", "--status=running", "--format", "json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise DiagnosticsUnavailable(f"docker compose ps failed: {result.stderr.strip()}")
    # One JSON object per line (docker compose ps's --format json), not a
    # single JSON array -- differs from `docker inspect`'s convention.
    names = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(json.loads(line)["Service"])
    return names


def check(lines: list[str], failures: list[str], label: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    lines.append(f"[{status}] {label}: {detail}")
    if not condition:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-minutes", type=int, default=120,
        help="window passed to /predictions/performance (max the endpoint allows; default 120)",
    )
    args = parser.parse_args()

    lines: list[str] = []
    failures: list[str] = []

    try:
        perf = _get_json(
            f"{MATERIALIZER_URL}/predictions/performance?window_minutes={args.window_minutes}"
        )
        version = _get_json(f"{API_URL}/version")
        services = _running_services()
    except DiagnosticsUnavailable as exc:
        print(f"Cannot run diagnostics: {exc}", file=sys.stderr)
        print("Is the stack up? `docker compose up -d`", file=sys.stderr)
        return 1

    window = perf["window"]

    # --- accounting -----------------------------------------------------
    check(
        lines, failures, "join accounting balances",
        window["n_joined"] == window["n_graded"] + window["n_scored_late"] + window["n_predictions_unmatched"],
        f"n_joined={window['n_joined']} == n_graded={window['n_graded']} + "
        f"n_scored_late={window['n_scored_late']} + n_predictions_unmatched={window['n_predictions_unmatched']}",
    )
    check(
        lines, failures, "no predictions scored late",
        window["n_scored_late"] == 0,
        f"n_scored_late={window['n_scored_late']}",
    )
    check(
        lines, failures, "labels present, not just predictions",
        window["n_joined"] > 0 and window["n_graded"] > 0,
        f"n_joined={window['n_joined']} n_graded={window['n_graded']}",
    )

    # --- lineage ----------------------------------------------------------
    check(
        lines, failures, "active lineage selected",
        window["stream_id"] is not None,
        f"stream_id={window['stream_id']!r}",
    )
    check(
        lines, failures, "no unknown-lineage rows silently dropped",
        window["n_unknown_lineage"] == 0,
        f"n_unknown_lineage={window['n_unknown_lineage']}",
    )

    sqlite_diag = None
    try:
        sqlite_diag = _sqlite_diagnostics(window["from_feature_ts"], window["stream_id"])
    except DiagnosticsUnavailable as exc:
        check(lines, failures, "raw SQLite diagnostics reachable", False, str(exc))

    if sqlite_diag is not None:
        # This is the check that actually validates the diagnostic's SQL
        # reproduces the live endpoint's real population -- comparing the
        # bucket sum against itself (below) can't do that, since the buckets
        # are partitions of that same sum by construction and would agree
        # even if the window were wrong.
        check(
            lines, failures, "SQLite active-stream count matches the live endpoint's n_joined (live-drift tolerant)",
            sqlite_diag["predictions_active_stream"] >= window["n_joined"],
            f"sqlite={sqlite_diag['predictions_active_stream']} >= http_n_joined={window['n_joined']} "
            "(sqlite ran a moment after the HTTP call on a live stream, so sqlite catching a few more "
            "rows is expected drift; sqlite < http would mean the diagnostic's window doesn't reproduce "
            "the endpoint's real population)",
        )

        # active/null collapse into the SAME bucket when stream_id is None
        # (the active segment IS the pre-migration legacy bucket -- see
        # _current_stream_segment's docstring) -- summing both then would
        # double-count those rows against predictions_total_from_cutoff.
        if window["stream_id"] is None:
            expected_total = sqlite_diag["predictions_active_stream"] + sqlite_diag["predictions_other_stream"]
            bucket_detail = (
                f"total={sqlite_diag['predictions_total_from_cutoff']} == active/null="
                f"{sqlite_diag['predictions_active_stream']} (same bucket, no active lineage) + other="
                f"{sqlite_diag['predictions_other_stream']}"
            )
        else:
            expected_total = (
                sqlite_diag["predictions_active_stream"]
                + sqlite_diag["predictions_null_stream"]
                + sqlite_diag["predictions_other_stream"]
            )
            bucket_detail = (
                f"total={sqlite_diag['predictions_total_from_cutoff']} == active="
                f"{sqlite_diag['predictions_active_stream']} + null={sqlite_diag['predictions_null_stream']} "
                f"+ other={sqlite_diag['predictions_other_stream']}"
            )
        check(
            lines, failures, "lineage buckets partition the cutoff population with no overlap or gap",
            sqlite_diag["predictions_total_from_cutoff"] == expected_total,
            bucket_detail
            + (
                f" (other lineages present: {sqlite_diag['distinct_other_stream_ids']} -- "
                "expected under replay, since every rerun shares feature_ts with prior runs; "
                "the check is that the active segment excludes them, not that they're absent)"
                if sqlite_diag["distinct_other_stream_ids"]
                else ""
            ),
        )
        check(
            lines, failures, "no replay/live interleave within the active segment",
            len(sqlite_diag["distinct_ingest_modes"]) <= 1,
            f"ingest_mode values in window={sqlite_diag['distinct_ingest_modes']}",
        )

        # --- duplicates -----------------------------------------------
        check(
            lines, failures, "predictions.feature_id unique (no duplicate joins)",
            sqlite_diag["predictions_total"] == sqlite_diag["predictions_distinct_feature_id"],
            f"total={sqlite_diag['predictions_total']} distinct={sqlite_diag['predictions_distinct_feature_id']}",
        )
        check(
            lines, failures, "outcomes.feature_id unique (no duplicate joins)",
            sqlite_diag["outcomes_total"] == sqlite_diag["outcomes_distinct_feature_id"],
            f"total={sqlite_diag['outcomes_total']} distinct={sqlite_diag['outcomes_distinct_feature_id']}",
        )

        # --- lead-time distribution -------------------------------------
        # The SQL query runs a moment after the HTTP snapshot above, and this
        # is a live, continuously-replaying stream -- so lead["n"] catching a
        # few more graded rows than window["n_graded"] is expected drift, not
        # a bug. What would be a bug: fewer rows (grading isn't monotonic /
        # something regressed between the two reads) or a negative lead.
        lead = sqlite_diag["lead_seconds"]
        check(
            lines, failures, "no negative lead time (nothing graded before it was scored)",
            lead["n"] > 0 and lead["min"] is not None and lead["min"] >= 0,
            f"n={lead['n']} min={lead['min']} p1={lead['p1']} p50={lead['p50']} p99={lead['p99']} max={lead['max']}",
        )
        check(
            lines, failures, "graded count monotonic vs. the earlier HTTP snapshot (live-drift tolerant)",
            lead["n"] >= window["n_graded"],
            f"sqlite n={lead['n']} >= http-snapshot n_graded={window['n_graded']} "
            "(sqlite query ran after the HTTP call, so >= is expected on a live replaying stream)",
        )
        check(
            lines, failures, "graded lead times cluster near the 60s label horizon (no bimodal split)",
            window["median_lead_seconds"] is not None
            and lead["p1"] is not None
            and lead["p99"] is not None
            and lead["p1"] > 0
            and lead["p99"] < 120,
            f"p1={lead['p1']} p50={lead['p50']} p99={lead['p99']} reported_median={window['median_lead_seconds']} "
            "(label horizon is 60s; a sane distribution should sit close to it, not spread toward 0 or far past it)",
        )

        # --- loop detection ---------------------------------------------
        # This is the check that would have caught the original incident
        # directly: a 10-minute fixture replayed 51x has zero positive
        # labels, but a *different* broken fixture with a couple of real
        # spikes would still pass a bare "prevalence > 0" check. Hashing
        # fixed-duration price buckets catches the actual pathology --
        # the same tick sequence recurring -- regardless of label balance.
        loop = sqlite_diag["loop_detection"]
        check(
            lines, failures, "no repeated price sequence across time buckets (replay loop detector)",
            len(loop["repeated_bucket_groups"]) == 0,
            f"n_buckets={loop['n_buckets']} bucket_seconds={loop['bucket_seconds']} "
            f"repeated_groups={loop['repeated_bucket_groups']} "
            "(non-adjacent buckets sharing an identical price sequence means the same tick "
            "window played out twice -- real market data doesn't repeat verbatim)",
        )

    # --- no replay/live interleave at the container level ------------------
    check(
        lines, failures, "ingestor and ws-ingestor are not both running",
        not ({"ingestor", "ws-ingestor"} <= services),
        f"running services matching ingest: {sorted(services & {'ingestor', 'ws-ingestor'})}",
    )

    # --- feature schema parity ---------------------------------------------
    run_id = version.get("run_id")
    if run_id is None:
        check(lines, failures, "serving model has a resolvable MLflow run_id", False, f"/version={version}")
    else:
        try:
            run = _get_json(f"{MLFLOW_URL}/api/2.0/mlflow/runs/get?run_id={run_id}")["run"]
        except DiagnosticsUnavailable as exc:
            check(lines, failures, "MLflow run reachable", False, str(exc))
            run = None

        if run is not None:
            params = {p["key"]: p["value"] for p in run["data"]["params"]}
            served_cols = params.get("feature_cols", "").split(",") if "feature_cols" in params else []
            check(
                lines, failures, "training feature_cols match the shipped feature spec",
                served_cols == CANONICAL_FEATURE_COLS,
                f"mlflow run {run_id} feature_cols={served_cols} "
                f"expected={CANONICAL_FEATURE_COLS} (handoff/docs/feature_spec.md, Variant B)",
            )

            legacy_selection = (version.get("model"), version.get("stage")) == ("btc-volatility-lr", "Production")
            if not legacy_selection:
                check(
                    lines, failures, "non-legacy model carries feature_set_id/feature_schema_version contract",
                    "feature_set_id" in params and "feature_schema_version" in params,
                    f"model={version.get('model')}/{version.get('stage')} params_present={sorted(params)}",
                )

    print("\n".join(lines))
    print()
    if failures:
        print(f"FAIL: {len(failures)}/{len(lines)} invariant(s) failed: {failures}")
        return 1
    print(f"PASS: all {len(lines)} invariants held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

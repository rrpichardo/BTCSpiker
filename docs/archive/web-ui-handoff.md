# Claude Code Handoff: BTCSpiker Web UI v2

## Start here

```bash
cd /Users/ricopichardo/Claude/BTCSpiker/.claude/worktrees/web-ui
git status --short --branch
git log -5 --oneline
docker compose ps
```

Work only in this existing worktree. Do not create another worktree and do not restart the implementation from the original base.

- Worktree: `/Users/ricopichardo/Claude/BTCSpiker/.claude/worktrees/web-ui`
- Branch: `claude/web-ui`
- Base commit: `d565e33` (`codex/v1`)
- Current HEAD: `121b1327964ca109a1f22d9f3fabfba1f9890962`
- Original plan: `/Users/ricopichardo/.claude/plans/delete-all-trace-of-tranquil-galaxy.md`
- Durable task ledger: `.superpowers/sdd/progress.md`
- Final-review findings: `.superpowers/sdd/final-review-findings.md`
- Final-fix evidence: `.superpowers/sdd/final-fix-report.md`

`CLAUDE_CODE_HANDOFF.md` is intentionally untracked. Preserve it during the handoff; do not include it in a product commit unless the user asks.

## Goal

Finish and verify BTCSpiker Web UI v2:

- Kafka-native `ticks.predictions` events from the prediction bridge.
- Kafka-to-SQLite materializer with duplicate safety and rebuildable projection semantics.
- Read-only Settings and System APIs.
- React/Vite/nginx UI on `http://localhost:3001` with Predictions, Settings, and System tabs.
- Full Compose, CI, README, and runbook integration.

## Completed and committed

Tasks 1-5 are implemented and individually reviewed. Important commits after the base:

```text
5cf87c8 Publish PredictionEvent to ticks.predictions from predict-bridge
8ab9635 Add web UI: Vite + React SPA with Predictions/Settings/System tabs
9dac97d Add materializer service: Kafka predictions -> SQLite read model + FastAPI
eb9fbe0 Add read-only settings and system APIs
737b4b6 Fix prediction bridge retry ordering
f288aaa Test prediction event publish contract
360a402 Harden settings and system status APIs
71b355a Fix materializer durability and recovery
69d0ed2 Fix web UI polling and production states
ff9da2f Harden materializer recovery commits
1a954a9 Treat missing prediction writes as stale
e2dc46e Integrate web stack compose CI and docs
939ec6f Fix UI healthcheck IPv4 target
9b0efc8 Harden runtime health and CI gates
e263aec Fix broker health detection and recovery via fresh-client probe
a5ea4ba Move broker probe to dedicated thread so blocked commits can't starve it
f518673 Document post-outage worker-stall failure mode in runbook
121b132 Re-resolve UI proxy upstreams per request; pad probe-round deadline
```

The product working tree is clean at `121b132`; this handoff file is the only
untracked file.

## Fresh verification already completed

Before `9b0efc8`:

- 95 non-integration Python tests passed.
- Black and Ruff passed for the original CI scope.
- UI production build passed.
- Full Compose build/start passed.
- Replay integration passed.
- Predictions flowed through `http://localhost:3001/api/predictions/recent`.
- Materializer restart preserved more than 60,000 rows.
- Stopping materializer produced a UI-proxy 502 and System reported materializer down.
- Saved `MODEL_VARIANT=baseline` showed `restart_required`; applying it produced binary baseline scores; `.env` and the API were restored to `MODEL_VARIANT=ml`.
- Embedded-browser acceptance was later completed. The chart updates live,
  degraded and recovered service states render correctly, Settings saved-vs-active
  behavior was exercised, mobile layout and focus-visible behavior were checked,
  and browser console logs were clean. Clipboard access was blocked by the
  embedded pane's permissions, while the UI degraded gracefully.

At `9b0efc8`:

- `python -m pytest tests --ignore=tests/test_replay_integration.py -q` -> `98 passed`.
- `python -m black --check api materializer scripts/feature_to_predict_bridge.py tests` -> passed after formatting.
- `python -m ruff check api materializer scripts/feature_to_predict_bridge.py tests` -> passed.
- `npm test` in `ui/` -> 3 passed.
- `npm run build` in `ui/` -> passed; existing Recharts chunk-size warning remains.
- `docker compose config --quiet` -> passed.
- Full images were rebuilt from this commit.
- `python -m pytest tests/test_replay_integration.py -v --tb=short` -> passed in 6.52s.
- UI/API/materializer were healthy and predictions flowed before the Kafka outage probe.

After `9b0efc8`:

- `e263aec` replaced the cached long-lived-consumer metadata check with a fresh
  Kafka client probe and added deterministic recovery tests.
- `a5ea4ba` moved the probe to its own thread so a blocked consumer commit cannot
  starve broker-health updates.
- Live verification observed materializer `ok: false` about 6 seconds after
  Kafka stopped and `ok: true` about 9 seconds after Kafka restarted, without
  restarting the materializer.
- `121b132` fixed nginx's stale Docker-upstream DNS caching and padded the system
  probe-round deadline. A final whole-branch review reported no remaining
  Critical or Important findings and marked the branch ready to merge.
- Final recorded verification: 101 non-integration Python tests, replay
  integration, three UI tests, Black, Ruff, production build, live E2E, and
  embedded-browser acceptance passed.

## Resolved blocker and remaining operational caveat

The final review identified that materializer health/supervision did not reliably reflect Kafka connectivity. Commit `9b0efc8` attempted to fix it with:

- `ConsumerState.broker_ok`
- periodic `consumer.list_topics(...)`
- post-readiness `supervise_consumer(...)`
- Compose healthcheck JSON validation (`ok` must be `true`)
- focused unit tests

The original live outage test proved the `9b0efc8` implementation was incomplete:

1. Ran `docker compose stop kafka`.
2. For at least 15 seconds, `GET http://127.0.0.1:8090/health` continued returning `ok: true` even though librdkafka logged connection-refused and DNS failures.
3. A fresh consumer created inside the materializer container failed `list_topics(timeout=1)` with `KafkaError{code=_TRANSPORT}`. The long-lived materializer consumer's periodic `list_topics` did not promptly mark health false, likely because it can return cached metadata.
4. Materializer eventually changed to `ok: false` after Kafka errors were surfaced through polling; `consume_errors` reached 133.
5. After `docker compose start kafka` and Kafka became healthy, materializer stayed `ok: false` for at least another 15 seconds. Health was sticky-false and did not recover automatically.
6. `docker compose restart materializer` restored `ok: true` immediately.

Commits `e263aec` and `a5ea4ba` resolved this materializer-specific blocker. On
2026-07-19 the lifecycle was rechecked live: Kafka was initially stopped,
materializer correctly reported `ok: false`, Kafka was started, and materializer
returned to `ok: true` and Docker `healthy` without a materializer restart.

A separate pre-existing operational caveat remains documented in
`docs/runbook.md`: after a Kafka outage, the long-running `ingestor`,
`featurizer`, and `predict-bridge` Kafka clients can stay wedged while their
processes remain up. Restart those three workers after broker recovery. On
2026-07-19, restarting only those workers restored fresh predictions; the
materializer remained running throughout and the browser returned from
`Degraded` to `Live`.

## If extending broker recovery further

The branch is already reviewed and ready to integrate. Only continue here if the
user explicitly expands scope to make the three upstream workers self-heal too.
If so, follow systematic debugging and TDD; do not simply lengthen timeouts.

1. Add a deterministic failing test for both transitions:
   - healthy -> Kafka unavailable -> materializer `ok: false`
   - Kafka restored -> materializer `ok: true` and consumption resumes without restarting the process
2. Avoid using the long-lived consumer's cached metadata as the sole liveness probe.
3. Evaluate one of these evidence-based approaches:
   - a fresh, short-lived Kafka/Admin metadata probe per health interval;
   - a configured librdkafka `error_cb` that updates broker state on transport/all-brokers-down events, combined with an explicit successful metadata/poll signal that clears the failure;
   - a supervised consumer recreation after a bounded consecutive-failure threshold.
4. Keep offset/durability guarantees intact. Never commit a later offset past a failed batch.
5. Ensure shutdown does not race the supervisor or leak consumers.
6. Keep the Compose healthcheck body validation added in `9b0efc8`.

## Verification to refresh before integration

Run all of these from the worktree:

```bash
python -m pytest tests --ignore=tests/test_replay_integration.py -q
python -m black --check api materializer scripts/feature_to_predict_bridge.py tests
python -m ruff check api materializer scripts/feature_to_predict_bridge.py tests
cd ui && npm test && npm run build && cd ..
docker compose config --quiet
docker compose up -d --build
python -m pytest tests/test_replay_integration.py -v --tb=short
```

If broker lifecycle code changes again, prove it live:

```bash
curl -sf http://127.0.0.1:8090/health | jq
docker compose stop kafka
# Within the designed bounded interval, health must return ok:false and the
# materializer Docker healthcheck must fail.
docker compose start kafka
# Without restarting materializer, health must return ok:true again.
# Restart ingestor/featurizer/predict-bridge if the documented upstream-client
# stall occurs, then confirm fresh rows resume through the UI proxy.
curl -sf http://127.0.0.1:3001/api/predictions/recent | jq '.count'
```

Current embedded-browser acceptance at `http://localhost:3001`:

- Predictions chart rendered with 500 points and updated while polling.
- The prediction table rendered the newest 50 events.
- The degraded banner appeared while materializer health was false and cleared
  to `Live` after broker/worker recovery.
- System identified materializer as the only failed service, then recovered.
- Settings displayed saved and active values plus apply commands.
- Desktop and 390x844 mobile layouts rendered acceptably; the wide event table
  scrolls horizontally on mobile.
- No browser console warnings or errors were observed.
- Clipboard permission prevented a reliable embedded-pane copy-button check;
  recheck the `Copied` feedback in a normal browser if desired.

## Runtime and repository notes

- The `web-ui` Compose stack is running. Kafka and materializer were healthy at
  the end of the 2026-07-19 live check, the three upstream workers were restarted
  per the runbook, and the Predictions view was `Live` with fresh scores.
- The older container `real-time-crypto-ml-main-api-1` was stopped to release host port 8000. Do not start it while `web-ui-api-1` owns port 8000.
- `.env` is restored to `MODEL_VARIANT=ml`.
- Branch is local, not merged or pushed by this handoff.
- Do not delete `.claude/worktrees/prediction-ui-settings-plan-e14f9f`; it was previously identified as stale, but cleanup was not authorized.

## Completion protocol

The broker lifecycle, final review, and browser acceptance are complete. Claude
Code should:

1. Read this file and `.superpowers/sdd/progress.md` completely.
2. Confirm `git status`, `HEAD=121b132`, and the live Compose state have not
   drifted.
3. Refresh the verification commands above if integration is requested.
4. Use the finishing-a-development-branch workflow and ask the user whether to
   merge, push/open a PR, keep the branch, or discard it.
5. Do not expand the upstream-worker self-healing scope without explicit user
   authorization.

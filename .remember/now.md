
## 15:13 | main
Added market_price field through bridge→materializer, built /timeline endpoint w/ 6-state outcome classification, replaced ScoreChart w/ TimelineCharts dual-chart + range-selector (1m–24h), fixture & live verified.
## 15:19 | main
Codex review found 2 P2 bugs (resolution bounds validation, horizon-boundary classification); verified findings; fixing timeline.py/materializer.py.
## 15:43 | main
Fixed both P2 bugs: materializer.py (resolution bounds validation), timeline.py (horizon-boundary classification); added regression tests; all 403 tests pass.
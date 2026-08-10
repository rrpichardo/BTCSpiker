
## 22:22 | fix/real-neural-stage-and-thread-cap
Cherry-picked real neural + thread-cap fixes to PR #6, fixed conflicts and pinned torch/threadpoolctl — but review found neural crashes in calibration wrapper (missing ClassifierMixin) and thread-cap races (process-global BLAS).
"""
Load test: fire 100 concurrent requests at /predict and report latency stats.

Usage:
    docker compose up -d
    python tests/load_test.py
"""

import concurrent.futures
import time

import requests

BASE_URL = "http://localhost:8000"

SAMPLE_ROW = {
    "log_return": 0.0001,
    "spread_bps": 1.5,
    "vol_60s": 0.00005,
    "mean_return_60s": 0.0,
    "trade_intensity_60s": 10.0,
    "n_ticks_60s": 50,
    "spread_mean_60s": 1.2,
}

N_REQUESTS = 100


def send_request(_):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/predict", json={"rows": [SAMPLE_ROW]})
    dt = time.perf_counter() - t0
    return r.status_code, dt


if __name__ == "__main__":
    print(f"Sending {N_REQUESTS} concurrent requests to {BASE_URL}/predict ...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_REQUESTS) as pool:
        results = list(pool.map(send_request, range(N_REQUESTS)))

    codes = [r[0] for r in results]
    latencies = sorted([r[1] for r in results])

    ok = codes.count(200)
    fail = N_REQUESTS - ok

    p50 = latencies[int(N_REQUESTS * 0.50)] * 1000
    p95 = latencies[int(N_REQUESTS * 0.95)] * 1000
    p99 = latencies[int(N_REQUESTS * 0.99)] * 1000
    max_ms = latencies[-1] * 1000

    print(f"Succeeded:    {ok}/{N_REQUESTS}")
    print(f"Failed:       {fail}/{N_REQUESTS}")
    print(f"Latency p50:  {p50:.1f} ms")
    print(f"Latency p95:  {p95:.1f} ms")
    print(f"Latency p99:  {p99:.1f} ms")
    print(f"Latency max:  {max_ms:.1f} ms")
    print(f"Target:       p95 <= 800 ms  {'PASS' if p95 <= 800 else 'FAIL'}")

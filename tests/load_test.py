"""Load-test the prediction endpoint with configurable request pressure."""

import argparse
import concurrent.futures
import math
import time

import requests

SAMPLE_ROW = {
    "log_return": 0.0001,
    "spread_bps": 1.5,
    "vol_60s": 0.00005,
    "mean_return_60s": 0.0,
    "trade_intensity_60s": 10.0,
    "n_ticks_60s": 50,
    "spread_mean_60s": 1.2,
}


def send_request(url: str, _index: int) -> tuple[int, float]:
    t0 = time.perf_counter()
    r = requests.post(f"{url}/predict", json={"rows": [SAMPLE_ROW]}, timeout=10)
    dt = time.perf_counter() - t0
    return r.status_code, dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    base_url = args.url.rstrip("/")
    print(
        f"Sending {args.requests} requests to {base_url}/predict at concurrency {args.concurrency} ...\n"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(lambda index: send_request(base_url, index), range(args.requests))
        )

    codes = [r[0] for r in results]
    latencies = sorted([r[1] for r in results])

    ok = codes.count(200)
    fail = args.requests - ok

    p50 = latencies[math.ceil(args.requests * 0.50) - 1] * 1000
    p95 = latencies[math.ceil(args.requests * 0.95) - 1] * 1000
    p99 = latencies[math.ceil(args.requests * 0.99) - 1] * 1000
    max_ms = latencies[-1] * 1000

    print(f"Succeeded:    {ok}/{args.requests}")
    print(f"Failed:       {fail}/{args.requests}")
    print(f"Latency p50:  {p50:.1f} ms")
    print(f"Latency p95:  {p95:.1f} ms")
    print(f"Latency p99:  {p99:.1f} ms")
    print(f"Latency max:  {max_ms:.1f} ms")
    passed = ok == args.requests and p95 <= 800
    print(f"Target:       p95 <= 800 ms  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

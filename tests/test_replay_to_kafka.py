import json

from scripts import replay_to_kafka


class FakeProducer:
    def __init__(self):
        self.produced = []

    def produce(self, topic, key=None, value=None, callback=None):
        self.produced.append((topic, key, value))
        if callback:
            callback(None, None)

    def poll(self, timeout):
        return 0

    def flush(self, timeout):
        return 0


def _write_ndjson(path, ticks):
    path.write_text("\n".join(json.dumps(t) for t in ticks) + "\n")


def _tick(ts, price="69700.0", bid="69699.9", ask="69700.1"):
    return {
        "product_id": "BTC-USD",
        "price": price,
        "best_bid": bid,
        "best_ask": ask,
        "volume_24_h": "1000.0",
        "timestamp": ts,
    }


def test_exact_duplicate_lines_are_not_republished(tmp_path, monkeypatch):
    fake = FakeProducer()
    monkeypatch.setattr(replay_to_kafka, "make_producer", lambda: fake)
    monkeypatch.setattr(replay_to_kafka, "wait_for_kafka", lambda producer: None)

    ticks = [
        _tick("2026-04-06T15:02:34.000000Z"),
        _tick("2026-04-06T15:02:34.000000Z"),  # exact duplicate
        _tick("2026-04-06T15:02:35.000000Z"),
    ]
    path = tmp_path / "ticks.ndjson"
    _write_ndjson(path, ticks)

    total = replay_to_kafka.replay(path, speed=1000.0, loop=False, stop={"flag": False})

    assert total == 2
    published = [json.loads(v) for _, _, v in fake.produced]
    assert [t["timestamp"] for t in published] == [
        "2026-04-06T15:02:34.000000Z",
        "2026-04-06T15:02:35.000000Z",
    ]


def test_same_timestamp_different_price_is_not_dropped(tmp_path, monkeypatch):
    fake = FakeProducer()
    monkeypatch.setattr(replay_to_kafka, "make_producer", lambda: fake)
    monkeypatch.setattr(replay_to_kafka, "wait_for_kafka", lambda producer: None)

    ticks = [
        _tick("2026-04-06T15:02:34.000000Z", price="69700.0"),
        _tick("2026-04-06T15:02:34.000000Z", price="69701.0"),  # same ts, real move
    ]
    path = tmp_path / "ticks.ndjson"
    _write_ndjson(path, ticks)

    total = replay_to_kafka.replay(path, speed=1000.0, loop=False, stop={"flag": False})

    assert total == 2
    published = [json.loads(v) for _, _, v in fake.produced]
    assert [t["price"] for t in published] == ["69700.0", "69701.0"]

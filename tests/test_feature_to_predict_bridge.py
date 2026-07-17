import json

import pytest

from scripts import feature_to_predict_bridge as bridge


FEATURE_MESSAGE = {
    "timestamp": "2026-07-16T19:00:00Z",
    "log_return": 0.001,
    "spread_bps": 2.5,
    "vol_60s": 0.02,
    "mean_return_60s": 0.0005,
    "trade_intensity_60s": 3.0,
    "n_ticks_60s": 180,
    "spread_mean_60s": 4.0,
}

VALID_RESPONSE = {
    "scores": [0.75],
    "model_variant": "ml",
    "version": "v1.0",
    "ts": "2026-07-16T19:00:01+00:00",
}

EXPECTED_EVENT = {
    "event_id": "ticks.features:2:41",
    "source_partition": 2,
    "source_offset": 41,
    "feature_ts": "2026-07-16T19:00:00Z",
    "api_ts": "2026-07-16T19:00:01+00:00",
    "score": 0.75,
    "model_variant": "ml",
    "model_version": "v1.0",
    "vol_60s": 0.02,
    "spread_bps": 2.5,
    "log_return": 0.001,
    "trade_intensity_60s": 3.0,
}


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeMessage:
    def __init__(self, offset=41):
        self._offset = offset

    def topic(self):
        return "ticks.features"

    def partition(self):
        return 2

    def offset(self):
        return self._offset

    def value(self):
        return json.dumps(FEATURE_MESSAGE).encode("utf-8")

    def timestamp(self):
        return (1, 1784232000000)

    def error(self):
        return None


def test_prediction_event_uses_original_feature_timestamp_and_exact_contract():
    msg = FakeMessage()
    row = bridge._build_row(FEATURE_MESSAGE, msg.timestamp()[1])

    event = bridge._build_prediction_event(
        msg,
        row,
        VALID_RESPONSE,
        feature_ts=FEATURE_MESSAGE["timestamp"],
    )

    assert event == EXPECTED_EVENT


def test_publish_prediction_uses_event_id_key_and_exact_payload():
    class FakeProducer:
        def __init__(self):
            self.produced = []

        def produce(self, topic, *, key, value, callback):
            self.produced.append((topic, key, value))
            callback(None, object())

        def poll(self, timeout):
            assert timeout == 0

        def flush(self, timeout):
            assert timeout == bridge.API_TIMEOUT
            return 0

    producer = FakeProducer()

    assert bridge._publish_prediction(producer, EXPECTED_EVENT) is True
    assert producer.produced == [
        (
            "ticks.predictions",
            "ticks.features:2:41",
            json.dumps(EXPECTED_EVENT).encode("utf-8"),
        )
    ]


def test_post_prediction_accepts_complete_single_score_response(monkeypatch):
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(VALID_RESPONSE),
    )

    success, detail, response = bridge._post_prediction(FEATURE_MESSAGE)

    assert success is True
    assert detail == ""
    assert response == VALID_RESPONSE


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"scores": [0.1, 0.2], "ts": "now", "model_variant": "ml", "version": "v1"},
        {"scores": [float("nan")], "ts": "now", "model_variant": "ml", "version": "v1"},
        {"scores": [float("inf")], "ts": "now", "model_variant": "ml", "version": "v1"},
        {"scores": [True], "ts": "now", "model_variant": "ml", "version": "v1"},
        {"scores": ["0.1"], "ts": "now", "model_variant": "ml", "version": "v1"},
        {"scores": [0.1], "ts": "", "model_variant": "ml", "version": "v1"},
        {"scores": [0.1], "ts": "   ", "model_variant": "ml", "version": "v1"},
        {"scores": [0.1], "ts": "now", "model_variant": "", "version": "v1"},
        {"scores": [0.1], "ts": "now", "model_variant": "ml", "version": ""},
        {"scores": [0.1], "ts": "now", "model_variant": 1, "version": "v1"},
        {"scores": [0.1], "ts": "now", "model_variant": "ml"},
    ],
)
def test_post_prediction_rejects_malformed_success_body(monkeypatch, payload):
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload),
    )

    success, detail, response = bridge._post_prediction(FEATURE_MESSAGE)

    assert success is False
    assert detail.startswith("malformed API response:")
    assert response is None


def test_main_retries_unconfirmed_publish_before_polling_next_record(monkeypatch):
    class FakeConsumer:
        instance = None

        def __init__(self, _config):
            self.poll_count = 0
            self.committed = []
            self.messages = [FakeMessage(offset=41), FakeMessage(offset=42)]
            FakeConsumer.instance = self

        def subscribe(self, _topics):
            pass

        def poll(self, timeout):
            assert timeout == 1.0
            self.poll_count += 1
            if self.poll_count <= len(self.messages):
                return self.messages[self.poll_count - 1]
            raise KeyboardInterrupt

        def commit(self, message):
            self.committed.append(message)

        def close(self):
            pass

    class FakeProducer:
        def __init__(self, _config):
            pass

        def flush(self, _timeout):
            return 0

    publish_results = iter([False, True, True])
    published_events = []

    def fake_publish(_producer, event):
        published_events.append(event)
        return next(publish_results)

    monkeypatch.setattr(bridge, "Consumer", FakeConsumer)
    monkeypatch.setattr(bridge, "Producer", FakeProducer)
    monkeypatch.setattr(bridge, "_wait_for_kafka", lambda *_args: None)
    monkeypatch.setattr(bridge, "_wait_for_api", lambda *_args: None)
    monkeypatch.setattr(
        bridge,
        "_post_prediction",
        lambda _row: (True, "", VALID_RESPONSE),
    )
    monkeypatch.setattr(bridge, "_publish_prediction", fake_publish)
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bridge.signal, "signal", lambda *_args: None)

    with pytest.raises(KeyboardInterrupt):
        bridge.main()

    consumer = FakeConsumer.instance
    assert consumer.poll_count == 3
    assert consumer.committed == consumer.messages
    assert len(published_events) == 3
    assert published_events[0] == published_events[1]
    assert [event["source_offset"] for event in published_events] == [41, 41, 42]


def test_main_retries_failed_post_before_polling_next_record(monkeypatch):
    class FakeConsumer:
        instance = None

        def __init__(self, _config):
            self.poll_count = 0
            self.committed = []
            self.messages = [FakeMessage(offset=41), FakeMessage(offset=42)]
            FakeConsumer.instance = self

        def subscribe(self, _topics):
            pass

        def poll(self, timeout):
            assert timeout == 1.0
            self.poll_count += 1
            if self.poll_count <= len(self.messages):
                return self.messages[self.poll_count - 1]
            raise KeyboardInterrupt

        def commit(self, message):
            self.committed.append(message)

        def close(self):
            pass

    class FakeProducer:
        def __init__(self, _config):
            pass

        def flush(self, _timeout):
            return 0

    post_results = iter(
        [
            (False, "API request failed", None),
            (True, "", VALID_RESPONSE),
            (True, "", VALID_RESPONSE),
        ]
    )
    posted_rows = []

    def fake_post(row):
        posted_rows.append(row)
        return next(post_results)

    monkeypatch.setattr(bridge, "Consumer", FakeConsumer)
    monkeypatch.setattr(bridge, "Producer", FakeProducer)
    monkeypatch.setattr(bridge, "_wait_for_kafka", lambda *_args: None)
    monkeypatch.setattr(bridge, "_wait_for_api", lambda *_args: None)
    monkeypatch.setattr(bridge, "_post_prediction", fake_post)
    monkeypatch.setattr(bridge, "_publish_prediction", lambda *_args: True)
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bridge.signal, "signal", lambda *_args: None)

    with pytest.raises(KeyboardInterrupt):
        bridge.main()

    consumer = FakeConsumer.instance
    assert consumer.poll_count == 3
    assert consumer.committed == consumer.messages
    assert len(posted_rows) == 3

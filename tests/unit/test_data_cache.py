"""Offline cache integrity, concurrency, expiry and HTTP retry regression tests."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from mqf_research.data import download_raw as raw


def test_atomic_writers_use_unique_staging_files(tmp_path, monkeypatch):
    """Simultaneous writes must publish one complete payload without temp collisions."""
    target = tmp_path / "payload.bin"
    replace = Path.replace
    barrier = threading.Barrier(8)
    staging_paths = []
    first_attempts = set()
    staging_lock = threading.Lock()

    def synchronized_replace(path, destination):
        """Hold all writers at publication to force overlapping temporary files."""
        with staging_lock:
            first_attempt = path not in first_attempts
            if first_attempt:
                first_attempts.add(path)
                staging_paths.append(path)
        if first_attempt:
            barrier.wait(timeout=10)
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    payloads = [bytes([index]) * 4096 for index in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda payload: raw.atomic_write(target, payload), payloads))
    assert target.read_bytes() in payloads
    assert len(set(staging_paths)) == len(payloads)
    assert sorted(tmp_path.iterdir()) == [target]


def test_failed_atomic_replace_retains_original_and_cleans_temp(tmp_path, monkeypatch):
    """A failed publication must preserve the old destination and remove only its temp."""
    target = tmp_path / "payload.bin"
    target.write_bytes(b"old result")

    def fail_replace(path, destination):
        """Simulate a filesystem publication failure after the temporary write."""
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        raw.atomic_write(target, b"new result")
    assert target.read_bytes() == b"old result"
    assert sorted(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.name != "nt", reason="Windows-only transient handle contention")
def test_atomic_replace_retries_transient_windows_denial(tmp_path, monkeypatch):
    """A temporary Windows sharing denial retries boundedly before publishing complete bytes."""
    target = tmp_path / "payload.bin"
    original_replace = Path.replace
    attempts = []
    sleep = Mock()

    def deny_twice(path, destination):
        """Simulate two transient Windows errors followed by a successful replacement."""
        attempts.append(path)
        if len(attempts) <= 2:
            raise PermissionError("transient handle contention")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", deny_twice)
    monkeypatch.setattr(raw.time, "sleep", sleep)
    raw.atomic_write(target, b"complete bytes")
    assert target.read_bytes() == b"complete bytes"
    assert len(attempts) == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.01, 0.02]


def test_cache_rejects_missing_or_mismatched_sidecar(tmp_path):
    """A payload is reusable only when the retained byte count and hash match."""
    target = tmp_path / "payload.json"
    raw.store(target, b'{"ok":true}', "TEST", "https://example.test/data")
    assert raw.cache_bytes(target) == b'{"ok":true}'
    target.write_bytes(b'{"ok":false}')
    assert raw.cache_bytes(target) is None
    raw.meta_path(target).unlink()
    assert raw.cache_bytes(target) is None


@pytest.mark.parametrize("bad_time", [None, 5, "bad", "2026-01-01", "2026-01-01T12:00:00"])
def test_expiry_rejects_invalid_or_naive_metadata(tmp_path, bad_time):
    """Malformed/unzoned cache timestamps are stale rather than exceptions or guesses."""
    target = tmp_path / "payload.json"
    raw.meta_path(target).write_text(json.dumps({"fetched_at_utc": bad_time}), encoding="utf-8")
    assert raw.fresh(target, 30) is False


def test_expiry_uses_aware_instants_and_rejects_future_times(tmp_path, monkeypatch):
    """TTL bounds apply to absolute instants, including offset and future-clock cases."""
    now = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(raw, "utcnow", lambda: now)
    target = tmp_path / "payload.json"
    for stamp, expected in [
        ("2026-01-09T07:00:00-05:00", True),
        ("2026-01-08T12:00:00Z", False),
        ((now + timedelta(seconds=1)).isoformat(), False),
    ]:
        raw.meta_path(target).write_text(json.dumps({"fetched_at_utc": stamp}), encoding="utf-8")
        assert raw.fresh(target, 2) is expected


def test_metadata_removes_credential_fields_but_preserves_filters(tmp_path):
    """Nested params and embedded URL credentials must not leak into cache sidecars."""
    target = tmp_path / "payload.json"
    endpoint = "https://user:PRIVATE@example.test/data?api_key=PRIVATE&symbol=ABC#PRIVATE"
    params = {
        "api_token": "PRIVATE",
        "from": "2020-01-01",
        "nested": [{"Authorization": "PRIVATE", "period": "d"}],
    }
    raw.store(target, b"{}", "TEST", endpoint, params)
    text = raw.meta_path(target).read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert "PRIVATE" not in text
    assert metadata["endpoint"] == "https://example.test/data?symbol=ABC"
    assert metadata["params"] == {"from": "2020-01-01", "nested": [{"period": "d"}]}
    assert raw.cache_bytes(target) == b"{}"


def test_http_retry_is_bounded_and_honors_retry_after(monkeypatch):
    """A throttled request waits once then succeeds without real network or sleep."""
    client = raw.HTTPClient("FAKE", "offline-test", max_retries=1)
    throttled = Mock(status_code=429, headers={"Retry-After": "3"})
    success = Mock(status_code=200, content=b"[]")
    client.eod.get = Mock(side_effect=[throttled, success])
    sleep = Mock()
    monkeypatch.setattr(raw.time, "sleep", sleep)
    assert client.get("https://example.test/data") == b"[]"
    assert client.eod.get.call_count == 2
    sleep.assert_called_once_with(3.0)


def test_http_quota_failure_does_not_retry_or_expose_key(monkeypatch):
    """Authentication failure terminates immediately with a credential-free message."""
    client = raw.HTTPClient("FAKE_PRIVATE_TOKEN", "offline-test", max_retries=4)
    client.eod.get = Mock(return_value=Mock(status_code=401))
    sleep = Mock()
    monkeypatch.setattr(raw.time, "sleep", sleep)
    with pytest.raises(raw.QuotaFailure) as failure:
        client.get("https://example.test/data")
    assert "FAKE_PRIVATE_TOKEN" not in str(failure.value)
    assert client.eod.get.call_count == 1
    sleep.assert_not_called()


def test_http_failure_and_retry_logs_redact_url_credentials(monkeypatch, caplog):
    """Provider errors must not expose userinfo, credential queries or URL fragments."""
    client = raw.HTTPClient("FAKE_PRIVATE_TOKEN", "offline-test", max_retries=1)
    client.eod.get = Mock(
        side_effect=[
            Mock(status_code=429, headers={"Retry-After": "0"}),
            Mock(status_code=503),
        ]
    )
    monkeypatch.setattr(raw.time, "sleep", Mock())
    with pytest.raises(raw.FetchFailure) as failure:
        client.get(
            "https://user:PRIVATE_PASSWORD@example.test/data?token=PRIVATE_QUERY#PRIVATE_FRAGMENT"
        )
    messages = caplog.text + str(failure.value)
    assert "PRIVATE" not in messages
    assert "https://example.test/data" in messages
    assert client.eod.get.call_count == 2

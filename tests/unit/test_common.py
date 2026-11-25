"""Shared infrastructure invariants and legacy byte-format parity, using synthetic data."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mqf_research.common import files, time
from mqf_research.common.hashing import bytes_sha256, file_sha256
from mqf_research.common.numeric_json import json_ready
from mqf_research.common.serialization import read_json, write_json


@pytest.mark.parametrize(
    "payload", [b"", b"abc", bytes(range(256)) * 9000], ids=["empty", "short", "multi_chunk"]
)
def test_digest_agrees_with_independent_full_byte_oracle(tmp_path, payload):
    """Streaming hashes agree for empty, short, and multi-chunk binary content."""
    target = tmp_path / "synthetic.bin"
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert file_sha256(target) == bytes_sha256(payload) == expected


def test_missing_digest_is_an_error(tmp_path):
    """A missing artifact cannot produce an apparently valid empty-file receipt."""
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_read_json_preserves_types_and_accepts_bom(tmp_path, encoding):
    """Windows BOM handling must not reinterpret identifiers or null/missing values."""
    target = tmp_path / "record.json"
    target.write_text('{"event_id":"001","label":null,"ok":true,"n":3}', encoding=encoding)
    assert read_json(target) == {"event_id": "001", "label": None, "ok": True, "n": 3}


def test_bad_json_remains_an_error(tmp_path):
    """Corrupt receipts are never silently replaced with an empty dictionary."""
    target = tmp_path / "record.json"
    target.write_text('{"incomplete":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_json(target)


@pytest.mark.parametrize("newline", [None, "\n", "\r\n"])
def test_json_writer_matches_original_write_text_bytes(tmp_path, newline):
    """Atomic publication preserves the prior native-line-ending JSON byte contract."""
    value = {"company": "財報", "value": 1.25, "missing": None, "lines": "a\nb"}
    expected = tmp_path / "old.json"
    expected.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline=newline
    )
    actual = tmp_path / "new.json"
    write_json(actual, value, newline=newline)
    assert actual.read_bytes() == expected.read_bytes()
    assert read_json(actual) == value


def test_json_writer_preserves_explicit_custom_type_policy(tmp_path):
    """Only callers requesting default=str stringify datetimes in audit receipts."""
    target = tmp_path / "time.json"
    value = {"at": datetime(2024, 1, 2, tzinfo=timezone.utc)}
    with pytest.raises(TypeError):
        write_json(target, value)
    assert not target.exists()
    write_json(target, value, default=str)
    assert read_json(target) == {"at": "2024-01-02 00:00:00+00:00"}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_json_does_not_damage_previous_file(tmp_path, invalid):
    """Strict encoding validates before touching an existing successful receipt."""
    target = tmp_path / "complete.json"
    target.write_bytes(b"previous successful receipt")
    with pytest.raises(ValueError):
        write_json(target, {"score": invalid})
    assert target.read_bytes() == b"previous successful receipt"
    assert list(tmp_path.iterdir()) == [target]


def test_json_directory_creation_is_explicit(tmp_path):
    """A writer cannot silently create an incorrectly spelled existing-run directory."""
    target = tmp_path / "new-run" / "receipt.json"
    with pytest.raises(FileNotFoundError):
        write_json(target, {"ok": True})
    assert not target.parent.exists()
    write_json(target, {"ok": True}, create_parents=True)
    assert read_json(target) == {"ok": True}


def test_json_formatting_options_are_explicit(tmp_path):
    """UTF-8 Unicode and optional trailing newline remain controlled by the caller."""
    target = tmp_path / "receipt.json"
    write_json(target, {"name": "財報"}, ensure_ascii=False, trailing_newline=False, newline="\n")
    assert target.read_bytes() == '{\n  "name": "財報"\n}'.encode("utf-8")
    with pytest.raises(ValueError, match="newline"):
        write_json(target, {}, newline="invalid")


def test_fsync_failure_preserves_previous_artifact_and_cleans_staging(tmp_path, monkeypatch):
    """Disk/flush errors cannot publish incomplete bytes or leave staging litter."""
    target = tmp_path / "receipt.json"
    target.write_bytes(b"old")

    def fail_fsync(descriptor):
        """Simulate a disk-full error after bytes were written but before publication."""
        raise OSError("disk full")

    monkeypatch.setattr(files.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="disk full"):
        write_json(target, {"new": "result"})
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.name != "nt", reason="Windows-only sharing-denial retry")
def test_permission_denial_has_bounded_retry_and_retains_previous_file(tmp_path, monkeypatch):
    """Permanent permission failure stops after six attempts and removes its own temp."""
    target = tmp_path / "receipt.json"
    target.write_bytes(b"old")
    attempts, waits = [], []

    def deny(source, destination):
        """Record publication attempts while reproducing a permanently locked target."""
        attempts.append((source, destination))
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", deny)
    monkeypatch.setattr(files.time, "sleep", waits.append)
    with pytest.raises(PermissionError, match="locked"):
        write_json(target, {})
    assert len(attempts) == 6
    assert waits == [0.01, 0.02, 0.04, 0.08, 0.16]
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_numeric_json_conversion_preserves_report_policy():
    """Missing scientific values become null only through the report-specific adapter."""
    value = {
        2: (np.int64(3), np.float32(1.5), np.float64("nan"), float("inf")),
        "ok": np.bool_(True),
    }
    assert json_ready(value) == {"2": [3, 1.5, None, None], "ok": True}
    assert isinstance(value[2], tuple)


def test_numeric_array_conversion_is_opt_in():
    """Report helpers still reject unexpected arrays; cell receipts explicitly accept them."""
    values = np.array([[1.0, np.nan], [np.inf, 3.5]], dtype=np.float32)
    assert json_ready(values) is values
    with pytest.raises(TypeError):
        json.dumps(json_ready({"value": values}), allow_nan=False)
    assert json_ready(values, allow_arrays=True) == [[1.0, None], [None, 3.5]]


def test_utc_metadata_clock_is_aware_and_second_precision(monkeypatch):
    """Audit metadata uses explicit UTC and drops microseconds without local-time guessing."""
    actual = time.utc_now()
    assert actual.tzinfo == timezone.utc
    frozen = datetime(2024, 3, 11, 13, 29, 59, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(time, "utc_now", lambda: frozen)
    assert time.utc_timestamp() == "2024-03-11T13:29:59+00:00"


def test_consumers_use_shared_file_and_time_contracts():
    """Data, text, and modeling input APIs share infrastructure without changing names."""
    from mqf_research.data import (
        build_abnormal_returns,
        build_market_features,
        build_model_dataset,
        download_raw,
        panel_scope,
    )
    from mqf_research.modeling.tuning import inputs
    from mqf_research.text import release

    assert download_raw.atomic_write is panel_scope.atomic_write is files.atomic_write
    assert download_raw.sha256 is bytes_sha256
    assert release.file_sha256 is inputs.sha is file_sha256
    assert inputs.read is read_json
    assert (
        build_market_features.now_utc
        is build_model_dataset.now_utc
        is build_abnormal_returns.utc_now
        is time.utc_timestamp
    )

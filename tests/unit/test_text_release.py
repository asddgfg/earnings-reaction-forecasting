"""Release and embedding tests using only small temporary synthetic artifacts."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mqf_research.text.__main__ import main
from mqf_research.text.release import (
    build_event_bags,
    file_sha256,
    load_repaired_inputs,
    validate_index,
    validate_release,
)


def _signal(position, event_id="A"):
    """Build an indexed signal with an exact digest of its embedding text."""
    text = f"Synthetic forecast information {position}."
    return {
        "row": position,
        "event_id": event_id,
        "signal_id": f"{event_id}_{position}",
        "embedding_text": text,
        "embedding_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _write_json(path, value):
    """Write test fixtures only; production release functions remain read-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    """Create a complete relocatable release with zero, missing and multi-signal events."""
    events = [
        {"event_id": "A", "extraction_status": "complete"},
        {"event_id": "B", "extraction_status": "complete"},
        {"event_id": "C", "extraction_status": "unresolved"},
    ]
    index = [_signal(0), _signal(1)]
    _write_json(tmp_path / "EVENTS.json", events)
    _write_json(tmp_path / "embeddings_minilm/INDEX.json", index)
    _write_json(
        tmp_path / "embeddings_minilm/COMPLETE.json",
        {
            "dimensions": 3,
            "signals": 2,
            "model": "synthetic-unit-test-encoder",
        },
    )
    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    np.save(tmp_path / "embeddings_minilm/SIGNAL_VECTORS.npy", vectors)
    (tmp_path / "EMBEDDING_INPUT.jsonl").write_text(
        "\n".join(json.dumps(row) for row in index) + "\n", encoding="utf-8"
    )
    data = pd.DataFrame(
        {
            "event_id": ["A", "B", "C"],
            "ticker": ["X", "Y", "Z"],
            "reaction_session": ["2024-01-02"] * 3,
            "feature": [1.0, 2.0, 3.0],
            "y": [0.01, -0.02, 0.03],
        }
    )
    name = "DATASET_financial_full.parquet"
    data.to_parquet(tmp_path / name, index=False)
    _write_json(
        tmp_path / "DATASET_BINDING.json",
        {
            "financial_features": ["feature"],
            "datasets": {"full": "C:/old/computer/path/" + name},
            "inherited_dataset_sha256": {"full": file_sha256(tmp_path / name)},
        },
    )
    members = {
        path.relative_to(tmp_path).as_posix(): file_sha256(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    _write_json(tmp_path / "RELEASE.json", {"batch_id": "synthetic", "files": members})
    return tmp_path


def test_full_release_maps_every_signal_and_distinguishes_empty_from_missing(release):
    """Full validation checks actual vectors and keeps empty-text cases separate."""
    summary = validate_release(release, verify_all_files=True)
    assert summary["signals"] == 2
    assert summary["all_signals_mapped"]
    assert summary["available_zero_signal_events"] == 1
    assert summary["unavailable_text_events"] == 1
    assert summary["vector_values_checked"]
    assert not summary["targets_loaded"]


def test_metadata_only_does_not_open_vectors(release):
    """A metadata-only pass explicitly does not certify missing vector bytes."""
    (release / "embeddings_minilm/SIGNAL_VECTORS.npy").unlink()
    summary = validate_release(release, metadata_only=True)
    assert summary["passed"] and not summary["vector_values_checked"]
    with pytest.raises(FileNotFoundError):
        validate_release(release)


def test_loader_is_relocatable_outcome_masked_and_read_only(release):
    """The maintained loader uses verified local files and excludes target columns by default."""
    data, features, bags, vectors = load_repaired_inputs(release)
    assert features == ["feature"] and "y" not in data.columns
    assert bags == {"A": [0, 1], "B": [], "C": []}
    assert data.text_available.tolist() == [True, True, False]
    assert data.has_signals.tolist() == [True, False, False]
    assert not vectors.flags.writeable
    with pytest.raises(ValueError):
        vectors[0, 0] = 0
    targets, _, _, _ = load_repaired_inputs(release, include_targets=True)
    assert "y" in targets.columns


@pytest.mark.parametrize("fault", ["duplicate_id", "reordered", "stale_text", "unknown_event"])
def test_index_or_event_identity_corruption_is_rejected(fault):
    """Signal identities cannot silently duplicate, reorder or attach to another event."""
    index = [_signal(0), _signal(1)]
    if fault == "duplicate_id":
        index[1]["signal_id"] = index[0]["signal_id"]
    elif fault == "reordered":
        index.reverse()
    elif fault == "stale_text":
        index[0]["embedding_text"] += " changed"
    else:
        index[1]["event_id"] = "UNKNOWN"
    with pytest.raises(ValueError):
        validate_index(index, dimensions=3)
        build_event_bags([{"event_id": "A"}], index)


@pytest.mark.parametrize("fault", ["dimensions", "dtype", "nan", "infinity", "zero_vector"])
def test_incompatible_or_corrupt_vectors_are_rejected(fault):
    """Detect wrong encoder dimensions, precision, numerical corruption and empty encodings."""
    vectors = np.array([[1, 0, 0]], dtype=np.float32)
    dimensions = 3
    if fault == "dimensions":
        dimensions = 384
    elif fault == "dtype":
        vectors = vectors.astype(np.float64)
    elif fault == "nan":
        vectors[0, 0] = np.nan
    elif fault == "infinity":
        vectors[0, 0] = np.inf
    else:
        vectors[:] = 0
    with pytest.raises(ValueError):
        validate_index([_signal(0)], dimensions, vectors)


def test_same_text_does_not_collapse_distinct_signals():
    """Retain distinct signal IDs even when their text or vectors happen to agree."""
    index = [_signal(0), _signal(1)]
    index[1]["embedding_text"] = index[0]["embedding_text"]
    index[1]["embedding_text_sha256"] = index[0]["embedding_text_sha256"]
    validate_index(index, 3)
    bags, _ = build_event_bags([{"event_id": "A", "extraction_status": "complete"}], index)
    assert bags["A"] == [0, 1]


def test_manual_review_can_restore_availability_without_inventing_a_signal():
    """A reviewed zero-signal event becomes available while an ineligible source stays unavailable."""
    events = [
        {"event_id": "A", "text_input_status": "manual_source_review_complete"},
        {"event_id": "B", "extraction_status": "complete", "text_source_eligible": False},
    ]
    bags, flags = build_event_bags(events, [])
    assert bags == {"A": [], "B": []}
    assert flags["A"]["text_available"] and not flags["A"]["has_signals"]
    assert not flags["B"]["text_available"]


@pytest.mark.parametrize("invalid_flag", ["false", 0, None])
def test_missing_flags_cannot_be_coerced_from_strings_or_numbers(invalid_flag):
    """JSON boolean flags must retain their meaning rather than use Python truthiness."""
    with pytest.raises(ValueError, match="JSON boolean"):
        build_event_bags(
            [
                {
                    "event_id": "A",
                    "extraction_status": "complete",
                    "text_source_eligible": invalid_flag,
                }
            ],
            [],
        )


def test_malformed_manifest_returns_cli_failure_instead_of_traceback(release, capsys):
    """Structurally invalid JSON artifacts produce an actionable validation error."""
    _write_json(release / "RELEASE.json", [])
    assert main([str(release), "--metadata-only"]) == 1
    assert not json.loads(capsys.readouterr().err)["passed"]


def test_release_manifest_and_member_checksums_fail_closed(release):
    """Changed manifests and changed payloads are rejected independently."""
    with pytest.raises(ValueError, match="frozen hash"):
        validate_release(release, expected_release_sha256="0" * 64)
    (release / "EVENTS.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_release(release)


@pytest.mark.parametrize("escape", ["../outside.txt", "C:/outside.txt", "C:relative.txt"])
def test_manifest_cannot_escape_release_directory(release, escape):
    """Hash verification must not follow a malicious absolute or parent-relative member."""
    manifest = json.loads((release / "RELEASE.json").read_text(encoding="utf-8"))
    manifest["files"][escape] = "0" * 64
    _write_json(release / "RELEASE.json", manifest)
    with pytest.raises(ValueError, match="relative path|escapes"):
        validate_release(release, verify_all_files=True)


def test_cli_returns_machine_readable_report_without_output_writes(release, capsys):
    """The explicit read-only CLI reports success and does not create artifacts."""
    before = sorted(path.relative_to(release) for path in release.rglob("*"))
    assert main([str(release), "--metadata-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "metadata_only" and report["files_modified"] == 0
    assert before == sorted(path.relative_to(release) for path in release.rglob("*"))

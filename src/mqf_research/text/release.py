"""Read and validate repaired signals and existing vectors without inference.

No function in this module writes a release, fits a model, or invokes an encoder.
The ``batch_032`` loader contract is preserved while adding explicit guards that
remain active under ``python -O`` and making the release directory relocatable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from mqf_research.common.hashing import bytes_sha256, file_sha256
from mqf_research.common.serialization import read_json as _read_json

_METADATA_FILES = (
    "EVENTS.json",
    "DATASET_BINDING.json",
    "EMBEDDING_INPUT.jsonl",
    "embeddings_minilm/INDEX.json",
    "embeddings_minilm/COMPLETE.json",
)
_HORIZONS = ("hour", "half", "full")


def _member_path(base: Path, relative: str) -> Path:
    """Resolve a manifest member while rejecting traversal and external links."""
    windows_path = PureWindowsPath(relative)
    if windows_path.drive or windows_path.is_absolute() or Path(relative).is_absolute():
        raise ValueError("Release member must be a relative path")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Release member escapes the release directory")
    return path


def _check_member(base: Path, manifest: dict, name: str) -> Path:
    """Verify one manifest-bound file before exposing its contents."""
    expected = manifest["files"].get(name)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing or invalid release checksum: {name}")
    path = _member_path(base, name)
    if file_sha256(path) != expected:
        raise ValueError(f"Release checksum mismatch: {name}")
    return path


def validate_index(index: list[dict], dimensions: int, vectors: np.ndarray | None = None) -> None:
    """Check ordered, unique signal identities and optional float32 vectors.

    Every row corresponds to one signal. Duplicate text is not deduplicated:
    distinct supported signals can legitimately share wording. Duplicate IDs,
    reordered row numbers, stale text hashes and nonfinite vectors are errors.
    """
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("Embedding dimensions must be a positive integer")
    if not isinstance(index, list):
        raise ValueError("Embedding index must be a JSON list")
    seen: set[str] = set()
    for position, row in enumerate(index):
        if not isinstance(row, dict):
            raise ValueError("Each embedding index row must be a JSON object")
        if type(row.get("row")) is not int or row["row"] != position:
            raise ValueError("Embedding row numbers must match index order")
        signal_id = row.get("signal_id")
        if not isinstance(signal_id, str) or not signal_id or signal_id in seen:
            raise ValueError("Missing or duplicate signal_id")
        seen.add(signal_id)
        if not isinstance(row.get("event_id"), str) or not row["event_id"]:
            raise ValueError("Missing event_id in embedding index")
        text = row.get("embedding_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Missing embedding_text")
        if bytes_sha256(text.encode("utf-8")) != row.get("embedding_text_sha256"):
            raise ValueError(f"Embedding text hash mismatch: {signal_id}")
    if vectors is None:
        return
    if vectors.shape != (len(index), dimensions):
        raise ValueError("Vector shape does not match index and encoder dimensions")
    if vectors.dtype != np.dtype("float32"):
        raise ValueError("Released signal vectors must use float32")
    # Read-only memmaps remain read-only. Chunking avoids a second full-size
    # temporary array merely to test finiteness and L2-normalization.
    for start in range(0, len(index), 4_096):
        chunk = vectors[start : start + 4_096]
        if not np.isfinite(chunk).all():
            raise ValueError("Signal vectors contain nonfinite values")
        if not np.allclose(np.linalg.norm(chunk, axis=1), 1.0, atol=1e-5):
            raise ValueError("Signal vectors are not L2-normalized")


def build_event_bags(
    events: list[dict], index: list[dict], event_ids: list[str] | None = None
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    """Map every selected event to all of its signal rows and availability flags.

    An eligible completed extraction with no signals has available text and an
    empty bag. Missing/ineligible text is a different condition. No averaging,
    truncation, sampling or implicit duplicate removal is performed.
    """
    metadata: dict[str, dict] = {}
    if not isinstance(events, list):
        raise ValueError("Event metadata must be a JSON list")
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
            raise ValueError("Event metadata needs a string event_id")
        event_id = event["event_id"]
        if not event_id:
            raise ValueError("Event metadata needs a nonempty event_id")
        if "text_source_eligible" in event and not isinstance(event["text_source_eligible"], bool):
            raise ValueError("text_source_eligible must be a JSON boolean")
        if event_id in metadata:
            raise ValueError("Duplicate event_id in event metadata")
        metadata[event_id] = event
    selected = list(metadata) if event_ids is None else list(event_ids)
    if len(selected) != len(set(selected)) or not set(selected) <= metadata.keys():
        raise ValueError("Requested event IDs are duplicated or absent from release")
    bags: dict[str, list[int]] = {event_id: [] for event_id in selected}
    for row in index:
        if row["event_id"] not in metadata:
            raise ValueError("Embedding index references an unknown event")
        if row["event_id"] in bags:
            bags[row["event_id"]].append(row["row"])
    flags = {}
    for event_id, rows in bags.items():
        event = metadata[event_id]
        complete = (
            event.get("extraction_status") == "complete"
            or event.get("text_input_status") == "manual_source_review_complete"
        )
        flags[event_id] = {
            "text_available": bool(complete and event.get("text_source_eligible", True)),
            "has_signals": bool(rows),
            "log_signal_count": float(np.log1p(len(rows))),
        }
    return bags, flags


def _check_embedding_inputs(path: Path, index: list[dict]) -> None:
    """Ensure the indexed text is the exact ordered text approved for embedding."""
    count = 0
    fields = ("signal_id", "event_id", "embedding_text", "embedding_text_sha256")
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if count >= len(index) or any(row.get(key) != index[count].get(key) for key in fields):
                raise ValueError("Embedding input and vector index disagree")
            count += 1
    if count != len(index):
        raise ValueError("Embedding input and vector index have different lengths")


def _artifacts(
    release_dir: str | Path,
    metadata_only: bool,
    expected_release_sha256: str | None,
    verify_all_files: bool,
) -> tuple[Path, dict, list[dict], list[dict], dict, np.ndarray | None]:
    """Load verified text artifacts, optionally omitting all vector-byte reads."""
    base = Path(release_dir).resolve()
    release_path = base / "RELEASE.json"
    if expected_release_sha256 is not None and file_sha256(release_path) != expected_release_sha256:
        raise ValueError("Release manifest differs from the expected frozen hash")
    manifest = _read_json(release_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Release manifest lacks its file checksum mapping")
    if metadata_only and verify_all_files:
        raise ValueError("metadata_only cannot verify every file, including vectors")
    members = list(manifest["files"]) if verify_all_files else list(_METADATA_FILES)
    if not metadata_only and "embeddings_minilm/SIGNAL_VECTORS.npy" not in members:
        members.append("embeddings_minilm/SIGNAL_VECTORS.npy")
    for member in members:
        _check_member(base, manifest, member)
    events = _read_json(base / "EVENTS.json")
    index = _read_json(base / "embeddings_minilm/INDEX.json")
    complete = _read_json(base / "embeddings_minilm/COMPLETE.json")
    if not isinstance(index, list) or not isinstance(complete, dict):
        raise ValueError("Encoder completion must be an object and index must be a list")
    if type(complete.get("signals")) is not int or complete["signals"] != len(index):
        raise ValueError("Encoder completion count does not match index")
    if "dimensions" not in complete:
        raise ValueError("Encoder completion lacks its dimensions")
    vectors = None
    if not metadata_only:
        vectors = np.load(
            base / "embeddings_minilm/SIGNAL_VECTORS.npy", mmap_mode="r", allow_pickle=False
        )
    validate_index(index, complete["dimensions"], vectors)
    _check_embedding_inputs(base / "EMBEDDING_INPUT.jsonl", index)
    return base, manifest, events, index, complete, vectors


def validate_release(
    release_dir: str | Path,
    *,
    metadata_only: bool = False,
    expected_release_sha256: str | None = None,
    verify_all_files: bool = False,
) -> dict[str, Any]:
    """Return a read-only release-validation summary without loading outcomes.

    Metadata-only mode verifies the text/index checksums and row identities but
    explicitly does not certify the vector bytes. The default also verifies and
    memory-maps existing vectors; it never re-embeds any text. ``verify_all_files``
    additionally hashes all release members, including original audit artifacts.
    """
    base, manifest, events, index, complete, vectors = _artifacts(
        release_dir, metadata_only, expected_release_sha256, verify_all_files
    )
    bags, flags = build_event_bags(events, index)
    return {
        "release": str(base),
        "batch_id": manifest.get("batch_id"),
        "passed": True,
        "mode": "metadata_only" if metadata_only else "text_and_existing_vectors",
        "release_manifest_sha256": file_sha256(base / "RELEASE.json"),
        "events": len(bags),
        "signals": len(index),
        "dimensions": complete["dimensions"],
        "encoder": complete.get("model"),
        "available_zero_signal_events": sum(
            flag["text_available"] and not flag["has_signals"] for flag in flags.values()
        ),
        "unavailable_text_events": sum(not flag["text_available"] for flag in flags.values()),
        "all_signals_mapped": sum(map(len, bags.values())) == len(index),
        "vector_values_checked": vectors is not None,
        "all_release_file_hashes_checked": verify_all_files,
        "targets_loaded": False,
        "llm_calls": 0,
        "embedding_calls": 0,
        "files_modified": 0,
    }


def load_repaired_inputs(
    release_dir: str | Path,
    horizon: str = "full",
    *,
    include_targets: bool = False,
    expected_release_sha256: str | None = None,
    verify_all_files: bool = True,
) -> tuple[Any, list[str], dict[str, list[int]], np.ndarray]:
    """Load the final MiniLM input contract, preserving every per-signal vector.

    Returns ``(dataframe, financial_features, event_bags, readonly_vectors)`` as
    the frozen batch loader does. Targets are excluded unless explicitly asked
    for. The known local dataset filename permits moving a complete release to
    another disk without trusting a stale absolute path stored in its binding.
    """
    if horizon not in _HORIZONS:
        raise ValueError(f"horizon must be one of {_HORIZONS}")
    base, manifest, events, index, _, vectors = _artifacts(
        release_dir, False, expected_release_sha256, verify_all_files
    )
    binding = _read_json(base / "DATASET_BINDING.json")
    features = binding["financial_features"]
    if not isinstance(features, list) or any(
        not isinstance(name, str) or not name for name in features
    ):
        raise ValueError("Financial features must be a list of nonempty column names")
    forbidden = {
        "y",
        "signed_y",
        "target",
        "stock_return",
        "iwv_return",
        "prediction",
        "open",
        "close",
    }
    if len(features) != len(set(features)) or forbidden.intersection(features):
        raise ValueError("Financial feature binding contains duplicates or outcome columns")
    dataset_name = f"DATASET_financial_{horizon}.parquet"
    dataset = _check_member(base, manifest, dataset_name)
    expected = binding["inherited_dataset_sha256"][horizon]
    if manifest["files"][dataset_name] != expected:
        raise ValueError("Financial binding disagrees with the released dataset checksum")
    import pandas as pd

    columns = None if include_targets else ["event_id", "ticker", "reaction_session"] + features
    data = pd.read_parquet(dataset, columns=columns)
    bags, flags = build_event_bags(events, index, data["event_id"].tolist())
    for name in ("text_available", "has_signals", "log_signal_count"):
        data[name] = [flags[event_id][name] for event_id in data["event_id"]]
    if vectors is None:
        raise RuntimeError("Internal loader error: vector validation was unexpectedly omitted")
    return data, features, bags, vectors

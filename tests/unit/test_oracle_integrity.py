"""Integrity checks for the bundled, observation-free historical engine snapshots."""

import hashlib
import json
from pathlib import Path


def test_historical_oracle_bytes_match_the_preserved_snapshot():
    """Formatting or accidentally replacing a reference implementation fails explicitly."""
    directory = Path(__file__).resolve().parents[1] / "oracles"
    manifest = json.loads((directory / "SHA256.json").read_text(encoding="utf-8-sig"))
    assert manifest["schema_version"] == 1
    assert {entry["path"] for entry in manifest["files"]} == {
        "tuning_engine.py.txt",
        "feature_engine.py.txt",
    }
    for entry in manifest["files"]:
        path = directory / entry["path"]
        content = path.read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]

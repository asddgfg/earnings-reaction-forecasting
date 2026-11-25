"""Small filesystem fixtures verify protocol locks and resumable completion receipts."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mqf_research.modeling._compatibility import bind_run_configuration
from mqf_research.modeling.feature_validation import inputs, study


def _json(path, value):
    """Write one temporary JSON fixture without accessing project datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_model_completion_receipt_rejects_changed_artifact(tmp_path):
    """Resuming a completed cell requires its artifact bytes to remain unchanged."""
    artifact = tmp_path / "small_result.txt"
    artifact.write_text("result", encoding="utf-8")
    study.complete(tmp_path, [artifact.name])
    receipt = study.check_complete(tmp_path)
    assert receipt["hashes"][artifact.name] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="small_result"):
        study.check_complete(tmp_path)


def test_model_selection_gate_requires_receipt_and_matching_hashes(tmp_path, monkeypatch):
    """A valid selection/protocol pair opens the gate; modifying either closes it again."""
    monkeypatch.setattr(inputs, "OUT", tmp_path)
    monkeypatch.setattr(inputs, "PROTOCOL", tmp_path / "PROTOCOL.json")
    bind_run_configuration(tmp_path, create=True)
    with pytest.raises(ValueError, match="remain closed"):
        inputs.verify_selection_gate()
    _json(inputs.PROTOCOL, {"fixed": "design"})
    selection = tmp_path / "LOCKED_SELECTION.json"
    _json(selection, {"development_only": True})
    _json(
        tmp_path / "SELECTION_COMPLETE.json",
        {
            "protocol_sha256": inputs.sha(inputs.PROTOCOL),
            "locked_selection_sha256": inputs.sha(selection),
            "development_only": True,
            "no_evaluation_outcomes_used": True,
        },
    )
    assert inputs.verify_selection_gate()[0]["development_only"] is True
    _json(selection, {"development_only": False})
    with pytest.raises(ValueError):
        inputs.verify_selection_gate()


def test_model_feature_protocol_is_explicit_and_cannot_be_replaced(tmp_path, monkeypatch):
    """Protocol initialization copies scientific settings and refuses a conflicting rerun."""
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "OUT", tmp_path / "new_runs/feature")
    original = tmp_path / "outputs/nn_feature_validation_20261123_b032/PROTOCOL.json"
    _json(original, {"input_horizons": ["hour", "half", "full"], "input_encoder": "minilm"})
    study.protocol()
    copied = study.read(study.OUT / "PROTOCOL.json")
    assert copied["input_horizons"] == ["hour", "half", "full"]
    assert copied["maintenance"]["scientific_settings_unchanged"] is True
    initial = (study.OUT / "PROTOCOL.json").read_bytes()
    study.protocol()
    assert (study.OUT / "PROTOCOL.json").read_bytes() == initial
    copied["input_encoder"] = "different"
    _json(study.OUT / "PROTOCOL.json", copied)
    with pytest.raises(ValueError, match="fresh output root"):
        study.protocol()


def test_model_partial_cell_is_rejected_before_any_data_access(tmp_path, monkeypatch):
    """A failed earlier fit cannot have its matrices replaced by a retry."""
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "CACHE", {"stage": "development"})
    destination = tmp_path / "development/hour/2023Q1/M"
    destination.mkdir(parents=True)
    checkpoint = destination / "seed42.pt"
    matrix = destination / "FULL.npz"
    checkpoint.write_bytes(b"keep-checkpoint")
    matrix.write_bytes(b"keep-matrix")
    with pytest.raises(FileExistsError, match="Incomplete cell"):
        study.cell(("hour", "2023Q1", "M"))
    assert checkpoint.read_bytes() == b"keep-checkpoint"
    assert matrix.read_bytes() == b"keep-matrix"


def test_model_self_check_uses_output_directory_and_persistent_gate(tmp_path, monkeypatch):
    """An absent selection is a passing gate test and receipts never modify package source."""
    monkeypatch.setattr(inputs, "OUT", tmp_path / "run")
    monkeypatch.setattr(inputs, "WORK", tmp_path / "source")
    monkeypatch.setattr(inputs, "PROTOCOL", tmp_path / "protocol.json")
    _json(inputs.PROTOCOL, {"fixed": True})
    bind_run_configuration(inputs.OUT, create=True)
    frames = {name: pd.DataFrame({"event_id": ["synthetic"]}) for name in inputs.HORIZONS}
    groups = {name: {"numeric": ["x"], "categorical": []} for name in ("M", "MF", "MFL")}
    audit = pd.DataFrame(
        {
            "empty_bag": [True],
            "source_eligible": [True],
            "text_available": [True],
            "has_signals": [False],
        }
    )

    def fake_load(*args):
        """Supply synthetic metadata while the real selection gate and output writer run."""
        return (
            frames,
            groups,
            {"synthetic": []},
            np.zeros((0, 384)),
            audit,
            pd.DataFrame({"row": []}),
        )

    monkeypatch.setattr(inputs, "_load", fake_load)
    checked = inputs.self_check_development()
    assert checked["evaluation_gate"].startswith("Passed:")
    assert (inputs.OUT / "checks/INPUT_HELPER_CHECK.json").is_file()
    assert not inputs.WORK.exists()


def test_model_shared_behavior_helpers_are_in_freeze_scope():
    """Source receipts bind path resolution and persistent guards as well as model code."""
    from mqf_research.modeling._compatibility import maintained_source_paths

    names = {path.name for path in maintained_source_paths()}
    assert {"_validation.py", "_compatibility.py", "config.py", "PROVENANCE.json"} <= names


def test_model_resume_rejects_changed_resolved_configuration(tmp_path, monkeypatch):
    """Changing an environment path cannot silently reuse receipts from another workspace."""
    receipt = bind_run_configuration(tmp_path, create=True)
    before = receipt.read_bytes()
    monkeypatch.setenv("MQF_RESEARCH_ROOT", str(tmp_path / "different_workspace"))
    with pytest.raises(ValueError, match="configuration changed"):
        bind_run_configuration(tmp_path)
    assert receipt.read_bytes() == before

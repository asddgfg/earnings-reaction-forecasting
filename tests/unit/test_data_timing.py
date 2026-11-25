"""Completion and time-boundary checks that do not require any research data."""

import json
import os
import sys
from pathlib import Path

import pytest

from mqf_research.data import build_abnormal_returns as ar
from mqf_research.data import build_market_features as market
from mqf_research.data import build_model_dataset as dataset
from mqf_research.data.batch_selection import choose_completed_batch
from mqf_research.data.legacy_tags import load_legacy_validator


def make_batch(data, name, *, done=True, incomplete=False):
    """Create one synthetic compatible batch with selectable completion markers."""
    batch = data / "event_text_batches" / name
    event = batch / "ai_inputs" / "ABC" / "event"
    event.mkdir(parents=True)
    (event / "manifest.json").write_text(
        json.dumps(
            {
                "packaging_policy_version": "actual_release_v1",
                "reaction_session": "2024-01-02",
            }
        ),
        encoding="utf-8",
    )
    if done:
        (batch / "DONE.json").write_text("{}", encoding="utf-8")
    if incomplete:
        (batch / "INCOMPLETE").touch()
    return batch


@pytest.mark.parametrize("choose", [ar.choose_batch, market.choose_batch, dataset.choose_batch])
def test_all_data_stages_ignore_uncompleted_batches(tmp_path, choose):
    """All three consumers select a completed batch despite newer unfinished folders."""
    completed = make_batch(tmp_path, "valid")
    make_batch(tmp_path, "unfinished", done=False)
    make_batch(tmp_path, "interrupted", incomplete=True)
    assert choose(tmp_path, None) == completed.resolve()


@pytest.mark.parametrize("done,incomplete", [(False, False), (True, True)])
def test_explicit_incomplete_batch_fails_closed(tmp_path, done, incomplete):
    """An explicit path does not waive completion requirements."""
    batch = make_batch(tmp_path, "bad", done=done, incomplete=incomplete)
    with pytest.raises(ValueError, match="not completed"):
        choose_completed_batch(tmp_path, batch)


def test_selection_uses_done_timestamp_not_directory_activity(tmp_path):
    """An old batch touched by later audit files must not become the latest completed run."""
    older = make_batch(tmp_path, "old")
    newer = make_batch(tmp_path, "new")
    os.utime(older / "DONE.json", ns=(1_000_000, 1_000_000))
    os.utime(newer / "DONE.json", ns=(2_000_000, 2_000_000))
    os.utime(older, ns=(9_000_000, 9_000_000))
    assert choose_completed_batch(tmp_path) == newer.resolve()


@pytest.mark.parametrize(
    "a,b",
    [
        ("2024-01-02", "2024-01-02T14:30:00Z"),
        ("2024-01-02T09:30:00", "2024-01-02T14:30:00Z"),
        ("2024-01-02T14:30:00Z", "2024-01-02T09:30:00"),
        ("2024-01-02", "2024-01-02"),
        (None, "2024-01-02T14:30:00Z"),
        ("invalid", "2024-01-02T14:30:00Z"),
    ],
)
def test_point_in_time_comparison_rejects_unverifiable_inputs(a, b):
    """No timezone assumption may turn missing or ambiguous evidence into PIT proof."""
    assert dataset.dt_leq(a, b) is None


def test_point_in_time_comparison_normalizes_equivalent_offsets():
    """Boundary equality succeeds across offsets, while one second late fails."""
    assert dataset.dt_leq("2024-01-02T09:30:00-05:00", "2024-01-02T14:30:00Z") is True
    assert dataset.dt_leq("2024-01-02T09:30:01-05:00", "2024-01-02T14:30:00Z") is False


def test_legacy_validator_load_isolated_and_never_calls_generation(tmp_path):
    """The compatibility adapter imports only the configured package with no path mutation."""
    package = tmp_path / "ai_tagger"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "config.py").write_text("VALIDATOR_SENTINEL = 42\n", encoding="utf-8")
    (package / "tag_events.py").write_text(
        "from .config import VALIDATOR_SENTINEL\n", encoding="utf-8"
    )
    before = list(sys.path)
    validator = load_legacy_validator(tmp_path)
    assert validator.VALIDATOR_SENTINEL == 42
    assert Path(validator.__file__).parent == package
    assert sys.path == before
    assert load_legacy_validator(tmp_path) is validator


def test_missing_legacy_source_has_actionable_error(tmp_path):
    """Missing historical source must fail explicitly rather than fabricate new tags."""
    with pytest.raises(FileNotFoundError, match="MQF_SOURCE_PROJECT"):
        load_legacy_validator(tmp_path)

"""Synthetic numeric/MIL contract tests; no research observations or training run."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mqf_research.modeling.feature_validation import engine as feature
from mqf_research.modeling.tuning import engine as tuning


def _frames():
    """Return tiny chronological frames with missing features and an unseen sector."""
    train = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "reaction_session": ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"],
            "x": [1.0, 3.0, np.nan, 5.0],
            "empty": [np.nan] * 4,
            "sector": ["A", "B", None, "A"],
            "y": [0.01, -0.02, 0.03, 0.04],
        }
    )
    test = pd.DataFrame(
        {
            "event_id": ["e", "f"],
            "reaction_session": ["2022-01-04", "2022-01-05"],
            "x": [1000.0, np.nan],
            "empty": [np.nan, np.nan],
            "sector": ["NEW", "A"],
            "y": [999.0, -999.0],
        }
    )
    return train, test


def _history():
    """Construct the minimum valid 200/100 chronological stopping partition."""
    return pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(300)],
            "quarter": ["2022Q1"] * 200 + ["2022Q2"] * 50 + ["2022Q3"] * 50,
            "reaction_session": ["2022-01-10"] * 200 + ["2022-04-11"] * 50 + ["2022-07-11"] * 50,
        }
    )


def _archived_engine(relative):
    """Import a byte-preserved, data-free snapshot bundled as a test oracle."""
    filenames = {
        "nn_tuning_20261123_b032": "tuning_engine.py.txt",
        "nn_feature_validation_20261123_b032": "feature_engine.py.txt",
    }
    location = Path(__file__).resolve().parents[1] / "oracles" / filenames[relative]
    loader = importlib.machinery.SourceFileLoader(f"_oracle_{relative}", str(location))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", [tuning, feature])
def test_model_preprocessing_uses_training_only(engine):
    """Extreme future features/targets cannot alter fitted statistics or train matrices."""
    train, test = _frames()
    result, fitted = engine.prepare(train, test, ["x", "empty"], ["sector"])
    changed = test.copy()
    changed["x"], changed["y"] = [-1e9, 1e9], [-1e99, 1e99]
    second, other = engine.prepare(train, changed, ["x", "empty"], ["sector"])
    np.testing.assert_array_equal(result["X_train"], second["X_train"])
    np.testing.assert_array_equal(fitted["scaler"].mean_, other["scaler"].mean_)
    assert result["target_mean"] == pytest.approx(0.015)
    assert fitted["imputer"].statistics_[0] == 3
    assert "NEW" not in fitted["onehot"].categories_[0]
    assert np.isfinite(result["X_test"]).all()
    assert result["X_train"].dtype == np.float32


@pytest.mark.parametrize(
    "engine, archived",
    [(tuning, "nn_tuning_20261123_b032"), (feature, "nn_feature_validation_20261123_b032")],
)
def test_model_preprocessing_matches_archived_valid_inputs(engine, archived):
    """Documentation and guard changes preserve exact valid-input preprocessing arithmetic."""
    train, test = _frames()
    expected, _ = _archived_engine(archived).prepare(train, test, ["x", "empty"], ["sector"])
    actual, _ = engine.prepare(train, test, ["x", "empty"], ["sector"])
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_model_all_numeric_recipes_preserve_archived_forward_arithmetic():
    """All twelve fixed recipes retain their original initialization and predictions."""
    archived = _archived_engine("nn_tuning_20261123_b032")
    assert tuning.recipes() == archived.recipes()
    values = torch.arange(21, dtype=torch.float32).reshape(3, 7) / 21
    for recipe in tuning.recipes():
        torch.manual_seed(42)
        old = archived.NumericNet(7, recipe).eval()
        torch.manual_seed(42)
        current = tuning.NumericNet(7, recipe).eval()
        torch.testing.assert_close(old(values), current(values), rtol=0, atol=0)


@pytest.mark.parametrize("engine", [tuning, feature])
def test_model_preprocessing_rejects_overlap_and_infinity(engine):
    """Invalid alignment or infinite features fail before any fitting can occur."""
    train, test = _frames()
    test.loc[0, "event_id"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        engine.prepare(train, test, ["x"], [])
    test.loc[0, "event_id"] = "e"
    train.loc[0, "x"] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        engine.prepare(train, test, ["x"], [])


@pytest.mark.parametrize("engine", [tuning, feature])
def test_model_preprocessing_rejects_future_history(engine):
    """Even disjoint IDs cannot place later observations in the fitted history."""
    train, test = _frames()
    train.loc[0, "reaction_session"] = "2023-01-01"
    with pytest.raises(ValueError, match="strictly earlier"):
        engine.prepare(train, test, ["x"], [])


@pytest.mark.parametrize("engine", [tuning, feature])
def test_model_split_small_history_fails_instead_of_looping(engine):
    """The historical fewer-than-100-row infinite-loop case now returns a clear error."""
    with pytest.raises(ValueError, match="at least 300"):
        engine.inner_split(_history().iloc[:50])


@pytest.mark.parametrize("engine", [tuning, feature])
def test_model_split_retains_chronology_and_all_events(engine):
    """The fixed minimum history gives 200 core and 100 later validation events."""
    history = _history()
    core, validation = engine.inner_split(history)
    assert (len(core), len(validation)) == (200, 100)
    assert core.reaction_session.max() < validation.reaction_session.min()
    assert set(core.event_id).isdisjoint(validation.event_id)
    assert set(core.event_id) | set(validation.event_id) == set(history.event_id)


def test_model_attention_matches_archived_and_ignores_padding():
    """Both model branches match old arithmetic, including empty bags and padded slots."""
    archived = _archived_engine("nn_feature_validation_20261123_b032")
    vectors = np.random.default_rng(7).normal(size=(4, 384)).astype(np.float32)
    bags = {"a": [0, 1, 2], "b": [], "c": [3]}
    embeddings, mask = feature.bag_batch(["a", "b", "c"], vectors, bags)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    for use_text in (False, True):
        torch.manual_seed(42)
        old = archived.Network(4, use_text).eval()
        torch.manual_seed(42)
        current = feature.Network(4, use_text).eval()
        expected, actual = old(x, embeddings, mask), current(x, embeddings, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        extra = torch.cat([embeddings, torch.full((3, 2, 384), 999.0)], dim=1)
        extra_mask = torch.cat([mask, torch.zeros((3, 2), dtype=torch.bool)], dim=1)
        torch.testing.assert_close(current(x, extra, extra_mask), actual, rtol=1e-6, atol=1e-7)
    pooled, weights = current.text_branch(embeddings, mask)
    torch.testing.assert_close(weights.sum(1), torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(pooled[1], torch.zeros(32), rtol=0, atol=0)
    actual.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in current.parameters()
    )


def test_model_attention_retains_signals_and_is_permutation_invariant():
    """Signal order changes neither the pooled representation nor the forecast."""
    vectors = np.random.default_rng(17).normal(size=(6, 384)).astype(np.float32)
    model = feature.Network(2, True).eval()
    x = torch.tensor([[0.1, 0.2]])
    original = feature.bag_batch(["event"], vectors, {"event": list(range(6))})
    reversed_bag = feature.bag_batch(["event"], vectors, {"event": list(reversed(range(6)))})
    assert original[1].sum() == 6
    torch.testing.assert_close(model(x, *original), model(x, *reversed_bag), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("indices", [[-1], [2], [0.5]])
def test_model_bags_reject_invalid_indices(indices):
    """Negative, out-of-bounds, and fractional indices cannot select the wrong signal."""
    with pytest.raises(ValueError, match="[Ii]ndex|indices"):
        feature.bag_batch(["event"], np.zeros((2, 384), dtype=np.float32), {"event": indices})


def test_model_bags_empty_batch_and_wrong_shape():
    """Empty batch is explicit; malformed embedding dimensions produce an actionable error."""
    emb, mask = feature.bag_batch([], np.zeros((0, 384), dtype=np.float32), {})
    assert tuple(emb.shape) == (0, 1, 384)
    assert tuple(mask.shape) == (0, 1)
    with pytest.raises(ValueError, match="384"):
        feature.bag_batch(["event"], np.zeros((1, 32)), {"event": [0]})


@pytest.mark.parametrize("engine", [tuning, feature])
def test_model_refit_never_overwrites_existing_checkpoint(engine, tmp_path):
    """Partial output is rejected before training or matrix access, preserving its bytes."""
    old = tmp_path / "seed42.pt"
    old.write_bytes(b"preserve-existing-checkpoint")
    with pytest.raises(FileExistsError, match="fresh run"):
        if engine is tuning:
            engine.refit({}, {}, 1, tmp_path)
        else:
            engine.refit({}, 1, False, None, None, tmp_path)
    assert old.read_bytes() == b"preserve-existing-checkpoint"


def test_model_early_stopping_selects_best_epoch_not_last(monkeypatch):
    """A known validation-loss curve stops after patience and retains its earlier minimum."""
    matrix = {
        "X_train": np.zeros((2, 1), np.float32),
        "X_test": np.zeros((2, 1), np.float32),
        "y_train_std": np.zeros(2, np.float32),
        "target_mean": 0.0,
        "target_sd": 1.0,
        "train_ids": np.array(["a", "b"]),
        "test_ids": np.array(["c", "d"]),
    }
    state = {"model": object(), "epoch": 0}
    curve = [3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    def fake_initialize(dim, use_text):
        """Return one controlled state so the test isolates the stopping rule."""
        return [state]

    def fake_epoch(*args):
        """Advance the synthetic curve without optimizing a model."""
        state["epoch"] += 1
        return 0.0

    def fake_predict(*args):
        """Return a constant whose squared error is the designed curve value."""
        return np.full(2, np.sqrt(curve[state["epoch"] - 1]))

    monkeypatch.setattr(feature, "initialize", fake_initialize)
    monkeypatch.setattr(feature, "epoch_step", fake_epoch)
    monkeypatch.setattr(feature, "predict", fake_predict)
    selected = feature.select_epochs(matrix, np.zeros(2), False, None, None)
    assert selected["selected_epoch"] == 2
    assert selected["stopped_epoch"] == 7
    assert selected["best_inner_mse"] == 1

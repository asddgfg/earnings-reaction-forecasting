"""Tiny synthetic training regressions; never load research observations or call APIs.

Six training rows, three forecast rows and two epochs exercise the real optimizer
and checkpoint paths. Assertions concern reproducibility and numerical contracts,
not whether an arbitrary miniature dataset becomes accurately predictable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mqf_research.modeling.feature_validation import engine as feature
from mqf_research.modeling.tuning import engine as tuning

MODES = ("tuning", "feature_numeric", "feature_text")


def _synthetic_case():
    """Build deterministic model-ready arrays and variable bags with empty events."""
    rng = np.random.default_rng(20261124)
    train_ids = np.array([f"train-{index}" for index in range(6)])
    test_ids = np.array([f"test-{index}" for index in range(3)])
    matrix = {
        "X_train": rng.normal(size=(6, 3)).astype(np.float32),
        "X_test": rng.normal(size=(3, 3)).astype(np.float32),
        "y_train_std": np.array([-1.0, 0.4, 0.8, -0.2, 1.1, -0.7], dtype=np.float32),
        "target_mean": 0.002,
        "target_sd": 0.015,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    bags = {}
    cursor = 0
    for event, count in zip(
        np.concatenate([train_ids, test_ids]), (2, 0, 1, 3, 1, 2, 1, 0, 2), strict=True
    ):
        bags[str(event)] = list(range(cursor, cursor + count))
        cursor += count
    vectors = rng.normal(size=(cursor, 384)).astype(np.float32)
    return matrix, vectors, bags


def _engine(mode):
    """Select the maintained engine without accessing a dataset or workflow stage."""
    return tuning if mode == "tuning" else feature


def _initialize(mode):
    """Create the same small recipe or fixed numeric/MIL architecture for both seeds."""
    if mode == "tuning":
        return tuning.initialize(3, tuning.recipes()[0])
    return feature.initialize(3, mode == "feature_text")


def _refit(engine, mode, matrix, vectors, bags, destination):
    """Run exactly two epochs on the six synthetic rows in an isolated temporary cell."""
    if mode == "tuning":
        return engine.refit(matrix, tuning.recipes()[0], 2, destination)
    return engine.refit(matrix, 2, mode == "feature_text", vectors, bags, destination)


def _replay(engine, mode, checkpoint, matrix, vectors, bags):
    """Replay a trusted checkpoint created by this test without another optimization step."""
    if mode == "tuning":
        return engine.replay(checkpoint, matrix)
    return engine.replay(checkpoint, matrix, vectors, bags)


def _archived_engine(mode):
    """Load the distributed data-free historical optimizer-loop oracle."""
    name = "tuning_engine.py.txt" if mode == "tuning" else "feature_engine.py.txt"
    source = Path(__file__).resolve().parents[1] / "oracles" / name
    loader = importlib.machinery.SourceFileLoader(f"_training_oracle_{mode}", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _step(mode, state, matrix, vectors, bags):
    """Advance one real epoch using only the synthetic arrays supplied by the fixture."""
    x = torch.from_numpy(matrix["X_train"])
    y = torch.from_numpy(matrix["y_train_std"])
    if mode == "tuning":
        return tuning.epoch_step(state, x, y)
    return feature.epoch_step(state, x, y, matrix["train_ids"], vectors, bags)


@pytest.mark.parametrize("mode", MODES)
def test_synthetic_fit_checkpoint_replay_and_archived_regression(mode, tmp_path):
    """Real tiny fits preserve archived updates, seed aggregation and exact saved replay."""
    matrix, vectors, bags = _synthetic_case()
    engine = _engine(mode)
    initial = {
        state["seed"]: state["model"].numeric[0].weight.detach().clone()
        for state in _initialize(mode)
    }
    destination = tmp_path / "first"
    forecast = _refit(engine, mode, matrix, vectors, bags, destination)
    seed_forecasts = np.load(destination / "SEED_PREDICTIONS.npy", allow_pickle=False)
    assert forecast.shape == (3,)
    assert seed_forecasts.shape == (2, 3)
    assert np.isfinite(seed_forecasts).all()
    np.testing.assert_array_equal(forecast, seed_forecasts.mean(axis=0))

    checkpoints = []
    for index, seed in enumerate((42, 43)):
        checkpoint_path = destination / f"seed{seed}.pt"
        # These are trusted temporary files written by the engine immediately above.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoints.append(checkpoint)
        assert checkpoint["seed"] == seed
        assert checkpoint["epochs"] == 2
        assert checkpoint["target_mean"] == matrix["target_mean"]
        assert checkpoint["target_sd"] == matrix["target_sd"]
        assert not torch.equal(checkpoint["state_dict"]["numeric.0.weight"], initial[seed])
        replayed = _replay(engine, mode, checkpoint_path, matrix, vectors, bags)
        np.testing.assert_array_equal(replayed, seed_forecasts[index])
    assert not torch.equal(
        checkpoints[0]["state_dict"]["numeric.0.weight"],
        checkpoints[1]["state_dict"]["numeric.0.weight"],
    )

    # A fresh cell reproduces the whole fit, while the frozen code independently
    # verifies that the maintained optimizer loop did not change valid arithmetic.
    repeated = _refit(engine, mode, matrix, vectors, bags, tmp_path / "repeat")
    archived = _refit(_archived_engine(mode), mode, matrix, vectors, bags, tmp_path / "archived")
    np.testing.assert_array_equal(forecast, repeated)
    np.testing.assert_array_equal(forecast, archived)
    archived_seeds = np.load(tmp_path / "archived/SEED_PREDICTIONS.npy", allow_pickle=False)
    np.testing.assert_array_equal(seed_forecasts, archived_seeds)


@pytest.mark.parametrize("mode", MODES)
def test_synthetic_seed_states_are_independent_of_execution_schedule(mode):
    """Interleaving seed updates or unrelated RNG draws cannot alter either fitted model."""
    matrix, vectors, bags = _synthetic_case()
    alternating = _initialize(mode)
    consecutive = _initialize(mode)
    for index in (0, 1, 0, 1):
        loss = _step(mode, alternating[index], matrix, vectors, bags)
        assert np.isfinite(loss)
    for index in (0, 0, 1, 1):
        # Simulate another worker consuming the global RNG between seeded steps.
        torch.rand(17)
        loss = _step(mode, consecutive[index], matrix, vectors, bags)
        assert np.isfinite(loss)
    for first, second in zip(alternating, consecutive, strict=True):
        for name, parameter in first["model"].state_dict().items():
            torch.testing.assert_close(
                parameter, second["model"].state_dict()[name], rtol=0, atol=0
            )
        torch.testing.assert_close(first["torch_state"], second["torch_state"], rtol=0, atol=0)
        assert first["rng"].bit_generator.state == second["rng"].bit_generator.state


@pytest.mark.parametrize("mode", ("tuning", "feature_numeric"))
@pytest.mark.parametrize("scenario", ("ties", "cap"))
def test_stopping_uses_ensemble_mse_earliest_tie_and_fixed_cap(mode, scenario, monkeypatch):
    """Known two-seed curves test aggregation, exact tie order and the 40-epoch budget."""
    matrix, vectors, bags = _synthetic_case()
    engine = _engine(mode)
    states = [{"model": SimpleNamespace(epoch=0, offset=offset)} for offset in (-0.5, 0.5)]

    def fake_initialize(*args):
        """Supply two controlled seed states without fitting any validation target."""
        return states

    def fake_epoch(state, *args):
        """Advance the known curve once per seed and return a finite synthetic fit loss."""
        state["model"].epoch += 1
        return 1.0

    def fake_predict(model, *args):
        """Keep seed offsets opposite so averaging forecasts removes them before MSE."""
        center = (3.0 if model.epoch == 1 else 1.0) if scenario == "ties" else 41.0 - model.epoch
        return np.full(3, center + model.offset)

    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(engine, "epoch_step", fake_epoch)
    monkeypatch.setattr(engine, "predict", fake_predict)
    if mode == "tuning":
        selected = engine.select_epochs(matrix, np.zeros(3), tuning.recipes()[0])
    else:
        selected = engine.select_epochs(matrix, np.zeros(3), False, vectors, bags)
    assert selected["best_inner_mse"] == 1.0
    assert selected["core_train_ids"] == matrix["train_ids"].tolist()
    assert selected["inner_validation_ids"] == matrix["test_ids"].tolist()
    if scenario == "ties":
        assert selected["selected_epoch"] == 2
        assert selected["stopped_epoch"] == 7
        assert selected["hit_cap"] is False
        # Mean(seed squared errors) would be 9.25; squared ensemble error is 9.
        assert selected["curve"][0]["validation_mse"] == 9.0
    else:
        assert selected["selected_epoch"] == 40
        assert selected["stopped_epoch"] == 40
        assert selected["hit_cap"] is True
        assert len(selected["curve"]) == 40
        assert selected["curve"][0]["validation_mse"] == 1600.0

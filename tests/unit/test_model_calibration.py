"""Known-answer and chronological tests for the frozen calibration specification."""

import numpy as np
import pandas as pd
import pytest

from mqf_research.modeling.feature_validation import calibration


def _panel():
    """Build a synthetic matched 463-event panel with the fixed 2023/2024 counts."""
    events = []
    for year, counts in ((2023, (60, 60, 60, 58)), (2024, (56, 56, 56, 57))):
        for quarter, count in enumerate(counts, 1):
            month = 3 * quarter - 2
            for event in range(count):
                events.append(
                    {
                        "event_id": f"{year}-{quarter}-{event}",
                        "ticker": "SYNTH",
                        "reaction_session": f"{year}-{month:02d}-{1 + event % 20:02d}",
                        "quarter": f"{year}Q{quarter}",
                        "year": year,
                        "y": 0.0,
                        "prediction": 0.0,
                    }
                )
    return pd.concat(
        [
            pd.DataFrame(events).assign(horizon=horizon, group=group)
            for horizon in calibration.HORIZONS
            for group in calibration.GROUPS
        ],
        ignore_index=True,
    )


def test_model_calibration_known_offset_and_identity():
    """An additive forecast bias has the known offset, while identity leaves forecasts intact."""
    prediction = np.linspace(-0.05, 0.05, 120)
    observed = prediction + 0.012
    coeff = calibration.fit("offset", observed, prediction)
    assert coeff["a"] == pytest.approx(0.012)
    assert coeff["b"] == 1
    np.testing.assert_allclose(calibration.apply(coeff, prediction), observed, atol=1e-15)
    unchanged = calibration.fit("none", observed, prediction)
    np.testing.assert_array_equal(calibration.apply(unchanged, prediction), prediction)


def test_model_affine_uses_declared_regularization_and_nonnegative_slope():
    """Closed-form affine coefficients follow the documented shrinkage and slope constraint."""
    prediction = np.linspace(-0.02, 0.02, 120)
    observed = 0.003 + 2 * prediction
    variance = float(np.mean(prediction**2))
    reg = 0.1 * 4 * variance
    expected_slope = (2 * variance + reg) / (variance + reg)
    coeff = calibration.fit("affine", observed, prediction)
    assert coeff["b"] == pytest.approx(expected_slope)
    assert coeff["a"] == pytest.approx(0.003)
    reversed_coeff = calibration.fit("affine", -prediction, prediction)
    assert reversed_coeff["b"] == 0


@pytest.mark.parametrize("method", calibration.METHODS)
def test_model_calibration_rejects_short_or_nonfinite_fit(method):
    """All candidates require the same minimum finite, aligned earlier observations."""
    with pytest.raises(ValueError, match="120"):
        calibration.fit(method, np.zeros(119), np.zeros(119))
    bad = np.zeros(120)
    bad[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibration.fit(method, bad, np.zeros(120))


def test_model_calibration_exact_ties_prefer_none_and_fit_strictly_earlier():
    """Equal zero-error methods select identity; every fold fits only previous quarters."""
    result = calibration.select(_panel())
    assert len(result["folds"]) == 108
    assert len(result["scores"]) == 27
    assert len(result["predictions"]) == 6075
    assert (result["folds"].train_end < result["folds"].test_start).all()
    assert result["scores"].n.eq(225).all()
    for row in result["folds"].itertuples():
        assert set(row.train_ids).isdisjoint(row.test_ids)
    for horizon in calibration.HORIZONS:
        for group in calibration.GROUPS:
            selection = result["summary"]["selections"][horizon][group]
            assert selection["method"] == "none"
            assert selection["n"] == 463
            assert selection["coefficients_frozen_for_evaluation"] is True


def test_model_calibration_rejects_future_outcomes_and_duplicate_events():
    """Evaluation rows and unequal sample multiplicity cannot enter calibration selection."""
    panel = _panel()
    future = panel.copy()
    selected = future.year.eq(2024)
    future.loc[selected, "year"] = 2025
    future.loc[selected, "reaction_session"] = future.loc[selected, "reaction_session"].str.replace(
        "2024", "2025"
    )
    future.loc[selected, "quarter"] = future.loc[selected, "quarter"].str.replace("2024", "2025")
    with pytest.raises(ValueError, match="2023-2024"):
        calibration.select(future)
    with pytest.raises(ValueError, match="Duplicate"):
        calibration.select(pd.concat([panel, panel.iloc[[0]]], ignore_index=True))


def test_model_calibration_rejects_changed_target_for_one_feature_group():
    """Feature comparisons must use identical realized targets for each matched event."""
    panel = _panel()
    index = panel.index[panel.group.eq("MF")][0]
    panel.loc[index, "y"] = 0.1
    with pytest.raises(ValueError, match="same horizon target"):
        calibration.select(panel)

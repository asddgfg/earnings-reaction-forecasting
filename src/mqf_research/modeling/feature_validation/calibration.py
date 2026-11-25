"""Development-only calibration for the frozen three-group, three-horizon study.

This module reads no files, fits no base model, and never accepts evaluation rows.
``select`` returns a dict containing a JSON-serializable ``summary`` and pandas
``predictions``, ``scores`` and ``folds`` tables. Raw predictions are preserved in
``prediction``; candidate outputs use ``calibrated_prediction``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("none", "offset", "affine")
HORIZONS = ("hour", "half", "full")
GROUPS = ("M", "MF", "MFL")
DEV_QUARTERS = tuple(f"{year}Q{quarter}" for year in (2023, 2024) for quarter in range(1, 5))
SELECTION_QUARTERS = tuple(f"2024Q{quarter}" for quarter in range(1, 5))
MIN_FIT_EVENTS = 120
DEVELOPMENT_EVENTS = 463
SELECTION_EVENTS = 225
AFFINE_REGULARIZATION = 0.1


def _array(values, name):
    """Convert a vector to finite floating-point values or raise a clear input error."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def fit(method, y, p):
    """Return a,b,n,reg for a + b*p; affine shrinks b toward one.

    Affine minimizes mean((y-a-b*p)**2) + reg*(b-1)**2, subject to
    b >= 0, with an unpenalized intercept and reg=0.1*population_variance(y).
    All three methods require the same minimum number of earlier OOF observations.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    y, p = _array(y, "y"), _array(p, "p")
    if len(y) != len(p) or len(y) < MIN_FIT_EVENTS:
        raise ValueError("Calibration requires at least 120 aligned earlier OOF observations")
    a, b, reg = 0.0, 1.0, 0.0
    if method == "offset":
        a = float(np.mean(y - p))
    elif method == "affine":
        reg = AFFINE_REGULARIZATION * float(np.var(y, ddof=0))
        denominator = float(np.var(p, ddof=0)) + reg
        covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
        b = max(0.0, (covariance + reg) / denominator) if denominator > 1e-20 else 1.0
        a = float(y.mean() - b * p.mean())
    if not np.isfinite([a, b, reg]).all():
        raise ValueError("Non-finite calibration coefficients")
    return dict(a=float(a), b=float(b), n=int(len(y)), reg=float(reg))


def apply(coeff, p):
    """Apply frozen coefficients to signed AR predictions, without clipping."""
    p = _array(p, "p")
    a, b = float(coeff["a"]), float(coeff["b"])
    if not np.isfinite([a, b]).all() or b < 0:
        raise ValueError("Calibration requires finite a and nonnegative b")
    result = a + b * p
    if not np.isfinite(result).all():
        raise ValueError("Non-finite calibrated predictions")
    return result


def _validated(dev_predictions):
    """Validate the exact development-only calibration panel and matched event targets.

    Reject evaluation years, duplicates, missing values, and inconsistent group/horizon metadata.
    """
    required = {
        "event_id",
        "ticker",
        "reaction_session",
        "quarter",
        "year",
        "horizon",
        "group",
        "y",
        "prediction",
    }
    if not required.issubset(dev_predictions.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(dev_predictions.columns))}")
    z = dev_predictions.copy(deep=True)
    if z[list(required)].isna().any().any():
        raise ValueError("Calibration inputs cannot contain missing required fields")
    if set(z.horizon) != set(HORIZONS) or set(z.group) != set(GROUPS):
        raise ValueError("Expected exactly the three fixed horizons and groups")
    if z.duplicated(["horizon", "group", "event_id"]).any():
        raise ValueError("Duplicate event within a group/horizon")
    dates = pd.to_datetime(z.reaction_session, errors="raise")
    z["reaction_session"] = dates.dt.strftime("%Y-%m-%d")
    years = pd.to_numeric(z.year, errors="raise")
    if not np.array_equal(years.to_numpy(), dates.dt.year.to_numpy()):
        raise ValueError("Year does not match reaction session")
    z["year"] = years.astype(int)
    if set(z.year) != {2023, 2024}:
        raise ValueError("Only 2023-2024 development OOF rows are permitted")
    derived_quarters = dates.dt.to_period("Q").astype(str)
    if not np.array_equal(z.quarter.astype(str).to_numpy(), derived_quarters.to_numpy()):
        raise ValueError("Quarter does not match reaction session")
    for name in ("y", "prediction"):
        z[name] = _array(z[name], name)
    cohort = None
    for horizon in HORIZONS:
        horizon_target = None
        for group in GROUPS:
            part = z[(z.horizon == horizon) & (z.group == group)]
            if (
                len(part) != DEVELOPMENT_EVENTS
                or int((part.year == 2024).sum()) != SELECTION_EVENTS
                or set(part.quarter) != set(DEV_QUARTERS)
            ):
                raise ValueError(f"Unexpected development/selection cohort: {horizon}/{group}")
            indexed = part.sort_values("event_id").reset_index(drop=True)
            metadata = indexed[["event_id", "ticker", "reaction_session", "quarter", "year"]]
            if cohort is None:
                cohort = metadata
            elif not metadata.equals(cohort):
                raise ValueError("Groups and horizons must use identical event metadata")
            if horizon_target is None:
                horizon_target = indexed.y.to_numpy()
            elif not np.array_equal(indexed.y.to_numpy(), horizon_target):
                raise ValueError("Feature groups must share exactly the same horizon target")
    return z.sort_values(["horizon", "group", "reaction_session", "event_id"]).reset_index(
        drop=True
    )


def select(dev_predictions):
    """Select by pooled 2024 MSE, then fit frozen coefficients on all 463 OOF rows.

    2023 supplies initial OOF calibration history. Each 2024 quarter uses strictly
    earlier OOF rows. The base recipe and stopping policy were selected using the
    development period, so these scores are selection diagnostics, not an unbiased
    test. No candidate, event, or quarter is silently omitted.
    """
    z = _validated(dev_predictions)
    predictions, folds, scores = [], [], []
    selections = {horizon: {} for horizon in HORIZONS}
    for horizon in HORIZONS:
        for group in GROUPS:
            part = z[(z.horizon == horizon) & (z.group == group)].copy()
            totals = {method: {"sse": 0.0, "n": 0} for method in METHODS}
            for quarter in SELECTION_QUARTERS:
                history, test = part[part.quarter < quarter], part[part.quarter == quarter]
                if (
                    len(history) < MIN_FIT_EVENTS
                    or len(test) == 0
                    or history.reaction_session.max() >= test.reaction_session.min()
                    or set(history.event_id) & set(test.event_id)
                ):
                    raise ValueError(f"Invalid calibration split: {horizon}/{group}/{quarter}")
                for method in METHODS:
                    coeff = fit(method, history.y, history.prediction)
                    calibrated = apply(coeff, test.prediction)
                    squared_error = np.square(test.y.to_numpy() - calibrated)
                    sse = float(squared_error.sum())
                    totals[method]["sse"] += sse
                    totals[method]["n"] += len(test)
                    row = dict(
                        horizon=horizon,
                        group=group,
                        quarter=quarter,
                        method=method,
                        **coeff,
                        train_start=str(history.reaction_session.min()),
                        train_end=str(history.reaction_session.max()),
                        test_start=str(test.reaction_session.min()),
                        test_end=str(test.reaction_session.max()),
                        train_ids=history.event_id.tolist(),
                        test_ids=test.event_id.tolist(),
                        test_n=int(len(test)),
                        sse=sse,
                    )
                    folds.append(row)
                    candidate = test.copy()
                    candidate["method"] = method
                    candidate["calibrated_prediction"] = calibrated
                    candidate["a"], candidate["b"] = coeff["a"], coeff["b"]
                    candidate["fit_n"], candidate["fit_reg"] = coeff["n"], coeff["reg"]
                    candidate["fit_end"] = row["train_end"]
                    predictions.append(candidate)
            mse = {}
            for method in METHODS:
                total = totals[method]
                if total["n"] != SELECTION_EVENTS:
                    raise ValueError("All candidates must score the same 225 development events")
                mse[method] = total["sse"] / total["n"]
                scores.append(
                    dict(
                        horizon=horizon,
                        group=group,
                        method=method,
                        n=total["n"],
                        mse=float(mse[method]),
                        sse=float(total["sse"]),
                    )
                )
            winner = min(METHODS, key=lambda method: (mse[method], METHODS.index(method)))
            final = fit(winner, part.y, part.prediction)
            selections[horizon][group] = dict(
                method=winner,
                **final,
                selection_mse=float(mse[winner]),
                selection_n=SELECTION_EVENTS,
                candidate_mse={m: float(mse[m]) for m in METHODS},
                fit_start=str(part.reaction_session.min()),
                fit_end=str(part.reaction_session.max()),
                fit_ids=part.event_id.tolist(),
                fit_quarters=list(DEV_QUARTERS),
                coefficients_frozen_for_evaluation=True,
            )
    score_table = pd.DataFrame(scores)
    score_table["selected"] = [
        selections[r.horizon][r.group]["method"] == r.method for r in score_table.itertuples()
    ]
    result = dict(
        summary=dict(
            methods=list(METHODS),
            horizons=list(HORIZONS),
            groups=list(GROUPS),
            selection_criterion="Lowest pooled 2024 MSE on chronological OOF calibration forecasts",
            exact_tie_order=list(METHODS),
            selection_quarters=list(SELECTION_QUARTERS),
            warmup_quarters=[f"2023Q{q}" for q in range(1, 5)],
            development_events_per_model=DEVELOPMENT_EVENTS,
            selection_events_per_model=SELECTION_EVENTS,
            minimum_fit_events=MIN_FIT_EVENTS,
            affine_regularization_multiplier=AFFINE_REGULARIZATION,
            affine_penalty="reg * (b - 1)^2; intercept unpenalized; b >= 0",
            decisions_use_development_only=True,
            evaluation_rows_read=0,
            development_scores_are_selection_diagnostics=True,
            selections=selections,
        ),
        predictions=pd.concat(predictions, ignore_index=True),
        scores=score_table,
        folds=pd.DataFrame(folds),
    )
    if (
        len(result["predictions"]) != 9 * 3 * SELECTION_EVENTS
        or len(result["scores"]) != 27
        or len(result["folds"]) != 108
    ):
        raise RuntimeError("Incomplete calibration candidate matrix")
    return result

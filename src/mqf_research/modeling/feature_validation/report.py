"""Frozen three-group NN forecast report. No model or calibrator is fitted here."""

from __future__ import annotations

import os

from mqf_research.common.hashing import file_sha256 as sha
from mqf_research.common.numeric_json import json_ready as clean
from mqf_research.common.serialization import read_json as read
from mqf_research.common.serialization import write_json
from mqf_research.config import get_settings
from mqf_research.modeling._compatibility import (
    activate_legacy_root,
    bind_run_configuration,
    load_legacy_module,
    require_run_destination,
)
from mqf_research.modeling._validation import require

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import argparse
import html
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = get_settings().research_root


def _inference():
    """Load preserved statistical code only for an actual report calculation."""
    return load_legacy_module("work.model_selection_20261118.statisticalinference", ROOT)


OUT = get_settings().output_root / "nn_feature_validation"
HELPER = ROOT / "work/model_selection_20261118/statisticalinference.py"
HORIZONS = ("hour", "half", "full")
GROUPS = ("M", "MF", "MFL")
PERIODS = ("2025", "2026", "2025-2026")
BOOTSTRAP_METHODS = ("ticker", "circular_5_observed_dates", "circular_10_observed_dates")
GROUP_NAMES = {"M": "Market", "MF": "Market + Financial", "MFL": "Market + Financial + LLM"}
WINDOW_NAMES = {
    "hour": "One hour: 09:30–10:30 ET",
    "half": "Half day: 09:30–12:45 ET",
    "full": "Full day: 09:30–16:00 ET",
}
SEED, REPLICATES = 20261123, 5000


def write(path, value):
    """Atomically publish the historical JSON schema in the isolated run directory."""
    write_json(path, clean(value))


class CachedBootstrap:
    """Run the original inference helper with reusable, identical random weights.

    Only a private in-memory copy of its resampling primitive is replaced. Its
    validation, status handling, p-values and intervals remain the original code.
    Weights retain the helper's 512-draw batches and RNG calls exactly. The shared
    imported helper and every source file remain unchanged.
    """

    def __init__(self):
        """Create the model/helper in the fixed parameter order used by the experiment."""
        self.weights = {}
        spec = importlib.util.spec_from_file_location("_nn_feature_bootstrap_helper", HELPER)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module._resampled_means = self.resampled_means

    def resampled_means(self, values, labels, reps, seed, block_length=0):
        """Reuse deterministic cluster/bootstrap weights while retaining original RNG draws.

        Cache keys include group count, replicate count, seed, and circular block length.
        """
        sums, counts, groups = self.module._group_sums(values, labels)
        key = (groups, int(reps), int(seed), int(block_length))
        if key not in self.weights:
            code = 0 if block_length == 0 else 100 + block_length
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), code]))
            batches = []
            for start in range(0, reps, 512):
                size = min(512, reps - start)
                if block_length == 0:
                    weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=size)
                else:
                    starts = rng.integers(0, groups, size=(size, math.ceil(groups / block_length)))
                    ix = ((starts[:, :, None] + np.arange(block_length)) % groups).reshape(
                        size, -1
                    )[:, :groups]
                    weights = np.zeros((size, groups), dtype=np.int64)
                    np.add.at(weights, (np.repeat(np.arange(size), groups), ix.ravel()), 1)
                batches.append(weights)
            self.weights[key] = batches
        result = np.empty(reps)
        start = 0
        for weights in self.weights[key]:
            denominator = weights @ counts
            if np.any(denominator <= 0):
                raise RuntimeError("Invalid bootstrap denominator")
            size = len(weights)
            result[start : start + size] = (weights @ sums) / denominator
            start += size
        return result

    def test(self, diff, tickers, dates):
        """Run all fixed paired-bootstrap checks on one aligned contrast vector."""
        return self.module.paired_cluster_bootstrap(
            diff, tickers, dates, reps=REPLICATES, seed=SEED
        )


def _validate_predictions(data, development):
    """Reject malformed, incomplete, or misaligned prediction panels before inference."""
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
        "history_mean",
        "history_majority_sign",
    }
    if not development:
        required |= {"calibrated_prediction", "calibration_method"}
    require((required <= set(data)), required - set(data))
    data = data.copy()
    require(
        (not data[list(required)].isna().any().any()),
        "Invariant failed: not data[list(required)].isna().any().any()",
    )
    data["reaction_session"] = pd.to_datetime(data.reaction_session).dt.strftime("%Y-%m-%d")
    dates = pd.to_datetime(data.reaction_session)
    require(
        (np.array_equal(dates.dt.year, data.year)),
        "Invariant failed: np.array_equal(dates.dt.year, data.year)",
    )
    require(
        (np.array_equal(dates.dt.to_period("Q").astype(str), data.quarter)),
        "Invariant failed: np.array_equal(dates.dt.to_period('Q').astype(str), data.quarter)",
    )
    require(
        (set(data.horizon) == set(HORIZONS) and set(data.group) == set(GROUPS)),
        "Invariant failed: set(data.horizon) == set(HORIZONS) and set(data.group) == set(GROUPS)",
    )
    require(
        (not data.duplicated(["horizon", "group", "event_id"]).any()),
        "Invariant failed: not data.duplicated(['horizon', 'group', 'event_id']).any()",
    )
    numbers = ["y", "prediction", "history_mean", "history_majority_sign"]
    if not development:
        numbers.append("calibrated_prediction")
    require(
        (np.isfinite(data[numbers].to_numpy(float)).all()),
        "Invariant failed: np.isfinite(data[numbers].to_numpy(float)).all()",
    )
    require(
        (data.history_majority_sign.isin([-1, 1]).all()),
        "Invariant failed: data.history_majority_sign.isin([-1, 1]).all()",
    )
    expected_counts = {2023: 238, 2024: 225} if development else {2025: 238, 2026: 259}
    n = 463 if development else 497
    require((len(data) == n * 9), "Invariant failed: len(data) == n * 9")
    reference = None
    for horizon in HORIZONS:
        target_reference = None
        for group in GROUPS:
            part = (
                data[(data.horizon == horizon) & (data.group == group)]
                .sort_values("event_id")
                .reset_index(drop=True)
            )
            require(
                (len(part) == n and part.year.value_counts().to_dict() == expected_counts),
                "Invariant failed: len(part) == n and part.year.value_counts().to_dict() == expected_counts",
            )
            meta = part[["event_id", "ticker", "reaction_session", "quarter", "year"]]
            if reference is None:
                reference = meta
            else:
                pd.testing.assert_frame_equal(meta, reference, check_exact=True, check_dtype=False)
            targets = part[["event_id", "y", "history_mean", "history_majority_sign"]]
            if target_reference is None:
                target_reference = targets
            else:
                pd.testing.assert_frame_equal(
                    targets, target_reference, check_exact=True, check_dtype=False
                )
            if not development:
                require(
                    (part.reaction_session.max() == "2026-10-02"),
                    "Invariant failed: part.reaction_session.max() == '2026-10-02'",
                )
    return data.sort_values(["horizon", "group", "reaction_session", "event_id"]).reset_index(
        drop=True
    )


def load(out):
    """Read completed prediction artifacts only after checking source and result receipts."""
    receipt = read(out / "SELECTION_COMPLETE.json")
    require(
        (sha(out / "LOCKED_SELECTION.json") == receipt["locked_selection_sha256"]),
        "Invariant failed: sha(out / 'LOCKED_SELECTION.json') == receipt['locked_selection_sha256']",
    )
    require(
        (sha(out / "PROTOCOL.json") == receipt["protocol_sha256"]),
        "Invariant failed: sha(out / 'PROTOCOL.json') == receipt['protocol_sha256']",
    )
    lock = read(out / "LOCKED_SELECTION.json")
    require(
        (lock["development_only"] and lock["no_evaluation_outcomes_used"]),
        "Invariant failed: lock['development_only'] and lock['no_evaluation_outcomes_used']",
    )
    frozen = read(out / "FROZEN_INPUTS.json")
    for path, digest in frozen.items():
        require((sha(path) == digest), path)
    dev = _validate_predictions(pd.read_parquet(out / "DEV_PREDICTIONS.parquet"), True)
    completed = read(out / "EVAL_COMPLETE.json")
    require(
        (completed["locked_selection_sha256"] == receipt["locked_selection_sha256"]),
        "Invariant failed: completed['locked_selection_sha256'] == receipt['locked_selection_sha256']",
    )
    require(
        (sha(out / "EVALUATION_PREDICTIONS.parquet") == completed["prediction_sha256"]),
        "Invariant failed: sha(out / 'EVALUATION_PREDICTIONS.parquet') == completed['prediction_sha256']",
    )
    # This is the first evaluation forecast/label read; the above barrier is mandatory.
    evaluation = _validate_predictions(
        pd.read_parquet(out / "EVALUATION_PREDICTIONS.parquet"), False
    )
    model_rows, selection_rows = [], []
    cal = lock["calibration"]
    require(
        (cal["decisions_use_development_only"] and cal["evaluation_rows_read"] == 0),
        "Invariant failed: cal['decisions_use_development_only'] and cal['evaluation_rows_read'] == 0",
    )
    require(
        (cal["selection_events_per_model"] == 225 and cal["development_events_per_model"] == 463),
        "Invariant failed: cal['selection_events_per_model'] == 225 and cal['development_events_per_model'] == 463",
    )
    for horizon in HORIZONS:
        for group in GROUPS:
            chosen = cal["selections"][horizon][group]
            part = evaluation[(evaluation.horizon == horizon) & (evaluation.group == group)]
            epoch = lock["model_selections"][horizon][group]["selected_epoch"]
            require(
                (isinstance(epoch, int) and 1 <= epoch <= 40),
                "Invariant failed: isinstance(epoch, int) and 1 <= epoch <= 40",
            )
            require(
                (chosen["method"] in ("none", "offset", "affine") and chosen["n"] == 463),
                "Invariant failed: chosen['method'] in ('none', 'offset', 'affine') and chosen['n'] == 463",
            )
            require(
                (chosen["selection_n"] == 225 and chosen["coefficients_frozen_for_evaluation"]),
                "Invariant failed: chosen['selection_n'] == 225 and chosen['coefficients_frozen_for_evaluation']",
            )
            require(
                (chosen["fit_end"] < "2025-01-01" and chosen["b"] >= 0),
                "Invariant failed: chosen['fit_end'] < '2025-01-01' and chosen['b'] >= 0",
            )
            require(
                (part.calibration_method.eq(chosen["method"]).all()),
                "Invariant failed: part.calibration_method.eq(chosen['method']).all()",
            )
            np.testing.assert_allclose(
                part.calibrated_prediction,
                chosen["a"] + chosen["b"] * part.prediction,
                rtol=1e-12,
                atol=1e-15,
            )
            dev_part = dev[(dev.horizon == horizon) & (dev.group == group)]
            require(
                (
                    set(chosen["fit_ids"]) == set(dev_part.event_id)
                    and len(chosen["fit_ids"]) == 463
                ),
                "Invariant failed: set(chosen['fit_ids']) == set(dev_part.event_id) and len(chosen['fit_ids']) == 463",
            )
            order = ("none", "offset", "affine")
            winner = min(
                order, key=lambda method: (chosen["candidate_mse"][method], order.index(method))
            )
            require((winner == chosen["method"]), "Invariant failed: winner == chosen['method']")
            model_rows.append(
                dict(
                    horizon=horizon,
                    group=group,
                    selected_epoch=epoch,
                    calibration_method=chosen["method"],
                    a=chosen["a"],
                    b=chosen["b"],
                    reg=chosen["reg"],
                    calibration_fit_n=chosen["n"],
                    calibration_fit_end=chosen["fit_end"],
                )
            )
            for method in order:
                selection_rows.append(
                    dict(
                        horizon=horizon,
                        group=group,
                        method=method,
                        n=225,
                        mse=chosen["candidate_mse"][method],
                        selected=method == winner,
                    )
                )
    names = [
        "PROTOCOL.json",
        "LOCKED_SELECTION.json",
        "SELECTION_COMPLETE.json",
        "FROZEN_INPUTS.json",
        "DEV_PREDICTIONS.parquet",
        "EVALUATION_PREDICTIONS.parquet",
        "EVAL_COMPLETE.json",
    ]
    return (
        dev,
        evaluation,
        lock,
        pd.DataFrame(model_rows),
        pd.DataFrame(selection_rows),
        {n: sha(out / n) for n in names},
        frozen,
    )


def metrics(part, prediction, horizon, group, period, variant):
    """Compute AR error and direction summaries on the supplied matched event cohort.

    AR is expressed in decimal returns; percentage-point displays multiply errors by 100.
    """
    y = part.y.to_numpy(float)
    p = np.asarray(prediction, float)
    error = p - y
    correct = (p > 0) == (y > 0)
    majority = (part.history_majority_sign.to_numpy() > 0) == (y > 0)
    return dict(
        horizon=horizon,
        group=group,
        period=period,
        variant=variant,
        n=len(part),
        mse=float(np.mean(error**2)),
        rmse_pp=100 * float(np.sqrt(np.mean(error**2))),
        mae_pp=100 * float(np.mean(np.abs(error))),
        bias_pp=100 * float(error.mean()),
        correct=int(correct.sum()),
        accuracy_pct=100 * float(correct.mean()),
        majority_correct=int(majority.sum()),
        majority_accuracy_pct=100 * float(majority.mean()),
        predicted_positive=int((p > 0).sum()),
        predicted_zero=int((p == 0).sum()),
        actual_positive=int((y > 0).sum()),
        ticker_clusters=part.ticker.nunique(),
        date_clusters=part.reaction_session.nunique(),
        first_event=part.reaction_session.min(),
        last_event=part.reaction_session.max(),
    )


def decorate(rows, metric):
    """Attach the predeclared multiple-testing family and Holm-adjusted interpretation."""
    rows = list(rows)
    for family in sorted({r["family"] for r in rows}):
        selected = [r for r in rows if r["family"] == family]
        require((len(selected) == 27), (family, len(selected)))
        _inference().holm(selected)
    factor = 10000 if metric == "mse" else 100
    unit = "pp2" if metric == "mse" else "pp"
    for row in rows:
        for field in ("effect", "ci_low", "ci_high"):
            row[field + "_" + unit] = factor * row[field]
        row["decision"] = (
            "not_estimable"
            if not row["valid"]
            else "significant_improvement"
            if row["reject_holm"] and row["effect"] > 0
            else "significant_deterioration"
            if row["reject_holm"] and row["effect"] < 0
            else "not_significant"
        )
        row["inference_scope"] = (
            "Conditional on saved forecasts; nominal intervals and family-adjusted p-values"
        )
    return pd.DataFrame(rows)


def build_tests(evaluation):
    """Construct all predeclared paired error, direction, and calibration contrasts.

    Keep the complete multiplicity family, including non-estimable constant differences.
    """
    boot = CachedBootstrap()
    rows = {name: [] for name in ("mse", "direction", "calibration")}
    boot_rows = {name: {method: [] for method in BOOTSTRAP_METHODS} for name in rows}
    metric_rows, baseline_rows = [], []

    def add(kind, horizon, period, candidate, comparator, diff, part):
        """Append one fully identified paired contrast to the current report's test family."""
        family = (
            "calibration_mse_all_horizons" if kind == "calibration" else f"raw_{kind}_{horizon}"
        )
        identity = dict(
            test_id=f"{kind}__{horizon}__{period}__{candidate}_vs_{comparator}",
            family=family,
            horizon=horizon,
            period=period,
            candidate=candidate,
            comparator=comparator,
        )
        rows[kind].append(
            dict(
                **identity,
                **_inference().paired_cluster_test(diff, part.ticker, part.reaction_session),
            )
        )
        for result in boot.test(diff, part.ticker, part.reaction_session):
            boot_rows[kind][result["method"]].append(dict(**identity, **result))

    for horizon in HORIZONS:
        for period in PERIODS:
            frame = evaluation[evaluation.horizon == horizon]
            if period != "2025-2026":
                frame = frame[frame.year == int(period)]
            parts = {
                group: frame[frame.group == group]
                .sort_values(["reaction_session", "event_id"])
                .reset_index(drop=True)
                for group in GROUPS
            }
            ref = parts["M"]
            y = ref.y.to_numpy(float)
            loss, correct = {}, {}
            for group, part in parts.items():
                pd.testing.assert_series_equal(part.event_id, ref.event_id, check_names=False)
                raw = part.prediction.to_numpy(float)
                calibrated = part.calibrated_prediction.to_numpy(float)
                loss[group] = (y - raw) ** 2
                correct[group] = ((raw > 0) == (y > 0)).astype(float)
                metric_rows.append(metrics(part, raw, horizon, group, period, "raw"))
                metric_rows.append(
                    metrics(part, calibrated, horizon, group, period, "selected_calibration")
                )
                add(
                    "calibration",
                    horizon,
                    period,
                    group + "_calibrated",
                    group + "_raw",
                    loss[group] - (y - calibrated) ** 2,
                    part,
                )
            for base, prediction in (
                ("zero_AR", np.zeros(len(ref))),
                ("history_mean", ref.history_mean.to_numpy(float)),
            ):
                result = metrics(ref, prediction, horizon, base, period, "baseline")
                if base == "zero_AR":
                    result["correct"] = result["accuracy_pct"] = None
                baseline_rows.append(result)
            for candidate, comparator in (("MF", "M"), ("MFL", "MF"), ("MFL", "M")):
                add(
                    "mse",
                    horizon,
                    period,
                    candidate,
                    comparator,
                    loss[comparator] - loss[candidate],
                    ref,
                )
                add(
                    "direction",
                    horizon,
                    period,
                    candidate,
                    comparator,
                    correct[candidate] - correct[comparator],
                    ref,
                )
            majority = ((ref.history_majority_sign.to_numpy() > 0) == (y > 0)).astype(float)
            for group in GROUPS:
                add("mse", horizon, period, group, "zero_AR", y**2 - loss[group], ref)
                add(
                    "mse",
                    horizon,
                    period,
                    group,
                    "history_mean",
                    (y - ref.history_mean.to_numpy()) ** 2 - loss[group],
                    ref,
                )
                add("direction", horizon, period, group, "chance_50", correct[group] - 0.5, ref)
                add(
                    "direction",
                    horizon,
                    period,
                    group,
                    "historical_majority",
                    correct[group] - majority,
                    ref,
                )
    tables = {}
    for kind in rows:
        metric = "direction" if kind == "direction" else "mse"
        tables[kind] = decorate(rows[kind], metric)
        tables[kind + "_bootstrap"] = pd.concat(
            [decorate(boot_rows[kind][method], metric) for method in BOOTSTRAP_METHODS],
            ignore_index=True,
        )
    require(
        ([len(tables[k]) for k in ("mse", "direction", "calibration")] == [81, 81, 27]),
        "Invariant failed: [len(tables[k]) for k in ('mse', 'direction', 'calibration')] == [81, 81, 27]",
    )
    require(
        (
            [len(tables[k + "_bootstrap"]) for k in ("mse", "direction", "calibration")]
            == [243, 243, 81]
        ),
        "Invariant failed: [len(tables[k + '_bootstrap']) for k in ('mse', 'direction', 'calibration')] == [243, 243, 81]",
    )
    return pd.DataFrame(metric_rows), pd.DataFrame(baseline_rows), tables


def summarize(tests, bootstrap):
    """Summarize clustered tests and resampling checks without selecting favorable methods."""
    lookup = {test_id: set(group.decision) for test_id, group in bootstrap.groupby("test_id")}
    return dict(
        significant_improvements=tests.loc[
            tests.decision.eq("significant_improvement"), "test_id"
        ].tolist(),
        significant_deteriorations=tests.loc[
            tests.decision.eq("significant_deterioration"), "test_id"
        ].tolist(),
        not_significant=tests.loc[tests.decision.eq("not_significant"), "test_id"].tolist(),
        not_estimable=tests.loc[tests.decision.eq("not_estimable"), "test_id"].tolist(),
        improvements_supported_by_all_bootstrap_methods=[
            r.test_id
            for r in tests.itertuples()
            if r.decision == "significant_improvement"
            and lookup[r.test_id] == {"significant_improvement"}
        ],
        cluster_bootstrap_decision_disagreements=[
            r.test_id for r in tests.itertuples() if lookup[r.test_id] != {r.decision}
        ],
    )


def table(data, digits=6):
    """Render a pandas table for the human-readable report without changing its data."""
    return (
        '<div class="scroll">'
        + data.to_html(
            index=False, border=0, na_rep="Not estimable", float_format=lambda x: f"{x:.{digits}f}"
        )
        + "</div>"
    )


def make_html(
    out, metric_table, baseline_table, tests, dev_metrics, cal_scores, locked_models, conclusion
):
    """Render the experiment narrative and numeric tables into its standalone HTML report."""
    primary = conclusion["primary_one_hour_llm_increment"]
    decision = primary["decision"]
    if decision == "significant_improvement":
        headline = (
            "The one-hour LLM package reduces MSE versus Market + Financial after Holm correction."
            if primary["all_bootstrap_methods_agree"]
            else "The one-hour LLM package reduces MSE under the primary test, but bootstrap decisions differ."
        )
    elif decision == "significant_deterioration":
        headline = "The one-hour LLM package increases MSE versus Market + Financial after Holm correction."
    elif decision == "not_estimable":
        headline = "The primary one-hour LLM contrast does not have an estimable significance test."
    else:
        headline = "The one-hour LLM package does not show a significant MSE improvement versus Market + Financial."
    sections = []
    for horizon in HORIZONS:
        raw = metric_table[(metric_table.horizon == horizon) & metric_table.variant.eq("raw")]
        display = raw[
            [
                "period",
                "group",
                "n",
                "mse",
                "rmse_pp",
                "mae_pp",
                "accuracy_pct",
                "majority_accuracy_pct",
            ]
        ].copy()
        display["group"] = display.group.map(GROUP_NAMES)
        sections.append(f"<h2>{html.escape(WINDOW_NAMES[horizon])}</h2>" + table(display))
        bases = baseline_table[baseline_table.horizon == horizon][
            ["period", "group", "n", "mse", "rmse_pp"]
        ]
        sections.append(
            "<details><summary>Numerical baselines</summary>" + table(bases) + "</details>"
        )
    cal_display = metric_table[
        [
            "horizon",
            "period",
            "group",
            "variant",
            "n",
            "mse",
            "rmse_pp",
            "mae_pp",
            "bias_pp",
            "accuracy_pct",
        ]
    ]
    test_sections = []
    for key, title, effect in (
        ("mse", "Raw MSE comparisons", "effect_pp2"),
        ("direction", "Raw direction comparisons", "effect_pp"),
        ("calibration", "Selected calibration versus raw MSE", "effect_pp2"),
    ):
        columns = [
            "horizon",
            "period",
            "candidate",
            "comparator",
            effect,
            "ci_low_pp" if effect.endswith("_pp") else "ci_low_pp2",
            "ci_high_pp" if effect.endswith("_pp") else "ci_high_pp2",
            "p_raw",
            "p_holm",
            "decision",
        ]
        test_sections.append(
            f"<details><summary>{title}: all {len(tests[key])} prespecified contrasts</summary>"
            + table(tests[key][columns])
            + "</details>"
        )
    links = [
        "METRICS.csv",
        "BASELINE_METRICS.csv",
        "DEV_METRICS.csv",
        "LOCKED_MODELS.csv",
        "CALIBRATION_SELECTION.csv",
        "RAW_MSE_TESTS.csv",
        "RAW_DIRECTION_TESTS.csv",
        "CALIBRATION_MSE_TESTS.csv",
        "RAW_MSE_BOOTSTRAP.csv",
        "RAW_DIRECTION_BOOTSTRAP.csv",
        "CALIBRATION_MSE_BOOTSTRAP.csv",
        "CONCLUSION.json",
        "PROTOCOL.json",
        "LOCKED_SELECTION.json",
    ]
    download = " · ".join(f'<a href="{name}">{name}</a>' for name in links)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fixed NN: features, horizons and calibration</title>
<style>body{{font:16px Arial,sans-serif;line-height:1.55;max-width:1300px;margin:36px auto;padding:0 24px;color:#193044}}
h1,h2{{color:#12324c}}h2{{margin-top:32px}}.note{{padding:18px;background:#edf3f7;border-left:4px solid #267293}}
.small{{font-size:14px;color:#53697a}}.scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;font-size:13px;width:100%}}
td,th{{padding:8px;border-bottom:1px solid #d7e0e8;text-align:right;vertical-align:top}}td:first-child,th:first-child{{text-align:left}}
th{{background:#edf3f7}}summary{{cursor:pointer;margin:16px 0}}a{{color:#155c83}}
@media print{{body{{margin:0;max-width:none}}.scroll{{overflow:visible}}tr{{break-inside:avoid}}}}</style></head><body>
<h1>Fixed neural-network framework: feature value and calibration</h1>
<p class="small">23 November 2026 · Batch 032 · Signed IWV-adjusted AR · 497 common later-period events</p>
<div class="note"><b>{html.escape(headline)}</b><br>Primary pooled contrast: MFL versus MF, one hour.
MSE difference = {primary["effect_pp2"]:.6f} pp²; Holm-adjusted p = {primary["p_holm"] if primary["p_holm"] is not None else "not estimable"}.
Positive differences favor the candidate. Direction accuracy is secondary.</div>
<h2>What is held fixed</h2>
<p>Market (M), Market + Financial (MF), and Market + Financial + LLM (MFL) use the same events,
time splits, 32-unit numeric branch, 64-unit prediction head, dropout 0.2, AdamW learning rate 0.002,
weight decay 0.05, and two seeds (42 and 43). MFL adds the existing 384-dimensional MiniLM signal vectors,
32-dimensional gated attention with 16 gate units, and three text availability/count inputs.
No new LLM extraction or embedding is run. MFL therefore tests the whole LLM feature package;
its effect cannot be attributed to semantic content alone.</p>
<p>The architecture was selected earlier using development MSE. Each group/window uses the same earlier-history
inner stopping rule, then a fresh refit. Final epoch counts are chosen separately on development data through
2024 and locked before evaluation. Later-quarter fits may update weights and preprocessing using completed
earlier quarters, but do not reselect epochs, hyperparameters, features, or calibration. This is a comparison
under a common neural framework, not identical parameter counts or a causal decomposition of information.</p>
{table(locked_models)}
<h2>Target, clock and coverage</h2>
<p>For each window, AR = stock open-to-end return − prior same-window alpha − beta × realized IWV open-to-end return.
The rolling model uses 252 eligible prior matched sessions within a 504-session search window. Each window
has its own alpha/beta. Realized IWV returns belong to the label, not the input. The information cutoff is
09:25 ET under the audited SEC availability rule. The windows start at 09:30 ET. These are endpoint returns,
not total overnight reactions, path volatility, or returns available from a pre-announcement trade.</p>
<p>The common intraday cohort has 1,341 events: 844 in 2021–2024 and 497 later events. The broader project
covers 2018–2026; those earlier years are not part of this matched intraday cohort. Development forecasts cover 463
events in 2023–2024. Evaluation contains 238 events in 2025 and 259 in 2026; 2026 ends on 2 October and is not
a complete calendar year. Coverage is shared across feature groups and windows.</p>
<h2>Validation before calibration</h2>
<p>Raw pooled MSE is the main development metric; direction is secondary. The following development scores
are selection diagnostics: the architecture and stopping decisions use development data. They are not an
unbiased outer test. The one-hour task is primary; half-day and full-day results are prespecified extensions.</p>
{table(dev_metrics)}
<p>MSE is in squared decimal-return units. RMSE, MAE and bias are in percentage points; accuracy is a percentage.
A positive forecast predicts positive AR; a nonpositive forecast predicts nonpositive AR. Zero AR is a numerical
forecast baseline, not a 50% direction classifier. The historical-majority direction is computed from earlier
training outcomes, with ties assigned positive.</p>
{"".join(sections)}
<h2>Calibration as a separate step</h2>
<p>Exactly three candidates are compared: raw (a=0,b=1), residual offset (a=mean(y−p),b=1), and nonnegative
regularized affine correction (a+b·p). Affine minimizes mean squared error plus 0.1·Var(y)·(b−1)², with b≥0
and an unpenalized intercept. The regularization is fixed, not tuned.</p>
<p>2023 OOF predictions supply the initial calibration history. Each 2024 quarter is transformed using only
earlier OOF predictions and labels (at least 120 events). Methods are selected on the same 225 pooled 2024
events; exact ties prefer raw, offset, then affine. The selected coefficients are then fitted on all 463
development OOF rows and frozen throughout 2025–2026. Raw feature comparisons remain primary, so separate
calibration choices do not masquerade as LLM feature gains.</p>
{table(cal_scores)}
<details><summary>Raw and selected-calibration performance</summary>{table(cal_display)}</details>
<h2>Significance and sensitivity</h2>
<p>All contrasts are paired by event. The primary two-sided t test uses stock/date two-way clustered variance
and min(cluster counts)−1 degrees of freedom. For each window, the 27 raw MSE contrasts form one Holm family;
the 27 direction contrasts form a separate family. The 27 selected-calibration versus raw MSE contrasts across
all windows form one family. Families include 2025, 2026, and pooled results; no favorable subset is selected.</p>
<p>Ticker, circular five-observed-date, and circular ten-observed-date bootstrap checks each use 5,000 draws
and seed 20261123. Each method receives its own complete-family Holm correction. Intervals are nominal 95%
intervals; p_holm is family adjusted. These bootstrap methods are separate sensitivity checks, not a joint
two-way bootstrap. Constant paired differences (including raw selected as calibration) have no estimable
test or confidence interval. They remain in the family using internal p=1, not evidence of equivalence.</p>
{"".join(test_sections)}
<p>Inference is conditional on the saved forecasts; training and selection are not bootstrapped. These years
have been examined in earlier project work, so this is a fixed retrospective comparison, not a newly untouched
test. Nonsignificance does not prove zero information. A significant feature increment would also not establish
profitability. No trading rule or model deployment is selected in this report.</p>
<h2>Files and reproducibility</h2><p>{download}</p>
<p class="small">Original data, prior results and model versions are preserved. The report performs no model,
calibration or embedding fits. The receipt records input/output hashes and the frozen statistical helper.</p>
</body></html>"""
    (out / "REPORT_EN.html").write_text(body, encoding="utf-8")


def main():
    """Parse this module's command-line options and execute the explicitly requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    activate_legacy_root(ROOT)
    out = require_run_destination(args.out)
    bind_run_configuration(out)
    dev, evaluation, lock, locked_models, cal_scores, input_hashes, frozen = load(out)
    metric_table, baseline_table, tests = build_tests(evaluation)
    dev_rows = []
    for horizon in HORIZONS:
        for group in GROUPS:
            part = dev[(dev.horizon == horizon) & (dev.group == group)]
            dev_rows.append(
                metrics(
                    part,
                    part.prediction.to_numpy(),
                    horizon,
                    group,
                    "2023-2024",
                    "raw_selection_diagnostic",
                )
            )
    dev_metrics = pd.DataFrame(dev_rows)
    conclusion = dict(
        primary_horizon="hour",
        extension_horizons=["half", "full"],
        main_metric="raw MSE",
        direction_secondary=True,
        feature_groups=GROUP_NAMES,
        evaluation_events_per_model=497,
        evaluation_year_counts={"2025": 238, "2026": 259},
        partial_2026=True,
        last_evaluation_event="2026-10-02",
        development_selection_events=463,
        calibration_selection_events=225,
        calibration_selection_year=2024,
        calibration_fit_events=463,
        all_decisions_development_only=True,
        later_stopping_or_calibration_updates=False,
        retrospective_evaluation=True,
        llm_gain_is_package_gain=True,
        semantic_content_is_not_isolated=True,
        inference_conditional_on_saved_forecasts=True,
        no_automatic_deployment=True,
        llm_calls=0,
        embedding_calls=0,
        report_fit_calls=0,
        bootstrap_seed=SEED,
        bootstrap_replicates=REPLICATES,
    )
    for kind in ("mse", "direction"):
        conclusion["raw_" + kind] = {
            horizon: summarize(
                tests[kind][tests[kind].horizon.eq(horizon)],
                tests[kind + "_bootstrap"][tests[kind + "_bootstrap"].horizon.eq(horizon)],
            )
            for horizon in HORIZONS
        }
    conclusion["calibration_mse"] = summarize(tests["calibration"], tests["calibration_bootstrap"])
    primary_id = "mse__hour__2025-2026__MFL_vs_MF"
    primary = tests["mse"].set_index("test_id").loc[primary_id]
    boot_primary = tests["mse_bootstrap"][tests["mse_bootstrap"].test_id.eq(primary_id)]
    conclusion["primary_one_hour_llm_increment"] = dict(
        test_id=primary_id,
        effect_pp2=float(primary.effect_pp2),
        p_holm=float(primary.p_holm),
        decision=primary.decision,
        all_bootstrap_methods_agree=bool(boot_primary.decision.eq(primary.decision).all()),
    )
    outputs = {
        "METRICS.csv": metric_table,
        "BASELINE_METRICS.csv": baseline_table,
        "DEV_METRICS.csv": dev_metrics,
        "LOCKED_MODELS.csv": locked_models,
        "CALIBRATION_SELECTION.csv": cal_scores,
        "RAW_MSE_TESTS.csv": tests["mse"],
        "RAW_DIRECTION_TESTS.csv": tests["direction"],
        "CALIBRATION_MSE_TESTS.csv": tests["calibration"],
        "RAW_MSE_BOOTSTRAP.csv": tests["mse_bootstrap"],
        "RAW_DIRECTION_BOOTSTRAP.csv": tests["direction_bootstrap"],
        "CALIBRATION_MSE_BOOTSTRAP.csv": tests["calibration_bootstrap"],
    }
    for name, data in outputs.items():
        data.to_csv(out / name, index=False)
    write(out / "CONCLUSION.json", conclusion)
    make_html(
        out,
        metric_table,
        baseline_table,
        tests,
        dev_metrics,
        cal_scores,
        locked_models,
        clean(conclusion),
    )
    for path, digest in frozen.items():
        require((sha(path) == digest), path)
    names = list(outputs) + ["CONCLUSION.json", "REPORT_EN.html"]
    write(
        out / "REPORT_COMPLETE.json",
        dict(
            complete=True,
            input_sha256=input_hashes,
            output_sha256={name: sha(out / name) for name in names},
            script_sha256=sha(__file__),
            statistical_helper_sha256=sha(HELPER),
            metric_rows=len(metric_table),
            baseline_rows=len(baseline_table),
            development_metric_rows=len(dev_metrics),
            raw_mse_tests=len(tests["mse"]),
            raw_direction_tests=len(tests["direction"]),
            calibration_mse_tests=len(tests["calibration"]),
            bootstrap_rows=sum(
                len(tests[k + "_bootstrap"]) for k in ("mse", "direction", "calibration")
            ),
            bootstrap_seed=SEED,
            bootstrap_replicates=REPLICATES,
            report_model_fits=0,
            report_calibration_fits=0,
            original_files_unchanged=True,
        ),
    )
    print(json.dumps(clean(conclusion["primary_one_hour_llm_increment"])), flush=True)


if __name__ == "__main__":
    main()

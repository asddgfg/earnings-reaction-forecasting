"""Development-only NN selection and locked later-period reporting; no fitting."""

import argparse
import html
import json
import os
from pathlib import Path

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
import numpy as np
import pandas as pd

ROOT = get_settings().research_root


def holm(*args, **kwargs):
    """Load preserved multiplicity code only when a real report requests it."""
    helper = load_legacy_module("work.model_selection_20261118.statisticalinference", ROOT)
    return helper.holm(*args, **kwargs)


def paired_cluster_bootstrap(*args, **kwargs):
    """Dispatch to the unchanged private bootstrap after its root has been checked."""
    helper = load_legacy_module("work.model_selection_20261118.statisticalinference", ROOT)
    return helper.paired_cluster_bootstrap(*args, **kwargs)


def paired_cluster_test(*args, **kwargs):
    """Dispatch to the unchanged private cluster test without import-time data access."""
    helper = load_legacy_module("work.model_selection_20261118.statisticalinference", ROOT)
    return helper.paired_cluster_test(*args, **kwargs)


OUT = get_settings().output_root / "nn_tuning"
PERIODS = ["2025", "2026", "2025-2026"]
MSE_BASES = ["original_nn", "zero_AR", "historical_mean", "market_ridge"]
DIR_BASES = ["chance_50", "historical_majority"]
METHODS = ["ticker", "circular_5_observed_dates", "circular_10_observed_dates"]
SEED = 20261118
NAMES = dict(
    selected_nn="Development-selected numeric NN",
    original_nn="Original NN pipeline",
    zero_AR="Zero AR",
    historical_mean="Earlier-history mean AR",
    market_ridge="Market Ridge",
)


def write(p, v):
    """Atomically publish the historical JSON schema in the isolated run directory."""
    write_json(p, clean(v))


def table(d, digits=6):
    """Render a pandas table for the human-readable report without changing its data."""
    return (
        '<div class="scroll">'
        + d.to_html(
            index=False, border=0, na_rep="Not estimable", float_format=lambda v: f"{v:.{digits}f}"
        )
        + "</div>"
    )


def verify_frozen(out):
    """Check every recorded input digest before permitting another experiment stage."""
    frozen = read(out / "FROZEN_INPUTS.json")
    for p, d in frozen.items():
        require((sha(p) == d), p)
    return frozen


def load(out):
    # Every later-period read is gated by a completed, hash-verified selection.
    """Read completed prediction artifacts only after checking source and result receipts."""
    lockreceipt = read(out / "SELECTION_COMPLETE.json")
    require(
        (sha(out / "LOCKED_SELECTION.json") == lockreceipt["locked_selection_sha256"]),
        "Invariant failed: sha(out / 'LOCKED_SELECTION.json') == lockreceipt['locked_selection_sha256']",
    )
    require(
        (sha(out / "PROTOCOL.json") == lockreceipt["protocol_sha256"]),
        "Invariant failed: sha(out / 'PROTOCOL.json') == lockreceipt['protocol_sha256']",
    )
    lock = read(out / "LOCKED_SELECTION.json")
    require(
        (lock["development_only"] and lock["no_evaluation_outcomes_used"]),
        "Invariant failed: lock['development_only'] and lock['no_evaluation_outcomes_used']",
    )
    require(
        (isinstance(lock["selected_epoch"], int) and 1 <= lock["selected_epoch"] <= 40),
        "Invariant failed: isinstance(lock['selected_epoch'], int) and 1 <= lock['selected_epoch'] <= 40",
    )
    frozen = verify_frozen(out)
    dev = pd.read_csv(out / "DEV_SCORES.csv", float_precision="round_trip")
    require(
        (len(dev) == 12 and dev.recipe_id.is_unique and dev.n.eq(463).all()),
        "Invariant failed: len(dev) == 12 and dev.recipe_id.is_unique and dev.n.eq(463).all()",
    )
    require(
        (dev.is_original.astype(bool).sum() == 1),
        "Invariant failed: dev.is_original.astype(bool).sum() == 1",
    )
    require(
        (np.isfinite(dev[["mse", "rmse_pp", "mae_pp", "accuracy_pct"]].to_numpy(float)).all()),
        "Invariant failed: np.isfinite(dev[['mse', 'rmse_pp', 'mae_pp', 'accuracy_pct']].to_numpy(float)).all()",
    )
    dev = (
        dev.assign(original_tie_order=np.where(dev.is_original, 0, 1))
        .sort_values(["mse", "original_tie_order", "numeric_width", "head_width", "recipe_id"])
        .reset_index(drop=True)
    )
    require(
        (str(dev.iloc[0].recipe_id) == lock["selected_recipe_id"]),
        "Invariant failed: str(dev.iloc[0].recipe_id) == lock['selected_recipe_id']",
    )
    np.testing.assert_allclose(dev.iloc[0].mse, lock["validation_mse"], rtol=1e-12, atol=1e-16)
    for k in ["numeric_width", "head_width", "dropout", "weight_decay"]:
        np.testing.assert_allclose(
            dev.iloc[0][k], lock["selected_recipe"][k], rtol=1e-12, atol=1e-15
        )
    receipt = read(out / "EVAL_COMPLETE.json")
    require(
        (sha(out / "EVALUATION_PREDICTIONS.parquet") == receipt["prediction_sha256"]),
        "Invariant failed: sha(out / 'EVALUATION_PREDICTIONS.parquet') == receipt['prediction_sha256']",
    )
    if "locked_selection_sha256" in receipt:
        require(
            (receipt["locked_selection_sha256"] == lockreceipt["locked_selection_sha256"]),
            "Invariant failed: receipt['locked_selection_sha256'] == lockreceipt['locked_selection_sha256']",
        )
    if "data_sha256" in receipt:
        require(
            (sha(out / "EVALUATION_DATA.parquet") == receipt["data_sha256"]),
            "Invariant failed: sha(out / 'EVALUATION_DATA.parquet') == receipt['data_sha256']",
        )
    data = pd.read_parquet(out / "EVALUATION_PREDICTIONS.parquet")
    keys = ["event_id", "ticker", "reaction_session", "quarter", "year", "y"]
    nums = [
        "y",
        "prediction",
        "original_nn_prediction",
        "market_ridge_prediction",
        "history_mean",
        "history_majority_sign",
    ]
    require(
        (set(keys + nums) <= set(data.columns)),
        "Invariant failed: set(keys + nums) <= set(data.columns)",
    )
    require(
        (len(data) == 497 and data.event_id.is_unique),
        "Invariant failed: len(data) == 497 and data.event_id.is_unique",
    )
    data["reaction_session"] = pd.to_datetime(data.reaction_session).dt.strftime("%Y-%m-%d")
    data = data.sort_values(["reaction_session", "event_id"]).reset_index(drop=True)
    dates = pd.to_datetime(data.reaction_session)
    require(
        (np.array_equal(dates.dt.year.to_numpy(), data.year.to_numpy())),
        "Invariant failed: np.array_equal(dates.dt.year.to_numpy(), data.year.to_numpy())",
    )
    require(
        ((dates.dt.to_period("Q").astype(str) == data.quarter).all()),
        "Invariant failed: (dates.dt.to_period('Q').astype(str) == data.quarter).all()",
    )
    require(
        (data.year.value_counts().to_dict() == {2025: 238, 2026: 259}),
        "Invariant failed: data.year.value_counts().to_dict() == {2025: 238, 2026: 259}",
    )
    require(
        (data.reaction_session.max() == "2026-10-02"),
        "Invariant failed: data.reaction_session.max() == '2026-10-02'",
    )
    require(
        (np.isfinite(data[nums].to_numpy(float)).all()),
        "Invariant failed: np.isfinite(data[nums].to_numpy(float)).all()",
    )
    require(
        (
            data[["y", "prediction", "original_nn_prediction", "market_ridge_prediction"]]
            .ne(0)
            .all()
            .all()
        ),
        "Invariant failed: data[['y', 'prediction', 'original_nn_prediction', 'market_ridge_prediction']].ne(0).all().all()",
    )
    source = pd.read_parquet(out / "EVALUATION_DATA.parquet", columns=keys)
    source["reaction_session"] = pd.to_datetime(source.reaction_session).dt.strftime("%Y-%m-%d")
    source = source.sort_values(["reaction_session", "event_id"]).reset_index(drop=True)
    require(
        (len(source) == 1341 and source.event_id.is_unique and source.y.ne(0).all()),
        "Invariant failed: len(source) == 1341 and source.event_id.is_unique and source.y.ne(0).all()",
    )
    require((np.isfinite(source.y).all()), "Invariant failed: np.isfinite(source.y).all()")
    pd.testing.assert_frame_equal(
        data[keys],
        source[source.year.ge(2025)][keys].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    baseline = []
    for q, g in data.groupby("quarter", sort=True):
        h = source[source.quarter.lt(q)]
        require(
            (h.reaction_session.max() < g.reaction_session.min()),
            "Invariant failed: h.reaction_session.max() < g.reaction_session.min()",
        )
        require(
            (set(h.event_id).isdisjoint(g.event_id)),
            "Invariant failed: set(h.event_id).isdisjoint(g.event_id)",
        )
        sign = 1 if 2 * int(h.y.gt(0).sum()) >= len(h) else -1
        mean = float(h.y.mean())
        require(
            (g.history_majority_sign.eq(sign).all()),
            "Invariant failed: g.history_majority_sign.eq(sign).all()",
        )
        np.testing.assert_allclose(g.history_mean, mean, rtol=1e-12, atol=1e-14)
        baseline.append(
            dict(
                quarter=q,
                history_events=len(h),
                history_positive=int(h.y.gt(0).sum()),
                majority_sign=sign,
                history_mean=mean,
                history_end=h.reaction_session.max(),
                evaluation_start=g.reaction_session.min(),
                evaluation_events=len(g),
            )
        )
    require((len(baseline) == 8), "Invariant failed: len(baseline) == 8")
    names = [
        "PROTOCOL.json",
        "LOCKED_SELECTION.json",
        "SELECTION_COMPLETE.json",
        "DEV_SCORES.csv",
        "EVAL_COMPLETE.json",
        "EVALUATION_PREDICTIONS.parquet",
        "EVALUATION_DATA.parquet",
        "FROZEN_INPUTS.json",
    ]
    nested = None
    if (out / "NESTED_DEV_PREDICTIONS.parquet").exists():
        names.append("NESTED_DEV_PREDICTIONS.parquet")
        n = pd.read_parquet(out / "NESTED_DEV_PREDICTIONS.parquet")
        require(
            (
                len(n) == 463
                and n.event_id.is_unique
                and np.isfinite(n[["y", "prediction"]].to_numpy()).all()
            ),
            "Invariant failed: len(n) == 463 and n.event_id.is_unique and np.isfinite(n[['y', 'prediction']].to_numpy()).all()",
        )
        require(
            (pd.to_datetime(n.reaction_session).min() >= pd.Timestamp("2023-01-01")),
            "Invariant failed: pd.to_datetime(n.reaction_session).min() >= pd.Timestamp('2023-01-01')",
        )
        require(
            (pd.to_datetime(n.reaction_session).max() < pd.Timestamp("2025-01-01")),
            "Invariant failed: pd.to_datetime(n.reaction_session).max() < pd.Timestamp('2025-01-01')",
        )
        expected = (
            source[source.year.between(2023, 2024)][["event_id", "y"]]
            .sort_values("event_id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            n[["event_id", "y"]].sort_values("event_id").reset_index(drop=True),
            expected,
            check_dtype=False,
            check_exact=True,
        )
        error = n.prediction.to_numpy() - n.y.to_numpy()
        nested = dict(
            n=len(n),
            mse=float(np.mean(error**2)),
            rmse_pp=100 * float(np.sqrt(np.mean(error**2))),
            mae_pp=100 * float(np.mean(abs(error))),
            accuracy_pct=100 * float((n.prediction.gt(0) == n.y.gt(0)).mean()),
            used_for_global_selection=False,
        )
    return data, pd.DataFrame(baseline), dev, lock, nested, {n: sha(out / n) for n in names}, frozen


def metrics(g, p, system, period):
    """Compute AR error and direction summaries on the supplied matched event cohort.

    AR is expressed in decimal returns; percentage-point displays multiply errors by 100.
    """
    y = g.y.to_numpy(float)
    e = p - y
    correct = (p > 0) == (y > 0)
    has_direction = bool((p != 0).all())
    recalls = [correct[(y > 0) == s].mean() for s in [False, True] if ((y > 0) == s).any()]
    majority = (g.history_majority_sign.to_numpy() > 0) == (y > 0)
    return dict(
        period=period,
        system=system,
        model=NAMES[system],
        n=len(g),
        mse=float(np.mean(e**2)),
        rmse_pp=100 * float(np.sqrt(np.mean(e**2))),
        mae_pp=100 * float(np.mean(abs(e))),
        bias_pp=100 * float(e.mean()),
        correct=int(correct.sum()) if has_direction else None,
        accuracy_pct=100 * float(correct.mean()) if has_direction else None,
        balanced_accuracy_pct=100 * float(np.mean(recalls)) if has_direction else None,
        majority_correct=int(majority.sum()),
        majority_accuracy_pct=100 * float(majority.mean()),
        prediction_mean_pp=100 * float(p.mean()),
        prediction_sd_pp=100 * float(p.std(ddof=0)),
        predicted_positive=int((p > 0).sum()),
        predicted_negative=int((p < 0).sum()),
        predicted_zero=int((p == 0).sum()),
        actual_positive=int((y > 0).sum()),
        actual_negative=int((y < 0).sum()),
        ticker_clusters=g.ticker.nunique(),
        date_clusters=g.reaction_session.nunique(),
        first_event=g.reaction_session.min(),
        last_event=g.reaction_session.max(),
    )


def decorate(rows, metric, scope):
    """Attach the predeclared multiple-testing family and Holm-adjusted interpretation."""
    unit, mult = ("pp2", 10000) if metric == "mse" else ("pp", 100)
    for r in rows:
        for k in ["effect", "ci_low", "ci_high"]:
            r[k + "_" + unit] = r[k] * mult
        r["metric"] = metric
        r["correction_scope"] = scope
        r["decision"] = (
            "inconclusive"
            if not r["valid"]
            else ("improvement" if r["effect"] > 0 else "deterioration")
            if r["reject_holm"]
            else "not_significant"
        )
    return rows


def interpretation(rows, boots):
    """Translate matched statistical results into a conservative research conclusion."""
    good = [r["test_id"] for r in rows if r["decision"] == "improvement"]
    return dict(
        significant_improvements=good,
        significant_deteriorations=[r["test_id"] for r in rows if r["decision"] == "deterioration"],
        inconclusive_tests=[r["test_id"] for r in rows if r["decision"] == "inconclusive"],
        improvements_supported_by_all_bootstrap_methods=[
            x
            for x in good
            if all(
                next(b for b in boots if b["test_id"] == x and b["method"] == m)["decision"]
                == "improvement"
                for m in METHODS
            )
        ],
        cluster_bootstrap_decision_disagreements=[
            r["test_id"]
            for r in rows
            if any(b["decision"] != r["decision"] for b in boots if b["test_id"] == r["test_id"])
        ],
    )


def report(out=OUT):
    """Generate the fixed model comparison tables and report from existing forecasts only."""
    out = require_run_destination(out)
    bind_run_configuration(out)
    out = Path(out)
    data, baseline, dev, lock, nested, inputs, frozen = load(out)
    summaries = []
    mt = []
    dt = []
    mb = []
    db = []
    for period in PERIODS:
        years = list(map(int, period.split("-")))
        g = data[data.year.between(min(years), max(years))]
        y = g.y.to_numpy(float)
        predictions = dict(
            selected_nn=g.prediction.to_numpy(float),
            original_nn=g.original_nn_prediction.to_numpy(float),
            zero_AR=np.zeros(len(g)),
            historical_mean=g.history_mean.to_numpy(float),
            market_ridge=g.market_ridge_prediction.to_numpy(float),
        )
        summaries.extend(metrics(g, p, s, period) for s, p in predictions.items())
        candidate_loss = (predictions["selected_nn"] - y) ** 2
        for comparator in MSE_BASES:
            loss = (predictions[comparator] - y) ** 2
            identity = dict(
                test_id=period + "__mse__" + comparator,
                period=period,
                candidate="selected_nn",
                comparator=comparator,
                metric="mse",
                candidate_mse=float(candidate_loss.mean()),
                comparator_mse=float(loss.mean()),
            )
            difference = loss - candidate_loss
            row = paired_cluster_test(difference, g.ticker, g.reaction_session)
            row.update(identity)
            mt.append(row)
            for b in paired_cluster_bootstrap(
                difference, g.ticker, g.reaction_session, reps=5000, seed=SEED
            ):
                b.update(identity)
                mb.append(b)
        correct = ((predictions["selected_nn"] > 0) == (y > 0)).astype(float)
        majority = ((g.history_majority_sign.to_numpy() > 0) == (y > 0)).astype(float)
        for comparator, difference in dict(
            chance_50=correct - 0.5, historical_majority=correct - majority
        ).items():
            identity = dict(
                test_id=period + "__direction__" + comparator,
                period=period,
                candidate="selected_nn",
                comparator=comparator,
                metric="direction",
            )
            row = paired_cluster_test(difference, g.ticker, g.reaction_session)
            row.update(identity)
            dt.append(row)
            for b in paired_cluster_bootstrap(
                difference, g.ticker, g.reaction_session, reps=5000, seed=SEED
            ):
                b.update(identity)
                db.append(b)
    require(
        (
            len(mt) == 12
            and len(dt) == 6
            and (len(mb) == 36)
            and (len(db) == 18)
            and (len(summaries) == 15)
        ),
        "Invariant failed: len(mt) == 12 and len(dt) == 6 and (len(mb) == 36) and (len(db) == 18) and (len(summaries) == 15)",
    )
    decorate(holm(mt), "mse", "Primary MSE Holm12: three periods x four fixed comparators")
    decorate(
        holm(dt), "direction", "Secondary direction Holm6: three periods x two fixed baselines"
    )
    for method in METHODS:
        decorate(
            holm([r for r in mb if r["method"] == method]),
            "mse",
            "MSE Holm12 within this bootstrap method",
        )
        decorate(
            holm([r for r in db if r["method"] == method]),
            "direction",
            "Direction Holm6 within this bootstrap method",
        )
    summary = pd.DataFrame(summaries)
    compact = summary[summary.system.eq("selected_nn")].copy()
    ml = {r["test_id"]: r for r in mt}
    dl = {r["test_id"]: r for r in dt}
    for c in MSE_BASES:
        compact[c + "_mse"] = [ml[p + "__mse__" + c]["comparator_mse"] for p in PERIODS]
        compact["mse_vs_" + c + "_p_holm12"] = [ml[p + "__mse__" + c]["p_holm"] for p in PERIODS]
        compact["mse_vs_" + c + "_decision"] = [ml[p + "__mse__" + c]["decision"] for p in PERIODS]
    for c in DIR_BASES:
        compact["direction_vs_" + c + "_p_holm6"] = [
            dl[p + "__direction__" + c]["p_holm"] for p in PERIODS
        ]
        compact["direction_vs_" + c + "_decision"] = [
            dl[p + "__direction__" + c]["decision"] for p in PERIODS
        ]
    compact["accuracy_nominal_ci_low_pct"] = [
        50 + dl[p + "__direction__chance_50"]["ci_low_pp"] for p in PERIODS
    ]
    compact["accuracy_nominal_ci_high_pct"] = [
        50 + dl[p + "__direction__chance_50"]["ci_high_pp"] for p in PERIODS
    ]
    mi = interpretation(mt, mb)
    di = interpretation(dt, db)
    original = dev[dev.is_original.astype(bool)].iloc[0]
    improvement = 100 * (float(original.mse) - float(lock["validation_mse"])) / float(original.mse)
    conclusion = dict(
        selected_recipe_id=lock["selected_recipe_id"],
        selected_recipe=lock["selected_recipe"],
        selected_epoch=lock["selected_epoch"],
        selection_criterion=lock["criterion"],
        development_validation_mse=lock["validation_mse"],
        development_mse_gain_vs_original_pct=improvement,
        development_selection_rows=12,
        development_selection_events=463,
        development_only_decisions=True,
        evaluation_used_for_selection=False,
        evaluation_events=497,
        evaluation_year_counts={2025: 238, 2026: 259},
        periods=PERIODS,
        mse=mi,
        direction=di,
        nested_development_diagnostic=nested,
        main_metric="MSE",
        direction_is_secondary=True,
        partial_2026=True,
        last_2026_event="2026-10-02",
        original_nn_comparator="Reused original pipeline; historical stopping policy may differ from new fixed-epoch policy.",
        inference_conditional_on_saved_forecasts=True,
        no_automatic_deployment=True,
        original_files_unchanged=True,
        evaluation_model_selection_changes=0,
        llm_calls=0,
        embedding_calls=0,
        report_fit_calls=0,
    )
    frames = [
        ("METRICS.csv", summary),
        ("YEARLY_RESULTS.csv", compact),
        ("MSE_TESTS.csv", pd.DataFrame(mt)),
        ("DIRECTION_TESTS.csv", pd.DataFrame(dt)),
        ("MSE_BOOTSTRAP_SENSITIVITY.csv", pd.DataFrame(mb)),
        ("DIRECTION_BOOTSTRAP_SENSITIVITY.csv", pd.DataFrame(db)),
        ("BASELINE_AUDIT.csv", baseline),
        ("DEVELOPMENT_RANKING.csv", dev.drop(columns="original_tie_order")),
    ]
    for name, frame in frames:
        frame.to_csv(out / name, index=False)
    write(out / "CONCLUSION.json", conclusion)
    make_html(out, dev, lock, nested, summary, compact, baseline, mt, dt, mb, db, conclusion)
    for name, digest in inputs.items():
        require((sha(out / name) == digest), name)
    require((verify_frozen(out) == frozen), "Invariant failed: verify_frozen(out) == frozen")
    names = [n for n, _ in frames] + ["CONCLUSION.json", "REPORT_EN.html"]
    write(
        out / "REPORT_COMPLETE.json",
        dict(
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            input_sha256=inputs,
            output_sha256={n: sha(out / n) for n in names},
            frozen_inputs_verified=len(frozen),
            report_script_sha256=sha(Path(__file__)),
            inference_script_sha256=sha(
                ROOT / "work/model_selection_20261118/statisticalinference.py"
            ),
            selected_recipe_id=lock["selected_recipe_id"],
            events=497,
            development_recipe_rows=12,
            metric_rows=15,
            mse_tests=12,
            direction_tests=6,
            mse_bootstrap_rows=36,
            direction_bootstrap_rows=18,
            bootstrap_replicates=5000,
            development_only_decisions=True,
            report_model_fit_calls=0,
            llm_calls=0,
            embedding_calls=0,
            original_files_unchanged=True,
            selection_lock_unchanged=True,
        ),
    )
    print(
        compact[
            [
                "period",
                "n",
                "accuracy_pct",
                "majority_accuracy_pct",
                "mse",
                "rmse_pp",
                "original_nn_mse",
                "zero_AR_mse",
                "historical_mean_mse",
                "market_ridge_mse",
            ]
        ].to_string(index=False)
    )
    print("MSE significant improvements: " + str(mi["significant_improvements"]))
    print("Direction significant improvements: " + str(di["significant_improvements"]))
    print("Report complete: " + str(out / "REPORT_EN.html"))


def make_html(out, dev, lock, nested, summary, compact, baseline, mt, dt, mb, db, conclusion):
    """Render the experiment narrative and numeric tables into its standalone HTML report."""
    mse_good = conclusion["mse"]["significant_improvements"]
    dir_good = conclusion["direction"]["significant_improvements"]
    main = (
        "No later-period MSE comparison shows significant improvement after Holm12 correction."
        if not mse_good
        else f"{len(mse_good)} of twelve MSE comparisons show improvement after Holm12 correction."
    )
    directional = (
        "No direction comparison shows significant improvement after Holm6 correction."
        if not dir_good
        else f"{len(dir_good)} of six direction comparisons show improvement after Holm6 correction."
    )
    disagreement = (
        conclusion["mse"]["cluster_bootstrap_decision_disagreements"]
        + conclusion["direction"]["cluster_bootstrap_decision_disagreements"]
    )
    robust = (
        "Cluster and bootstrap decisions agree for all contrasts."
        if not disagreement
        else "Cluster and bootstrap decisions differ for: "
        + ", ".join(disagreement)
        + ". All methods are shown below."
    )
    mcols = [
        "period",
        "comparator",
        "effect_pp2",
        "ci_low_pp2",
        "ci_high_pp2",
        "p_raw",
        "p_holm",
        "decision",
    ]
    dcols = [
        "period",
        "comparator",
        "effect_pp",
        "ci_low_pp",
        "ci_high_pp",
        "p_raw",
        "p_holm",
        "decision",
    ]
    selected_display = compact[
        [
            "period",
            "n",
            "correct",
            "accuracy_pct",
            "majority_accuracy_pct",
            "mse",
            "rmse_pp",
            "mae_pp",
            "original_nn_mse",
            "zero_AR_mse",
            "historical_mean_mse",
            "market_ridge_mse",
        ]
    ].rename(
        columns={
            "period": "Period",
            "n": "Events",
            "correct": "Correct",
            "accuracy_pct": "Direction accuracy (%)",
            "majority_accuracy_pct": "History-majority accuracy (%)",
            "mse": "MSE (AR²)",
            "rmse_pp": "RMSE (pp)",
            "mae_pp": "MAE (pp)",
            "original_nn_mse": "Original NN MSE",
            "zero_AR_mse": "Zero-AR MSE",
            "historical_mean_mse": "History-mean MSE",
            "market_ridge_mse": "Market Ridge MSE",
        }
    )
    nested_html = (
        table(pd.DataFrame([nested]))
        + "<p>At each development quarter, this diagnostic chooses a recipe from earlier inner-validation "
        "losses only. It was not used to choose the globally selected recipe.</p>"
        if nested
        else "<p>No separate nested-selection diagnostic file was supplied.</p>"
    )
    links = " · ".join(
        f'<a href="{n}">{html.escape(label)}</a>'
        for n, label in [
            ("YEARLY_RESULTS.csv", "Compact annual results"),
            ("DEVELOPMENT_RANKING.csv", "All twelve development recipes"),
            ("METRICS.csv", "All metrics"),
            ("MSE_TESTS.csv", "MSE tests"),
            ("DIRECTION_TESTS.csv", "Direction tests"),
            ("MSE_BOOTSTRAP_SENSITIVITY.csv", "MSE bootstrap"),
            ("DIRECTION_BOOTSTRAP_SENSITIVITY.csv", "Direction bootstrap"),
            ("BASELINE_AUDIT.csv", "Historical baselines"),
            ("LOCKED_SELECTION.json", "Selection lock"),
            ("PROTOCOL.json", "Protocol"),
        ]
    )
    chosen = html.escape(json.dumps(lock["selected_recipe"], sort_keys=True))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Development-only NN tuning</title>
<style>body{{font:16px Arial,sans-serif;line-height:1.55;max-width:1240px;margin:36px auto;padding:0 24px;color:#193044}}
h1,h2{{color:#12324c}}h2{{margin-top:30px}}.note{{padding:18px;background:#edf3f7;border-left:4px solid #267293}}
.small{{font-size:14px;color:#53697a}}.scroll{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;font-size:13px;width:100%}}
td,th{{padding:9px;border-bottom:1px solid #d7e0e8;text-align:right;vertical-align:top}}td:first-child,th:first-child{{text-align:left}}
th{{background:#edf3f7}}pre{{white-space:pre-wrap;font-size:13px}}a{{color:#155c83}}
@media print{{body{{margin:0;max-width:none}}.scroll{{overflow:visible}}tr{{break-inside:avoid}}}}</style></head><body>
<h1>Development-only neural-network tuning</h1>
<p class="small">23 November 2026 · Batch 032 · One-hour signed IWV-adjusted AR · 12 fixed recipes · 497 later-period events</p>
<div class="note"><b>{main}</b><br>{directional}<br>{html.escape(robust)}</div>
<h2>Decision rule and locked configuration</h2>
<p>All model-selection decisions in this experiment use data through 2024 only. The winning recipe minimizes
pooled 2023–2024 development MSE over the twelve prespecified recipes. Exact ties prefer the original recipe,
then smaller capacity, then recipe ID. The final training epoch count is chosen from an inner chronological
split of all 844 development events and locked before any 2025–2026 evaluation loading.</p>
<p><b>Selected recipe:</b> {html.escape(lock["selected_recipe_id"])}.
<b>Fixed evaluation epochs:</b> {lock["selected_epoch"]}.
<b>Development MSE:</b> {float(lock["validation_mse"]):.10f}.
<b>Observed development MSE gain versus original:</b> {conclusion["development_mse_gain_vs_original_pct"]:+.3f}%.</p>
<pre>{chosen}</pre>
<h2>All twelve development results</h2>{table(dev.drop(columns="original_tie_order"), 8)}
<p>The grid combines numeric/head widths 8/16, 16/32 and 32/64; dropout 0.2 or 0.4; and weight decay
0.05 or 0.2. Learning rate 0.002, maximum 40 epochs, patience 5, minimum 5 epochs, batch size 128,
gradient clipping 5 and seeds 42/43 remain fixed. Two seed predictions are averaged; no seed is selected.
Each recipe’s development-quarter stopping choice uses earlier inner data, followed by refitting on all
earlier history. The original recipe reuses its validated forecasts, with a separate smoke reproduction
check. These development scores were used to choose the recipe; they are selection results, not an
unbiased estimate of the chosen model’s performance.</p>
<h2>Earlier-inner-selection diagnostic</h2>{nested_html}
<h2>Locked 2025–2026 performance</h2>{table(selected_display, 8)}
<p>MSE uses squared decimal AR; RMSE and MAE use percentage points. The 2026 sample ends on
02 October 2026 and is not a full year. Its fourth quarter contains one event. Every system is scored
on the same 497 events. The zero-AR reference makes no directional call.</p>
<h2>Primary MSE significance: twelve planned comparisons</h2>{table(pd.DataFrame(mt)[mcols])}
<p>Effects are comparator squared error minus selected-network squared error; positive values favor
the selected network. Effects and confidence intervals use squared percentage points (pp²).
One Holm12 correction covers three periods × four fixed comparators. Confidence intervals are nominal
95%, not simultaneous family-adjusted intervals.</p>
<h2>Secondary direction significance: six comparisons</h2>{table(pd.DataFrame(dt)[dcols])}
<p>Effects are gains in direction accuracy in percentage points. Direction uses the unchanged zero threshold.
A separately prespecified Holm6 family covers three periods × two baselines: 50% and the causal historical
majority. Secondary direction results do not override MSE selection.</p>
<h2>MSE bootstrap sensitivity: every planned method</h2>
{table(pd.DataFrame(mb)[["period", "comparator", "method", "effect_pp2", "ci_low_pp2", "ci_high_pp2", "p_raw", "p_holm", "decision"]])}
<h2>Direction bootstrap sensitivity: every planned method</h2>
{table(pd.DataFrame(db)[["period", "comparator", "method", "effect_pp", "ci_low_pp", "ci_high_pp", "p_raw", "p_holm", "decision"]])}
<p>The primary tests use two-sided paired stock/reaction-date cluster t statistics. Bootstrap sensitivities
use 5,000 resamples with seed {SEED}: ticker resampling and separate circular blocks of five or ten observed
reaction dates. Each method corrects its complete MSE or direction family separately. These are separate
dependence assumptions, not a joint two-way bootstrap. No favorable method is selected.
Inference conditions on saved forecasts and does not resample model fitting or prior research choices.
Constant differences or unusable variance estimates receive no estimated p-value or confidence interval;
their Holm bookkeeping entry remains in the family rather than silently dropping the comparison.</p>
<h2>Quarterly information boundaries and historical baselines</h2>{table(baseline, 8)}
<p>The new network uses the locked recipe and fixed epoch count for every later quarter. Weights and
training-only preprocessing are refitted with all strictly earlier eligible events. Earlier elapsed
2025–2026 quarters may enter later training histories under this fixed rule; the current and future quarters
do not. No later-period outcome selects hyperparameters, epochs, thresholds, features or calibration.</p>
<p>Historical mean and majority use the same earlier history; majority ties are positive. The baseline sign
is global, not per-stock, and is never chosen from the scored quarter. The original NN comparison reuses
the original forecasting pipeline, including its historical stopping policy. The comparison is therefore
between complete forecasting policies, not a pure architecture or single-parameter ablation.</p>
<h2>Complete benchmark metrics</h2>
{table(summary[["period", "model", "n", "mse", "rmse_pp", "mae_pp", "accuracy_pct", "balanced_accuracy_pct", "bias_pp", "prediction_sd_pp"]], 8)}
<h2>Interpretation and preservation</h2>
<p>This run separates development-only decisions from later-period evaluation. The 2025–2026 years were
reported in earlier project work; this report documents a new locked evaluation and makes no claim that
these years are newly observed blind data. They are not used to revise this run’s recipe or search space.
The existing common cohort and cached features are preserved; historical vendor vintages are not newly certified.</p>
<p>A lower observed MSE alone does not establish a significant improvement. Nonsignificance is not proof of
equivalence or impossibility. Direction accuracy is distinct from return-magnitude accuracy and trading
profitability. No LLM tagging or embedding generation is performed. Old models, data, reports and selection
pointers remain unchanged; the newly selected recipe is recorded for this experiment and is not automatically deployed.</p>
<p><a href="https://cameron.econ.ucdavis.edu/research/CGM_twoway_ALL_13may2008.pdf">Multi-way clustering method</a> ·
<a href="https://www.jmlr.org/papers/v11/cawley10a.html">Model-selection bias</a>.</p>
<h2>Files</h2><p>{links}</p></body></html>"""
    (out / "REPORT_EN.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    activate_legacy_root(ROOT)
    report(args.out)

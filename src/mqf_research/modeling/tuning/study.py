"""Twelve prespecified NN recipes; all choices locked before evaluation access."""

import os

from mqf_research.common.hashing import file_sha256 as sha
from mqf_research.common.serialization import read_json as read
from mqf_research.common.serialization import write_json
from mqf_research.config import get_settings
from mqf_research.modeling._compatibility import (
    activate_legacy_root,
    bind_run_configuration,
    maintained_source_paths,
)
from mqf_research.modeling._validation import ensure_empty_cell, require

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import argparse
import concurrent.futures
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

from mqf_research.modeling.tuning import engine

ROOT = get_settings().research_root

WORK = Path(__file__).resolve().parent
OUT = get_settings().output_root / "nn_tuning"
PARENT = ROOT / "outputs/small_sample_ai_20261118_b032"
OLDNN = ROOT / "outputs/valuation_industry_20261113_b032"
DEVQ = [f"{y}Q{q}" for y in [2023, 2024] for q in range(1, 5)]
EVALQ = [f"{y}Q{q}" for y in [2025, 2026] for q in range(1, 5)]
ORIGINAL = next(r for r in engine.recipes() if r["is_original"])


def write(p, value):
    """Atomically publish the historical JSON schema in the isolated run directory."""
    write_json(p, value, default=str)


def oldfold(q):
    """Locate the read-only historical numeric-model fold used as a reference."""
    return OLDNN / "models/hour/validation" / q / "context_numeric_nn"


def verify_frozen():
    """Check every recorded input digest before permitting another experiment stage."""
    bind_run_configuration(OUT)
    for p, d in read(OUT / "FROZEN_INPUTS.json").items():
        require((sha(p) == d), p)


def protocol():
    """Record the prespecified study design in a new run; reject a conflicting protocol."""
    OUT.mkdir(parents=True, exist_ok=True)
    bind_run_configuration(OUT, create=True)
    spec = dict(
        version="nn_tuning_20261123_b032",
        recipes=engine.recipes(),
        seeds=engine.SEEDS,
        request="Only development-period data choose model/settings. Run fixed12 NN recipes, lock all decisions before2025-2026 evaluation. Preserve old outputs; noLLM.",
        target="One-hour signed IWV-adjusted AR; samebatch03209:25ET origin and09:30-10:30target.",
        features="Exact77numeric+2categorical preonly_numeric_nn fields; noembeddingvectors.3textavailability/count controls retained.",
        architecture="Same original inert384-to32projection/v/u/a initialization and zero32textbranch; numericTanh/headReLU, gridvaries numeric/headwidth, dropout andAdamWweightdecay only.",
        fixed_training=dict(
            learning_rate=0.002,
            optimizer="AdamW",
            batch_size=128,
            gradient_clip=5.0,
            max_epochs=40,
            min_epochs=5,
            patience=5,
            min_delta=0.0,
            seeds=[42, 43],
            aggregation="meanrawARprediction",
        ),
        data="Samecommoncohort;844developmentrows2021-2024,463scored2023-2024.497evaluationrows2025-2026availablethrough2026-10-02.",
        development_quarters=DEVQ,
        evaluation_quarters=EVALQ,
        preprocessing="Training-only median/missing indicators, StandardScalerclipÂ±10 andonehotindustry. Targetmean/populationSD floor1e-6. Freshcore-only transform forinnerstopping andfull-earlier-history transform forrefit.",
        development="For each fixedrecipe/quarter, chooseepochs on strictlyearlierinnerhistory thenfreshrefitallpriorhistory. SameoriginalinnerIDs; >=200core,>=100inneracross>=2quarters. Bothseedforecasts averaged beforeinnerMSE.",
        selection="Lowest pooled raw2023-2024 development MSE over12 recipes. Exactties preferoriginal, thensmallernumericwidth, thenrecipe_id. Direction neverselects. These scores aredevelopmentselection, not an unbiased outertest.",
        nested_diagnostic="For each developmentouterquarter, independentlychoose recipe usingonly earlierinnerbestMSE, thenuse its savedouterforecast. Summarize this nestedselectionpolicy separately; neveruse it tochange search orfinalcriterion.",
        final_epoch="After recipe selection, run identicalinnerstopping rule onall844pre2025developmentrows forwinner;freezeits selectedepoch togetherwithrecipe before anyevaluationtarget access. Noevaluation-period earlystopping.",
        evaluation="Onlylockedwinner,8expanding-quarterweightrefits, fixedrecipe+epochs+seeds. Newlyelapsed2025-2026labels may enterlaterweighttraining only; neverrecipe/epoch/threshold/calibration decisions.",
        original_comparator="Reuseoriginalsavedpreonly_numeric_nn forecasts. Its older pipeline used chronologicalrollingearlystopping; comparisonincludesstopping-policy difference, not a pureparameterablation.",
        baselines=["original_nn", "zero_AR", "history_mean", "market_ridge"],
        inference="Pairedstock/datecluster two-sidedt. PrimaryMSEHolm12 (3periodsÃ—4baselines); separate secondarydirectionHolm6 (3periodsÃ—50%/causalglobalmajority).",
        inference_periods=["2025", "2026", "2025-2026"],
        bootstrap="5000drawsseed20261118; ticker,circular5/10observeddates; fullHolm12MSE/Holm6direction separatelypermethod. No favorablemethod selection.",
        interpretation="Thisrounddecisionsuseonlydevelopment. Evaluationperiodsreportedpreviously; retainfullresearchhistory andcallnewresultslockedtime-out evaluation, not firstunseenblindtest.",
        stopping_budget="Exactly12recipesincludingoriginal;do notexpandsearchorcontinueuntilsignificant.",
        constraints=dict(
            llm_calls=0,
            embedding_calls=0,
            new_feature_search=0,
            target_search=0,
            evaluation_hyperparameter_search=0,
            evaluation_early_stopping=0,
            old_outputs_overwritten=False,
        ),
    )
    path = OUT / "PROTOCOL.json"
    if path.exists():
        previous = read(path)
        previous.pop("recorded_utc")
        require((previous == spec), "Invariant failed: previous == spec")
    else:
        write(path, dict(spec, recorded_utc=pd.Timestamp.now(tz="UTC").isoformat()))
    print("Protocolfixed;12recipes;development-onlydecisions.", flush=True)


def prepare():
    """Prepare the stage's inputs and provenance receipts without using later targets."""
    protocol()
    from mqf_research.modeling.tuning.inputs import load_development, source_paths_development

    if not (OUT / "FROZEN_INPUTS.json").exists():
        preserved = dict(read(PARENT / "FROZEN_INPUTS.json"))
        receipt = read(PARENT / "REPORT_COMPLETE.json")
        for group in ["input_sha256", "output_sha256"]:
            preserved.update({str(PARENT / name): d for name, d in receipt[group].items()})
        for p, d in preserved.items():
            require((sha(p) == d), p)
        paths = (
            source_paths_development()
            + maintained_source_paths()
            + [WORK / n for n in ["study.py", "engine.py", "inputs.py", "report.py"]]
        )
        paths += [
            OUT / "RUN_CONFIG.json",
            OUT / "PROTOCOL.json",
            ROOT / "work/model_selection_20261118/statisticalinference.py",
            ROOT / "work/tabpfn_fixed_20261119_b032/inputs.py",
            ROOT / "work/mil_models_b032.py",
            ROOT / "work/compare_feature_horizons_b032.py",
            ROOT / "work/compare_valuation_industry_20261113.py",
        ]
        for q in DEVQ:
            paths += [
                oldfold(q) / n
                for n in [
                    "INNER_SELECTION.json",
                    "PREDICTIONS.parquet",
                    "seed42.pt",
                    "seed43.pt",
                    "preprocessing.joblib",
                    "COMPLETE.json",
                ]
            ]
        paths += list(ROOT.glob("CURRENT_*.json"))
        # Hash-only preservation of prior evaluation files; no forecasts/labels parsed.
        paths += [
            OLDNN / "EVALUATION_PREDICTIONS.parquet",
            ROOT / "outputs/market_ridge_fixed_20261118_b032/PREDICTIONS.parquet",
            ROOT / "outputs/tabpfn_fixed_20261119_b032/PREDICTIONS.parquet",
            ROOT / "outputs/tabpfn_fixed_20261119_b032/REPORT_EN.html",
        ]
        for p in paths:
            preserved[str(p)] = sha(p)
        write(OUT / "FROZEN_INPUTS.json", preserved)
    verify_frozen()
    if (OUT / "PREPARED.json").exists():
        receipt = read(OUT / "PREPARED.json")
        require(
            (sha(OUT / "DEV_DATA.parquet") == receipt["data_sha256"]),
            "Invariant failed: sha(OUT / 'DEV_DATA.parquet') == receipt['data_sha256']",
        )
        print("Development preparationalreadycomplete.", flush=True)
        return
    data, num, cat, ref = load_development()
    require(
        (data.year.max() == 2024 and len(data) == 844),
        "Invariant failed: data.year.max() == 2024 and len(data) == 844",
    )
    data.to_parquet(OUT / "DEV_DATA.parquet", index=False)
    ref.to_parquet(OUT / "DEV_REFERENCE.parquet", index=False)
    write(OUT / "FEATURES.json", dict(numeric=num, categorical=cat))
    folds = {}
    schedule = []
    from work.small_sample_ai_20261118.inputs import transform_saved_nn

    for q in DEVQ:
        hist = data[data.quarter.lt(q)].copy()
        test = data[data.quarter.eq(q)].copy()
        core, inner = engine.inner_split(hist)
        old = read(oldfold(q) / "INNER_SELECTION.json")
        require(
            (core.event_id.tolist() == old["core_train_ids"]),
            "Invariant failed: core.event_id.tolist() == old['core_train_ids']",
        )
        require(
            (inner.event_id.tolist() == old["inner_validation_ids"]),
            "Invariant failed: inner.event_id.tolist() == old['inner_validation_ids']",
        )
        require(
            (hist.reaction_session.max() < test.reaction_session.min()),
            "Invariant failed: hist.reaction_session.max() < test.reaction_session.min()",
        )
        full, pre = engine.prepare(hist, test, num, cat)
        np.testing.assert_array_equal(full["X_train"], transform_saved_nn(hist, q, hist))
        np.testing.assert_array_equal(full["X_test"], transform_saved_nn(test, q, hist))
        ins, ipre = engine.prepare(core, inner, num, cat)
        ins["y_valid"] = inner.y.to_numpy()
        dest = OUT / "development" / q
        dest.mkdir(parents=True, exist_ok=True)
        ensure_empty_cell(dest)
        np.savez_compressed(dest / "FULL.npz", **full)
        np.savez_compressed(dest / "INNER.npz", **ins)
        joblib.dump(pre, dest / "preprocessing.joblib", compress=3)
        joblib.dump(ipre, dest / "inner_preprocessing.joblib", compress=3)
        audit = dict(
            quarter=q,
            history_ids=hist.event_id.tolist(),
            test_ids=test.event_id.tolist(),
            core_ids=core.event_id.tolist(),
            inner_ids=inner.event_id.tolist(),
            history_end=hist.reaction_session.max(),
            test_start=test.reaction_session.min(),
            source_original=str(oldfold(q)),
            exact_original_input_matrices=True,
        )
        write(dest / "INPUT_AUDIT.json", audit)
        names = [
            "FULL.npz",
            "INNER.npz",
            "preprocessing.joblib",
            "inner_preprocessing.joblib",
            "INPUT_AUDIT.json",
        ]
        folds[q] = {n: sha(dest / n) for n in names}
        schedule.append(
            dict(quarter=q, history=len(hist), core=len(core), inner=len(inner), outer=len(test))
        )
    pd.DataFrame(schedule).to_csv(OUT / "FOLD_SCHEDULE.csv", index=False)
    write(
        OUT / "PREPARED.json",
        dict(
            data_sha256=sha(OUT / "DEV_DATA.parquet"),
            reference_sha256=sha(OUT / "DEV_REFERENCE.parquet"),
            fold_inputs=folds,
            development_only=True,
            future_outcomes_read=False,
            recipes=12,
        ),
    )
    verify_frozen()
    print(pd.DataFrame(schedule).to_string(index=False), flush=True)


def complete(dest, names, **extra):
    """Write a completion receipt only after the listed artifacts have been produced."""
    write(dest / "COMPLETE.json", dict(hashes={n: sha(dest / n) for n in names}, **extra))


def check_complete(dest):
    """Validate every file hash in a completed cell before reusing its outputs."""
    for n, d in read(dest / "COMPLETE.json")["hashes"].items():
        require((sha(dest / n) == d), str(dest / n))


def smoke():
    """Compare the original recipe's synthetic and historical setup before full development."""
    verify_frozen()
    dest = OUT / "original_reproduction"
    dest.mkdir(exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        check_complete(dest)
        return
    ensure_empty_cell(dest)
    root = OUT / "development/2023Q1"
    inner = np.load(root / "INNER.npz", allow_pickle=False)
    full = np.load(root / "FULL.npz", allow_pickle=False)
    selected = engine.select_epochs(inner, inner["y_valid"], ORIGINAL)
    old = read(oldfold("2023Q1") / "INNER_SELECTION.json")
    require(
        (
            selected["selected_epoch"] == old["selected_epoch"]
            and selected["stopped_epoch"] == old["stopped_epoch"]
        ),
        "Invariant failed: selected['selected_epoch'] == old['selected_epoch'] and selected['stopped_epoch'] == old['stopped_epoch']",
    )
    for actual, reference in zip(selected["curve"], old["curve"]):
        for field in ["validation_mse", "training_standardized_loss"]:
            np.testing.assert_allclose(actual[field], reference[field], atol=1e-12, rtol=1e-9)
    p = engine.refit(full, ORIGINAL, selected["selected_epoch"], dest)
    reference = (
        pd.read_parquet(oldfold("2023Q1") / "PREDICTIONS.parquet")
        .set_index("event_id")
        .loc[full["test_ids"]]
        .prediction.to_numpy()
    )
    np.testing.assert_allclose(p, reference, atol=1e-8, rtol=1e-6)
    write(
        dest / "CHECK.json",
        dict(
            passed=True,
            quarter="2023Q1",
            selected_epoch=selected["selected_epoch"],
            max_prediction_difference=float(np.max(abs(p - reference))),
            original_initialization_and_stopping_reproduced=True,
            no_future_outcomes_read=True,
        ),
    )
    complete(dest, ["CHECK.json", "seed42.pt", "seed43.pt", "SEED_PREDICTIONS.npy"])
    print("Originalsettingreproduced;validationsmokecheckpassed.", flush=True)


def fit_development(task):
    """Fit one prespecified recipe/quarter using nested chronological early stopping.

    A matching complete receipt resumes the cell; incomplete or changed artifacts must not be trusted.
    """
    q, recipe = task
    root = OUT / "development" / q
    dest = root / recipe["recipe_id"]
    dest.mkdir(exist_ok=True)
    with FileLock(str(dest / "RUN.lock"), timeout=0):
        if (dest / "COMPLETE.json").exists():
            check_complete(dest)
            return q, recipe["recipe_id"], "cached"
        ensure_empty_cell(dest)
        pre = read(OUT / "PREPARED.json")
        for n, d in pre["fold_inputs"][q].items():
            require((sha(root / n) == d), "Invariant failed: sha(root / n) == d")
        start = time.time()
        full = np.load(root / "FULL.npz", allow_pickle=False)
        if recipe["is_original"]:
            selected = read(oldfold(q) / "INNER_SELECTION.json")
            selected["best_inner_mse"] = min(c["validation_mse"] for c in selected["curve"])
            ref = (
                pd.read_parquet(oldfold(q) / "PREDICTIONS.parquet")
                .set_index("event_id")
                .loc[full["test_ids"]]
            )
            p = ref.prediction.to_numpy()
            origin = dict(
                reused_original=True,
                source=str(oldfold(q)),
                checkpoint_paths=[str(oldfold(q) / f"seed{s}.pt") for s in engine.SEEDS],
            )
            names = ["INNER_SELECTION.json", "PREDICTIONS.npy", "FIT_AUDIT.json"]
        else:
            inner = np.load(root / "INNER.npz", allow_pickle=False)
            selected = engine.select_epochs(inner, inner["y_valid"], recipe)
            p = engine.refit(full, recipe, selected["selected_epoch"], dest)
            names = [
                "INNER_SELECTION.json",
                "PREDICTIONS.npy",
                "FIT_AUDIT.json",
                "SEED_PREDICTIONS.npy",
                "seed42.pt",
                "seed43.pt",
            ]
            origin = dict(reused_original=False)
        write(dest / "INNER_SELECTION.json", selected)
        np.save(dest / "PREDICTIONS.npy", p)
        write(
            dest / "FIT_AUDIT.json",
            dict(
                quarter=q,
                recipe=recipe,
                epochs=selected["selected_epoch"],
                train_ids=full["train_ids"].tolist(),
                test_ids=full["test_ids"].tolist(),
                input_sha256=sha(root / "FULL.npz"),
                inner_sha256=sha(root / "INNER.npz"),
                no_future_outcomes_read=True,
                seconds=time.time() - start,
                **origin,
            ),
        )
        complete(dest, names)
        return q, recipe["recipe_id"], f"{time.time() - start:.1f}s"


def development(workers):
    """Run the declared development cells and collect their matched out-of-time forecasts."""
    verify_frozen()
    smoke()
    tasks = [(q, r) for q in DEVQ for r in engine.recipes()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_development, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            q, name, state = future.result()
            print(f"{q} {name} complete {state}", flush=True)
    verify_frozen()
    write(
        OUT / "DEV_FIT_COMPLETE.json",
        dict(
            folds=8,
            recipes=12,
            cells=96,
            development_only=True,
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )


def ranking_key(row):
    """Rank candidates by development MSE with the predeclared deterministic tie order."""
    return (row["mse"], not row["is_original"], row["numeric_width"], row["recipe_id"])


def select():
    """Lock choices using development forecasts only, before permitting evaluation access."""
    verify_frozen()
    require(
        ((OUT / "DEV_FIT_COMPLETE.json").exists()),
        "Invariant failed: (OUT / 'DEV_FIT_COMPLETE.json').exists()",
    )
    if (OUT / "SELECTION_COMPLETE.json").exists():
        from mqf_research.modeling.tuning.inputs import verify_selection_gate

        verify_selection_gate()
        print("Selectionalreadylocked;unchanged.", flush=True)
        return
    ensure_empty_cell(OUT / "final_development_stopping")
    data = pd.read_parquet(OUT / "DEV_DATA.parquet")
    require((data.year.max() == 2024), "Invariant failed: data.year.max() == 2024")
    allpred = []
    nested = []
    inners = []
    for q in DEVQ:
        test = data[data.quarter.eq(q)][
            ["event_id", "ticker", "reaction_session", "quarter", "year", "y"]
        ].copy()
        candidates = []
        for r in engine.recipes():
            dest = OUT / "development" / q / r["recipe_id"]
            check_complete(dest)
            frame = test.copy()
            frame["prediction"] = np.load(dest / "PREDICTIONS.npy", allow_pickle=False)
            frame["recipe_id"] = r["recipe_id"]
            allpred.append(frame)
            info = read(dest / "INNER_SELECTION.json")
            score = info["best_inner_mse"]
            candidates.append(dict(recipe=r, frame=frame, inner_mse=score))
            inners.append(
                dict(
                    quarter=q,
                    recipe_id=r["recipe_id"],
                    selected_epoch=info["selected_epoch"],
                    stopped_epoch=info["stopped_epoch"],
                    inner_mse=score,
                )
            )
        winner = min(
            candidates,
            key=lambda a: (
                a["inner_mse"],
                not a["recipe"]["is_original"],
                a["recipe"]["numeric_width"],
                a["recipe"]["recipe_id"],
            ),
        )
        nested.append(winner["frame"])
    predictions = pd.concat(allpred, ignore_index=True)
    scores = []
    for r in engine.recipes():
        p = predictions[predictions.recipe_id.eq(r["recipe_id"])]
        require(
            (len(p) == 463 and p.event_id.is_unique),
            "Invariant failed: len(p) == 463 and p.event_id.is_unique",
        )
        error = p.prediction.to_numpy() - p.y.to_numpy()
        scores.append(
            dict(
                **r,
                n=len(p),
                mse=float(np.mean(error**2)),
                rmse_pp=float(np.sqrt(np.mean(error**2)) * 100),
                mae_pp=float(np.mean(abs(error)) * 100),
                accuracy_pct=float(np.mean((p.prediction > 0) == (p.y > 0)) * 100),
            )
        )
    scores = sorted(scores, key=ranking_key)
    pd.DataFrame(scores).to_csv(OUT / "DEV_SCORES.csv", index=False)
    predictions.to_parquet(OUT / "DEV_PREDICTIONS.parquet", index=False)
    pd.concat(nested, ignore_index=True).to_parquet(
        OUT / "NESTED_DEV_PREDICTIONS.parquet", index=False
    )
    pd.DataFrame(inners).to_csv(OUT / "DEV_INNER_SELECTIONS.csv", index=False)
    selected = next(r for r in engine.recipes() if r["recipe_id"] == scores[0]["recipe_id"])
    num = read(OUT / "FEATURES.json")["numeric"]
    cat = read(OUT / "FEATURES.json")["categorical"]
    core, inner = engine.inner_split(data)
    matrix, prep = engine.prepare(core, inner, num, cat)
    final = OUT / "final_development_stopping"
    final.mkdir(exist_ok=True)
    np.savez_compressed(final / "INNER.npz", **matrix, y_valid=inner.y.to_numpy())
    joblib.dump(prep, final / "preprocessing.joblib", compress=3)
    stopping = engine.select_epochs(matrix, inner.y.to_numpy(), selected)
    write(final / "INNER_SELECTION.json", stopping)
    complete(final, ["INNER.npz", "preprocessing.joblib", "INNER_SELECTION.json"])
    locked = dict(
        selected_recipe=selected,
        selected_recipe_id=selected["recipe_id"],
        selected_epoch=stopping["selected_epoch"],
        criterion="Lowestpooled2023-2024developmentMSE",
        validation_mse=scores[0]["mse"],
        development_only=True,
        no_evaluation_outcomes_used=True,
        epoch_selection_ids=matrix["test_ids"].tolist(),
        epoch_core_ids=matrix["train_ids"].tolist(),
        all12_recipes_complete=True,
        evaluation_early_stopping=False,
        seeds=engine.SEEDS,
        selection_utc=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    write(OUT / "LOCKED_SELECTION.json", locked)
    artifacts = [
        "LOCKED_SELECTION.json",
        "DEV_SCORES.csv",
        "DEV_PREDICTIONS.parquet",
        "NESTED_DEV_PREDICTIONS.parquet",
        "DEV_INNER_SELECTIONS.csv",
        "DEV_DATA.parquet",
        "final_development_stopping/COMPLETE.json",
    ]
    write(
        OUT / "SELECTION_COMPLETE.json",
        dict(
            locked_selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
            protocol_sha256=sha(OUT / "PROTOCOL.json"),
            development_only=True,
            no_evaluation_outcomes_used=True,
            hashes={n: sha(OUT / n) for n in artifacts},
            frozen_model_code_sha256={
                n: sha(WORK / n) for n in ["engine.py", "study.py", "inputs.py"]
            },
            locked_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    verify_frozen()
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(
        f"SelectionLOCKED: {selected['recipe_id']}; evaluationepochs={stopping['selected_epoch']}.",
        flush=True,
    )


def evaluation():
    """Evaluate only the locked recipe, duration, and calibration policy.

    Earlier elapsed outcomes may update weights in later quarters; they cannot change model selection.
    """
    verify_frozen()
    from mqf_research.modeling.tuning.inputs import (
        load_evaluation,
        source_paths_evaluation,
        verify_selection_gate,
    )

    verify_selection_gate()
    lock = read(OUT / "LOCKED_SELECTION.json")
    recipe = lock["selected_recipe"]
    epochs = lock["selected_epoch"]
    gate = read(OUT / "SELECTION_COMPLETE.json")
    for n, d in gate["hashes"].items():
        require((sha(OUT / n) == d), n)
    if not (OUT / "EVALUATION_INPUT_FREEZE.json").exists():
        paths = source_paths_evaluation() + [
            OLDNN / "EVALUATION_PREDICTIONS.parquet",
            ROOT / "outputs/market_ridge_fixed_20261118_b032/PREDICTIONS.parquet",
        ]
        write(OUT / "EVALUATION_INPUT_FREEZE.json", {str(p): sha(p) for p in paths})
    for p, d in read(OUT / "EVALUATION_INPUT_FREEZE.json").items():
        require((sha(p) == d), p)
    data, num, cat = load_evaluation()
    data.to_parquet(OUT / "EVALUATION_DATA.parquet", index=False)
    # Existing comparator forecasts are read only after recipe+duration lock.
    original = pd.read_parquet(
        OLDNN / "EVALUATION_PREDICTIONS.parquet",
        filters=[("horizon", "==", "hour"), ("candidate", "==", "context_numeric_nn")],
    ).set_index("event_id")
    ridge = pd.read_parquet(
        ROOT / "outputs/market_ridge_fixed_20261118_b032/PREDICTIONS.parquet",
        filters=[("year", ">=", 2025)],
    ).set_index("event_id")
    require(
        (len(original) == len(ridge) == 497), "Invariant failed: len(original) == len(ridge) == 497"
    )
    parts = []
    for q in EVALQ:
        hist = data[data.quarter.lt(q)].copy()
        test = data[data.quarter.eq(q)].copy()
        require(
            (hist.reaction_session.max() < test.reaction_session.min()),
            "Invariant failed: hist.reaction_session.max() < test.reaction_session.min()",
        )
        dest = OUT / "evaluation" / q
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "COMPLETE.json").exists():
            check_complete(dest)
            parts.append(pd.read_parquet(dest / "PREDICTIONS.parquet"))
            continue
        ensure_empty_cell(dest)
        matrix, prep = engine.prepare(hist, test, num, cat)
        np.savez_compressed(dest / "FULL.npz", **matrix)
        joblib.dump(prep, dest / "preprocessing.joblib", compress=3)
        p = engine.refit(matrix, recipe, epochs, dest)
        frame = test[["event_id", "ticker", "reaction_session", "quarter", "year", "y"]].copy()
        frame["prediction"] = p
        frame["recipe_id"] = recipe["recipe_id"]
        for reference, col in [
            (original, "original_nn_prediction"),
            (ridge, "market_ridge_prediction"),
        ]:
            selected = reference.loc[frame.event_id]
            np.testing.assert_array_equal(selected.y.to_numpy(), frame.y.to_numpy())
            frame[col] = selected.prediction.to_numpy()
        frame["history_mean"] = float(hist.y.mean())
        frame["history_majority_sign"] = 1 if int(hist.y.gt(0).sum()) * 2 >= len(hist) else -1
        frame.to_parquet(dest / "PREDICTIONS.parquet", index=False)
        write(
            dest / "FIT_AUDIT.json",
            dict(
                quarter=q,
                recipe=recipe,
                epochs=epochs,
                seeds=engine.SEEDS,
                history_ids=hist.event_id.tolist(),
                test_ids=test.event_id.tolist(),
                historical_weight_updates_only=True,
                early_stopping_performed=False,
                selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
                recipe_changed=False,
            ),
        )
        complete(
            dest,
            [
                "FULL.npz",
                "preprocessing.joblib",
                "seed42.pt",
                "seed43.pt",
                "SEED_PREDICTIONS.npy",
                "PREDICTIONS.parquet",
                "FIT_AUDIT.json",
            ],
        )
        parts.append(frame)
        print(f"{q}evaluationcomplete n={len(frame)} fixedeepochs={epochs}", flush=True)
    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["reaction_session", "event_id"])
        .reset_index(drop=True)
    )
    require(
        (len(result) == 497 and result.event_id.is_unique),
        "Invariant failed: len(result) == 497 and result.event_id.is_unique",
    )
    result.to_parquet(OUT / "EVALUATION_PREDICTIONS.parquet", index=False)
    verify_frozen()
    verify_selection_gate()
    for n, d in gate["hashes"].items():
        require((sha(OUT / n) == d), n)
    write(
        OUT / "EVAL_COMPLETE.json",
        dict(
            prediction_sha256=sha(OUT / "EVALUATION_PREDICTIONS.parquet"),
            data_sha256=sha(OUT / "EVALUATION_DATA.parquet"),
            events=497,
            selected_recipe_id=recipe["recipe_id"],
            fixed_epochs=epochs,
            locked_selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
            evaluation_hyperparameter_choices=0,
            evaluation_epoch_choices=0,
            old_results_preserved=True,
            llm_calls=0,
            embedding_calls=0,
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    print("Evaluationcompletewithlockedrecipe/duration. Nofurtherselection.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["protocol", "prepare", "smoke", "development", "select", "evaluation"]
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    activate_legacy_root(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT / "RUN.lock"), timeout=0):
        if args.stage == "development":
            development(args.workers)
        else:
            {
                "protocol": protocol,
                "prepare": prepare,
                "smoke": smoke,
                "select": select,
                "evaluation": evaluation,
            }[args.stage]()

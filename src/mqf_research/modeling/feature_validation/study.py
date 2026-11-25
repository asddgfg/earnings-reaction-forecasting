"""Run the prespecified three-feature, three-horizon NN experiment without old-file edits."""

import argparse
import json
import os
import time
from pathlib import Path

from mqf_research.common.hashing import file_sha256 as sha
from mqf_research.common.numeric_json import json_ready
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

from mqf_research.modeling.feature_validation import calibration, engine, inputs

ROOT = get_settings().research_root

WORK = Path(__file__).resolve().parent
OUT = get_settings().output_root / "nn_feature_validation"
OLD = ROOT / "outputs/nn_tuning_20261123_b032"
HORIZONS = ["hour", "half", "full"]
GROUPS = ["M", "MF", "MFL"]
DEVQ = [f"{year}Q{q}" for year in [2023, 2024] for q in range(1, 5)]
EVALQ = [f"{year}Q{q}" for year in [2025, 2026] for q in range(1, 5)]
CACHE = {}


def clean(value):
    """Preserve the cell receipt policy accepting NumPy arrays as JSON lists."""
    return json_ready(value, allow_arrays=True)


def write(path, value):
    """Atomically publish the historical JSON schema in the isolated run directory."""
    write_json(path, clean(value), create_parents=True)


def complete(destination, names):
    """Write a completion receipt only after the listed artifacts have been produced."""
    write(
        destination / "COMPLETE.json",
        dict(
            hashes={n: sha(destination / n) for n in names},
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )


def check_complete(destination):
    """Validate every file hash in a completed cell before reusing its outputs."""
    receipt = read(destination / "COMPLETE.json")
    for name, digest in receipt["hashes"].items():
        require((sha(destination / name) == digest), str(destination / name))
    return receipt


def verify_frozen():
    """Check every recorded input digest before permitting another experiment stage."""
    bind_run_configuration(OUT)
    for path, digest in read(OUT / "FROZEN_INPUTS.json").items():
        require((sha(path) == digest), path)


def cache_paths(stage):
    """List the expected cached data artifacts for the requested authorized stage."""
    prefix = "DEV" if stage == "development" else "EVAL"
    return [
        *[OUT / f"{prefix}_DATA_{h}.parquet" for h in HORIZONS],
        OUT / f"{prefix}_BAGS.json",
        OUT / f"{prefix}_VECTORS.npy",
        OUT / "FEATURE_GROUPS.json",
    ]


def protocol():
    """Copy the frozen scientific design into a new, isolated maintained run.

    The archived experiment had a separately authored protocol. This explicit
    stage reads no outcomes and changes no scientific setting. A conflicting
    existing protocol is rejected rather than silently replaced.
    """
    bind_run_configuration(OUT, create=True)
    source = ROOT / "outputs/nn_feature_validation_20261123_b032/PROTOCOL.json"
    specification = read(source)
    specification["maintenance"] = {
        "source_protocol": str(source),
        "source_protocol_sha256": sha(source),
        "source_code": str(WORK),
        "scientific_settings_unchanged": True,
    }
    target = OUT / "PROTOCOL.json"
    if target.exists():
        require(
            read(target) == specification, "Existing protocol differs; choose a fresh output root"
        )
    else:
        write(target, specification)


def prepare():
    """Prepare the stage's inputs and provenance receipts without using later targets."""
    protocol()
    require(
        ((OUT / "PROTOCOL.json").is_file()), "Invariant failed: (OUT / 'PROTOCOL.json').is_file()"
    )
    require((not (OUT / "PREPARED.json").exists()), "Already prepared; use existing receipts.")
    old = read(OLD / "LOCKED_SELECTION.json")
    require(
        (old["selected_recipe_id"] == "n32_h64_d20_w5"),
        "Invariant failed: old['selected_recipe_id'] == 'n32_h64_d20_w5'",
    )
    for name, value in [
        ("numeric_width", 32),
        ("head_width", 64),
        ("dropout", 0.2),
        ("weight_decay", 0.05),
    ]:
        require(
            (old["selected_recipe"][name] == value),
            "Invariant failed: old['selected_recipe'][name] == value",
        )
    frames, groups, bags, vectors = inputs.load_development(HORIZONS, "minilm")
    require(
        (set(frames) == set(HORIZONS) and set(groups) == set(GROUPS)),
        "Invariant failed: set(frames) == set(HORIZONS) and set(groups) == set(GROUPS)",
    )
    require(
        (len(groups["M"]["numeric"]) == 23 and groups["M"]["categorical"] == []),
        "Invariant failed: len(groups['M']['numeric']) == 23 and groups['M']['categorical'] == []",
    )
    require(
        (len(groups["MF"]["numeric"]) == 74 and len(groups["MFL"]["numeric"]) == 77),
        "Invariant failed: len(groups['MF']['numeric']) == 74 and len(groups['MFL']['numeric']) == 77",
    )
    require(
        (
            set(groups["M"]["numeric"])
            <= set(groups["MF"]["numeric"])
            < set(groups["MFL"]["numeric"])
        ),
        "Invariant failed: set(groups['M']['numeric']) <= set(groups['MF']['numeric']) < set(groups['MFL']['numeric'])",
    )
    controls = {"text_available", "has_signals", "log_signal_count"}
    require(
        (set(groups["MFL"]["numeric"]) - set(groups["MF"]["numeric"]) == controls),
        "Invariant failed: set(groups['MFL']['numeric']) - set(groups['MF']['numeric']) == controls",
    )
    require(
        (not controls & set(groups["MF"]["numeric"])),
        "Invariant failed: not controls & set(groups['MF']['numeric'])",
    )
    require(
        (
            not groups["M"]["use_signals"]
            and (not groups["MF"]["use_signals"])
            and groups["MFL"]["use_signals"]
        ),
        "Invariant failed: not groups['M']['use_signals'] and (not groups['MF']['use_signals']) and groups['MFL']['use_signals']",
    )
    write(OUT / "FEATURE_GROUPS.json", groups)
    write(OUT / "DEV_BAGS.json", bags)
    np.save(OUT / "DEV_VECTORS.npy", vectors)
    for h, data in frames.items():
        require(
            (len(data) == 844 and data.year.le(2024).all() and data.event_id.is_unique),
            "Invariant failed: len(data) == 844 and data.year.le(2024).all() and data.event_id.is_unique",
        )
        data.to_parquet(OUT / f"DEV_DATA_{h}.parquet", index=False)
    same_ids = frames["hour"].event_id.tolist()
    require(
        (all((d.event_id.tolist() == same_ids for d in frames.values()))),
        "Invariant failed: all((d.event_id.tolist() == same_ids for d in frames.values()))",
    )
    require(
        (set(bags) == set(same_ids) and vectors.shape[1] == 384),
        "Invariant failed: set(bags) == set(same_ids) and vectors.shape[1] == 384",
    )
    paths = list(inputs.source_paths(HORIZONS, "minilm"))
    paths += [
        WORK / n for n in ["engine.py", "inputs.py", "calibration.py", "study.py", "report.py"]
    ]
    paths += [OUT / "PROTOCOL.json", OUT / "RUN_CONFIG.json"]
    paths += maintained_source_paths()
    # Preserve the prior run and its already recorded underlying originals, hash only.
    paths += [Path(p) for p in read(OLD / "FROZEN_INPUTS.json")]
    paths += [
        p
        for p in OLD.rglob("*")
        if p.is_file() and p.name not in ["RUN.lock", "RUN_COORDINATION.json"]
    ]
    paths = list(dict.fromkeys(Path(p) for p in paths))
    write(OUT / "FROZEN_INPUTS.json", {str(p): sha(p) for p in paths})
    smoke = engine.self_check()
    write(OUT / "ENGINE_SELF_CHECK.json", smoke)
    write(
        OUT / "PREPARED.json",
        dict(
            rows_per_horizon=844,
            scoring_events_per_horizon=463,
            horizons=HORIZONS,
            groups=GROUPS,
            vector_shape=list(vectors.shape),
            hashes={str(p.relative_to(OUT)): sha(p) for p in cache_paths("development")},
            source_count=len(paths),
            no_future_outcomes_read=True,
            prepared_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    verify_frozen()
    print(
        "Prepared 844 development events, three horizons, three clean feature groups.", flush=True
    )


def load_cache(stage):
    """Load one stage's verified feature frames and signal arrays into the worker cache."""
    global CACHE
    if stage == "evaluation":
        inputs.verify_selection_gate()
    prefix = "DEV" if stage == "development" else "EVAL"
    CACHE = dict(
        stage=stage,
        groups=read(OUT / "FEATURE_GROUPS.json"),
        frames={h: pd.read_parquet(OUT / f"{prefix}_DATA_{h}.parquet") for h in HORIZONS},
        bags=read(OUT / f"{prefix}_BAGS.json"),
        vectors=np.load(OUT / f"{prefix}_VECTORS.npy", mmap_mode="r"),
    )
    if stage == "evaluation":
        CACHE["lock"] = read(OUT / "LOCKED_SELECTION.json")


def make_frame(test, history, h, g, prediction):
    """Attach forecasts and strictly earlier-history baselines to aligned event metadata."""
    frame = test[["event_id", "ticker", "reaction_session", "quarter", "year", "y"]].copy()
    frame["horizon"], frame["group"] = h, g
    frame["prediction"] = prediction
    frame["history_mean"] = float(history.y.mean())
    frame["history_majority_sign"] = 1 if int(history.y.gt(0).sum()) * 2 >= len(history) else -1
    return frame


def cell(job):
    """Run or verify one horizon/quarter/feature-group cell under an exclusive file lock."""
    h, q, g = job
    stage = CACHE["stage"]
    dest = OUT / stage / h / q / g
    dest.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with FileLock(str(dest / "RUN.lock"), timeout=0):
        if (dest / "COMPLETE.json").exists():
            check_complete(dest)
            return dict(horizon=h, quarter=q, group=g, resumed=True)
        ensure_empty_cell(dest)
        frame = CACHE["frames"][h]
        history = frame[frame.quarter.lt(q)].copy()
        test = frame[frame.quarter.eq(q)].copy()
        require(
            (history.reaction_session.max() < test.reaction_session.min()),
            "Invariant failed: history.reaction_session.max() < test.reaction_session.min()",
        )
        spec = CACHE["groups"][g]
        use_text = g == "MFL"
        files = []
        if stage == "development":
            core, valid = engine.inner_split(history)
            matrix, prep = engine.prepare(core, valid, spec["numeric"], spec["categorical"])
            np.savez_compressed(dest / "INNER.npz", **matrix)
            joblib.dump(prep, dest / "inner_preprocessing.joblib", compress=3)
            stopped = engine.select_epochs(
                matrix, valid.y.to_numpy(), use_text, CACHE["vectors"], CACHE["bags"]
            )
            write(dest / "INNER_SELECTION.json", stopped)
            epochs = stopped["selected_epoch"]
            files += ["INNER.npz", "inner_preprocessing.joblib", "INNER_SELECTION.json"]
        else:
            epochs = CACHE["lock"]["model_selections"][h][g]["selected_epoch"]
        matrix, prep = engine.prepare(history, test, spec["numeric"], spec["categorical"])
        np.savez_compressed(dest / "FULL.npz", **matrix)
        joblib.dump(prep, dest / "preprocessing.joblib", compress=3)
        p = engine.refit(matrix, epochs, use_text, CACHE["vectors"], CACHE["bags"], dest)
        result = make_frame(test, history, h, g, p)
        if stage == "evaluation":
            coefficients = CACHE["lock"]["calibration"]["selections"][h][g]
            result["calibrated_prediction"] = calibration.apply(coefficients, p)
            result["calibration_method"] = coefficients["method"]
        result.to_parquet(dest / "PREDICTIONS.parquet", index=False)
        audit = dict(
            stage=stage,
            horizon=h,
            group=g,
            quarter=q,
            epochs=epochs,
            recipe=engine.RECIPE,
            seeds=engine.SEEDS,
            numeric_features=spec["numeric"],
            categorical_features=spec["categorical"],
            use_signals=use_text,
            history_ids=history.event_id.tolist(),
            test_ids=test.event_id.tolist(),
            early_stopping_performed=stage == "development",
            historical_weight_updates_only=stage == "evaluation",
            no_signal_truncation=True,
            history_signal_count=sum(len(CACHE["bags"][i]) for i in history.event_id)
            if use_text
            else 0,
            test_signal_count=sum(len(CACHE["bags"][i]) for i in test.event_id) if use_text else 0,
        )
        if stage == "evaluation":
            audit["selection_sha256"] = sha(OUT / "LOCKED_SELECTION.json")
        write(dest / "FIT_AUDIT.json", audit)
        files += [
            "FULL.npz",
            "preprocessing.joblib",
            "seed42.pt",
            "seed43.pt",
            "SEED_PREDICTIONS.npy",
            "PREDICTIONS.parquet",
            "FIT_AUDIT.json",
        ]
        complete(dest, files)
        return dict(
            horizon=h,
            quarter=q,
            group=g,
            epochs=epochs,
            seconds=round(time.monotonic() - start, 1),
            resumed=False,
        )


def run_jobs(stage, quarters, workers):
    """Execute the fixed cell grid with bounded worker concurrency and propagate failures."""
    jobs = [(h, q, g) for h in HORIZONS for q in quarters for g in GROUPS]
    if workers == 1:
        load_cache(stage)
        for job in jobs:
            print(json.dumps(cell(job)), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=load_cache,
            initargs=(stage,),
        ) as pool:
            pending = [pool.submit(cell, job) for job in jobs]
            for f in as_completed(pending):
                print(json.dumps(f.result()), flush=True)
    for h, q, g in jobs:
        check_complete(OUT / stage / h / q / g)


def collect(stage, quarters):
    """Load completed cell forecasts into one deterministic, event-aligned panel."""
    frames = []
    for h in HORIZONS:
        for q in quarters:
            for g in GROUPS:
                dest = OUT / stage / h / q / g
                check_complete(dest)
                frames.append(pd.read_parquet(dest / "PREDICTIONS.parquet"))
    data = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["horizon", "group", "reaction_session", "event_id"])
        .reset_index(drop=True)
    )
    require(
        (not data.duplicated(["horizon", "group", "event_id"]).any()),
        "Invariant failed: not data.duplicated(['horizon', 'group', 'event_id']).any()",
    )
    return data


def development(workers):
    """Run the declared development cells and collect their matched out-of-time forecasts."""
    verify_frozen()
    for name, digest in read(OUT / "PREPARED.json")["hashes"].items():
        require((sha(OUT / name) == digest), "Invariant failed: sha(OUT / name) == digest")
    run_jobs("development", DEVQ, workers)
    predictions = collect("development", DEVQ)
    require(
        (len(predictions) == 4167 and predictions.year.le(2024).all()),
        "Invariant failed: len(predictions) == 4167 and predictions.year.le(2024).all()",
    )
    predictions.to_parquet(OUT / "DEV_PREDICTIONS.parquet", index=False)
    verify_frozen()
    write(
        OUT / "DEV_FIT_COMPLETE.json",
        dict(
            events_per_system=463,
            systems=9,
            cells=72,
            prediction_sha256=sha(OUT / "DEV_PREDICTIONS.parquet"),
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    print("Development fitting complete: 72 cells, 4167 forecasts.", flush=True)


def select():
    """Lock choices using development forecasts only, before permitting evaluation access."""
    verify_frozen()
    for name, digest in read(OUT / "PREPARED.json")["hashes"].items():
        require((sha(OUT / name) == digest), "Invariant failed: sha(OUT / name) == digest")
    require(((WORK / "report.py").is_file()), "Freeze reporting source before any evaluation.")
    require(
        (
            sha(OUT / "DEV_PREDICTIONS.parquet")
            == read(OUT / "DEV_FIT_COMPLETE.json")["prediction_sha256"]
        ),
        "Invariant failed: sha(OUT / 'DEV_PREDICTIONS.parquet') == read(OUT / 'DEV_FIT_COMPLETE.json')['prediction_sha256']",
    )
    require(
        (not (OUT / "SELECTION_COMPLETE.json").exists()),
        "Selection already locked; do not overwrite.",
    )
    load_cache("development")
    predictions = pd.read_parquet(OUT / "DEV_PREDICTIONS.parquet")
    models = {h: {} for h in HORIZONS}
    final_receipts = []
    for h in HORIZONS:
        for g in GROUPS:
            dest = OUT / "final_stopping" / h / g
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "COMPLETE.json").exists():
                check_complete(dest)
                stopped = read(dest / "INNER_SELECTION.json")
            else:
                ensure_empty_cell(dest)
                core, valid = engine.inner_split(CACHE["frames"][h])
                spec = CACHE["groups"][g]
                matrix, prep = engine.prepare(core, valid, spec["numeric"], spec["categorical"])
                np.savez_compressed(dest / "INNER.npz", **matrix)
                joblib.dump(prep, dest / "inner_preprocessing.joblib", compress=3)
                stopped = engine.select_epochs(
                    matrix, valid.y.to_numpy(), g == "MFL", CACHE["vectors"], CACHE["bags"]
                )
                write(dest / "INNER_SELECTION.json", stopped)
                complete(dest, ["INNER.npz", "inner_preprocessing.joblib", "INNER_SELECTION.json"])
            models[h][g] = dict(
                selected_epoch=stopped["selected_epoch"],
                best_inner_mse=stopped["best_inner_mse"],
                recipe=engine.RECIPE,
                epoch_core_ids=stopped["core_train_ids"],
                epoch_validation_ids=stopped["inner_validation_ids"],
                epoch_receipt=str((dest / "COMPLETE.json").relative_to(OUT)),
            )
            final_receipts.append(str((dest / "COMPLETE.json").relative_to(OUT)))
            print(
                f"Final development duration {h}/{g}: {stopped['selected_epoch']} epochs",
                flush=True,
            )
    calibrated = calibration.select(predictions)
    calibrated["predictions"].to_parquet(OUT / "DEV_CALIBRATION_PREDICTIONS.parquet", index=False)
    calibrated["scores"].to_csv(OUT / "DEV_CALIBRATION_SCORES.csv", index=False)
    write(OUT / "DEV_CALIBRATION_FOLDS.json", calibrated["folds"].to_dict("records"))
    write(
        OUT / "LOCKED_SELECTION.json",
        dict(
            development_only=True,
            no_evaluation_outcomes_used=True,
            architecture=engine.RECIPE,
            model_selections=models,
            calibration=calibrated["summary"],
            input_encoder="minilm",
            input_horizons=HORIZONS,
            groups=GROUPS,
            evaluation_early_stopping=False,
            calibration_evaluation_updates=False,
            llm_calls=0,
            embedding_calls=0,
            selection_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    artifacts = [
        "LOCKED_SELECTION.json",
        "DEV_PREDICTIONS.parquet",
        "DEV_CALIBRATION_PREDICTIONS.parquet",
        "DEV_CALIBRATION_SCORES.csv",
        "DEV_CALIBRATION_FOLDS.json",
        *final_receipts,
        *[str(p.relative_to(OUT)) for p in cache_paths("development")],
    ]
    write(
        OUT / "SELECTION_COMPLETE.json",
        dict(
            locked_selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
            protocol_sha256=sha(OUT / "PROTOCOL.json"),
            development_only=True,
            no_evaluation_outcomes_used=True,
            report_source_sha256=sha(WORK / "report.py"),
            hashes={n: sha(OUT / n) for n in artifacts},
            locked_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    verify_frozen()
    print(
        "All nine durations and calibration methods/coefficients locked before evaluation.",
        flush=True,
    )


def evaluation(workers):
    """Evaluate only the locked recipe, duration, and calibration policy.

    Earlier elapsed outcomes may update weights in later quarters; they cannot change model selection.
    """
    verify_frozen()
    inputs.verify_selection_gate()
    # Loader independently verifies the protocol and lock receipt before any future target read.
    gate = read(OUT / "SELECTION_COMPLETE.json")
    require(
        (sha(WORK / "report.py") == gate["report_source_sha256"]),
        "Invariant failed: sha(WORK / 'report.py') == gate['report_source_sha256']",
    )
    for name, digest in gate["hashes"].items():
        require((sha(OUT / name) == digest), "Invariant failed: sha(OUT / name) == digest")
    require(
        ((OUT / "INDEPENDENT_DEVELOPMENT_AUDIT.json").exists()),
        "Independent development audit required.",
    )
    require(
        (read(OUT / "INDEPENDENT_DEVELOPMENT_AUDIT.json")["passed"] is True),
        "Invariant failed: read(OUT / 'INDEPENDENT_DEVELOPMENT_AUDIT.json')['passed'] is True",
    )
    require(
        (
            read(OUT / "INDEPENDENT_DEVELOPMENT_AUDIT.json")["locked_selection_sha256"]
            == sha(OUT / "LOCKED_SELECTION.json")
        ),
        "Invariant failed: read(OUT / 'INDEPENDENT_DEVELOPMENT_AUDIT.json')['locked_selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
    )
    if not (OUT / "EVALUATION_PREPARED.json").exists():
        frames, groups, bags, vectors = inputs.load_evaluation(HORIZONS, "minilm")
        require(
            (groups == read(OUT / "FEATURE_GROUPS.json")),
            "Invariant failed: groups == read(OUT / 'FEATURE_GROUPS.json')",
        )
        for h, data in frames.items():
            require(
                (len(data) == 1341 and len(data[data.year.ge(2025)]) == 497),
                "Invariant failed: len(data) == 1341 and len(data[data.year.ge(2025)]) == 497",
            )
            old = pd.read_parquet(OUT / f"DEV_DATA_{h}.parquet")
            pd.testing.assert_frame_equal(
                data[data.year.le(2024)].reset_index(drop=True),
                old,
                check_dtype=False,
                check_exact=True,
            )
            data.to_parquet(OUT / f"EVAL_DATA_{h}.parquet", index=False)
        write(OUT / "EVAL_BAGS.json", bags)
        np.save(OUT / "EVAL_VECTORS.npy", vectors)
        write(
            OUT / "EVALUATION_PREPARED.json",
            dict(
                hashes={str(p.relative_to(OUT)): sha(p) for p in cache_paths("evaluation")},
                locked_selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
            ),
        )
    require(
        (
            read(OUT / "EVALUATION_PREPARED.json")["locked_selection_sha256"]
            == sha(OUT / "LOCKED_SELECTION.json")
        ),
        "Invariant failed: read(OUT / 'EVALUATION_PREPARED.json')['locked_selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
    )
    for name, digest in read(OUT / "EVALUATION_PREPARED.json")["hashes"].items():
        require((sha(OUT / name) == digest), "Invariant failed: sha(OUT / name) == digest")
    run_jobs("evaluation", EVALQ, workers)
    predictions = collect("evaluation", EVALQ)
    require(
        (len(predictions) == 4473 and predictions.year.ge(2025).all()),
        "Invariant failed: len(predictions) == 4473 and predictions.year.ge(2025).all()",
    )
    require(
        (predictions.groupby(["horizon", "group"]).size().eq(497).all()),
        "Invariant failed: predictions.groupby(['horizon', 'group']).size().eq(497).all()",
    )
    predictions.to_parquet(OUT / "EVALUATION_PREDICTIONS.parquet", index=False)
    verify_frozen()
    for name, digest in gate["hashes"].items():
        require((sha(OUT / name) == digest), "Invariant failed: sha(OUT / name) == digest")
    write(
        OUT / "EVAL_COMPLETE.json",
        dict(
            prediction_sha256=sha(OUT / "EVALUATION_PREDICTIONS.parquet"),
            locked_selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
            events_per_system=497,
            systems=9,
            cells=72,
            hyperparameter_choices=0,
            epoch_choices=0,
            calibration_choices=0,
            old_results_preserved=True,
            llm_calls=0,
            embedding_calls=0,
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    print("Evaluation complete: 497 matched events x nine systems; no revised choices.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["protocol", "prepare", "development", "select", "evaluation"]
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    activate_legacy_root(ROOT)
    require((args.workers in [1, 2]), "Invariant failed: args.workers in [1, 2]")
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT / "RUN.lock"), timeout=0):
        if args.stage == "protocol":
            protocol()
        elif args.stage == "prepare":
            prepare()
        elif args.stage == "development":
            development(args.workers)
        elif args.stage == "select":
            select()
        else:
            evaluation(args.workers)

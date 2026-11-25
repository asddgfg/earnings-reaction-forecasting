"""Independent verification: no production engine/report imports or training."""

import os

from mqf_research.common.hashing import file_sha256 as sha
from mqf_research.common.serialization import read_json as read
from mqf_research.common.serialization import write_json
from mqf_research.config import get_settings
from mqf_research.modeling._compatibility import activate_legacy_root, bind_run_configuration
from mqf_research.modeling._validation import require, validate_history

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

ROOT = get_settings().research_root

OUT = get_settings().output_root / "nn_tuning"
PARENT = ROOT / "outputs/small_sample_ai_20261118_b032"
OLDNN = ROOT / "outputs/valuation_industry_20261113_b032"
RELEASE = ROOT / "outputs/source_repair_batches/batch_032"
DEVQ = [f"{year}Q{q}" for year in (2023, 2024) for q in range(1, 5)]
EVALQ = [f"{year}Q{q}" for year in (2025, 2026) for q in range(1, 5)]
torch.set_num_threads(1)


def write(path, value):
    """Atomically publish the historical JSON schema in the isolated run directory."""
    write_json(path, value, default=str)


def close(a, b, atol=1e-12, rtol=1e-9):
    """Compare independently computed arrays using the declared numeric tolerances."""
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


def hash_receipt(folder):
    """Check that a saved cell is complete and its named artifacts are unchanged."""
    receipt = read(folder / "COMPLETE.json")
    for name, digest in receipt["hashes"].items():
        require((sha(folder / name) == digest), str(folder / name))
    return receipt


def frozen():
    """Verify that all provenance-bound source files still match their frozen digests."""
    bind_run_configuration(OUT)
    entries = read(OUT / "FROZEN_INPUTS.json")
    for path, digest in entries.items():
        require((sha(path) == digest), path)
    return len(entries)


def expected_recipes():
    """Independently enumerate the fixed twelve-candidate recipe grid for audit."""
    result = []
    for n, h in [(8, 16), (16, 32), (32, 64)]:
        for d in [0.2, 0.4]:
            for w in [0.05, 0.2]:
                result.append(
                    dict(
                        recipe_id=f"n{n}_h{h}_d{int(d * 100)}_w{int(w * 100)}",
                        numeric_width=n,
                        head_width=h,
                        dropout=d,
                        weight_decay=w,
                        is_original=(n == 16 and d == 0.2 and w == 0.05),
                    )
                )
    return result


def oldfold(q):
    """Locate the read-only historical numeric-model fold used as a reference."""
    return OLDNN / "models/hour/validation" / q / "context_numeric_nn"


def reconstruct_source(data):
    """Filters before reading target values; development mode never parses future labels."""
    ids = data.event_id.tolist()
    binding = read(RELEASE / "DATASET_BINDING.json")
    base = binding["financial_features"]
    fields = read(OUT / "FEATURES.json")
    num = fields["numeric"]
    cat = fields["categorical"]
    require(
        (len(base) == 67 and len(num) == 77 and (len(cat) == 2)),
        "Invariant failed: len(base) == 67 and len(num) == 77 and (len(cat) == 2)",
    )
    require(
        (fields == read(PARENT / "FEATURES.json")),
        "Invariant failed: fields == read(PARENT / 'FEATURES.json')",
    )
    meta = [
        "event_id",
        "ticker",
        "reaction_session",
        "effective_utc",
        "cutoff_utc",
        "estimation_end",
    ]
    source = pd.read_parquet(
        binding["datasets"]["hour"], columns=meta + ["y"] + base, filters=[("event_id", "in", ids)]
    )
    source["reaction_session"] = pd.to_datetime(source.reaction_session).dt.strftime("%Y-%m-%d")
    source["quarter"] = pd.to_datetime(source.reaction_session).dt.to_period("Q").astype(str)
    source["year"] = pd.to_datetime(source.reaction_session).dt.year
    context_fields = [
        x
        for x in num
        if x not in base and x not in ["text_available", "has_signals", "log_signal_count"]
    ]
    context = pd.read_parquet(
        OLDNN / "CONTEXT_FEATURES.parquet",
        columns=["event_id"] + context_fields + cat,
        filters=[("event_id", "in", ids)],
    )
    source = source.merge(context, on="event_id", how="left", validate="one_to_one")
    events = {
        row["event_id"]: row for row in read(RELEASE / "EVENTS.json") if row["event_id"] in set(ids)
    }
    counts = {}
    for row in read(RELEASE / "embeddings_minilm/INDEX.json"):
        key = row["event_id"]
        if key in events:
            counts[key] = counts.get(key, 0) + 1
    require((set(events) == set(ids)), "Invariant failed: set(events) == set(ids)")
    require(
        (all((counts.get(key, 0) == events[key]["ml_signal_count"] for key in ids))),
        "Invariant failed: all((counts.get(key, 0) == events[key]['ml_signal_count'] for key in ids))",
    )
    source["text_available"] = [
        (
            (
                events[e]["extraction_status"] == "complete"
                or events[e].get("text_input_status") == "manual_source_review_complete"
            )
            and events[e].get("text_source_eligible", True)
        )
        for e in source.event_id
    ]
    source["has_signals"] = [bool(counts.get(e, 0)) for e in source.event_id]
    source["log_signal_count"] = [np.log1p(counts.get(e, 0)) for e in source.event_id]
    source = source.sort_values(["reaction_session", "event_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(source[data.columns], data, check_dtype=False, check_exact=True)
    require(
        (
            (
                pd.to_datetime(data.effective_utc, utc=True)
                <= pd.to_datetime(data.cutoff_utc, utc=True)
            ).all()
        ),
        "Invariant failed: (pd.to_datetime(data.effective_utc, utc=True) <= pd.to_datetime(data.cutoff_utc, utc=True)).all()",
    )
    require(
        ((pd.to_datetime(data.estimation_end) < pd.to_datetime(data.reaction_session)).all()),
        "Invariant failed: (pd.to_datetime(data.estimation_end) < pd.to_datetime(data.reaction_session)).all()",
    )
    require(
        (np.isfinite(data.y).all() and data.event_id.is_unique),
        "Invariant failed: np.isfinite(data.y).all() and data.event_id.is_unique",
    )
    return num, cat


def partition(history):
    # Independently find the earliest of the final two-or-more quarters needed for100 stopping events.
    """Independently reconstruct the training-history split used for early stopping."""
    validate_history(history)
    quarters = sorted(history.quarter.unique())
    cut = quarters[-2]
    while len(history[history.quarter.ge(cut)]) < 100:
        cut = quarters[quarters.index(cut) - 1]
    core = history[history.quarter.lt(cut)].copy()
    inner = history[history.quarter.ge(cut)].copy()
    require(
        (len(core) >= 200 and len(inner) >= 100 and (inner.quarter.nunique() >= 2)),
        "Invariant failed: len(core) >= 200 and len(inner) >= 100 and (inner.quarter.nunique() >= 2)",
    )
    require(
        (core.reaction_session.max() < inner.reaction_session.min()),
        "Invariant failed: core.reaction_session.max() < inner.reaction_session.min()",
    )
    require(
        (set(core.event_id).isdisjoint(inner.event_id)),
        "Invariant failed: set(core.event_id).isdisjoint(inner.event_id)",
    )
    return core, inner


def transforms(train, test, num, cat):
    """Independently fit training-only imputation, scaling, and category encoding for audit."""
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scale = StandardScaler()
    tr = scale.fit_transform(imp.fit_transform(train[num]))
    te = scale.transform(imp.transform(test[num]))
    hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    ctr = hot.fit_transform(train[cat].fillna("UNKNOWN").astype(str))
    cte = hot.transform(test[cat].fillna("UNKNOWN").astype(str))
    a = np.column_stack([np.clip(tr, -10, 10), ctr]).astype("float32")
    b = np.column_stack([np.clip(te, -10, 10), cte]).astype("float32")
    mu = float(train.y.mean())
    sd = max(float(train.y.std(ddof=0)), 1e-6)
    target = ((train.y.to_numpy() - mu) / sd).astype("float32")
    return a, b, target, mu, sd


def matrix_check(path, train, test, num, cat, prep_path=None):
    """Compare saved model matrices and preprocessing statistics with an independent reconstruction."""
    matrix = np.load(path, allow_pickle=False)
    expected = transforms(train, test, num, cat)
    for name, want in zip(
        ["X_train", "X_test", "y_train_std", "target_mean", "target_sd"], expected
    ):
        np.testing.assert_array_equal(matrix[name], want)
    np.testing.assert_array_equal(matrix["train_ids"], train.event_id.to_numpy(str))
    np.testing.assert_array_equal(matrix["test_ids"], test.event_id.to_numpy(str))
    require(
        (train.reaction_session.max() < test.reaction_session.min()),
        "Invariant failed: train.reaction_session.max() < test.reaction_session.min()",
    )
    require(("y_test" not in matrix.files), "Invariant failed: 'y_test' not in matrix.files")
    if "y_valid" in matrix.files:
        np.testing.assert_array_equal(matrix["y_valid"], test.y.to_numpy())
    if prep_path:
        prep = joblib.load(prep_path)
        require(
            (prep["fit_event_ids"] == train.event_id.tolist()),
            "Invariant failed: prep['fit_event_ids'] == train.event_id.tolist()",
        )
        require(
            (prep["numeric_features"] == num and prep["categorical_features"] == cat),
            "Invariant failed: prep['numeric_features'] == num and prep['categorical_features'] == cat",
        )
        for column, levels in zip(cat, prep["onehot"].categories_):
            require(
                (set(levels) == set(train[column].fillna("UNKNOWN").astype(str))),
                "Invariant failed: set(levels) == set(train[column].fillna('UNKNOWN').astype(str))",
            )
        transformed = np.column_stack(
            [
                np.clip(prep["scaler"].transform(prep["imputer"].transform(test[num])), -10, 10),
                prep["onehot"].transform(test[cat].fillna("UNKNOWN").astype(str)),
            ]
        ).astype("float32")
        np.testing.assert_array_equal(transformed, expected[1])
    return matrix


def inputs_check():
    """Verify canonical input identity, chronology, feature membership, and preprocessing."""
    frozen_count = frozen()
    proto = read(OUT / "PROTOCOL.json")
    require(
        (proto["recipes"] == expected_recipes() and proto["seeds"] == [42, 43]),
        "Invariant failed: proto['recipes'] == expected_recipes() and proto['seeds'] == [42, 43]",
    )
    require(
        (proto["development_quarters"] == DEVQ and proto["evaluation_quarters"] == EVALQ),
        "Invariant failed: proto['development_quarters'] == DEVQ and proto['evaluation_quarters'] == EVALQ",
    )
    receipt = read(OUT / "PREPARED.json")
    require(
        (sha(OUT / "DEV_DATA.parquet") == receipt["data_sha256"]),
        "Invariant failed: sha(OUT / 'DEV_DATA.parquet') == receipt['data_sha256']",
    )
    require(
        (sha(OUT / "DEV_REFERENCE.parquet") == receipt["reference_sha256"]),
        "Invariant failed: sha(OUT / 'DEV_REFERENCE.parquet') == receipt['reference_sha256']",
    )
    data = pd.read_parquet(OUT / "DEV_DATA.parquet")
    require(
        (
            len(data) == 844
            and data.year.max() == 2024
            and (data.reaction_session.max() == "2024-12-20")
        ),
        "Invariant failed: len(data) == 844 and data.year.max() == 2024 and (data.reaction_session.max() == '2024-12-20')",
    )
    pd.testing.assert_frame_equal(
        data, pd.read_parquet(PARENT / "DATA.parquet"), check_exact=True, check_dtype=False
    )
    num, cat = reconstruct_source(data)
    reference = pd.read_parquet(OUT / "DEV_REFERENCE.parquet")
    require(
        (len(reference) == 463 and reference.year.max() == 2024 and reference.event_id.is_unique),
        "Invariant failed: len(reference) == 463 and reference.year.max() == 2024 and reference.event_id.is_unique",
    )
    test = data[data.year.ge(2023)]
    pd.testing.assert_frame_equal(
        reference[["event_id", "y"]].reset_index(drop=True),
        test[["event_id", "y"]].reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    for q in DEVQ:
        root = OUT / "development" / q
        for name, digest in receipt["fold_inputs"][q].items():
            require((sha(root / name) == digest), "Invariant failed: sha(root / name) == digest")
        history = data[data.quarter.lt(q)]
        test = data[data.quarter.eq(q)]
        core, inner = partition(history)
        audit = read(root / "INPUT_AUDIT.json")
        require(
            (
                audit["history_ids"] == history.event_id.tolist()
                and audit["test_ids"] == test.event_id.tolist()
            ),
            "Invariant failed: audit['history_ids'] == history.event_id.tolist() and audit['test_ids'] == test.event_id.tolist()",
        )
        require(
            (
                audit["core_ids"] == core.event_id.tolist()
                and audit["inner_ids"] == inner.event_id.tolist()
            ),
            "Invariant failed: audit['core_ids'] == core.event_id.tolist() and audit['inner_ids'] == inner.event_id.tolist()",
        )
        old = read(oldfold(q) / "INNER_SELECTION.json")
        require(
            (
                old["core_train_ids"] == core.event_id.tolist()
                and old["inner_validation_ids"] == inner.event_id.tolist()
            ),
            "Invariant failed: old['core_train_ids'] == core.event_id.tolist() and old['inner_validation_ids'] == inner.event_id.tolist()",
        )
        matrix_check(root / "FULL.npz", history, test, num, cat, root / "preprocessing.joblib")
        matrix_check(root / "INNER.npz", core, inner, num, cat, root / "inner_preprocessing.joblib")
    return (
        data,
        num,
        cat,
        dict(
            passed=True,
            development_rows=844,
            scored_events=463,
            source_rows_independently_reconstructed=844,
            input_matrix_pairs=16,
            frozen_hashes_verified=frozen_count,
            no_evaluation_outcomes_read=True,
            no_model_fits=True,
            recipe_count=12,
        ),
    )


def stopping_check(info):
    """Recompute epoch selection and patience from the complete recorded validation curve."""
    curve = info["curve"]
    best = float("inf")
    chosen = None
    stale = 0
    stop = None
    for i, row in enumerate(curve, 1):
        require((row["epoch"] == i), "Invariant failed: row['epoch'] == i")
        loss = float(row["validation_mse"])
        require(
            (np.isfinite(loss) and loss >= 0), "Invariant failed: np.isfinite(loss) and loss >= 0"
        )
        close(row["validation_rmse_pp"], 100 * np.sqrt(loss))
        if loss < best:
            best = loss
            chosen = i
            stale = 0
        else:
            stale += 1
        if i >= 5 and stale >= 5:
            stop = i
            require((i == len(curve)), "Curve continued after the frozen stopping rule")
            break
    if stop is None:
        require((len(curve) == 40), "Invariant failed: len(curve) == 40")
        stop = 40
    require(
        (chosen == info["selected_epoch"] and stop == info["stopped_epoch"]),
        "Invariant failed: chosen == info['selected_epoch'] and stop == info['stopped_epoch']",
    )
    if "best_inner_mse" in info:
        close(best, info["best_inner_mse"])
    require(
        (bool(info["hit_cap"]) == (stop == 40)),
        "Invariant failed: bool(info['hit_cap']) == (stop == 40)",
    )
    return best


class IndependentNet(nn.Module):
    """Independent numeric architecture used to audit trusted saved checkpoints."""

    def __init__(self, dim, recipe):
        """Create the model/helper in the fixed parameter order used by the experiment."""
        super().__init__()
        self.proj = nn.Linear(384, 32)
        self.v = nn.Linear(32, 16)
        self.u = nn.Linear(32, 16)
        self.a = nn.Linear(16, 1, bias=False)
        self.numeric = nn.Sequential(nn.Linear(dim, recipe["numeric_width"]), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(recipe["numeric_width"] + 32, recipe["head_width"]),
            nn.ReLU(),
            nn.Dropout(recipe["dropout"]),
            nn.Linear(recipe["head_width"], 1),
        )

    def forward(self, x):
        """Return one signed-AR forecast per event from numeric features and optional signal bags."""
        return self.head(torch.cat([self.numeric(x), x.new_zeros((len(x), 32))], dim=1)).squeeze(1)


def checkpoint_predict(path, matrix, recipe, epochs, seed):
    """Rebuild a trusted numeric checkpoint and verify its frozen recipe and epoch count."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(
        (saved["seed"] == seed and saved["epochs"] == epochs),
        "Invariant failed: saved['seed'] == seed and saved['epochs'] == epochs",
    )
    if "recipe" in saved:
        require((saved["recipe"] == recipe), "Invariant failed: saved['recipe'] == recipe")
    else:
        require(
            (recipe["is_original"] and saved["mode"] == "numeric_nn"),
            "Invariant failed: recipe['is_original'] and saved['mode'] == 'numeric_nn'",
        )
        require(
            (saved["embedding_dim"] == 384 and saved["projection_dim"] == 32),
            "Invariant failed: saved['embedding_dim'] == 384 and saved['projection_dim'] == 32",
        )
    require(
        (saved["numeric_dim"] == matrix["X_train"].shape[1]),
        "Invariant failed: saved['numeric_dim'] == matrix['X_train'].shape[1]",
    )
    close(saved["target_mean"], float(matrix["target_mean"]), atol=0, rtol=0)
    close(saved["target_sd"], float(matrix["target_sd"]), atol=0, rtol=0)
    net = IndependentNet(saved["numeric_dim"], recipe)
    net.load_state_dict(saved["state_dict"])
    net.eval()
    x = torch.from_numpy(matrix["X_test"])
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            out = net(x[start : start + 128]).numpy()
            values.extend((out * saved["target_sd"] + saved["target_mean"]).tolist())
    result = np.asarray(values)
    require((np.isfinite(result).all()), "Invariant failed: np.isfinite(result).all()")
    return result


def score(frame):
    """Calculate a saved fold's error summary without refitting or changing its event cohort."""
    error = frame.prediction.to_numpy() - frame.y.to_numpy()
    return dict(
        n=len(frame),
        mse=float(np.mean(error**2)),
        rmse_pp=float(np.sqrt(np.mean(error**2)) * 100),
        mae_pp=float(np.mean(abs(error)) * 100),
        accuracy_pct=float(np.mean((frame.prediction > 0) == (frame.y > 0)) * 100),
    )


def selection_check():
    """Verify recipe ranking, tie-breaking, and the locked development-only selection."""
    require(
        ((OUT / "SELECTION_COMPLETE.json").exists()),
        "Invariant failed: (OUT / 'SELECTION_COMPLETE.json').exists()",
    )
    gate = read(OUT / "SELECTION_COMPLETE.json")
    require(
        (gate["development_only"] and gate["no_evaluation_outcomes_used"]),
        "Invariant failed: gate['development_only'] and gate['no_evaluation_outcomes_used']",
    )
    require(
        (sha(OUT / "LOCKED_SELECTION.json") == gate["locked_selection_sha256"]),
        "Invariant failed: sha(OUT / 'LOCKED_SELECTION.json') == gate['locked_selection_sha256']",
    )
    require(
        (sha(OUT / "PROTOCOL.json") == gate["protocol_sha256"]),
        "Invariant failed: sha(OUT / 'PROTOCOL.json') == gate['protocol_sha256']",
    )
    for name, digest in gate["hashes"].items():
        require((sha(OUT / name) == digest), name)
    for name, digest in gate["frozen_model_code_sha256"].items():
        require(
            (sha(Path(__file__).parent / name) == digest),
            "Invariant failed: sha(Path(__file__).parent / name) == digest",
        )
    return read(OUT / "LOCKED_SELECTION.json")


def development_check():
    """Audit development data, model choices, stopping receipts, and calibration provenance."""
    data, num, cat, summary = inputs_check()
    require(
        ((OUT / "DEV_FIT_COMPLETE.json").exists()),
        "Invariant failed: (OUT / 'DEV_FIT_COMPLETE.json').exists()",
    )
    lock = selection_check()
    recipes = expected_recipes()
    fullpred = pd.read_parquet(OUT / "DEV_PREDICTIONS.parquet")
    nested = pd.read_parquet(OUT / "NESTED_DEV_PREDICTIONS.parquet")
    scores = pd.read_csv(OUT / "DEV_SCORES.csv")
    innercsv = pd.read_csv(OUT / "DEV_INNER_SELECTIONS.csv")
    require(
        (len(fullpred) == 463 * 12 and len(nested) == 463 and nested.event_id.is_unique),
        "Invariant failed: len(fullpred) == 463 * 12 and len(nested) == 463 and nested.event_id.is_unique",
    )
    require(
        (len(scores) == 12 and len(innercsv) == 96),
        "Invariant failed: len(scores) == 12 and len(innercsv) == 96",
    )
    require(
        (fullpred.year.max() == nested.year.max() == 2024),
        "Invariant failed: fullpred.year.max() == nested.year.max() == 2024",
    )
    replay_count = 0
    maxdiff = 0.0
    nested_ids = {}
    for q in DEVQ:
        root = OUT / "development" / q
        matrix = np.load(root / "FULL.npz", allow_pickle=False)
        expected_test = data[data.quarter.eq(q)]
        choices = []
        for recipe in recipes:
            dest = root / recipe["recipe_id"]
            hash_receipt(dest)
            info = read(dest / "INNER_SELECTION.json")
            best = stopping_check(info)
            inputaudit = read(root / "INPUT_AUDIT.json")
            require(
                (info["core_train_ids"] == inputaudit["core_ids"]),
                "Invariant failed: info['core_train_ids'] == inputaudit['core_ids']",
            )
            require(
                (info["inner_validation_ids"] == inputaudit["inner_ids"]),
                "Invariant failed: info['inner_validation_ids'] == inputaudit['inner_ids']",
            )
            audit = read(dest / "FIT_AUDIT.json")
            require(
                (audit["recipe"] == recipe and audit["epochs"] == info["selected_epoch"]),
                "Invariant failed: audit['recipe'] == recipe and audit['epochs'] == info['selected_epoch']",
            )
            require(
                (
                    audit["train_ids"] == matrix["train_ids"].tolist()
                    and audit["test_ids"] == matrix["test_ids"].tolist()
                ),
                "Invariant failed: audit['train_ids'] == matrix['train_ids'].tolist() and audit['test_ids'] == matrix['test_ids'].tolist()",
            )
            require(
                (audit["no_future_outcomes_read"]),
                "Invariant failed: audit['no_future_outcomes_read']",
            )
            values = np.load(dest / "PREDICTIONS.npy", allow_pickle=False)
            saved_seed = []
            for seed in [42, 43]:
                path = (oldfold(q) if recipe["is_original"] else dest) / f"seed{seed}.pt"
                p = checkpoint_predict(path, matrix, recipe, info["selected_epoch"], seed)
                saved_seed.append(p)
                replay_count += 1
            calculated = np.mean(saved_seed, axis=0)
            close(calculated, values, atol=1e-8, rtol=1e-6)
            maxdiff = max(maxdiff, float(np.max(abs(calculated - values))))
            if not recipe["is_original"]:
                close(
                    np.asarray(saved_seed),
                    np.load(dest / "SEED_PREDICTIONS.npy"),
                    atol=1e-8,
                    rtol=1e-6,
                )
            frame = fullpred[fullpred.quarter.eq(q) & fullpred.recipe_id.eq(recipe["recipe_id"])]
            pd.testing.assert_frame_equal(
                frame[
                    ["event_id", "ticker", "reaction_session", "quarter", "year", "y"]
                ].reset_index(drop=True),
                expected_test[
                    ["event_id", "ticker", "reaction_session", "quarter", "year", "y"]
                ].reset_index(drop=True),
                check_exact=True,
                check_dtype=False,
            )
            np.testing.assert_array_equal(frame.prediction.to_numpy(), values)
            row = innercsv[
                innercsv.quarter.eq(q) & innercsv.recipe_id.eq(recipe["recipe_id"])
            ].iloc[0]
            require(
                (
                    row.selected_epoch == info["selected_epoch"]
                    and row.stopped_epoch == info["stopped_epoch"]
                ),
                "Invariant failed: row.selected_epoch == info['selected_epoch'] and row.stopped_epoch == info['stopped_epoch']",
            )
            close(row.inner_mse, best)
            choices.append(
                (best, not recipe["is_original"], recipe["numeric_width"], recipe["recipe_id"])
            )
        picked = min(choices)[3]
        nested_ids[q] = picked
        chosen = nested[nested.quarter.eq(q)]
        require(
            (chosen.recipe_id.eq(picked).all()),
            "Invariant failed: chosen.recipe_id.eq(picked).all()",
        )
        want = fullpred[fullpred.quarter.eq(q) & fullpred.recipe_id.eq(picked)]
        pd.testing.assert_frame_equal(
            chosen.reset_index(drop=True), want.reset_index(drop=True), check_exact=True
        )
    rankings = []
    for recipe in recipes:
        current = fullpred[fullpred.recipe_id.eq(recipe["recipe_id"])]
        metrics = score(current)
        row = scores[scores.recipe_id.eq(recipe["recipe_id"])].iloc[0]
        require(
            (len(current) == 463 and current.event_id.is_unique),
            "Invariant failed: len(current) == 463 and current.event_id.is_unique",
        )
        for key, value in metrics.items():
            close(row[key], value)
        for key, value in recipe.items():
            require((row[key] == value), "Invariant failed: row[key] == value")
        rankings.append(
            (
                metrics["mse"],
                not recipe["is_original"],
                recipe["numeric_width"],
                recipe["recipe_id"],
            )
        )
    expected_order = [row[3] for row in sorted(rankings)]
    require(
        (scores.recipe_id.tolist() == expected_order),
        "Invariant failed: scores.recipe_id.tolist() == expected_order",
    )
    require(
        (lock["selected_recipe_id"] == expected_order[0]),
        "Invariant failed: lock['selected_recipe_id'] == expected_order[0]",
    )
    recipe = next(r for r in recipes if r["recipe_id"] == expected_order[0])
    require(
        (lock["selected_recipe"] == recipe and lock["seeds"] == [42, 43]),
        "Invariant failed: lock['selected_recipe'] == recipe and lock['seeds'] == [42, 43]",
    )
    require(
        (lock["development_only"] and lock["no_evaluation_outcomes_used"]),
        "Invariant failed: lock['development_only'] and lock['no_evaluation_outcomes_used']",
    )
    require(
        (not lock["evaluation_early_stopping"]),
        "Invariant failed: not lock['evaluation_early_stopping']",
    )
    close(lock["validation_mse"], scores.iloc[0].mse)
    final = OUT / "final_development_stopping"
    hash_receipt(final)
    core, inner = partition(data)
    matrix = matrix_check(
        final / "INNER.npz", core, inner, num, cat, final / "preprocessing.joblib"
    )
    info = read(final / "INNER_SELECTION.json")
    stopping_check(info)
    require(
        (info["core_train_ids"] == core.event_id.tolist()),
        "Invariant failed: info['core_train_ids'] == core.event_id.tolist()",
    )
    require(
        (info["inner_validation_ids"] == inner.event_id.tolist()),
        "Invariant failed: info['inner_validation_ids'] == inner.event_id.tolist()",
    )
    require(
        (lock["selected_epoch"] == info["selected_epoch"]),
        "Invariant failed: lock['selected_epoch'] == info['selected_epoch']",
    )
    require(
        (
            lock["epoch_core_ids"] == core.event_id.tolist()
            and lock["epoch_selection_ids"] == inner.event_id.tolist()
        ),
        "Invariant failed: lock['epoch_core_ids'] == core.event_id.tolist() and lock['epoch_selection_ids'] == inner.event_id.tolist()",
    )
    smoke = OUT / "original_reproduction"
    hash_receipt(smoke)
    smoke_info = read(smoke / "CHECK.json")
    require(
        (smoke_info["passed"] and smoke_info["quarter"] == "2023Q1"),
        "Invariant failed: smoke_info['passed'] and smoke_info['quarter'] == '2023Q1'",
    )
    require(
        (
            smoke_info["no_future_outcomes_read"]
            and smoke_info["original_initialization_and_stopping_reproduced"]
        ),
        "Invariant failed: smoke_info['no_future_outcomes_read'] and smoke_info['original_initialization_and_stopping_reproduced']",
    )
    original = next(r for r in recipes if r["is_original"])
    original_info = read(oldfold("2023Q1") / "INNER_SELECTION.json")
    require(
        (smoke_info["selected_epoch"] == original_info["selected_epoch"]),
        "Invariant failed: smoke_info['selected_epoch'] == original_info['selected_epoch']",
    )
    smoke_matrix = np.load(OUT / "development/2023Q1/FULL.npz", allow_pickle=False)
    smoke_seeds = np.asarray(
        [
            checkpoint_predict(
                smoke / f"seed{s}.pt", smoke_matrix, original, original_info["selected_epoch"], s
            )
            for s in [42, 43]
        ]
    )
    close(smoke_seeds, np.load(smoke / "SEED_PREDICTIONS.npy"), atol=1e-8, rtol=1e-6)
    smoke_reference = np.load(
        OUT / "development/2023Q1" / original["recipe_id"] / "PREDICTIONS.npy"
    )
    close(smoke_seeds.mean(axis=0), smoke_reference, atol=1e-8, rtol=1e-6)
    close(
        smoke_info["max_prediction_difference"],
        float(np.max(abs(smoke_seeds.mean(axis=0) - smoke_reference))),
        atol=1e-8,
        rtol=1e-6,
    )
    summary.update(
        development_predictions=5556,
        model_checkpoints_replayed=replay_count,
        maximum_replayed_prediction_difference=maxdiff,
        stopping_rules_verified=97,
        original_smoke_receipt_passed=True,
        original_smoke_checkpoints_replayed=2,
        nested_recipe_choices_verified=nested_ids,
        selected_recipe_id=lock["selected_recipe_id"],
        selected_epoch=lock["selected_epoch"],
        selection_only_development_verified=True,
        training_not_repeated=True,
        inner_curve_values_not_independently_retrained=True,
    )
    frozen()
    return summary


def later_models():
    # The first later-target read is strictly after both selection hash checks.
    """Audit expanding-quarter evaluation checkpoints using only previously elapsed history."""
    lock = selection_check()
    require(
        ((OUT / "EVAL_COMPLETE.json").exists()),
        "Invariant failed: (OUT / 'EVAL_COMPLETE.json').exists()",
    )
    receipt = read(OUT / "EVAL_COMPLETE.json")
    require(
        (sha(OUT / "EVALUATION_DATA.parquet") == receipt["data_sha256"]),
        "Invariant failed: sha(OUT / 'EVALUATION_DATA.parquet') == receipt['data_sha256']",
    )
    require(
        (sha(OUT / "EVALUATION_PREDICTIONS.parquet") == receipt["prediction_sha256"]),
        "Invariant failed: sha(OUT / 'EVALUATION_PREDICTIONS.parquet') == receipt['prediction_sha256']",
    )
    require(
        (receipt["locked_selection_sha256"] == sha(OUT / "LOCKED_SELECTION.json")),
        "Invariant failed: receipt['locked_selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
    )
    require(
        (
            receipt["evaluation_hyperparameter_choices"] == 0
            and receipt["evaluation_epoch_choices"] == 0
        ),
        "Invariant failed: receipt['evaluation_hyperparameter_choices'] == 0 and receipt['evaluation_epoch_choices'] == 0",
    )
    for path, digest in read(OUT / "EVALUATION_INPUT_FREEZE.json").items():
        require((sha(path) == digest), path)
    data = pd.read_parquet(OUT / "EVALUATION_DATA.parquet")
    require(
        (
            len(data) == 1341
            and data.event_id.is_unique
            and (data.reaction_session.max() == "2026-10-02")
        ),
        "Invariant failed: len(data) == 1341 and data.event_id.is_unique and (data.reaction_session.max() == '2026-10-02')",
    )
    require(
        (
            data.groupby("year").size().to_dict()
            == {2021: 105, 2022: 276, 2023: 238, 2024: 225, 2025: 238, 2026: 259}
        ),
        "Invariant failed: data.groupby('year').size().to_dict() == {2021: 105, 2022: 276, 2023: 238, 2024: 225, 2025: 238, 2026: 259}",
    )
    pd.testing.assert_frame_equal(
        data[data.year.le(2024)].reset_index(drop=True),
        pd.read_parquet(OUT / "DEV_DATA.parquet"),
        check_dtype=False,
        check_exact=True,
    )
    num, cat = reconstruct_source(data)
    pred = pd.read_parquet(OUT / "EVALUATION_PREDICTIONS.parquet")
    require(
        (len(pred) == 497 and pred.event_id.is_unique),
        "Invariant failed: len(pred) == 497 and pred.event_id.is_unique",
    )
    cols = ["event_id", "ticker", "reaction_session", "quarter", "year", "y"]
    pd.testing.assert_frame_equal(
        pred[cols],
        data[data.year.ge(2025)][cols].reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    require(
        (
            np.isfinite(
                pred[
                    [
                        "y",
                        "prediction",
                        "original_nn_prediction",
                        "market_ridge_prediction",
                        "history_mean",
                    ]
                ].to_numpy()
            ).all()
        ),
        "Invariant failed: np.isfinite(pred[['y', 'prediction', 'original_nn_prediction', 'market_ridge_prediction', 'history_mean']].to_numpy()).all()",
    )
    require(
        (
            pred[["y", "prediction", "original_nn_prediction", "market_ridge_prediction"]]
            .ne(0)
            .all()
            .all()
        ),
        "Invariant failed: pred[['y', 'prediction', 'original_nn_prediction', 'market_ridge_prediction']].ne(0).all().all()",
    )
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
    for frame, column in [(original, "original_nn_prediction"), (ridge, "market_ridge_prediction")]:
        aligned = frame.loc[pred.event_id]
        np.testing.assert_array_equal(aligned.y.to_numpy(), pred.y.to_numpy())
        np.testing.assert_array_equal(aligned.prediction.to_numpy(), pred[column].to_numpy())
    recipe = lock["selected_recipe"]
    epochs = lock["selected_epoch"]
    histories = []
    replays = 0
    maxdiff = 0.0
    for q in EVALQ:
        folder = OUT / "evaluation" / q
        hash_receipt(folder)
        history = data[data.quarter.lt(q)]
        current = data[data.quarter.eq(q)]
        matrix = matrix_check(
            folder / "FULL.npz", history, current, num, cat, folder / "preprocessing.joblib"
        )
        audit = read(folder / "FIT_AUDIT.json")
        require(
            (
                audit["history_ids"] == history.event_id.tolist()
                and audit["test_ids"] == current.event_id.tolist()
            ),
            "Invariant failed: audit['history_ids'] == history.event_id.tolist() and audit['test_ids'] == current.event_id.tolist()",
        )
        require(
            (
                audit["recipe"] == recipe
                and audit["epochs"] == epochs
                and (audit["seeds"] == [42, 43])
            ),
            "Invariant failed: audit['recipe'] == recipe and audit['epochs'] == epochs and (audit['seeds'] == [42, 43])",
        )
        require(
            (audit["historical_weight_updates_only"] and (not audit["early_stopping_performed"])),
            "Invariant failed: audit['historical_weight_updates_only'] and (not audit['early_stopping_performed'])",
        )
        require(
            (
                not audit["recipe_changed"]
                and audit["selection_sha256"] == sha(OUT / "LOCKED_SELECTION.json")
            ),
            "Invariant failed: not audit['recipe_changed'] and audit['selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
        )
        saved = pd.read_parquet(folder / "PREDICTIONS.parquet")
        expected = pred[pred.quarter.eq(q)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, expected, check_exact=True, check_dtype=False)
        seedpred = []
        for seed in [42, 43]:
            seedpred.append(
                checkpoint_predict(folder / f"seed{seed}.pt", matrix, recipe, epochs, seed)
            )
            replays += 1
        close(np.asarray(seedpred), np.load(folder / "SEED_PREDICTIONS.npy"), atol=1e-8, rtol=1e-6)
        p = np.mean(seedpred, axis=0)
        close(p, saved.prediction.to_numpy(), atol=1e-8, rtol=1e-6)
        maxdiff = max(maxdiff, float(np.max(abs(p - saved.prediction.to_numpy()))))
        sign = 1 if 2 * int(history.y.gt(0).sum()) >= len(history) else -1
        close(saved.history_mean, float(history.y.mean()), atol=0, rtol=0)
        require(
            (saved.history_majority_sign.eq(sign).all()),
            "Invariant failed: saved.history_majority_sign.eq(sign).all()",
        )
        histories.append(
            dict(
                quarter=q,
                history_events=len(history),
                history_positive=int(history.y.gt(0).sum()),
                majority_sign=sign,
                history_mean=float(history.y.mean()),
                history_end=history.reaction_session.max(),
                evaluation_start=current.reaction_session.min(),
                evaluation_events=len(current),
            )
        )
    require(
        (pred.recipe_id.eq(lock["selected_recipe_id"]).all()),
        "Invariant failed: pred.recipe_id.eq(lock['selected_recipe_id']).all()",
    )
    return (
        data,
        pred,
        histories,
        dict(
            evaluation_rows=497,
            evaluation_checkpoint_replays=replays,
            maximum_evaluation_prediction_difference=maxdiff,
            evaluation_epochs_frozen=epochs,
            evaluation_model_choices=0,
            evaluation_stopping_choices=0,
            chronological_weight_refits=8,
        ),
    )


def metric_check(frame, p):
    """Independently calculate forecast metrics for comparison with the saved report."""
    y = frame.y.to_numpy(float)
    error = p - y
    actual = y > 0
    calls = p > 0
    correct = actual == calls
    has_direction = bool((p != 0).all())
    majority = (frame.history_majority_sign.to_numpy() > 0) == actual
    return dict(
        n=len(frame),
        mse=float(np.mean(error**2)),
        rmse_pp=100 * float(np.sqrt(np.mean(error**2))),
        mae_pp=100 * float(np.mean(abs(error))),
        bias_pp=100 * float(error.mean()),
        correct=int(correct.sum()) if has_direction else np.nan,
        accuracy_pct=100 * float(correct.mean()) if has_direction else np.nan,
        balanced_accuracy_pct=100
        * float(
            np.mean([correct[actual == s].mean() for s in [False, True] if (actual == s).any()])
        )
        if has_direction
        else np.nan,
        majority_correct=int(majority.sum()),
        majority_accuracy_pct=100 * float(majority.mean()),
        prediction_mean_pp=100 * float(p.mean()),
        prediction_sd_pp=100 * float(p.std(ddof=0)),
        predicted_positive=int((p > 0).sum()),
        predicted_negative=int((p < 0).sum()),
        predicted_zero=int((p == 0).sum()),
        actual_positive=int((y > 0).sum()),
        actual_negative=int((y < 0).sum()),
        ticker_clusters=int(frame.ticker.nunique()),
        date_clusters=int(frame.reaction_session.nunique()),
        first_event=frame.reaction_session.min(),
        last_event=frame.reaction_session.max(),
    )


def independent_decision(row):
    """Apply the declared corrected-significance rule without consulting model performance ranks."""
    if not row["valid"]:
        return "inconclusive"
    if not row["reject_holm"]:
        return "not_significant"
    return "improvement" if row["effect"] > 0 else "deterioration"


def reporting_check(pred, histories):
    # These separately implemented helpers contain no data loading at import.
    """Recompute report metrics and statistical decisions from the unchanged matched forecasts."""
    from work.market_ridge_fixed_20261118.verify import (
        independent_bootstrap,
        independent_holm,
        independent_test,
    )

    receipt = read(OUT / "REPORT_COMPLETE.json")
    for group in ["input_sha256", "output_sha256"]:
        for name, digest in receipt[group].items():
            require((sha(OUT / name) == digest), name)
    require(
        (receipt["report_script_sha256"] == sha(Path(__file__).with_name("report.py"))),
        "Invariant failed: receipt['report_script_sha256'] == sha(Path(__file__).with_name('report.py'))",
    )
    require(
        (
            receipt["inference_script_sha256"]
            == sha(ROOT / "work/model_selection_20261118/statisticalinference.py")
        ),
        "Invariant failed: receipt['inference_script_sha256'] == sha(ROOT / 'work/model_selection_20261118/statisticalinference.py')",
    )
    periods = ["2025", "2026", "2025-2026"]
    methods = ["ticker", "circular_5_observed_dates", "circular_10_observed_dates"]
    tables = {
        "mse": pd.read_csv(OUT / "MSE_TESTS.csv", float_precision="round_trip"),
        "direction": pd.read_csv(OUT / "DIRECTION_TESTS.csv", float_precision="round_trip"),
    }
    boot_tables = {
        "mse": pd.read_csv(OUT / "MSE_BOOTSTRAP_SENSITIVITY.csv", float_precision="round_trip"),
        "direction": pd.read_csv(
            OUT / "DIRECTION_BOOTSTRAP_SENSITIVITY.csv", float_precision="round_trip"
        ),
    }
    metrics = pd.read_csv(OUT / "METRICS.csv", float_precision="round_trip")
    yearly = pd.read_csv(OUT / "YEARLY_RESULTS.csv", float_precision="round_trip")
    require(
        (len(metrics) == 15 and len(yearly) == 3),
        "Invariant failed: len(metrics) == 15 and len(yearly) == 3",
    )
    require(
        (len(tables["mse"]) == 12 and len(tables["direction"]) == 6),
        "Invariant failed: len(tables['mse']) == 12 and len(tables['direction']) == 6",
    )
    require(
        (len(boot_tables["mse"]) == 36 and len(boot_tables["direction"]) == 18),
        "Invariant failed: len(boot_tables['mse']) == 36 and len(boot_tables['direction']) == 18",
    )
    baseline = pd.read_csv(OUT / "BASELINE_AUDIT.csv", float_precision="round_trip")
    require((len(baseline) == 8), "Invariant failed: len(baseline) == 8")
    for item in histories:
        row = baseline[baseline.quarter.eq(item["quarter"])].iloc[0]
        for key, value in item.items():
            if isinstance(value, str):
                require((row[key] == value), "Invariant failed: row[key] == value")
            else:
                close(row[key], value)
    computed = {"mse": [], "direction": []}
    boots = {"mse": [], "direction": []}
    for period in periods:
        frame = pred if period == "2025-2026" else pred[pred.year.eq(int(period))]
        y = frame.y.to_numpy(float)
        forecasts = dict(
            selected_nn=frame.prediction.to_numpy(float),
            original_nn=frame.original_nn_prediction.to_numpy(float),
            zero_AR=np.zeros(len(frame)),
            historical_mean=frame.history_mean.to_numpy(float),
            market_ridge=frame.market_ridge_prediction.to_numpy(float),
        )
        for system, p in forecasts.items():
            want = metric_check(frame, p)
            row = metrics[metrics.period.eq(period) & metrics.system.eq(system)].iloc[0]
            for key, value in want.items():
                if isinstance(value, str):
                    require((row[key] == value), "Invariant failed: row[key] == value")
                else:
                    close(row[key], value)
            if system == "selected_nn":
                yr = yearly[yearly.period.eq(period)].iloc[0]
                for key, value in want.items():
                    if key not in yr.index:
                        continue
                    if isinstance(value, str):
                        require((yr[key] == value), "Invariant failed: yr[key] == value")
                    else:
                        close(yr[key], value)
        loss = (forecasts["selected_nn"] - y) ** 2
        differences = {
            "mse": {
                name: (forecasts[name] - y) ** 2 - loss
                for name in ["original_nn", "zero_AR", "historical_mean", "market_ridge"]
            },
            "direction": {
                "chance_50": ((forecasts["selected_nn"] > 0) == (y > 0)).astype(float) - 0.5,
                "historical_majority": ((forecasts["selected_nn"] > 0) == (y > 0)).astype(float)
                - ((frame.history_majority_sign.to_numpy() > 0) == (y > 0)).astype(float),
            },
        }
        for metric, contrasts in differences.items():
            for comparator, diff in contrasts.items():
                identity = dict(
                    test_id=period + "__" + metric + "__" + comparator,
                    period=period,
                    comparator=comparator,
                    metric=metric,
                )
                result = independent_test(diff, frame.ticker, frame.reaction_session)
                result.update(identity)
                computed[metric].append(result)
                for method in methods:
                    boot = independent_bootstrap(diff, frame.ticker, frame.reaction_session, method)
                    boot.update(identity)
                    boots[metric].append(boot)
    for metric in ["mse", "direction"]:
        independent_holm(computed[metric])
        for method in methods:
            independent_holm([r for r in boots[metric] if r["method"] == method])
        for rows, table in [
            (computed[metric], tables[metric]),
            (boots[metric], boot_tables[metric]),
        ]:
            for result in rows:
                matched = table[table.test_id.eq(result["test_id"])]
                if "method" in result:
                    matched = matched[matched.method.eq(result["method"])]
                require((len(matched) == 1), "Invariant failed: len(matched) == 1")
                row = matched.iloc[0]
                for key, value in result.items():
                    if isinstance(value, str):
                        require((row[key] == value), "Invariant failed: row[key] == value")
                    elif isinstance(value, bool):
                        require(
                            (bool(row[key]) == value), "Invariant failed: bool(row[key]) == value"
                        )
                    else:
                        close(row[key], value, atol=1e-12, rtol=1e-9)
                factor = 10000 if metric == "mse" else 100
                suffix = "_pp2" if metric == "mse" else "_pp"
                for key in ["effect", "ci_low", "ci_high"]:
                    close(row[key + suffix], result[key] * factor)
                require(
                    (row["decision"] == independent_decision(result)),
                    "Invariant failed: row['decision'] == independent_decision(result)",
                )
                require(
                    (row.holm_family_size == (12 if metric == "mse" else 6)),
                    "Invariant failed: row.holm_family_size == (12 if metric == 'mse' else 6)",
                )
                if "method" in result:
                    require(
                        (row.replicates == 5000 and row.seed == 20261118),
                        "Invariant failed: row.replicates == 5000 and row.seed == 20261118",
                    )
        for period in periods:
            row = yearly[yearly.period.eq(period)].iloc[0]
            for result in [r for r in computed[metric] if r["period"] == period]:
                prefix = ("mse_vs_" if metric == "mse" else "direction_vs_") + result["comparator"]
                close(
                    row[prefix + ("_p_holm12" if metric == "mse" else "_p_holm6")], result["p_holm"]
                )
                require(
                    (row[prefix + "_decision"] == independent_decision(result)),
                    "Invariant failed: row[prefix + '_decision'] == independent_decision(result)",
                )
    for period in periods:
        chance = next(
            r
            for r in computed["direction"]
            if r["period"] == period and r["comparator"] == "chance_50"
        )
        row = yearly[yearly.period.eq(period)].iloc[0]
        close(row.accuracy_nominal_ci_low_pct, 50 + 100 * chance["ci_low"])
        close(row.accuracy_nominal_ci_high_pct, 50 + 100 * chance["ci_high"])
    ranking = pd.read_csv(OUT / "DEVELOPMENT_RANKING.csv", float_precision="round_trip")
    original_ranking = pd.read_csv(OUT / "DEV_SCORES.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        ranking[original_ranking.columns].reset_index(drop=True),
        original_ranking.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    conclusion = read(OUT / "CONCLUSION.json")
    lock = selection_check()
    for name in ["selected_recipe_id", "selected_recipe", "selected_epoch"]:
        require(
            (conclusion[name] == lock[name]), "Invariant failed: conclusion[name] == lock[name]"
        )
    require(
        (conclusion["selection_criterion"] == lock["criterion"]),
        "Invariant failed: conclusion['selection_criterion'] == lock['criterion']",
    )
    close(conclusion["development_validation_mse"], lock["validation_mse"])
    original_mse = float(original_ranking.loc[original_ranking.is_original, "mse"].iloc[0])
    close(
        conclusion["development_mse_gain_vs_original_pct"],
        100 * (original_mse - float(lock["validation_mse"])) / original_mse,
    )
    require(
        (
            conclusion["development_selection_rows"] == 12
            and conclusion["development_selection_events"] == 463
        ),
        "Invariant failed: conclusion['development_selection_rows'] == 12 and conclusion['development_selection_events'] == 463",
    )
    require(
        (
            conclusion["development_only_decisions"]
            and (not conclusion["evaluation_used_for_selection"])
        ),
        "Invariant failed: conclusion['development_only_decisions'] and (not conclusion['evaluation_used_for_selection'])",
    )
    require(
        (conclusion["evaluation_events"] == len(pred) == 497),
        "Invariant failed: conclusion['evaluation_events'] == len(pred) == 497",
    )
    require(
        (
            conclusion["evaluation_year_counts"]
            == {str(k): int(v) for k, v in pred.groupby("year").size().items()}
        ),
        "Invariant failed: conclusion['evaluation_year_counts'] == {str(k): int(v) for k, v in pred.groupby('year').size().items()}",
    )
    require(
        (conclusion["periods"] == periods and conclusion["main_metric"] == "MSE"),
        "Invariant failed: conclusion['periods'] == periods and conclusion['main_metric'] == 'MSE'",
    )
    require(
        (conclusion["direction_is_secondary"] and conclusion["partial_2026"]),
        "Invariant failed: conclusion['direction_is_secondary'] and conclusion['partial_2026']",
    )
    require(
        (conclusion["last_2026_event"] == pred.loc[pred.year.eq(2026), "reaction_session"].max()),
        "Invariant failed: conclusion['last_2026_event'] == pred.loc[pred.year.eq(2026), 'reaction_session'].max()",
    )
    require(
        (
            conclusion["inference_conditional_on_saved_forecasts"]
            and conclusion["no_automatic_deployment"]
        ),
        "Invariant failed: conclusion['inference_conditional_on_saved_forecasts'] and conclusion['no_automatic_deployment']",
    )
    require(
        (conclusion["original_files_unchanged"]),
        "Invariant failed: conclusion['original_files_unchanged']",
    )
    for name in [
        "evaluation_model_selection_changes",
        "llm_calls",
        "embedding_calls",
        "report_fit_calls",
    ]:
        require((conclusion[name] == 0), "Invariant failed: conclusion[name] == 0")
    for metric in ["mse", "direction"]:
        rows = computed[metric]
        samples = boots[metric]
        decisions = {r["test_id"]: independent_decision(r) for r in rows}
        good = [key for key, value in decisions.items() if value == "improvement"]
        expected = dict(
            significant_improvements=good,
            significant_deteriorations=[
                key for key, value in decisions.items() if value == "deterioration"
            ],
            inconclusive_tests=[key for key, value in decisions.items() if value == "inconclusive"],
            improvements_supported_by_all_bootstrap_methods=[
                key
                for key in good
                if all(
                    independent_decision(
                        next(b for b in samples if b["test_id"] == key and b["method"] == method)
                    )
                    == "improvement"
                    for method in methods
                )
            ],
            cluster_bootstrap_decision_disagreements=[
                key
                for key, value in decisions.items()
                if any(independent_decision(b) != value for b in samples if b["test_id"] == key)
            ],
        )
        require(
            (conclusion[metric] == expected), "Invariant failed: conclusion[metric] == expected"
        )
    nested = pd.read_parquet(OUT / "NESTED_DEV_PREDICTIONS.parquet")
    for key, value in score(nested).items():
        close(conclusion["nested_development_diagnostic"][key], value)
    require(
        (not conclusion["nested_development_diagnostic"]["used_for_global_selection"]),
        "Invariant failed: not conclusion['nested_development_diagnostic']['used_for_global_selection']",
    )
    return dict(
        passed=True,
        metric_rows=15,
        mse_tests=12,
        direction_tests=6,
        mse_bootstrap_rows=36,
        direction_bootstrap_rows=18,
        bootstrap_replicates=5000,
        all_pvalues_recomputed_independently=True,
        all_confidence_intervals_verified=True,
        conclusion_consistent_with_independent_metrics=True,
        mse_significant_improvements=[
            r["test_id"] for r in computed["mse"] if independent_decision(r) == "improvement"
        ],
        direction_significant_improvements=[
            r["test_id"] for r in computed["direction"] if independent_decision(r) == "improvement"
        ],
        independent_inference_helpers_sha256=sha(
            ROOT / "work/market_ridge_fixed_20261118/verify.py"
        ),
    )


def all_check():
    """Run the complete read-only audit of inputs, development, evaluation, and reporting."""
    result = development_check()
    data, pred, histories, later = later_models()
    report = reporting_check(pred, histories)
    result.update(later)
    result.update(
        report=report,
        no_evaluation_outcomes_read=False,
        evaluation_read_only_after_verified_lock=True,
        old_files_preserved=True,
    )
    frozen()
    selection_check()
    return result


def main():
    """Parse this module's command-line options and execute the explicitly requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["inputs", "development", "all"])
    args = parser.parse_args()
    activate_legacy_root(ROOT)
    if args.stage == "inputs":
        result = inputs_check()[3]
        name = "INDEPENDENT_DEV_INPUT_CHECK.json"
    elif args.stage == "development":
        result = development_check()
        name = "INDEPENDENT_DEVELOPMENT_AUDIT.json"
    else:
        result = all_check()
        name = "INDEPENDENT_AUDIT.json"
    result.update(
        verifier_sha256=sha(Path(__file__)), verified_utc=pd.Timestamp.now(tz="UTC").isoformat()
    )
    write(OUT / name, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

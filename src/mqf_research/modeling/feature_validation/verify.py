"""Independent checks: event-wise Attention oracle, saved-model replay, no training.

The engine check imports the production engine only as the object under test.
Its reference network uses explicit separate event bags, never padded tensors.
No evaluation labels or forecasts are read by the engine check.
"""

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
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

ROOT = get_settings().research_root

OUT = get_settings().output_root / "nn_feature_validation"
OLD = ROOT / "outputs/nn_tuning_20261123_b032"
DEVQ = [f"{year}Q{quarter}" for year in (2023, 2024) for quarter in range(1, 5)]
FLAGS = ["text_available", "has_signals", "log_signal_count"]
HORIZONS = ["hour", "half", "full"]
GROUPS = ["M", "MF", "MFL"]
EVALQ = [f"{year}Q{quarter}" for year in (2025, 2026) for quarter in range(1, 5)]
RELEASE = ROOT / "outputs/source_repair_batches/batch_032"
CONTEXT = ROOT / "outputs/valuation_industry_20261113_b032"
TAXONOMY = ROOT / "outputs/model_selection_20261118_b032/FEATURE_GROUPS.json"
COMMON = ROOT / "outputs/feature_horizon_comparison_20261111_b032/COMMON_EVENTS.csv"
RECIPE = dict(
    numeric_width=32,
    head_width=64,
    dropout=0.2,
    weight_decay=0.05,
    learning_rate=0.002,
    projection_width=32,
    gate_width=16,
    batch_size=128,
    max_epochs=40,
    patience=5,
    minimum_search_epochs=5,
)
torch.set_num_threads(1)


def write(path, value):
    """Atomically publish the historical JSON schema in the isolated run directory."""
    write_json(path, value, default=str, create_parents=True)


def close(actual, expected, atol=1e-12, rtol=1e-9):
    """Compare independently computed arrays using the declared numeric tolerances."""
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


class EventWiseNetwork(nn.Module):
    """Reference implementation; each event's real signals are processed alone."""

    def __init__(self, numeric_dim, use_text):
        """Create the model/helper in the fixed parameter order used by the experiment."""
        super().__init__()
        self.use_text = bool(use_text)
        self.proj = nn.Linear(384, 32)
        self.v = nn.Linear(32, 16)
        self.u = nn.Linear(32, 16)
        self.a = nn.Linear(16, 1, bias=False)
        self.numeric = nn.Sequential(nn.Linear(numeric_dim, 32), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, signal_lists=None, return_details=False):
        """Return one signed-AR forecast per event from numeric features and optional signal bags."""
        pool = []
        weights = []
        for event in range(len(x)):
            z = signal_lists[event] if self.use_text else None
            if z is None or len(z) == 0:
                pool.append(x.new_zeros(32))
                weights.append(x.new_zeros(0))
            else:
                h = self.proj(z)
                logits = self.a(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).reshape(-1)
                w = torch.softmax(logits, dim=0)
                pool.append((w[:, None] * h).sum(dim=0))
                weights.append(w)
        latent = torch.stack(pool)
        prediction = self.head(torch.cat([self.numeric(x), torch.tanh(latent)], dim=1)).reshape(-1)
        return (prediction, latent, weights) if return_details else prediction


def individual_signals(ids, vectors, bags):
    """Extract each event's true signal vectors for the independent unpadded oracle."""
    return [
        torch.from_numpy(np.asarray(vectors[bags[str(event)]], dtype=np.float32)) for event in ids
    ]


def scalar_stopping_check(info):
    """Recompute the stopping decision, not the underlying fitted curve values."""
    curve = info["curve"]
    best, best_epoch, stale, halted = float("inf"), None, 0, None
    for epoch, row in enumerate(curve, 1):
        require((row["epoch"] == epoch), "Invariant failed: row['epoch'] == epoch")
        loss = float(row["validation_mse"])
        require(
            (np.isfinite(loss) and loss >= 0), "Invariant failed: np.isfinite(loss) and loss >= 0"
        )
        close(row["validation_rmse_pp"], 100 * np.sqrt(loss))
        if loss < best:
            best, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        if epoch >= 5 and stale >= 5:
            halted = epoch
            require((epoch == len(curve)), "Invariant failed: epoch == len(curve)")
            break
    if halted is None:
        require((len(curve) == 40), "Invariant failed: len(curve) == 40")
        halted = 40
    require(
        (info["selected_epoch"] == best_epoch and info["stopped_epoch"] == halted),
        "Invariant failed: info['selected_epoch'] == best_epoch and info['stopped_epoch'] == halted",
    )
    require(
        (bool(info["hit_cap"]) == (halted == 40)),
        "Invariant failed: bool(info['hit_cap']) == (halted == 40)",
    )
    close(info["best_inner_mse"], best)
    return best


def engine_check():
    """Check the maintained engine against an event-wise oracle using synthetic signals."""
    from mqf_research.modeling.feature_validation import engine as production

    engine_path = Path(production.__file__)
    initial_hash = sha(engine_path)
    require((production.SEEDS == [42, 43]), "Invariant failed: production.SEEDS == [42, 43]")
    require(
        (
            production.RECIPE
            == dict(
                numeric_width=32,
                head_width=64,
                dropout=0.2,
                weight_decay=0.05,
                learning_rate=0.002,
                projection_width=32,
                gate_width=16,
                batch_size=128,
                max_epochs=40,
                patience=5,
                minimum_search_epochs=5,
            )
        ),
        "Invariant failed: production.RECIPE == dict(numeric_width=32, head_width=64, dropout=0.2, weight_decay=0.05, learning_rate=0.002, projection_width=32, gate_width=16, batch_size=128, max_epochs=40, patience=5, minimum_search_epochs=5)",
    )
    torch.manual_seed(1849)
    tested = production.Network(7, True).eval()
    oracle = EventWiseNetwork(7, True).eval()
    oracle.load_state_dict(tested.state_dict())
    rng = np.random.default_rng(1849)
    x = torch.from_numpy(rng.normal(size=(4, 7)).astype("float32"))
    y = torch.from_numpy(rng.normal(size=4).astype("float32"))
    vectors = rng.normal(size=(7, 384)).astype("float32")
    ids = np.asarray(["alpha", "empty", "single", "beta"])
    bags = dict(alpha=[0, 1, 2, 3], empty=[], single=[4], beta=[5, 6])
    padded, mask = production.bag_batch(ids, vectors, bags)
    require(
        (mask.dtype == torch.bool and mask.sum().item() == 7),
        "Invariant failed: mask.dtype == torch.bool and mask.sum().item() == 7",
    )
    for i, event in enumerate(ids):
        np.testing.assert_array_equal(padded[i, mask[i]].numpy(), vectors[bags[event]])
        require(
            (mask[i].sum().item() == len(bags[event])),
            "Invariant failed: mask[i].sum().item() == len(bags[event])",
        )
    real = individual_signals(ids, vectors, bags)
    expected, reference_pool, reference_weights = oracle(x, real, True)
    actual = tested(x, padded, mask)
    pool, weights = tested.text_branch(padded, mask)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(pool, reference_pool, atol=2e-7, rtol=1e-6)
    for i in range(len(ids)):
        torch.testing.assert_close(weights[i, mask[i]], reference_weights[i], atol=1e-7, rtol=1e-6)
    require(
        (torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))),
        "Invariant failed: torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))",
    )
    require(
        (torch.equal(pool[1], torch.zeros(32))),
        "Invariant failed: torch.equal(pool[1], torch.zeros(32))",
    )
    torch.testing.assert_close(weights.sum(1), torch.tensor([1.0, 0.0, 1.0, 1.0]))
    ((actual - y) ** 2).mean().backward()
    ((expected - y) ** 2).mean().backward()
    for name, param in tested.named_parameters():
        expected_grad = dict(oracle.named_parameters())[name].grad
        require(
            (param.grad is not None and expected_grad is not None),
            "Invariant failed: param.grad is not None and expected_grad is not None",
        )
        require(
            (torch.isfinite(param.grad).all()), "Invariant failed: torch.isfinite(param.grad).all()"
        )
        torch.testing.assert_close(param.grad, expected_grad, atol=2e-7, rtol=2e-5)
    # Batched padding must not mix information or weights across events.
    with torch.inference_mode():
        for i in range(len(ids)):
            emb_one, mask_one = production.bag_batch(ids[i : i + 1], vectors, bags)
            torch.testing.assert_close(
                actual[i : i + 1], tested(x[i : i + 1], emb_one, mask_one), atol=1e-7, rtol=1e-6
            )
        replaced = padded.clone()
        replaced[0] = 0
        torch.testing.assert_close(actual[1:], tested(x, replaced, mask)[1:], atol=1e-7, rtol=1e-6)
        reversed_bags = {event: list(reversed(indices)) for event, indices in bags.items()}
        reverse_emb, reverse_mask = production.bag_batch(ids, vectors, reversed_bags)
        torch.testing.assert_close(
            actual, tested(x, reverse_emb, reverse_mask), atol=1e-7, rtol=1e-6
        )
        extra_pad = torch.cat([padded, torch.full((4, 3, 384), 999.0)], dim=1)
        extra_mask = torch.cat([mask, torch.zeros((4, 3), dtype=torch.bool)], dim=1)
        torch.testing.assert_close(actual, tested(x, extra_pad, extra_mask), atol=1e-7, rtol=1e-6)
    # Entirely empty batches need finite gradients, not just an empty event among nonempty events.
    tested.zero_grad(set_to_none=True)
    empty_bags = {event: [] for event in ids}
    empty_emb, empty_mask = production.bag_batch(ids, vectors, empty_bags)
    require(
        (empty_emb.shape == (4, 1, 384) and (not empty_mask.any())),
        "Invariant failed: empty_emb.shape == (4, 1, 384) and (not empty_mask.any())",
    )
    empty_pool, empty_weights = tested.text_branch(empty_emb, empty_mask)
    require(
        (torch.equal(empty_pool, torch.zeros((4, 32)))),
        "Invariant failed: torch.equal(empty_pool, torch.zeros((4, 32)))",
    )
    require(
        (torch.equal(empty_weights, torch.zeros((4, 1)))),
        "Invariant failed: torch.equal(empty_weights, torch.zeros((4, 1)))",
    )
    empty_prediction = tested(x, empty_emb, empty_mask)
    numeric = EventWiseNetwork(7, False).eval()
    numeric.load_state_dict(tested.state_dict())
    torch.testing.assert_close(empty_prediction, numeric(x), atol=0, rtol=0)
    empty_prediction.square().mean().backward()
    require(
        (all((p.grad is None or torch.isfinite(p.grad).all() for p in tested.parameters()))),
        "Invariant failed: all((p.grad is None or torch.isfinite(p.grad).all() for p in tested.parameters()))",
    )
    for name, param in tested.named_parameters():
        if name.startswith(("proj.", "v.", "u.", "a.")):
            require(
                (param.grad is None or torch.count_nonzero(param.grad) == 0),
                "Invariant failed: param.grad is None or torch.count_nonzero(param.grad) == 0",
            )
    # Replay only old DEVELOPMENT checkpoints. No fitting and no later-period read.
    replayed = 0
    max_difference = 0.0
    protected = {}
    for quarter in DEVQ:
        folder = OLD / "development" / quarter
        matrix_path = folder / "FULL.npz"
        protected[str(matrix_path)] = sha(matrix_path)
        matrix = np.load(matrix_path, allow_pickle=False)
        dest = folder / "n32_h64_d20_w5"
        source_predictions = np.load(dest / "SEED_PREDICTIONS.npy", allow_pickle=False)
        protected[str(dest / "SEED_PREDICTIONS.npy")] = sha(dest / "SEED_PREDICTIONS.npy")
        for j, seed in enumerate([42, 43]):
            checkpoint_path = dest / f"seed{seed}.pt"
            protected[str(checkpoint_path)] = sha(checkpoint_path)
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = production.Network(saved["numeric_dim"], False)
            model.load_state_dict(saved["state_dict"])
            result = production.predict(
                model,
                torch.from_numpy(matrix["X_test"]),
                matrix["test_ids"],
                None,
                {},
                saved["target_mean"],
                saved["target_sd"],
            )
            reference = EventWiseNetwork(saved["numeric_dim"], False).eval()
            reference.load_state_dict(saved["state_dict"])
            raw = []
            with torch.inference_mode():
                for start in range(0, len(matrix["X_test"]), 128):
                    values = reference(
                        torch.from_numpy(matrix["X_test"][start : start + 128])
                    ).numpy()
                    raw.extend((values * saved["target_sd"] + saved["target_mean"]).tolist())
            np.testing.assert_array_equal(result, np.asarray(raw))
            np.testing.assert_array_equal(result, source_predictions[j])
            max_difference = max(max_difference, float(np.max(abs(result - source_predictions[j]))))
            replayed += 1
    require(
        (initial_hash == sha(engine_path)), "Invariant failed: initial_hash == sha(engine_path)"
    )
    require(
        (all((sha(path) == digest for path, digest in protected.items()))),
        "Invariant failed: all((sha(path) == digest for path, digest in protected.items()))",
    )
    return dict(
        passed=True,
        independent_eventwise_attention_formula=True,
        independent_attention_gradient_formula=True,
        attention_normalization=True,
        true_signal_order_and_no_truncation=True,
        empty_event_and_all_empty_batch=True,
        empty_text_gradient_zero=True,
        event_isolation=True,
        permutation_invariance=True,
        padding_invariance=True,
        numeric_development_checkpoint_replays=replayed,
        maximum_numeric_replay_difference=max_difference,
        no_evaluation_outcomes_read=True,
        model_fit_calls=0,
        optimizer_steps=0,
        source_hashes_verified=len(protected),
        source_files_preserved=True,
        engine_sha256=initial_hash,
    )


def frozen_check():
    """Reject any change to inputs or source code recorded by this run."""
    bind_run_configuration(OUT)
    entries = read(OUT / "FROZEN_INPUTS.json")
    for path, digest in entries.items():
        require((sha(path) == digest), path)
    return len(entries)


def complete_check(folder):
    """Verify a completed model cell against its per-file hash receipt."""
    receipt = read(folder / "COMPLETE.json")
    for name, digest in receipt["hashes"].items():
        require((sha(folder / name) == digest), str(folder / name))
    return receipt


def selection_gate():
    """Verify the completed development selection before accessing evaluation artifacts."""
    receipt = read(OUT / "SELECTION_COMPLETE.json")
    require(
        (receipt["development_only"] and receipt["no_evaluation_outcomes_used"]),
        "Invariant failed: receipt['development_only'] and receipt['no_evaluation_outcomes_used']",
    )
    require(
        (sha(OUT / "PROTOCOL.json") == receipt["protocol_sha256"]),
        "Invariant failed: sha(OUT / 'PROTOCOL.json') == receipt['protocol_sha256']",
    )
    require(
        (sha(OUT / "LOCKED_SELECTION.json") == receipt["locked_selection_sha256"]),
        "Invariant failed: sha(OUT / 'LOCKED_SELECTION.json') == receipt['locked_selection_sha256']",
    )
    for name, digest in receipt["hashes"].items():
        require((sha(OUT / name) == digest), name)
    lock = read(OUT / "LOCKED_SELECTION.json")
    require(
        (lock["development_only"] and lock["no_evaluation_outcomes_used"]),
        "Invariant failed: lock['development_only'] and lock['no_evaluation_outcomes_used']",
    )
    require((lock["architecture"] == RECIPE), "Invariant failed: lock['architecture'] == RECIPE")
    require(
        (
            lock["input_encoder"] == "minilm"
            and lock["input_horizons"] == HORIZONS
            and (lock["groups"] == GROUPS)
        ),
        "Invariant failed: lock['input_encoder'] == 'minilm' and lock['input_horizons'] == HORIZONS and (lock['groups'] == GROUPS)",
    )
    require(
        (not lock["evaluation_early_stopping"] and (not lock["calibration_evaluation_updates"])),
        "Invariant failed: not lock['evaluation_early_stopping'] and (not lock['calibration_evaluation_updates'])",
    )
    require(
        (lock["llm_calls"] == lock["embedding_calls"] == 0),
        "Invariant failed: lock['llm_calls'] == lock['embedding_calls'] == 0",
    )
    require(
        (set(lock["model_selections"]) == set(HORIZONS)),
        "Invariant failed: set(lock['model_selections']) == set(HORIZONS)",
    )
    require(
        (all((set(lock["model_selections"][h]) == set(GROUPS) for h in HORIZONS))),
        "Invariant failed: all((set(lock['model_selections'][h]) == set(GROUPS) for h in HORIZONS))",
    )
    return lock


def canonical_check(stage):
    """Reconstruct permitted data and every signal directly, without the production loader."""
    if stage == "evaluation":
        selection_gate()
    else:
        require((stage == "development"), "Invariant failed: stage == 'development'")
    prefix = "DEV" if stage == "development" else "EVAL"
    receipt = read(
        OUT / ("PREPARED.json" if stage == "development" else "EVALUATION_PREPARED.json")
    )
    for name, digest in receipt["hashes"].items():
        require((sha(OUT / name) == digest), name)
    if stage == "evaluation":
        require(
            (receipt["locked_selection_sha256"] == sha(OUT / "LOCKED_SELECTION.json")),
            "Invariant failed: receipt['locked_selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
        )
    protocol = read(OUT / "PROTOCOL.json")
    require(
        (protocol["input_horizons"] == HORIZONS and protocol["input_encoder"] == "minilm"),
        "Invariant failed: protocol['input_horizons'] == HORIZONS and protocol['input_encoder'] == 'minilm'",
    )
    require(
        (
            protocol["primary_horizon"] == "hour"
            and protocol["secondary_horizons"] == ["half", "full"]
        ),
        "Invariant failed: protocol['primary_horizon'] == 'hour' and protocol['secondary_horizons'] == ['half', 'full']",
    )
    require(
        (
            protocol["architecture"]["seeds"] == [42, 43]
            and (not protocol["architecture"]["signal_truncation"])
        ),
        "Invariant failed: protocol['architecture']['seeds'] == [42, 43] and (not protocol['architecture']['signal_truncation'])",
    )
    require(
        (protocol["training_policy"]["maximum_epochs"] == 40),
        "Invariant failed: protocol['training_policy']['maximum_epochs'] == 40",
    )
    require(
        (
            protocol["training_policy"]["patience"]
            == protocol["training_policy"]["minimum_search_epochs"]
            == 5
        ),
        "Invariant failed: protocol['training_policy']['patience'] == protocol['training_policy']['minimum_search_epochs'] == 5",
    )
    require(
        (not protocol["training_policy"]["evaluation_early_stopping"]),
        "Invariant failed: not protocol['training_policy']['evaluation_early_stopping']",
    )
    groups = read(OUT / "FEATURE_GROUPS.json")
    taxonomy = read(TAXONOMY)
    require((set(groups) == set(GROUPS)), "Invariant failed: set(groups) == set(GROUPS)")
    for group, count in [("M", 23), ("MF", 74), ("MFL", 77)]:
        require(
            (groups[group]["numeric"] == taxonomy[group]["numeric"]),
            "Invariant failed: groups[group]['numeric'] == taxonomy[group]['numeric']",
        )
        require(
            (groups[group]["categorical"] == taxonomy[group]["categorical"]),
            "Invariant failed: groups[group]['categorical'] == taxonomy[group]['categorical']",
        )
        require(
            (len(groups[group]["numeric"]) == len(set(groups[group]["numeric"])) == count),
            "Invariant failed: len(groups[group]['numeric']) == len(set(groups[group]['numeric'])) == count",
        )
        require(
            (groups[group]["use_signals"] == (group == "MFL")),
            "Invariant failed: groups[group]['use_signals'] == (group == 'MFL')",
        )
    require(
        (groups["M"]["categorical"] == []), "Invariant failed: groups['M']['categorical'] == []"
    )
    categories = ["context_sector_ff12", "context_industry_sic"]
    require(
        (groups["MF"]["categorical"] == groups["MFL"]["categorical"] == categories),
        "Invariant failed: groups['MF']['categorical'] == groups['MFL']['categorical'] == categories",
    )
    require(
        (groups["MFL"]["numeric"] == groups["MF"]["numeric"] + FLAGS),
        "Invariant failed: groups['MFL']['numeric'] == groups['MF']['numeric'] + FLAGS",
    )
    require(
        (not set(FLAGS) & set(groups["MF"]["numeric"])),
        "Invariant failed: not set(FLAGS) & set(groups['MF']['numeric'])",
    )
    require(
        (set(groups["M"]["numeric"]) < set(groups["MF"]["numeric"])),
        "Invariant failed: set(groups['M']['numeric']) < set(groups['MF']['numeric'])",
    )
    cohort = (
        pd.read_csv(COMMON).sort_values(["reaction_session", "event_id"]).reset_index(drop=True)
    )
    require(
        (len(cohort) == 1341 and cohort.event_id.is_unique),
        "Invariant failed: len(cohort) == 1341 and cohort.event_id.is_unique",
    )
    if stage == "development":
        cohort = cohort[cohort.year.le(2024)].reset_index(drop=True)
    require(
        (len(cohort) == (844 if stage == "development" else 1341)),
        "Invariant failed: len(cohort) == (844 if stage == 'development' else 1341)",
    )
    ids = cohort.event_id.tolist()
    permitted = set(ids)
    if stage == "development":
        require((cohort.year.max() == 2024), "Invariant failed: cohort.year.max() == 2024")
    bag = read(OUT / f"{prefix}_BAGS.json")
    vectors = np.load(OUT / f"{prefix}_VECTORS.npy", allow_pickle=False)
    events = {
        row["event_id"]: row
        for row in read(RELEASE / "EVENTS.json")
        if row["event_id"] in permitted
    }
    require(
        (set(events) == set(bag) == permitted),
        "Invariant failed: set(events) == set(bag) == permitted",
    )
    index = read(RELEASE / "embeddings_minilm/INDEX.json")
    require(
        ([row["row"] for row in index] == list(range(len(index)))),
        "Invariant failed: [row['row'] for row in index] == list(range(len(index)))",
    )
    require(
        (len({row["signal_id"] for row in index}) == len(index)),
        "Invariant failed: len({row['signal_id'] for row in index}) == len(index)",
    )
    selected = [row for row in index if row["event_id"] in permitted]
    expected_bags = {event: [] for event in ids}
    for compact, row in enumerate(selected):
        expected_bags[row["event_id"]].append(compact)
    require((bag == expected_bags), "Invariant failed: bag == expected_bags")
    require(
        (
            sorted((row for entries in bag.values() for row in entries))
            == list(range(len(selected)))
        ),
        "Invariant failed: sorted((row for entries in bag.values() for row in entries)) == list(range(len(selected)))",
    )
    source_vectors = np.load(
        RELEASE / "embeddings_minilm/SIGNAL_VECTORS.npy", mmap_mode="r", allow_pickle=False
    )
    np.testing.assert_array_equal(vectors, source_vectors[[row["row"] for row in selected]])
    require(
        (
            vectors.dtype == np.float32
            and vectors.shape == (len(selected), 384)
            and np.isfinite(vectors).all()
        ),
        "Invariant failed: vectors.dtype == np.float32 and vectors.shape == (len(selected), 384) and np.isfinite(vectors).all()",
    )
    del source_vectors
    control = []
    for event in ids:
        item = events[event]
        count = len(bag[event])
        require(
            (item["ml_signal_count"] == count), "Invariant failed: item['ml_signal_count'] == count"
        )
        eligible = bool(item.get("text_source_eligible", True))
        available = bool(
            (
                item["extraction_status"] == "complete"
                or item.get("text_input_status") == "manual_source_review_complete"
            )
            and eligible
        )
        require(
            (not count or (eligible and available)),
            "Invariant failed: not count or (eligible and available)",
        )
        control.append(
            dict(
                event_id=event,
                text_available=available,
                has_signals=bool(count),
                log_signal_count=np.log1p(count),
            )
        )
    control = pd.DataFrame(control)
    binding = read(RELEASE / "DATASET_BINDING.json")
    base = binding["financial_features"]
    require(
        (len(base) == 67 and groups["M"]["numeric"] == base[:23]),
        "Invariant failed: len(base) == 67 and groups['M']['numeric'] == base[:23]",
    )
    extras = [field for field in groups["MF"]["numeric"] if field not in base]
    require((len(extras) == 7), "Invariant failed: len(extras) == 7")
    context = pd.read_parquet(
        CONTEXT / "CONTEXT_FEATURES.parquet",
        columns=["event_id"] + extras + categories,
        filters=[("event_id", "in", ids)],
    )
    require(
        (context.event_id.is_unique and set(context.event_id) == permitted),
        "Invariant failed: context.event_id.is_unique and set(context.event_id) == permitted",
    )
    frames = {}
    meta = [
        "event_id",
        "ticker",
        "reaction_session",
        "effective_utc",
        "cutoff_utc",
        "estimation_end",
    ]
    for horizon in HORIZONS:
        if stage == "evaluation":
            selection_gate()
        cached = pd.read_parquet(OUT / f"{prefix}_DATA_{horizon}.parquet")
        require(
            (cached.event_id.tolist() == ids), "Invariant failed: cached.event_id.tolist() == ids"
        )
        source = pd.read_parquet(
            binding["datasets"][horizon],
            columns=meta + ["y"] + base,
            filters=[("event_id", "in", ids)],
        )
        require(
            (len(source) == len(ids) and source.event_id.is_unique),
            "Invariant failed: len(source) == len(ids) and source.event_id.is_unique",
        )
        source["reaction_session"] = pd.to_datetime(source.reaction_session).dt.strftime("%Y-%m-%d")
        source["quarter"] = pd.to_datetime(source.reaction_session).dt.to_period("Q").astype(str)
        source["year"] = pd.to_datetime(source.reaction_session).dt.year
        source = source.merge(context, on="event_id", validate="one_to_one").merge(
            control, on="event_id", validate="one_to_one"
        )
        source = source.sort_values(["reaction_session", "event_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            cached, source[cached.columns], check_dtype=False, check_exact=True
        )
        require(
            (
                (
                    pd.to_datetime(cached.effective_utc, utc=True)
                    <= pd.to_datetime(cached.cutoff_utc, utc=True)
                ).all()
            ),
            "Invariant failed: (pd.to_datetime(cached.effective_utc, utc=True) <= pd.to_datetime(cached.cutoff_utc, utc=True)).all()",
        )
        require(
            (
                (
                    pd.to_datetime(cached.estimation_end) < pd.to_datetime(cached.reaction_session)
                ).all()
            ),
            "Invariant failed: (pd.to_datetime(cached.estimation_end) < pd.to_datetime(cached.reaction_session)).all()",
        )
        require((np.isfinite(cached.y).all()), "Invariant failed: np.isfinite(cached.y).all()")
        pd.testing.assert_frame_equal(
            cached[["event_id", "ticker", "reaction_session", "quarter", "year"]],
            cohort[["event_id", "ticker", "reaction_session", "quarter", "year"]],
            check_dtype=False,
            check_exact=True,
        )
        if stage == "evaluation":
            pd.testing.assert_frame_equal(
                cached[cached.year.le(2024)].reset_index(drop=True),
                pd.read_parquet(OUT / f"DEV_DATA_{horizon}.parquet"),
                check_dtype=False,
                check_exact=True,
            )
        frames[horizon] = cached
    pd.testing.assert_frame_equal(
        frames["hour"][frames["hour"].year.le(2024)].reset_index(drop=True),
        pd.read_parquet(ROOT / "outputs/small_sample_ai_20261118_b032/DATA.parquet"),
        check_dtype=False,
        check_exact=True,
    )
    return (
        frames,
        groups,
        bag,
        vectors,
        dict(
            canonical_source_rows_per_horizon=len(ids),
            signal_rows=len(selected),
            empty_bags=sum(not entries for entries in bag.values()),
            vector_values_bitwise_identical=True,
            all_signal_event_mappings_verified=True,
            text_controls_excluded_from_M_and_MF=True,
        ),
    )


def inputs_check():
    """Verify canonical input identity, chronology, feature membership, and preprocessing."""
    frozen_count = frozen_check()
    frames, groups, bags, vectors, summary = canonical_check("development")
    require(
        (summary["signal_rows"] == 5044 and summary["empty_bags"] == 2),
        "Invariant failed: summary['signal_rows'] == 5044 and summary['empty_bags'] == 2",
    )
    summary.update(
        passed=True,
        frozen_hashes_verified=frozen_count,
        no_evaluation_outcomes_read=True,
        model_fit_calls=0,
        horizons=HORIZONS,
        group_numeric_counts={g: len(groups[g]["numeric"]) for g in GROUPS},
    )
    return frames, groups, bags, vectors, summary


def chronological_partition(data):
    """Reconstruct the core/validation quarter boundary independently of the engine."""
    validate_history(data)
    boundary = pd.Period(data.quarter.max(), freq="Q") - 1
    while int((data.quarter >= str(boundary)).sum()) < 100:
        boundary -= 1
    core = data[data.quarter < str(boundary)].copy()
    valid = data[data.quarter >= str(boundary)].copy()
    require(
        (len(core) >= 200 and len(valid) >= 100 and (valid.quarter.nunique() >= 2)),
        "Invariant failed: len(core) >= 200 and len(valid) >= 100 and (valid.quarter.nunique() >= 2)",
    )
    require(
        (core.reaction_session.max() < valid.reaction_session.min()),
        "Invariant failed: core.reaction_session.max() < valid.reaction_session.min()",
    )
    require(
        (set(core.event_id).isdisjoint(valid.event_id)),
        "Invariant failed: set(core.event_id).isdisjoint(valid.event_id)",
    )
    return core, valid


def matrix_check(path, prep_path, history, test, spec):
    """Compare saved model matrices and preprocessing statistics with an independent reconstruction."""
    numeric = spec["numeric"]
    categorical = spec["categorical"]
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    train_numeric = np.clip(scaler.fit_transform(imputer.fit_transform(history[numeric])), -10, 10)
    test_numeric = np.clip(scaler.transform(imputer.transform(test[numeric])), -10, 10)
    if categorical:
        hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        train_categories = hot.fit_transform(history[categorical].fillna("UNKNOWN").astype(str))
        test_categories = hot.transform(test[categorical].fillna("UNKNOWN").astype(str))
    else:
        train_categories = np.empty((len(history), 0), np.float32)
        test_categories = np.empty((len(test), 0), np.float32)
    x = np.column_stack([train_numeric, train_categories]).astype("float32")
    xt = np.column_stack([test_numeric, test_categories]).astype("float32")
    mean = float(history.y.mean())
    sd = max(float(history.y.std(ddof=0)), 1e-6)
    y = ((history.y.to_numpy() - mean) / sd).astype("float32")
    expected = dict(
        X_train=x,
        X_test=xt,
        y_train_std=y,
        target_mean=mean,
        target_sd=sd,
        train_ids=history.event_id.to_numpy(str),
        test_ids=test.event_id.to_numpy(str),
    )
    saved = np.load(path, allow_pickle=False)
    require(
        (set(saved.files) == set(expected)), "Invariant failed: set(saved.files) == set(expected)"
    )
    for name, value in expected.items():
        np.testing.assert_array_equal(saved[name], value)
    require(
        (history.reaction_session.max() < test.reaction_session.min()),
        "Invariant failed: history.reaction_session.max() < test.reaction_session.min()",
    )
    prep = joblib.load(prep_path)
    require(
        (prep["fit_event_ids"] == history.event_id.tolist()),
        "Invariant failed: prep['fit_event_ids'] == history.event_id.tolist()",
    )
    require(
        (prep["numeric_features"] == numeric and prep["categorical_features"] == categorical),
        "Invariant failed: prep['numeric_features'] == numeric and prep['categorical_features'] == categorical",
    )
    transformed = np.clip(
        prep["scaler"].transform(prep["imputer"].transform(test[numeric])), -10, 10
    )
    if categorical:
        for name, levels in zip(categorical, prep["onehot"].categories_):
            require(
                (set(levels) == set(history[name].fillna("UNKNOWN").astype(str))),
                "Invariant failed: set(levels) == set(history[name].fillna('UNKNOWN').astype(str))",
            )
        transformed = np.column_stack(
            [transformed, prep["onehot"].transform(test[categorical].fillna("UNKNOWN").astype(str))]
        )
    else:
        require((prep["onehot"] is None), "Invariant failed: prep['onehot'] is None")
    np.testing.assert_array_equal(transformed.astype("float32"), xt)
    return saved


def replay_checkpoint(path, matrix, seed, epochs, use_text, bags, vectors):
    """Replay a trusted saved model with independent event-wise attention, without fitting."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(
        (saved["seed"] == seed and saved["epochs"] == epochs and (saved["use_text"] == use_text)),
        "Invariant failed: saved['seed'] == seed and saved['epochs'] == epochs and (saved['use_text'] == use_text)",
    )
    require(
        (saved["recipe"] == RECIPE and saved["numeric_dim"] == matrix["X_train"].shape[1]),
        "Invariant failed: saved['recipe'] == RECIPE and saved['numeric_dim'] == matrix['X_train'].shape[1]",
    )
    close(saved["target_mean"], float(matrix["target_mean"]), atol=0, rtol=0)
    close(saved["target_sd"], float(matrix["target_sd"]), atol=0, rtol=0)
    model = EventWiseNetwork(saved["numeric_dim"], use_text).eval()
    model.load_state_dict(saved["state_dict"])
    result = []
    with torch.inference_mode():
        for start in range(0, len(matrix["X_test"]), 128):
            x = torch.from_numpy(matrix["X_test"][start : start + 128])
            ids = matrix["test_ids"][start : start + 128]
            signal_list = individual_signals(ids, vectors, bags) if use_text else None
            p = model(x, signal_list).numpy()
            result.extend((p * saved["target_sd"] + saved["target_mean"]).tolist())
    return np.asarray(result)


def cells_check(stage, frames, groups, bags, vectors, lock=None):
    """Audit every completed cell's split, checkpoint, preprocessing, and stored predictions."""
    quarters = DEVQ if stage == "development" else EVALQ
    combined = pd.read_parquet(
        OUT
        / (
            "DEV_PREDICTIONS.parquet"
            if stage == "development"
            else "EVALUATION_PREDICTIONS.parquet"
        )
    )
    expected_n = 4167 if stage == "development" else 4473
    require(
        (
            len(combined) == expected_n
            and (not combined.duplicated(["horizon", "group", "event_id"]).any())
        ),
        "Invariant failed: len(combined) == expected_n and (not combined.duplicated(['horizon', 'group', 'event_id']).any())",
    )
    maximum = 0.0
    replays = 0
    stops = 0
    for horizon in HORIZONS:
        data = frames[horizon]
        for quarter in quarters:
            history = data[data.quarter < quarter]
            test = data[data.quarter == quarter]
            for group in GROUPS:
                folder = OUT / stage / horizon / quarter / group
                complete_check(folder)
                spec = groups[group]
                use_text = group == "MFL"
                matrix = matrix_check(
                    folder / "FULL.npz", folder / "preprocessing.joblib", history, test, spec
                )
                audit = read(folder / "FIT_AUDIT.json")
                require(
                    (
                        audit["stage"] == stage
                        and audit["horizon"] == horizon
                        and (audit["quarter"] == quarter)
                        and (audit["group"] == group)
                    ),
                    "Invariant failed: audit['stage'] == stage and audit['horizon'] == horizon and (audit['quarter'] == quarter) and (audit['group'] == group)",
                )
                require(
                    (audit["recipe"] == RECIPE and audit["seeds"] == [42, 43]),
                    "Invariant failed: audit['recipe'] == RECIPE and audit['seeds'] == [42, 43]",
                )
                require(
                    (
                        audit["numeric_features"] == spec["numeric"]
                        and audit["categorical_features"] == spec["categorical"]
                    ),
                    "Invariant failed: audit['numeric_features'] == spec['numeric'] and audit['categorical_features'] == spec['categorical']",
                )
                require(
                    (
                        audit["history_ids"] == history.event_id.tolist()
                        and audit["test_ids"] == test.event_id.tolist()
                    ),
                    "Invariant failed: audit['history_ids'] == history.event_id.tolist() and audit['test_ids'] == test.event_id.tolist()",
                )
                require(
                    (audit["use_signals"] == use_text and audit["no_signal_truncation"]),
                    "Invariant failed: audit['use_signals'] == use_text and audit['no_signal_truncation']",
                )
                require(
                    (
                        audit["history_signal_count"]
                        == (sum((len(bags[e]) for e in history.event_id)) if use_text else 0)
                    ),
                    "Invariant failed: audit['history_signal_count'] == (sum((len(bags[e]) for e in history.event_id)) if use_text else 0)",
                )
                require(
                    (
                        audit["test_signal_count"]
                        == (sum((len(bags[e]) for e in test.event_id)) if use_text else 0)
                    ),
                    "Invariant failed: audit['test_signal_count'] == (sum((len(bags[e]) for e in test.event_id)) if use_text else 0)",
                )
                if stage == "development":
                    core, valid = chronological_partition(history)
                    matrix_check(
                        folder / "INNER.npz",
                        folder / "inner_preprocessing.joblib",
                        core,
                        valid,
                        spec,
                    )
                    stopped = read(folder / "INNER_SELECTION.json")
                    scalar_stopping_check(stopped)
                    stops += 1
                    require(
                        (
                            stopped["core_train_ids"] == core.event_id.tolist()
                            and stopped["inner_validation_ids"] == valid.event_id.tolist()
                        ),
                        "Invariant failed: stopped['core_train_ids'] == core.event_id.tolist() and stopped['inner_validation_ids'] == valid.event_id.tolist()",
                    )
                    epochs = stopped["selected_epoch"]
                    require(
                        (
                            audit["early_stopping_performed"]
                            and (not audit["historical_weight_updates_only"])
                        ),
                        "Invariant failed: audit['early_stopping_performed'] and (not audit['historical_weight_updates_only'])",
                    )
                else:
                    epochs = lock["model_selections"][horizon][group]["selected_epoch"]
                    require(
                        (
                            not audit["early_stopping_performed"]
                            and audit["historical_weight_updates_only"]
                        ),
                        "Invariant failed: not audit['early_stopping_performed'] and audit['historical_weight_updates_only']",
                    )
                    require(
                        (audit["selection_sha256"] == sha(OUT / "LOCKED_SELECTION.json")),
                        "Invariant failed: audit['selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
                    )
                require((audit["epochs"] == epochs), "Invariant failed: audit['epochs'] == epochs")
                replay = np.asarray(
                    [
                        replay_checkpoint(
                            folder / f"seed{seed}.pt", matrix, seed, epochs, use_text, bags, vectors
                        )
                        for seed in [42, 43]
                    ]
                )
                saved = np.load(folder / "SEED_PREDICTIONS.npy", allow_pickle=False)
                close(replay, saved, atol=3e-8, rtol=2e-6)
                if not use_text:
                    np.testing.assert_array_equal(replay, saved)
                maximum = max(maximum, float(np.max(abs(replay - saved))))
                replays += 2
                prediction = pd.read_parquet(folder / "PREDICTIONS.parquet")
                columns = ["event_id", "ticker", "reaction_session", "quarter", "year", "y"]
                pd.testing.assert_frame_equal(
                    prediction[columns].reset_index(drop=True),
                    test[columns].reset_index(drop=True),
                    check_dtype=False,
                    check_exact=True,
                )
                require(
                    (prediction.horizon.eq(horizon).all() and prediction.group.eq(group).all()),
                    "Invariant failed: prediction.horizon.eq(horizon).all() and prediction.group.eq(group).all()",
                )
                np.testing.assert_array_equal(prediction.prediction.to_numpy(), saved.mean(axis=0))
                close(prediction.history_mean, float(history.y.mean()), atol=0, rtol=0)
                sign = 1 if int(history.y.gt(0).sum()) * 2 >= len(history) else -1
                require(
                    (prediction.history_majority_sign.eq(sign).all()),
                    "Invariant failed: prediction.history_majority_sign.eq(sign).all()",
                )
                if stage == "evaluation":
                    coefficients = lock["calibration"]["selections"][horizon][group]
                    require(
                        (prediction.calibration_method.eq(coefficients["method"]).all()),
                        "Invariant failed: prediction.calibration_method.eq(coefficients['method']).all()",
                    )
                    close(
                        prediction.calibrated_prediction,
                        coefficients["a"] + coefficients["b"] * prediction.prediction,
                        atol=0,
                        rtol=0,
                    )
                piece = combined[
                    (combined.horizon == horizon)
                    & (combined.group == group)
                    & (combined.quarter == quarter)
                ]
                pd.testing.assert_frame_equal(
                    piece.reset_index(drop=True),
                    prediction.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=True,
                )
    return combined, dict(
        checkpoints_replayed=replays,
        stopping_rules_verified=stops,
        maximum_replayed_prediction_difference=maximum,
        cells_verified=72,
    )


def calibration_coefficients(method, y, p):
    """Independently compute the declared identity, offset, or regularized affine map."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    require(
        (len(y) == len(p) and len(y) >= 120), "Invariant failed: len(y) == len(p) and len(y) >= 120"
    )
    a = 0.0
    b = 1.0
    reg = 0.0
    if method == "offset":
        a = float((y - p).mean())
    elif method == "affine":
        reg = 0.1 * float(np.var(y, ddof=0))
        denominator = float(np.var(p, ddof=0)) + reg
        covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
        b = max(0.0, (covariance + reg) / denominator) if denominator > 1e-20 else 1.0
        a = float(y.mean() - b * p.mean())
    else:
        require((method == "none"), "Invariant failed: method == 'none'")
    return dict(a=a, b=b, n=len(y), reg=reg)


def calibration_check(predictions, lock):
    """Verify chronological calibration fits, equal candidate cohorts, and frozen choices."""
    choices = ["none", "offset", "affine"]
    summary = lock["calibration"]
    require(
        (summary["methods"] == summary["exact_tie_order"] == choices),
        "Invariant failed: summary['methods'] == summary['exact_tie_order'] == choices",
    )
    require(
        (summary["selection_quarters"] == [f"2024Q{q}" for q in range(1, 5)]),
        "Invariant failed: summary['selection_quarters'] == [f'2024Q{q}' for q in range(1, 5)]",
    )
    require(
        (summary["warmup_quarters"] == [f"2023Q{q}" for q in range(1, 5)]),
        "Invariant failed: summary['warmup_quarters'] == [f'2023Q{q}' for q in range(1, 5)]",
    )
    require(
        (
            summary["development_events_per_model"] == 463
            and summary["selection_events_per_model"] == 225
        ),
        "Invariant failed: summary['development_events_per_model'] == 463 and summary['selection_events_per_model'] == 225",
    )
    require(
        (
            summary["minimum_fit_events"] == 120
            and summary["affine_regularization_multiplier"] == 0.1
        ),
        "Invariant failed: summary['minimum_fit_events'] == 120 and summary['affine_regularization_multiplier'] == 0.1",
    )
    require(
        (summary["decisions_use_development_only"] and summary["evaluation_rows_read"] == 0),
        "Invariant failed: summary['decisions_use_development_only'] and summary['evaluation_rows_read'] == 0",
    )
    folds = read(OUT / "DEV_CALIBRATION_FOLDS.json")
    scores = pd.read_csv(OUT / "DEV_CALIBRATION_SCORES.csv", float_precision="round_trip")
    outputs = pd.read_parquet(OUT / "DEV_CALIBRATION_PREDICTIONS.parquet")
    require(
        (len(folds) == 108 and len(scores) == 27 and (len(outputs) == 6075)),
        "Invariant failed: len(folds) == 108 and len(scores) == 27 and (len(outputs) == 6075)",
    )
    for horizon in HORIZONS:
        for group in GROUPS:
            part = (
                predictions[(predictions.horizon == horizon) & (predictions.group == group)]
                .sort_values(["reaction_session", "event_id"])
                .reset_index(drop=True)
            )
            require(
                (len(part) == 463 and set(part.year) == {2023, 2024}),
                "Invariant failed: len(part) == 463 and set(part.year) == {2023, 2024}",
            )
            losses = {method: 0.0 for method in choices}
            for quarter in summary["selection_quarters"]:
                history = part[part.quarter < quarter]
                test = part[part.quarter == quarter]
                require(
                    (
                        len(history) >= 120
                        and history.reaction_session.max() < test.reaction_session.min()
                    ),
                    "Invariant failed: len(history) >= 120 and history.reaction_session.max() < test.reaction_session.min()",
                )
                for method in choices:
                    coefficient = calibration_coefficients(method, history.y, history.prediction)
                    matches = [
                        row
                        for row in folds
                        if row["horizon"] == horizon
                        and row["group"] == group
                        and row["quarter"] == quarter
                        and row["method"] == method
                    ]
                    require((len(matches) == 1), "Invariant failed: len(matches) == 1")
                    row = matches[0]
                    for field, value in coefficient.items():
                        close(row[field], value)
                    require(
                        (
                            row["train_ids"] == history.event_id.tolist()
                            and row["test_ids"] == test.event_id.tolist()
                        ),
                        "Invariant failed: row['train_ids'] == history.event_id.tolist() and row['test_ids'] == test.event_id.tolist()",
                    )
                    for field, value in dict(
                        train_start=history.reaction_session.min(),
                        train_end=history.reaction_session.max(),
                        test_start=test.reaction_session.min(),
                        test_end=test.reaction_session.max(),
                    ).items():
                        require((row[field] == value), "Invariant failed: row[field] == value")
                    require(
                        (row["test_n"] == len(test)), "Invariant failed: row['test_n'] == len(test)"
                    )
                    p = coefficient["a"] + coefficient["b"] * test.prediction.to_numpy()
                    loss = float(np.square(p - test.y.to_numpy()).sum())
                    losses[method] += loss
                    close(row["sse"], loss)
                    subset = outputs[
                        (outputs.horizon == horizon)
                        & (outputs.group == group)
                        & (outputs.quarter == quarter)
                        & (outputs.method == method)
                    ]
                    require(
                        (len(subset) == len(test)), "Invariant failed: len(subset) == len(test)"
                    )
                    pd.testing.assert_frame_equal(
                        subset[part.columns].reset_index(drop=True),
                        test[part.columns].reset_index(drop=True),
                        check_dtype=False,
                        check_exact=True,
                    )
                    close(subset.calibrated_prediction, p, atol=0, rtol=0)
                    close(subset.a, coefficient["a"])
                    close(subset.b, coefficient["b"])
                    require(
                        (
                            subset.fit_n.eq(len(history)).all()
                            and subset.fit_end.eq(history.reaction_session.max()).all()
                        ),
                        "Invariant failed: subset.fit_n.eq(len(history)).all() and subset.fit_end.eq(history.reaction_session.max()).all()",
                    )
                    close(subset.fit_reg, coefficient["reg"])
            mse = {method: value / 225 for method, value in losses.items()}
            winner = min(choices, key=lambda method: (mse[method], choices.index(method)))
            chosen = summary["selections"][horizon][group]
            require(
                (chosen["method"] == winner and chosen["selection_n"] == 225),
                "Invariant failed: chosen['method'] == winner and chosen['selection_n'] == 225",
            )
            close(chosen["selection_mse"], mse[winner])
            for method in choices:
                close(chosen["candidate_mse"][method], mse[method])
                row = scores[
                    (scores.horizon == horizon)
                    & (scores.group == group)
                    & (scores.method == method)
                ].iloc[0]
                require(
                    (row.n == 225 and bool(row.selected) == (method == winner)),
                    "Invariant failed: row.n == 225 and bool(row.selected) == (method == winner)",
                )
                close(row.mse, mse[method])
                close(row.sse, losses[method])
            final = calibration_coefficients(winner, part.y, part.prediction)
            for field, value in final.items():
                close(chosen[field], value)
            require(
                (chosen["fit_ids"] == part.event_id.tolist() and chosen["fit_quarters"] == DEVQ),
                "Invariant failed: chosen['fit_ids'] == part.event_id.tolist() and chosen['fit_quarters'] == DEVQ",
            )
            require(
                (
                    chosen["fit_start"] == part.reaction_session.min()
                    and chosen["fit_end"] == part.reaction_session.max()
                ),
                "Invariant failed: chosen['fit_start'] == part.reaction_session.min() and chosen['fit_end'] == part.reaction_session.max()",
            )
            require(
                (chosen["coefficients_frozen_for_evaluation"]),
                "Invariant failed: chosen['coefficients_frozen_for_evaluation']",
            )
    return dict(
        calibration_folds=108,
        calibration_candidate_scores=27,
        calibration_selection_events=225,
        final_calibration_fit_events=463,
        calibration_models_locked=9,
        calibration_methods_and_coefficients_independently_verified=True,
    )


def development_check():
    """Audit development data, model choices, stopping receipts, and calibration provenance."""
    frames, groups, bags, vectors, summary = inputs_check()
    lock = selection_gate()
    fit = read(OUT / "DEV_FIT_COMPLETE.json")
    require(
        (fit["cells"] == 72 and fit["systems"] == 9 and (fit["events_per_system"] == 463)),
        "Invariant failed: fit['cells'] == 72 and fit['systems'] == 9 and (fit['events_per_system'] == 463)",
    )
    require(
        (sha(OUT / "DEV_PREDICTIONS.parquet") == fit["prediction_sha256"]),
        "Invariant failed: sha(OUT / 'DEV_PREDICTIONS.parquet') == fit['prediction_sha256']",
    )
    predictions, cells = cells_check("development", frames, groups, bags, vectors)
    epochs = {h: {} for h in HORIZONS}
    for horizon in HORIZONS:
        core, valid = chronological_partition(frames[horizon])
        for group in GROUPS:
            folder = OUT / "final_stopping" / horizon / group
            complete_check(folder)
            matrix_check(
                folder / "INNER.npz",
                folder / "inner_preprocessing.joblib",
                core,
                valid,
                groups[group],
            )
            stopped = read(folder / "INNER_SELECTION.json")
            best = scalar_stopping_check(stopped)
            require(
                (
                    stopped["core_train_ids"] == core.event_id.tolist()
                    and stopped["inner_validation_ids"] == valid.event_id.tolist()
                ),
                "Invariant failed: stopped['core_train_ids'] == core.event_id.tolist() and stopped['inner_validation_ids'] == valid.event_id.tolist()",
            )
            selected = lock["model_selections"][horizon][group]
            require(
                (
                    selected["recipe"] == RECIPE
                    and selected["selected_epoch"] == stopped["selected_epoch"]
                ),
                "Invariant failed: selected['recipe'] == RECIPE and selected['selected_epoch'] == stopped['selected_epoch']",
            )
            close(selected["best_inner_mse"], best)
            require(
                (
                    selected["epoch_core_ids"] == core.event_id.tolist()
                    and selected["epoch_validation_ids"] == valid.event_id.tolist()
                ),
                "Invariant failed: selected['epoch_core_ids'] == core.event_id.tolist() and selected['epoch_validation_ids'] == valid.event_id.tolist()",
            )
            require(
                (OUT / selected["epoch_receipt"] == folder / "COMPLETE.json"),
                "Invariant failed: OUT / selected['epoch_receipt'] == folder / 'COMPLETE.json'",
            )
            epochs[horizon][group] = stopped["selected_epoch"]
    calibrated = calibration_check(predictions, lock)
    engine_receipt = read(OUT / "ENGINE_INDEPENDENT_AUDIT.json")
    require(
        (
            engine_receipt["passed"]
            and engine_receipt["engine_sha256"] == sha(Path(__file__).parent / "engine.py")
        ),
        "Invariant failed: engine_receipt['passed'] and engine_receipt['engine_sha256'] == sha(Path(__file__).parent / 'engine.py')",
    )
    summary.update(
        development=cells,
        final_stopping_rules_verified=9,
        frozen_epochs=epochs,
        calibration=calibrated,
        selection_only_development_verified=True,
        locked_selection_sha256=sha(OUT / "LOCKED_SELECTION.json"),
        all_stopping_rules_verified=81,
        model_fit_calls=0,
        inner_curve_values_not_independently_retrained=True,
    )
    frozen_check()
    selection_gate()
    return summary


BOOT_WEIGHTS = {}


def independent_bootstrap(x, tickers, dates, method):
    """Independent resampling and null-centered inference with frozen 20261123 seed."""
    x = np.asarray(x, float)
    effect = float(x.mean())
    block = {"ticker": 0, "circular_5_observed_dates": 5, "circular_10_observed_dates": 10}[method]
    labels = (
        np.asarray(tickers, str)
        if not block
        else pd.to_datetime(pd.Series(dates)).dt.strftime("%Y-%m-%d").to_numpy()
    )
    grouped = (
        pd.DataFrame({"label": labels, "value": x})
        .groupby("label", sort=True)
        .value.agg(["sum", "size"])
    )
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["size"].to_numpy(float)
    groups = len(grouped)
    valid = bool(not np.all(x == x[0]) and groups >= 2 and (not block or groups > block))
    if valid:
        key = (groups, block)
        if key not in BOOT_WEIGHTS:
            rng = np.random.default_rng(
                np.random.SeedSequence([20261123, 0 if block == 0 else 100 + block])
            )
            chunks = []
            for start in range(0, 5000, 512):
                size = min(512, 5000 - start)
                if block == 0:
                    w = rng.multinomial(groups, np.full(groups, 1 / groups), size=size)
                else:
                    starts = rng.integers(0, groups, size=(size, math.ceil(groups / block)))
                    samples = ((starts[:, :, None] + np.arange(block)) % groups).reshape(size, -1)[
                        :, :groups
                    ]
                    w = np.zeros((size, groups), dtype=np.int64)
                    np.add.at(w, (np.repeat(np.arange(size), groups), samples.ravel()), 1)
                chunks.append(w)
            BOOT_WEIGHTS[key] = chunks
        draws = np.concatenate([(w @ sums) / (w @ counts) for w in BOOT_WEIGHTS[key]])
        scale = max(float(np.max(abs(x))), float(np.max(abs(draws))), np.finfo(float).tiny)
        tolerance = 32 * np.finfo(float).eps * scale
        valid = bool(np.ptp(draws) > tolerance)
    if valid:
        count = int(np.sum(abs(draws - effect) >= abs(effect) - tolerance))
        low, high = np.quantile(draws, [0.025, 0.975])
        p = (count + 1) / 5001
        se = float(np.std(draws, ddof=1))
    else:
        count = np.nan
        low = high = p = se = np.nan
    return dict(
        method=method,
        effect=effect,
        p_raw=float(p),
        ci_low=float(low),
        ci_high=float(high),
        se_bootstrap=se,
        null_exceedances=count,
        valid=valid,
    )


def metric_values(part, p):
    """Independently calculate decimal-AR errors and zero-threshold direction accuracy."""
    y = part.y.to_numpy(float)
    p = np.asarray(p, float)
    error = p - y
    right = (p > 0) == (y > 0)
    majority = (part.history_majority_sign.to_numpy() > 0) == (y > 0)
    return dict(
        n=len(part),
        mse=float(np.mean(error**2)),
        rmse_pp=float(np.sqrt(np.mean(error**2)) * 100),
        mae_pp=float(np.mean(abs(error)) * 100),
        bias_pp=float(error.mean() * 100),
        correct=int(right.sum()),
        accuracy_pct=float(right.mean() * 100),
        majority_correct=int(majority.sum()),
        majority_accuracy_pct=float(majority.mean() * 100),
        predicted_positive=int((p > 0).sum()),
        predicted_zero=int((p == 0).sum()),
        actual_positive=int((y > 0).sum()),
        ticker_clusters=int(part.ticker.nunique()),
        date_clusters=int(part.reaction_session.nunique()),
        first_event=part.reaction_session.min(),
        last_event=part.reaction_session.max(),
    )


def check_values(row, expected):
    """Compare a saved report row to independently recomputed scalar metrics."""
    for key, value in expected.items():
        if isinstance(value, str):
            require((row[key] == value), (key, row[key], value))
        elif isinstance(value, (bool, np.bool_)):
            require((bool(row[key]) == value), (key, row[key], value))
        else:
            close(row[key], value)


def inference_decision(row):
    """Derive the declared significance decision from a row's test status and corrected p-value."""
    if not row["valid"]:
        return "not_estimable"
    if not row["reject_holm"]:
        return "not_significant"
    if row["effect"] > 0:
        return "significant_improvement"
    if row["effect"] < 0:
        return "significant_deterioration"
    return "not_significant"


def inference_summary(results, bootstrap):
    """Summarize all prespecified inference families without dropping unfavorable contrasts."""
    by_id = {r["test_id"]: inference_decision(r) for r in results}
    methods = {
        test_id: {inference_decision(row) for row in bootstrap if row["test_id"] == test_id}
        for test_id in by_id
    }
    return dict(
        significant_improvements=[k for k, v in by_id.items() if v == "significant_improvement"],
        significant_deteriorations=[
            k for k, v in by_id.items() if v == "significant_deterioration"
        ],
        not_significant=[k for k, v in by_id.items() if v == "not_significant"],
        not_estimable=[k for k, v in by_id.items() if v == "not_estimable"],
        improvements_supported_by_all_bootstrap_methods=[
            k
            for k, v in by_id.items()
            if v == "significant_improvement" and methods[k] == {"significant_improvement"}
        ],
        cluster_bootstrap_decision_disagreements=[k for k, v in by_id.items() if methods[k] != {v}],
    )


def reporting_check(predictions, lock):
    """Recompute report metrics and statistical decisions from the unchanged matched forecasts."""
    from work.market_ridge_fixed_20261118.verify import independent_holm, independent_test

    receipt = read(OUT / "REPORT_COMPLETE.json")
    require(
        (
            receipt["complete"]
            and receipt["bootstrap_seed"] == 20261123
            and (receipt["bootstrap_replicates"] == 5000)
        ),
        "Invariant failed: receipt['complete'] and receipt['bootstrap_seed'] == 20261123 and (receipt['bootstrap_replicates'] == 5000)",
    )
    require(
        (
            receipt["report_model_fits"] == receipt["report_calibration_fits"] == 0
            and receipt["original_files_unchanged"]
        ),
        "Invariant failed: receipt['report_model_fits'] == receipt['report_calibration_fits'] == 0 and receipt['original_files_unchanged']",
    )
    for key in ["input_sha256", "output_sha256"]:
        for name, digest in receipt[key].items():
            require((sha(OUT / name) == digest), name)
    require(
        (receipt["script_sha256"] == sha(Path(__file__).parent / "report.py")),
        "Invariant failed: receipt['script_sha256'] == sha(Path(__file__).parent / 'report.py')",
    )
    require(
        (
            receipt["statistical_helper_sha256"]
            == sha(ROOT / "work/model_selection_20261118/statisticalinference.py")
        ),
        "Invariant failed: receipt['statistical_helper_sha256'] == sha(ROOT / 'work/model_selection_20261118/statisticalinference.py')",
    )
    metric_table = pd.read_csv(OUT / "METRICS.csv", float_precision="round_trip")
    baseline_table = pd.read_csv(OUT / "BASELINE_METRICS.csv", float_precision="round_trip")
    dev_table = pd.read_csv(OUT / "DEV_METRICS.csv", float_precision="round_trip")
    locked_table = pd.read_csv(OUT / "LOCKED_MODELS.csv", float_precision="round_trip")
    cal_table = pd.read_csv(OUT / "CALIBRATION_SELECTION.csv", float_precision="round_trip")
    require(
        (
            [len(x) for x in [metric_table, baseline_table, dev_table, locked_table, cal_table]]
            == [54, 18, 9, 9, 27]
        ),
        "Invariant failed: [len(x) for x in [metric_table, baseline_table, dev_table, locked_table, cal_table]] == [54, 18, 9, 9, 27]",
    )
    names = dict(mse="RAW_MSE", direction="RAW_DIRECTION", calibration="CALIBRATION_MSE")
    tables = {
        kind: pd.read_csv(OUT / (name + "_TESTS.csv"), float_precision="round_trip")
        for kind, name in names.items()
    }
    boot_tables = {
        kind: pd.read_csv(OUT / (name + "_BOOTSTRAP.csv"), float_precision="round_trip")
        for kind, name in names.items()
    }
    require(
        ([len(tables[k]) for k in names] == [81, 81, 27]),
        "Invariant failed: [len(tables[k]) for k in names] == [81, 81, 27]",
    )
    require(
        ([len(boot_tables[k]) for k in names] == [243, 243, 81]),
        "Invariant failed: [len(boot_tables[k]) for k in names] == [243, 243, 81]",
    )
    results = {kind: [] for kind in names}
    bootstrap = {kind: [] for kind in names}
    methods = ["ticker", "circular_5_observed_dates", "circular_10_observed_dates"]
    periods = ["2025", "2026", "2025-2026"]
    dev = pd.read_parquet(OUT / "DEV_PREDICTIONS.parquet")

    def add(kind, horizon, period, candidate, comparator, diff, part):
        """Append one fully identified paired contrast to the current report's test family."""
        identity = dict(
            test_id=f"{kind}__{horizon}__{period}__{candidate}_vs_{comparator}",
            family="calibration_mse_all_horizons"
            if kind == "calibration"
            else f"raw_{kind}_{horizon}",
            horizon=horizon,
            period=period,
            candidate=candidate,
            comparator=comparator,
        )
        value = independent_test(diff, part.ticker, part.reaction_session)
        value.update(identity)
        results[kind].append(value)
        for method in methods:
            value = independent_bootstrap(diff, part.ticker, part.reaction_session, method)
            value.update(identity)
            bootstrap[kind].append(value)

    for horizon in HORIZONS:
        for group in GROUPS:
            part = dev[(dev.horizon == horizon) & (dev.group == group)]
            row = dev_table[(dev_table.horizon == horizon) & (dev_table.group == group)].iloc[0]
            require(
                (row.period == "2023-2024" and row.variant == "raw_selection_diagnostic"),
                "Invariant failed: row.period == '2023-2024' and row.variant == 'raw_selection_diagnostic'",
            )
            check_values(row, metric_values(part, part.prediction))
            selected = lock["model_selections"][horizon][group]
            calibration = lock["calibration"]["selections"][horizon][group]
            row = locked_table[
                (locked_table.horizon == horizon) & (locked_table.group == group)
            ].iloc[0]
            check_values(
                row,
                dict(
                    selected_epoch=selected["selected_epoch"],
                    calibration_method=calibration["method"],
                    a=calibration["a"],
                    b=calibration["b"],
                    reg=calibration["reg"],
                    calibration_fit_n=463,
                    calibration_fit_end=calibration["fit_end"],
                ),
            )
            for method in ["none", "offset", "affine"]:
                row = cal_table[
                    (cal_table.horizon == horizon)
                    & (cal_table.group == group)
                    & (cal_table.method == method)
                ].iloc[0]
                check_values(
                    row,
                    dict(
                        n=225,
                        mse=calibration["candidate_mse"][method],
                        selected=method == calibration["method"],
                    ),
                )
        for period in periods:
            frame = predictions[predictions.horizon == horizon]
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
            loss = {}
            correct = {}
            for group, part in parts.items():
                pd.testing.assert_frame_equal(
                    part[["event_id", "y"]], ref[["event_id", "y"]], check_exact=True
                )
                loss[group] = (part.prediction.to_numpy() - y) ** 2
                correct[group] = ((part.prediction.to_numpy() > 0) == (y > 0)).astype(float)
                for variant, column in [
                    ("raw", "prediction"),
                    ("selected_calibration", "calibrated_prediction"),
                ]:
                    matches = metric_table[
                        (metric_table.horizon == horizon)
                        & (metric_table.group == group)
                        & (metric_table.period == period)
                        & (metric_table.variant == variant)
                    ]
                    require((len(matches) == 1), "Invariant failed: len(matches) == 1")
                    check_values(matches.iloc[0], metric_values(part, part[column]))
                add(
                    "calibration",
                    horizon,
                    period,
                    group + "_calibrated",
                    group + "_raw",
                    loss[group] - (part.calibrated_prediction.to_numpy() - y) ** 2,
                    part,
                )
            for base, p in [
                ("zero_AR", np.zeros(len(ref))),
                ("history_mean", ref.history_mean.to_numpy()),
            ]:
                values = metric_values(ref, p)
                if base == "zero_AR":
                    values["correct"] = np.nan
                    values["accuracy_pct"] = np.nan
                row = baseline_table[
                    (baseline_table.horizon == horizon)
                    & (baseline_table.group == base)
                    & (baseline_table.period == period)
                ].iloc[0]
                require((row.variant == "baseline"), "Invariant failed: row.variant == 'baseline'")
                check_values(row, values)
            for candidate, comparator in [("MF", "M"), ("MFL", "MF"), ("MFL", "M")]:
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
                    (ref.history_mean.to_numpy() - y) ** 2 - loss[group],
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
    for kind in names:
        families = sorted({r["family"] for r in results[kind]})
        for family in families:
            subset = [r for r in results[kind] if r["family"] == family]
            require((len(subset) == 27), "Invariant failed: len(subset) == 27")
            independent_holm(subset)
            for method in methods:
                subset = [
                    r for r in bootstrap[kind] if r["family"] == family and r["method"] == method
                ]
                require((len(subset) == 27), "Invariant failed: len(subset) == 27")
                independent_holm(subset)
        for expected, table in [
            (results[kind], tables[kind]),
            (bootstrap[kind], boot_tables[kind]),
        ]:
            for item in expected:
                rows = table[table.test_id == item["test_id"]]
                if "method" in item:
                    rows = rows[rows.method == item["method"]]
                require((len(rows) == 1), "Invariant failed: len(rows) == 1")
                row = rows.iloc[0]
                check_values(row, item)
                factor = 100 if kind == "direction" else 10000
                unit = "pp" if kind == "direction" else "pp2"
                for field in ["effect", "ci_low", "ci_high"]:
                    close(row[field + "_" + unit], item[field] * factor)
                require(
                    (row.decision == inference_decision(item) and row.holm_family_size == 27),
                    "Invariant failed: row.decision == inference_decision(item) and row.holm_family_size == 27",
                )
                if "method" in item:
                    require(
                        (row.replicates == 5000 and row.seed == 20261123),
                        "Invariant failed: row.replicates == 5000 and row.seed == 20261123",
                    )
    conclusion = read(OUT / "CONCLUSION.json")
    for field, value in dict(
        primary_horizon="hour",
        extension_horizons=["half", "full"],
        main_metric="raw MSE",
        direction_secondary=True,
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
        bootstrap_seed=20261123,
        bootstrap_replicates=5000,
    ).items():
        require((conclusion[field] == value), field)
    for kind in ["mse", "direction"]:
        for horizon in HORIZONS:
            calculated = inference_summary(
                [r for r in results[kind] if r["horizon"] == horizon],
                [r for r in bootstrap[kind] if r["horizon"] == horizon],
            )
            require(
                (conclusion["raw_" + kind][horizon] == calculated),
                "Invariant failed: conclusion['raw_' + kind][horizon] == calculated",
            )
    require(
        (
            conclusion["calibration_mse"]
            == inference_summary(results["calibration"], bootstrap["calibration"])
        ),
        "Invariant failed: conclusion['calibration_mse'] == inference_summary(results['calibration'], bootstrap['calibration'])",
    )
    identity = "mse__hour__2025-2026__MFL_vs_MF"
    primary = next(row for row in results["mse"] if row["test_id"] == identity)
    primary_boot = [row for row in bootstrap["mse"] if row["test_id"] == identity]
    stated = conclusion["primary_one_hour_llm_increment"]
    require(
        (stated["test_id"] == identity and stated["decision"] == inference_decision(primary)),
        "Invariant failed: stated['test_id'] == identity and stated['decision'] == inference_decision(primary)",
    )
    close(stated["effect_pp2"], 10000 * primary["effect"])
    close(np.nan if stated["p_holm"] is None else stated["p_holm"], primary["p_holm"])
    require(
        (
            stated["all_bootstrap_methods_agree"]
            == all((inference_decision(row) == inference_decision(primary) for row in primary_boot))
        ),
        "Invariant failed: stated['all_bootstrap_methods_agree'] == all((inference_decision(row) == inference_decision(primary) for row in primary_boot))",
    )
    return dict(
        passed=True,
        metric_rows=54,
        baseline_rows=18,
        development_metrics=9,
        raw_mse_tests=81,
        raw_direction_tests=81,
        calibration_mse_tests=27,
        bootstrap_comparisons=567,
        bootstrap_replicates=5000,
        bootstrap_seed=20261123,
        all_pvalues_confidence_intervals_and_holm_families_recomputed=True,
        conclusion_consistent_with_independent_metrics=True,
        primary_one_hour_llm_increment=stated,
        independent_cluster_helper_sha256=sha(ROOT / "work/market_ridge_fixed_20261118/verify.py"),
    )


def all_check():
    """Run the complete read-only audit of inputs, development, evaluation, and reporting."""
    summary = development_check()
    lock = selection_gate()
    fit = read(OUT / "EVAL_COMPLETE.json")
    require(
        (fit["locked_selection_sha256"] == sha(OUT / "LOCKED_SELECTION.json")),
        "Invariant failed: fit['locked_selection_sha256'] == sha(OUT / 'LOCKED_SELECTION.json')",
    )
    require(
        (fit["events_per_system"] == 497 and fit["systems"] == 9 and (fit["cells"] == 72)),
        "Invariant failed: fit['events_per_system'] == 497 and fit['systems'] == 9 and (fit['cells'] == 72)",
    )
    require(
        (fit["hyperparameter_choices"] == fit["epoch_choices"] == fit["calibration_choices"] == 0),
        "Invariant failed: fit['hyperparameter_choices'] == fit['epoch_choices'] == fit['calibration_choices'] == 0",
    )
    require(
        (fit["old_results_preserved"] and fit["llm_calls"] == fit["embedding_calls"] == 0),
        "Invariant failed: fit['old_results_preserved'] and fit['llm_calls'] == fit['embedding_calls'] == 0",
    )
    require(
        (fit["prediction_sha256"] == sha(OUT / "EVALUATION_PREDICTIONS.parquet")),
        "Invariant failed: fit['prediction_sha256'] == sha(OUT / 'EVALUATION_PREDICTIONS.parquet')",
    )
    frames, groups, bags, vectors, inputs = canonical_check("evaluation")
    predictions, cells = cells_check("evaluation", frames, groups, bags, vectors, lock)
    report = reporting_check(predictions, lock)
    summary.update(
        evaluation_input=inputs,
        evaluation=cells,
        report=report,
        no_evaluation_outcomes_read=False,
        evaluation_read_only_after_verified_lock=True,
        evaluation_parameter_choices=0,
        evaluation_epoch_choices=0,
        evaluation_calibration_updates=0,
        model_fit_calls=0,
        old_files_preserved=True,
    )
    frozen_check()
    selection_gate()
    return summary


def main():
    """Parse this module's command-line options and execute the explicitly requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["engine", "inputs", "development", "all"])
    args = parser.parse_args()
    activate_legacy_root(ROOT)
    if args.stage == "engine":
        result = engine_check()
        name = "ENGINE_INDEPENDENT_AUDIT.json"
    elif args.stage == "inputs":
        result = inputs_check()[-1]
        name = "INDEPENDENT_INPUT_AUDIT.json"
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

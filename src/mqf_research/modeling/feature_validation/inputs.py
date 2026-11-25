"""Guarded canonical inputs for the fixed NN feature-group comparison.

Only the pre-2025 common cohort is loaded for development. Reading any later
target requires this experiment's own completed, hashed selection receipt.
This module performs no fitting, PCA, LLM calls, embedding inference or scoring.
Existing MiniLM signal vectors are preserved individually and byte-for-byte.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from mqf_research.common.hashing import file_sha256 as sha
from mqf_research.common.serialization import read_json as read
from mqf_research.config import get_settings
from mqf_research.modeling._compatibility import bind_run_configuration
from mqf_research.modeling._validation import require

ROOT = get_settings().research_root
WORK = Path(__file__).resolve().parent
OUT = get_settings().output_root / "nn_feature_validation"
PROTOCOL = OUT / "PROTOCOL.json"
RELEASE = ROOT / "outputs/source_repair_batches/batch_032"
CONTEXT = ROOT / "outputs/valuation_industry_20261113_b032"
PRIOR = ROOT / "outputs/small_sample_ai_20261118_b032"
TAXONOMY = ROOT / "outputs/model_selection_20261118_b032/FEATURE_GROUPS.json"
COMMON = ROOT / "outputs/feature_horizon_comparison_20261111_b032/COMMON_EVENTS.csv"
HORIZONS = ("hour", "half", "full")
ENCODER = "minilm"
DEVELOPMENT_COUNTS = {2021: 105, 2022: 276, 2023: 238, 2024: 225}
ALL_COUNTS = {**DEVELOPMENT_COUNTS, 2025: 238, 2026: 259}
DEVELOPMENT_QUARTERS = [f"{y}Q{q}" for y in (2023, 2024) for q in range(1, 5)]
EVALUATION_QUARTERS = [f"{y}Q{q}" for y in (2025, 2026) for q in range(1, 5)]
METADATA = [
    "event_id",
    "ticker",
    "reaction_session",
    "effective_utc",
    "cutoff_utc",
    "estimation_end",
]
ORDERED_METADATA = [
    "event_id",
    "ticker",
    "reaction_session",
    "quarter",
    "year",
    "effective_utc",
    "cutoff_utc",
    "estimation_end",
]
CONTEXT_NUMERIC = [
    "context_log_pe_fy",
    "context_earnings_yield_fy",
    "context_fy_nonpositive",
    "context_fy_age_days",
    "context_log_pe_ttm",
    "context_earnings_yield_ttm",
    "context_ttm_nonpositive",
]
CATEGORICAL = ["context_sector_ff12", "context_industry_sic"]
CONTROLS = ["text_available", "has_signals", "log_signal_count"]


def _parameters(horizons, encoder, require_protocol=False):
    """Reject horizons, encoder choices, or protocols outside this fixed experiment."""
    require((tuple(horizons) == HORIZONS), "All three predeclared windows must be retained.")
    require((encoder == ENCODER), "This experiment freezes the existing MiniLM encoder.")
    if require_protocol:
        require((PROTOCOL.is_file()), "Write the fixed protocol before loading development inputs.")
        protocol = read(PROTOCOL)
        require(
            (tuple(protocol["input_horizons"]) == HORIZONS),
            "Invariant failed: tuple(protocol['input_horizons']) == HORIZONS",
        )
        require(
            (protocol["input_encoder"] == ENCODER),
            "Invariant failed: protocol['input_encoder'] == ENCODER",
        )


def feature_groups():
    """Return the pure M23, MF74 and MFL77 taxonomy, without a PCA transform."""
    source = read(TAXONOMY)
    base = read(RELEASE / "DATASET_BINDING.json")["financial_features"]
    require(
        (len(base) == len(set(base)) == 67), "Invariant failed: len(base) == len(set(base)) == 67"
    )
    require(
        (source["M"]["numeric"] == base[:23]),
        "Invariant failed: source['M']['numeric'] == base[:23]",
    )
    require(
        (source["MF"]["numeric"] == base + CONTEXT_NUMERIC),
        "Invariant failed: source['MF']['numeric'] == base + CONTEXT_NUMERIC",
    )
    require(
        (source["MFL"]["numeric"] == base + CONTEXT_NUMERIC + CONTROLS),
        "Invariant failed: source['MFL']['numeric'] == base + CONTEXT_NUMERIC + CONTROLS",
    )
    require(
        (source["M"]["categorical"] == []), "Invariant failed: source['M']['categorical'] == []"
    )
    require(
        (source["MF"]["categorical"] == source["MFL"]["categorical"] == CATEGORICAL),
        "Invariant failed: source['MF']['categorical'] == source['MFL']['categorical'] == CATEGORICAL",
    )
    names = {"M": "Market", "MF": "Market + Financial", "MFL": "Market + Financial + LLM"}
    groups = {
        key: dict(
            name=names[key],
            numeric=list(source[key]["numeric"]),
            categorical=list(source[key]["categorical"]),
            use_signals=(key == "MFL"),
        )
        for key in ("M", "MF", "MFL")
    }
    for key, count in [("M", 23), ("MF", 74), ("MFL", 77)]:
        require(
            (len(groups[key]["numeric"]) == len(set(groups[key]["numeric"])) == count),
            "Invariant failed: len(groups[key]['numeric']) == len(set(groups[key]['numeric'])) == count",
        )
    require(
        (not set(CONTROLS).intersection(groups["M"]["numeric"])),
        "Invariant failed: not set(CONTROLS).intersection(groups['M']['numeric'])",
    )
    require(
        (not set(CONTROLS).intersection(groups["MF"]["numeric"])),
        "Invariant failed: not set(CONTROLS).intersection(groups['MF']['numeric'])",
    )
    require(
        (groups["MFL"]["numeric"][-3:] == CONTROLS),
        "Invariant failed: groups['MFL']['numeric'][-3:] == CONTROLS",
    )
    original = read(PRIOR / "FEATURES.json")
    require(
        (original["numeric"] == groups["MFL"]["numeric"]),
        "Invariant failed: original['numeric'] == groups['MFL']['numeric']",
    )
    require(
        (original["categorical"] == groups["MFL"]["categorical"]),
        "Invariant failed: original['categorical'] == groups['MFL']['categorical']",
    )
    return groups


def bag_source(encoder=ENCODER):
    """Describe frozen signal inputs; no arrays or outcome values are read."""
    require((encoder == ENCODER), "Invariant failed: encoder == ENCODER")
    folder = RELEASE / "embeddings_minilm"
    return dict(
        encoder=ENCODER,
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        normalization="preserve_existing_vectors",
        pooling="none_individual_signals",
        index=folder / "INDEX.json",
        vectors=folder / "SIGNAL_VECTORS.npy",
        complete=folder / "COMPLETE.json",
        events=RELEASE / "EVENTS.json",
    )


def source_paths(horizons=HORIZONS, encoder=ENCODER):
    """Paths to freeze before development; whole-file hashing is not target use."""
    _parameters(horizons, encoder)
    binding = read(RELEASE / "DATASET_BINDING.json")
    bags = bag_source(encoder)
    files = [
        Path(__file__),
        PROTOCOL,
        TAXONOMY,
        COMMON,
        RELEASE / "DATASET_BINDING.json",
        RELEASE / "RELEASE.json",
        bags["events"],
        bags["index"],
        bags["vectors"],
        bags["complete"],
        CONTEXT / "CONTEXT_FEATURES.parquet",
        CONTEXT / "FROZEN_HASHES.json",
        PRIOR / "DATA.parquet",
        PRIOR / "PREPARED.json",
        PRIOR / "FEATURES.json",
    ]
    files.extend(Path(binding["datasets"][h]) for h in horizons)
    return list(dict.fromkeys(files))


def _verify_sources(horizons, encoder):
    """Validate release, dataset, context, and vector digests before reading inputs."""
    binding = read(RELEASE / "DATASET_BINDING.json")
    release = read(RELEASE / "RELEASE.json")
    bag = bag_source(encoder)
    complete = read(bag["complete"])
    require(
        (complete["model"] == bag["model"]), "Invariant failed: complete['model'] == bag['model']"
    )
    require((complete["dimensions"] == 384), "Invariant failed: complete['dimensions'] == 384")
    for name in ["DATASET_BINDING.json", "EVENTS.json"]:
        require((sha(RELEASE / name) == release["files"][name]), name)
    for horizon in horizons:
        path = Path(binding["datasets"][horizon])
        require(
            (path == RELEASE / f"DATASET_financial_{horizon}.parquet"),
            "Invariant failed: path == RELEASE / f'DATASET_financial_{horizon}.parquet'",
        )
        require(
            (sha(path) == binding["hashes"][horizon] == release["files"][path.name]),
            "Invariant failed: sha(path) == binding['hashes'][horizon] == release['files'][path.name]",
        )
    require(
        (sha(bag["index"]) == complete["index_sha256"]),
        "Invariant failed: sha(bag['index']) == complete['index_sha256']",
    )
    require(
        (sha(bag["vectors"]) == complete["vectors_sha256"]),
        "Invariant failed: sha(bag['vectors']) == complete['vectors_sha256']",
    )
    require(
        (sha(COMMON) == "3c6b37350f76676142b77fcc0be196b47cd774642ad6c2ca4df0fdafaa59f1e8"),
        "Invariant failed: sha(COMMON) == '3c6b37350f76676142b77fcc0be196b47cd774642ad6c2ca4df0fdafaa59f1e8'",
    )
    context_path = CONTEXT / "CONTEXT_FEATURES.parquet"
    require(
        (sha(context_path) == read(CONTEXT / "FROZEN_HASHES.json")[str(context_path)]),
        "Invariant failed: sha(context_path) == read(CONTEXT / 'FROZEN_HASHES.json')[str(context_path)]",
    )
    require(
        (sha(PRIOR / "DATA.parquet") == read(PRIOR / "PREPARED.json")["data_sha256"]),
        "Invariant failed: sha(PRIOR / 'DATA.parquet') == read(PRIOR / 'PREPARED.json')['data_sha256']",
    )


def verify_selection_gate():
    """Only this experiment's completed, hash-bound selection unlocks evaluation."""
    bind_run_configuration(OUT)
    selection_path = OUT / "LOCKED_SELECTION.json"
    receipt_path = OUT / "SELECTION_COMPLETE.json"
    require(
        (selection_path.is_file() and receipt_path.is_file()),
        "2025-2026 targets remain closed until this experiment completes selection.",
    )
    receipt = read(receipt_path)
    require(
        (receipt.get("protocol_sha256") == sha(PROTOCOL)),
        "Invariant failed: receipt.get('protocol_sha256') == sha(PROTOCOL)",
    )
    require(
        (receipt.get("locked_selection_sha256") == sha(selection_path)),
        "Invariant failed: receipt.get('locked_selection_sha256') == sha(selection_path)",
    )
    require(
        (receipt.get("development_only") is True),
        "Invariant failed: receipt.get('development_only') is True",
    )
    require(
        (receipt.get("no_evaluation_outcomes_used") is True),
        "Invariant failed: receipt.get('no_evaluation_outcomes_used') is True",
    )
    selection = read(selection_path)
    require(
        (isinstance(selection, dict) and selection.get("development_only") is True),
        "Invariant failed: isinstance(selection, dict) and selection.get('development_only') is True",
    )
    return selection, receipt


def _cohort(stage):
    # COMMON contains dates and identifiers only, never realized outcomes.
    """Return the permitted common event metadata, applying the evaluation lock first."""
    cohort = pd.read_csv(COMMON)
    require(
        (cohort.event_id.is_unique and len(cohort) == 1341),
        "Invariant failed: cohort.event_id.is_unique and len(cohort) == 1341",
    )
    require(
        (cohort.groupby("year").size().to_dict() == ALL_COUNTS),
        "Invariant failed: cohort.groupby('year').size().to_dict() == ALL_COUNTS",
    )
    if stage == "development":
        cohort = cohort[cohort.year.le(2024)].copy()
        require((len(cohort) == 844), "Invariant failed: len(cohort) == 844")
    else:
        require((stage == "evaluation"), "Invariant failed: stage == 'evaluation'")
        verify_selection_gate()
    return cohort.sort_values(["reaction_session", "event_id"]).reset_index(drop=True)


def _signal_inputs(ids, encoder):
    """Gather vectors for the permitted IDs only, retaining every original signal."""
    ids = list(ids)
    permitted = set(ids)
    source = bag_source(encoder)
    event_rows = read(source["events"])
    events = {row["event_id"]: row for row in event_rows if row["event_id"] in permitted}
    require((set(events) == permitted), "Invariant failed: set(events) == permitted")
    index = read(source["index"])
    require(
        ([row["row"] for row in index] == list(range(len(index)))),
        "Invariant failed: [row['row'] for row in index] == list(range(len(index)))",
    )
    require(
        (len({row["signal_id"] for row in index}) == len(index)),
        "Invariant failed: len({row['signal_id'] for row in index}) == len(index)",
    )
    selected = [row for row in index if row["event_id"] in permitted]
    counts = Counter(row["event_id"] for row in selected)
    require(
        (all((counts[event] == events[event]["ml_signal_count"] for event in ids))),
        "Invariant failed: all((counts[event] == events[event]['ml_signal_count'] for event in ids))",
    )
    source_rows = [row["row"] for row in selected]
    full_vectors = np.load(source["vectors"], mmap_mode="r", allow_pickle=False)
    require(
        (full_vectors.shape == (len(index), 384)),
        "Invariant failed: full_vectors.shape == (len(index), 384)",
    )
    require(
        (full_vectors.dtype == np.dtype("float32")),
        "Invariant failed: full_vectors.dtype == np.dtype('float32')",
    )
    vectors = np.array(full_vectors[source_rows], dtype=np.float32, copy=True, order="C")
    require((np.isfinite(vectors).all()), "Invariant failed: np.isfinite(vectors).all()")
    # Fancy indexing copies the selected rows without transforming their values.
    require(
        (np.array_equal(vectors, full_vectors[source_rows])),
        "Invariant failed: np.array_equal(vectors, full_vectors[source_rows])",
    )
    del full_vectors
    bags = {event: [] for event in ids}
    signal_records = []
    for compact_row, row in enumerate(selected):
        bags[row["event_id"]].append(compact_row)
        signal_records.append(
            dict(
                compact_row=compact_row,
                source_row=row["row"],
                event_id=row["event_id"],
                signal_id=row["signal_id"],
                embedding_text_sha256=row["embedding_text_sha256"],
            )
        )
    require(
        (sorted((i for bag in bags.values() for i in bag)) == list(range(len(vectors)))),
        "Invariant failed: sorted((i for bag in bags.values() for i in bag)) == list(range(len(vectors)))",
    )
    event_records = []
    for event in ids:
        record = events[event]
        available = (
            record["extraction_status"] == "complete"
            or record.get("text_input_status") == "manual_source_review_complete"
        ) and record.get("text_source_eligible", True)
        source_eligible = bool(record.get("text_source_eligible", True))
        has_signals = bool(bags[event])
        # A missing bag is retained as a valid event with zero text contribution.
        # A populated bag must never pass a source-eligibility exclusion.
        require((not has_signals or source_eligible), event)
        require((not has_signals or available), event)
        event_records.append(
            dict(
                event_id=event,
                extraction_status=record["extraction_status"],
                text_input_status=record.get("text_input_status", ""),
                source_eligible=source_eligible,
                text_available=bool(available),
                has_signals=has_signals,
                signal_count=len(bags[event]),
                log_signal_count=np.log1p(len(bags[event])),
                empty_bag=not has_signals,
                original_ml_signal_count=int(record["ml_signal_count"]),
            )
        )
    audit = pd.DataFrame(event_records)
    index_audit = pd.DataFrame(
        signal_records,
        columns=["compact_row", "source_row", "event_id", "signal_id", "embedding_text_sha256"],
    )
    return bags, vectors, audit, index_audit


def _load(stage, horizons, encoder):
    """Build aligned features, targets, and signal bags for one authorized data stage.

    Verify the point-in-time cutoff, release hashes, identical event cohort, and feature taxonomy.
    """
    _parameters(horizons, encoder, require_protocol=True)
    if stage == "evaluation":
        verify_selection_gate()
    else:
        require((stage == "development"), "Invariant failed: stage == 'development'")
    _verify_sources(horizons, encoder)
    groups = feature_groups()
    base = read(RELEASE / "DATASET_BINDING.json")["financial_features"]
    binding = read(RELEASE / "DATASET_BINDING.json")
    cohort = _cohort(stage)
    ids = cohort.event_id.tolist()
    require(
        (stage != "development" or cohort.year.le(2024).all()),
        "Invariant failed: stage != 'development' or cohort.year.le(2024).all()",
    )
    context = pd.read_parquet(
        CONTEXT / "CONTEXT_FEATURES.parquet",
        columns=["event_id", *CONTEXT_NUMERIC, *CATEGORICAL],
        filters=[("event_id", "in", ids)],
    )
    require(
        (context.event_id.is_unique and set(context.event_id) == set(ids)),
        "Invariant failed: context.event_id.is_unique and set(context.event_id) == set(ids)",
    )
    bags, vectors, audit, index_audit = _signal_inputs(ids, encoder)
    controls = audit[["event_id", *CONTROLS]]
    frames = {}
    for horizon in horizons:
        if stage == "evaluation":
            verify_selection_gate()  # Check again immediately before each target read.
        frame = pd.read_parquet(
            binding["datasets"][horizon],
            columns=METADATA + ["y"] + base,
            filters=[("event_id", "in", ids)],
        )
        require(
            (frame.event_id.is_unique and set(frame.event_id) == set(ids)),
            "Invariant failed: frame.event_id.is_unique and set(frame.event_id) == set(ids)",
        )
        frame["reaction_session"] = pd.to_datetime(frame.reaction_session).dt.strftime("%Y-%m-%d")
        frame["quarter"] = pd.to_datetime(frame.reaction_session).dt.to_period("Q").astype(str)
        frame["year"] = pd.to_datetime(frame.reaction_session).dt.year
        frame = frame.merge(context, on="event_id", how="left", validate="one_to_one")
        frame = frame.merge(controls, on="event_id", how="left", validate="one_to_one")
        frame = frame.sort_values(["reaction_session", "event_id"]).reset_index(drop=True)
        frame = frame[ORDERED_METADATA + ["y"] + groups["MFL"]["numeric"] + CATEGORICAL]
        require(
            (
                frame.groupby("year").size().to_dict()
                == (DEVELOPMENT_COUNTS if stage == "development" else ALL_COUNTS)
            ),
            "Invariant failed: frame.groupby('year').size().to_dict() == (DEVELOPMENT_COUNTS if stage == 'development' else ALL_COUNTS)",
        )
        require(
            (np.isfinite(frame.y.to_numpy(float)).all()),
            "Invariant failed: np.isfinite(frame.y.to_numpy(float)).all()",
        )
        require(
            (not np.isinf(frame[groups["MFL"]["numeric"]].to_numpy(float)).any()),
            "Invariant failed: not np.isinf(frame[groups['MFL']['numeric']].to_numpy(float)).any()",
        )
        require(
            (frame[CATEGORICAL].notna().all().all()),
            "Invariant failed: frame[CATEGORICAL].notna().all().all()",
        )
        require(
            (
                (
                    pd.to_datetime(frame.effective_utc, utc=True)
                    <= pd.to_datetime(frame.cutoff_utc, utc=True)
                ).all()
            ),
            "Invariant failed: (pd.to_datetime(frame.effective_utc, utc=True) <= pd.to_datetime(frame.cutoff_utc, utc=True)).all()",
        )
        require(
            ((pd.to_datetime(frame.estimation_end) < pd.to_datetime(frame.reaction_session)).all()),
            "Invariant failed: (pd.to_datetime(frame.estimation_end) < pd.to_datetime(frame.reaction_session)).all()",
        )
        compare = ["event_id", "ticker", "reaction_session", "quarter", "year"]
        pd.testing.assert_frame_equal(frame[compare], cohort[compare], check_dtype=False)
        frames[horizon] = frame
    # Exact comparison reads an old file containing only the 844 development rows.
    old = pd.read_parquet(PRIOR / "DATA.parquet")
    require(
        (len(old) == 844 and old.year.le(2024).all()),
        "Invariant failed: len(old) == 844 and old.year.le(2024).all()",
    )
    prior_hour = frames["hour"].loc[frames["hour"].year.le(2024)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prior_hour, old, check_dtype=False, check_exact=True)
    if stage == "evaluation":
        verify_selection_gate()
    return frames, groups, bags, vectors, audit, index_audit


def load_development(horizons=HORIZONS, encoder=ENCODER):
    """Return (frames, groups, bags, vectors) for exactly 844 pre-2025 events.

    Frames contain all potential input columns. Consumers must select numeric
    and categorical columns from the group definition; M/MF have no LLM control
    columns and must use a zero text branch. Empty bags remain in every cohort.
    """
    return _load("development", horizons, encoder)[:4]


def load_evaluation(horizons=HORIZONS, encoder=ENCODER):
    """Return all 1,341 events only after this experiment locks its selection."""
    verify_selection_gate()
    return _load("evaluation", horizons, encoder)[:4]


def self_check_development():
    """Validate development-only inputs and persist event/signal provenance."""
    frames, groups, bags, vectors, audit, index_audit = _load("development", HORIZONS, ENCODER)
    check_dir = OUT / "checks"
    check_dir.mkdir(parents=True, exist_ok=True)
    audit_path = check_dir / "DEVELOPMENT_BAG_AUDIT.csv"
    index_path = check_dir / "DEVELOPMENT_SIGNAL_INDEX.csv"
    audit.to_csv(audit_path, index=False)
    index_audit.to_csv(index_path, index=False)
    if (OUT / "LOCKED_SELECTION.json").exists() and (OUT / "SELECTION_COMPLETE.json").exists():
        gate_check = "Not exercised: completed selection files already exist"
    else:
        try:
            load_evaluation()
        except ValueError:
            gate_check = "Passed: absent selection receipt prevents evaluation target reads"
        else:
            raise AssertionError("Evaluation unexpectedly bypassed the new selection gate")
    result = dict(
        passed=True,
        horizons=list(HORIZONS),
        encoder=ENCODER,
        dimensions=int(vectors.shape[1]),
        signal_rows=len(vectors),
        development_rows={h: len(frame) for h, frame in frames.items()},
        year_counts={str(k): v for k, v in DEVELOPMENT_COUNTS.items()},
        group_numeric_counts={k: len(g["numeric"]) for k, g in groups.items()},
        group_categorical_counts={k: len(g["categorical"]) for k, g in groups.items()},
        empty_bags=int(audit.empty_bag.sum()),
        eligible_source_events=int(audit.source_eligible.sum()),
        text_available_events=int(audit.text_available.sum()),
        populated_bags=int(audit.has_signals.sum()),
        original_hour844_all_columns_exact=True,
        selected_signal_vectors_bitwise_identical=True,
        common_event_order_across_horizons=True,
        llm_controls_excluded_from_M_and_MF=True,
        model_fits=0,
        llm_calls=0,
        embedding_calls=0,
        evaluation_outcomes_loaded=False,
        evaluation_gate=gate_check,
        script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(PROTOCOL),
        bag_audit_sha256=sha(audit_path),
        signal_index_sha256=sha(index_path),
    )
    (check_dir / "INPUT_HELPER_CHECK.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    """Run only the requested input check; --help does not load or write datasets."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check"])
    parser.parse_args()
    self_check_development()


if __name__ == "__main__":
    main()

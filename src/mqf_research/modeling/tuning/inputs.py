"""Guarded inputs for development-only numeric-neural-network tuning.

Development loading reads only the exact pre-2025 cohort and its already saved
reference forecasts. The full-period loader is imported only after a new model
selection and its hash receipt have been written. No model fitting, LLM calls,
embedding-array reads or performance computations occur in this module.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mqf_research.common.hashing import file_sha256 as sha
from mqf_research.common.serialization import read_json as read
from mqf_research.config import get_settings
from mqf_research.modeling._compatibility import (
    activate_legacy_root,
    bind_run_configuration,
    load_legacy_module,
)
from mqf_research.modeling._validation import require

ROOT = get_settings().research_root
WORK = Path(__file__).resolve().parent
OUT = get_settings().output_root / "nn_tuning"
PROTOCOL = OUT / "PROTOCOL.json"
PARENT = ROOT / "outputs/small_sample_ai_20261118_b032"
DEVELOPMENT_MODULE = "work.small_sample_ai_20261118.inputs"
EVALUATION_MODULE = "work.tabpfn_fixed_20261119_b032.inputs"
DEVELOPMENT_QUARTERS = [f"{year}Q{q}" for year in [2023, 2024] for q in range(1, 5)]
EVALUATION_QUARTERS = [f"{year}Q{q}" for year in [2025, 2026] for q in range(1, 5)]
DEVELOPMENT_COUNTS = {2021: 105, 2022: 276, 2023: 238, 2024: 225}
ALL_COUNTS = {**DEVELOPMENT_COUNTS, 2025: 238, 2026: 259}


def feature_lists():
    """Return the unchanged original 77 numeric and two categorical fields."""
    original = load_legacy_module(DEVELOPMENT_MODULE, ROOT)
    numeric, categorical = original.feature_lists()
    expected = read(PARENT / "FEATURES.json")
    require(
        (numeric == expected["numeric"] and categorical == expected["categorical"]),
        "Invariant failed: numeric == expected['numeric'] and categorical == expected['categorical']",
    )
    require(
        (len(numeric) == len(set(numeric)) == 77),
        "Invariant failed: len(numeric) == len(set(numeric)) == 77",
    )
    require(
        (len(categorical) == len(set(categorical)) == 2),
        "Invariant failed: len(categorical) == len(set(categorical)) == 2",
    )
    return list(numeric), list(categorical)


def source_paths_development():
    """Enumerate source files without parsing any 2025+ target values.

    Some source parquet files contain later rows; hashing an entire source file
    is a provenance check, while load_development() filters before target reads.
    """
    original = load_legacy_module(DEVELOPMENT_MODULE, ROOT)
    paths = original.source_paths() + [
        Path(__file__),
        PROTOCOL,
        PARENT / "DATA.parquet",
        PARENT / "PREPARED.json",
        PARENT / "FEATURES.json",
    ]
    return list(dict.fromkeys(Path(path) for path in paths))


def load_development():
    """Return (844 development rows, 77 numeric fields, 2 categories, 463 NN references)."""
    require(
        (PROTOCOL.is_file() and read(PROTOCOL)),
        "Record the new fixed tuning protocol before loading development data.",
    )
    original = load_legacy_module(DEVELOPMENT_MODULE, ROOT)
    data, numeric, categorical, reference = original.load()
    expected_numeric, expected_categorical = feature_lists()
    require(
        (numeric == expected_numeric and categorical == expected_categorical),
        "Invariant failed: numeric == expected_numeric and categorical == expected_categorical",
    )
    require(
        (data.event_id.is_unique and len(data) == 844),
        "Invariant failed: data.event_id.is_unique and len(data) == 844",
    )
    require(
        (data.groupby("year").size().to_dict() == DEVELOPMENT_COUNTS),
        "Invariant failed: data.groupby('year').size().to_dict() == DEVELOPMENT_COUNTS",
    )
    require(
        (data.reaction_session.max() == "2024-12-20"),
        "Invariant failed: data.reaction_session.max() == '2024-12-20'",
    )
    require(
        (data.year.le(2024).all() and reference.year.le(2024).all()),
        "Invariant failed: data.year.le(2024).all() and reference.year.le(2024).all()",
    )
    require(
        (len(reference) == 463 and reference.event_id.is_unique),
        "Invariant failed: len(reference) == 463 and reference.event_id.is_unique",
    )
    require(
        (np.isfinite(data.y.to_numpy(float)).all()),
        "Invariant failed: np.isfinite(data.y.to_numpy(float)).all()",
    )
    require(
        (not np.isinf(data[numeric].to_numpy(float)).any()),
        "Invariant failed: not np.isinf(data[numeric].to_numpy(float)).any()",
    )
    require(
        (sha(PARENT / "DATA.parquet") == read(PARENT / "PREPARED.json")["data_sha256"]),
        "Invariant failed: sha(PARENT / 'DATA.parquet') == read(PARENT / 'PREPARED.json')['data_sha256']",
    )
    saved = pd.read_parquet(PARENT / "DATA.parquet")
    pd.testing.assert_frame_equal(data, saved, check_dtype=False, check_exact=True)
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
    return data.copy(), list(numeric), list(categorical), reference.copy()


def verify_selection_gate():
    """Verify the selection receipt before importing or reading full-period inputs."""
    bind_run_configuration(OUT)
    selection_path = OUT / "LOCKED_SELECTION.json"
    receipt_path = OUT / "SELECTION_COMPLETE.json"
    require(
        (selection_path.is_file() and receipt_path.is_file()),
        "2025-2026 inputs remain closed until LOCKED_SELECTION.json and SELECTION_COMPLETE.json exist.",
    )
    receipt = read(receipt_path)
    require(
        (receipt.get("locked_selection_sha256") == sha(selection_path)),
        "The model-selection hash does not match the completed selection receipt.",
    )
    require(
        (receipt.get("development_only") is True),
        "Invariant failed: receipt.get('development_only') is True",
    )
    require(
        (receipt.get("no_evaluation_outcomes_used") is True),
        "Invariant failed: receipt.get('no_evaluation_outcomes_used') is True",
    )
    require(
        (receipt.get("protocol_sha256") == sha(PROTOCOL)),
        "The completed selection does not belong to the currently frozen protocol.",
    )
    selection = read(selection_path)
    require(
        (isinstance(selection, dict) and selection),
        "Invariant failed: isinstance(selection, dict) and selection",
    )
    return selection, receipt


def source_paths_evaluation():
    """Enumerate full-period input sources only after the completed selection gate."""
    verify_selection_gate()
    full = load_legacy_module(EVALUATION_MODULE, ROOT)
    paths = full.source_paths() + [
        Path(__file__),
        PROTOCOL,
        OUT / "LOCKED_SELECTION.json",
        OUT / "SELECTION_COMPLETE.json",
    ]
    return list(dict.fromkeys(Path(path) for path in paths))


def load_evaluation():
    """Return (all 1,341 rows, 77 numeric fields, 2 categories) after locking selection.

    The returned history supports fixed expanding-quarter evaluation: earlier
    elapsed evaluation outcomes may enter a later quarter's training history,
    but this function cannot be reached before development-only selection.
    """
    verify_selection_gate()
    development, numeric, categorical, _ = load_development()
    # Deliberately defer this import as well as its full-period read until the
    # new locked-selection receipt has passed every gate above.
    full = load_legacy_module(EVALUATION_MODULE, ROOT)
    data, full_numeric, full_categorical = full.load()
    require(
        (full_numeric == numeric and full_categorical == categorical),
        "Invariant failed: full_numeric == numeric and full_categorical == categorical",
    )
    require(
        (len(data) == 1341 and data.event_id.is_unique),
        "Invariant failed: len(data) == 1341 and data.event_id.is_unique",
    )
    require(
        (data.groupby("year").size().to_dict() == ALL_COUNTS),
        "Invariant failed: data.groupby('year').size().to_dict() == ALL_COUNTS",
    )
    require(
        (data.reaction_session.max() == "2026-10-02"),
        "Invariant failed: data.reaction_session.max() == '2026-10-02'",
    )
    prior = data[data.year.le(2024)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prior, development, check_dtype=False, check_exact=True)
    require(
        (len(data[data.year.ge(2025)]) == 497),
        "Invariant failed: len(data[data.year.ge(2025)]) == 497",
    )
    verify_selection_gate()
    return data.copy(), list(numeric), list(categorical)


def self_check_development():
    """Check only development inputs and write a compact receipt; never load evaluation."""
    evaluation_imported_before = EVALUATION_MODULE in sys.modules
    data, numeric, categorical, reference = load_development()
    require(
        ((EVALUATION_MODULE in sys.modules) == evaluation_imported_before),
        "Invariant failed: (EVALUATION_MODULE in sys.modules) == evaluation_imported_before",
    )
    gate_result = "Not exercised: selection receipt already exists"
    if (
        not (OUT / "LOCKED_SELECTION.json").exists()
        or not (OUT / "SELECTION_COMPLETE.json").exists()
    ):
        try:
            verify_selection_gate()
        except ValueError:
            gate_result = "Passed: missing selection receipt blocks evaluation"
        else:
            raise AssertionError("Evaluation gate unexpectedly accepted missing selection receipt")
    result = dict(
        passed=True,
        development_rows=len(data),
        reference_rows=len(reference),
        numeric_features=len(numeric),
        categorical_features=len(categorical),
        year_counts={str(k): int(v) for k, v in data.groupby("year").size().items()},
        first_date=data.reaction_session.min(),
        last_date=data.reaction_session.max(),
        original844_all_columns_exact=True,
        evaluation_targets_loaded=False,
        full_period_loader_imported_by_check=False,
        evaluation_gate=gate_result,
        model_fits=0,
        llm_calls=0,
        embedding_calls=0,
        script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(PROTOCOL),
    )
    check_dir = OUT / "checks"
    check_dir.mkdir(parents=True, exist_ok=True)
    (check_dir / "INPUT_HELPER_CHECK.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    """Run the explicit development-input check; --help never reads research data."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check"])
    parser.parse_args()
    activate_legacy_root(ROOT)
    self_check_development()


if __name__ == "__main__":
    main()

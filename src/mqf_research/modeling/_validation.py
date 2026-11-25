"""Persistent scientific input checks shared by the maintained model engines.

Unlike ``assert``, these guards remain active when Python uses ``-O``. They
reject malformed inputs without changing arithmetic for valid historical data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import pandas as pd


def require(condition: Any, message: Any) -> None:
    """Raise ValueError when an input/provenance invariant is not satisfied.

    ``condition`` must be scalar, just as in the original assertion. Keeping the
    supplied message preserves useful file paths and identifiers in failures.
    """
    if not condition:
        raise ValueError(str(message))


def validate_history(history: pd.DataFrame) -> None:
    """Reject impossible or malformed histories before searching quarter cutoffs.

    At least 300 observations are necessary for the fixed 200/100 split. Without
    this check the historical loop never terminated when fewer than 100 rows
    were available. The engine still verifies the stricter realized partition.
    """
    import numpy as np
    import pandas as pd

    required = {"quarter", "reaction_session"}
    require(required.issubset(history.columns), "History needs quarter and reaction_session")
    require(len(history) >= 300, "Chronological stopping requires at least 300 history events")
    require(
        not history[list(required)].isna().any().any(), "History dates/quarters cannot be missing"
    )
    dates = pd.to_datetime(history["reaction_session"], errors="raise")
    quarters = dates.dt.to_period("Q").astype(str)
    require(
        np.array_equal(quarters.to_numpy(), history["quarter"].astype(str).to_numpy()),
        "History quarter must agree with reaction_session",
    )
    require(quarters.nunique() >= 3, "Stopping needs a core quarter and two validation quarters")


def validate_preprocessing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> None:
    """Validate event alignment and chronology before any fitted preprocessing.

    Only training targets are checked or used. Evaluation targets deliberately
    do not influence imputation, scaling, target moments, or category vocabulary.
    Numeric missing values remain allowed because median imputation handles them.
    """
    import numpy as np
    import pandas as pd

    require(len(train) > 0 and len(test) > 0, "Preprocessing needs nonempty train and test frames")
    require(len(numeric) > 0, "At least one numeric feature is required")
    features = list(numeric) + list(categorical)
    require(len(features) == len(set(features)), "Feature names must be unique across groups")
    required = {"event_id", "reaction_session", *features}
    for name, frame in (("train", train), ("test", test)):
        require(
            required.issubset(frame.columns),
            f"Missing {name} columns: {sorted(required - set(frame.columns))}",
        )
        require(
            frame.event_id.notna().all() and frame.event_id.is_unique,
            f"{name} event IDs must be unique and present",
        )
        require(frame.reaction_session.notna().all(), f"{name} reaction dates cannot be missing")
        require(
            not np.isinf(frame[list(numeric)].to_numpy(dtype=float)).any(),
            f"{name} numeric features cannot contain infinity",
        )
    require(set(train.event_id).isdisjoint(test.event_id), "Train/test events overlap")
    require(
        pd.to_datetime(train.reaction_session).max() < pd.to_datetime(test.reaction_session).min(),
        "Training events must be strictly earlier than test events",
    )
    require(
        "y" in train and np.isfinite(train.y.to_numpy(float)).all(),
        "Training targets must be present and finite",
    )


def ensure_new_checkpoints(destination: str | Path, seeds: Sequence[int]) -> Path:
    """Refuse to overwrite completed or partial model outputs in an existing run.

    Resume is handled by a higher-level completion receipt. A partial fit is not
    silently replaced: the caller must choose a fresh run directory explicitly.
    """
    destination = Path(destination)
    names = [*(f"seed{seed}.pt" for seed in seeds), "SEED_PREDICTIONS.npy"]
    existing = [str(destination / name) for name in names if (destination / name).exists()]
    if existing:
        raise FileExistsError(
            "Checkpoint outputs already exist; use a fresh run: " + ", ".join(existing)
        )
    return destination


def ensure_empty_cell(destination: str | Path) -> None:
    """Reject partial cell artifacts before matrices or audit metadata can be replaced.

    A completion receipt must be checked by the caller first. The exclusive
    ``RUN.lock`` file is allowed because the caller holds it while checking.
    """
    destination = Path(destination)
    remaining = (
        [path.name for path in destination.iterdir() if path.name != "RUN.lock"]
        if destination.exists()
        else []
    )
    if remaining:
        raise FileExistsError(
            f"Incomplete cell contains existing artifacts: {destination}; "
            "choose a fresh output root instead of replacing " + ", ".join(sorted(remaining))
        )

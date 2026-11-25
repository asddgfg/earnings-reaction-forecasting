"""Report-only conversion of NumPy values; never transform training inputs here."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def json_ready(value: Any, *, allow_arrays: bool = False) -> Any:
    """Convert scalar NumPy values and nonfinite report values to strict JSON types.

    Dictionary keys become strings, tuples become lists, and nonfinite floats
    become null. Array conversion is opt-in: the two historical report writers
    reject unexpected ndarrays, whereas cell completion receipts accept them.
    This policy applies only to serialization and does not alter model arrays.
    """
    if isinstance(value, dict):
        return {
            str(key): json_ready(item, allow_arrays=allow_arrays) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_ready(item, allow_arrays=allow_arrays) for item in value]
    if allow_arrays and isinstance(value, np.ndarray):
        return json_ready(value.tolist(), allow_arrays=True)
    if isinstance(value, np.generic):
        return json_ready(value.item(), allow_arrays=allow_arrays)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

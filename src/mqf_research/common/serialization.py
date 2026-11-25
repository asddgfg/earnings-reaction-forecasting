"""Explicit JSON encoding and atomic artifact writes without scientific coercions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .files import atomic_write


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document, accepting the BOM emitted by Windows editors."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(
    path: str | Path,
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    create_parents: bool = False,
    ensure_ascii: bool = True,
    trailing_newline: bool = True,
    newline: str | None = None,
) -> None:
    """Atomically publish strict indented JSON with caller-controlled formatting.

    Nonfinite numbers raise before any file is touched. Use an explicit numeric
    adapter when a report intentionally represents missing values as JSON null.
    ``newline=None`` matches ``Path.write_text``'s native newline conversion,
    preserving historical Windows receipt bytes; ``newline='\n'`` is portable LF.
    """
    text = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=ensure_ascii, default=default)
    if trailing_newline:
        text += "\n"
    line_separator = os.linesep if newline is None else newline
    if line_separator not in ("\n", "\r\n", "\r"):
        raise ValueError("newline must be LF, CRLF, CR, or None")
    content = text.replace("\n", line_separator).encode("utf-8")
    atomic_write(path, content, create_parents=create_parents)

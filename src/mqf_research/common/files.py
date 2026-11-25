"""Failure-safe single-file publication shared by caches and research artifacts."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write(path: str | Path, content: bytes, *, create_parents: bool = True) -> None:
    """Flush bytes to a unique same-directory temporary file, then replace the target.

    Readers observe a complete old or new file. Concurrent writers own distinct
    staging files, and the last successful replacement wins. A payload and its
    checksum sidecar are still two separate publications, not a transaction.
    ``create_parents=False`` preserves callers that require an existing run dir.
    """
    path = Path(path)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Staging on the destination volume is required for an atomic rename.
        # Never use a shared `.tmp` name: parallel workers could publish or
        # remove each other's partially written bytes.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                # Windows scanners/readers can briefly hold the destination.
                # Retry only sharing/access denial, bounded to 310 ms; other
                # I/O failures and permanent denial propagate to the caller.
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(0.01 * 2**attempt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

"""Content hashes for cache integrity and reproducible research receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def bytes_sha256(content: bytes) -> str:
    """Hash unchanged bytes; callers decide text encoding or JSON canonicalization."""
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file in bounded 1 MiB chunks, including empty and large vector files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

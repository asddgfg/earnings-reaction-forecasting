"""Resolve archived source files without rewriting immutable experiment manifests.

The relocation catalog is an explicit migration receipt, not a search path. Each
entry binds an old absolute filename to identical bytes under the project archive.
Missing, changed, or escaping entries fail instead of selecting a similar file.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common.hashing import file_sha256
from .common.serialization import read_json


def _valid_digest(value: object) -> bool:
    """Recognize the canonical lowercase SHA-256 representation used by manifests."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relocation_entries(project_root: Path) -> dict[Path, dict[str, str]]:
    """Load only a completed, unambiguous migration catalog with valid entry shapes.

    Keys are resolved original absolute paths. Returning validated entries lets a
    package loader check every preserved module before executing an initializer.
    Destination containment and actual content hashes are checked by the resolver.
    """
    catalog = project_root.resolve() / "configs/source_relocations.json"
    payload = read_json(catalog)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported source relocation catalog")
    if payload.get("state") != "executed_and_verified":
        raise ValueError("Source relocation has not been executed and verified")
    records = payload.get("files")
    if not isinstance(records, list):
        raise ValueError("Relocation catalog must contain a files list")
    entries = {}
    for entry in records:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(name), str) or not entry[name]
            for name in ("source", "destination")
        ):
            raise ValueError("Relocation entries require nonempty source and destination strings")
        source = Path(entry["source"])
        if not source.is_absolute():
            raise ValueError("Relocation source must be an absolute historical path")
        source = source.resolve()
        if not _valid_digest(entry.get("sha256")):
            raise ValueError("Relocated source has an invalid SHA-256 digest")
        if source in entries:
            raise ValueError(f"Duplicate source relocation: {source}")
        entries[source] = entry
    return entries


def resolve_preserved_source(
    source: Path,
    project_root: Path,
    expected_sha256: str | None = None,
) -> Path:
    """Find a preserved source at its original path or a verified archive location.

    Existing files never silently fall back to archived copies: their hashes must
    still match when a digest is supplied. Archive entries must stay inside this
    project's archive and carry a SHA-256 digest, including for legacy imports.
    """
    source = source.resolve()
    root = project_root.resolve()
    destination = source
    digest = expected_sha256
    if digest is not None and not _valid_digest(digest):
        raise ValueError("Expected source hash must be a lowercase SHA-256 digest")
    if not source.is_file():
        entries = relocation_entries(root)
        entry = entries.get(source)
        if entry is None:
            raise FileNotFoundError(f"No unique relocation for preserved source: {source}")
        destination = Path(entry["destination"])
        if not destination.is_absolute():
            destination = root / destination
        destination = destination.resolve()
        archive = root / "archive"
        # Resolve both the root and destination to reject junction/symlink escape.
        if archive.resolve() != archive or not destination.is_relative_to(archive):
            raise ValueError("Relocated source escapes the project archive")
        recorded = entry["sha256"]
        if digest is not None and recorded != digest:
            raise ValueError("Relocation digest conflicts with frozen provenance")
        digest = recorded
    if not destination.is_file():
        raise FileNotFoundError(destination)
    if digest is not None and file_sha256(destination) != digest:
        raise ValueError(f"Preserved source hash changed: {source}")
    return destination

"""Fail-closed relocation checks for immutable historical sources and legacy validators."""

import hashlib
import json
from pathlib import Path

import pytest

from mqf_research.data.legacy_tags import load_legacy_validator
from mqf_research.provenance import resolve_preserved_source


def _digest(content):
    """Produce an independent expected digest for synthetic source bytes."""
    return hashlib.sha256(content).hexdigest()


def _catalog(root, entries, **overrides):
    """Write a migration receipt without changing the historical source manifest."""
    target = root / "configs/source_relocations.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "state": "executed_and_verified", "files": entries}
    payload.update(overrides)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _relocation(root, name="legacy.py", content=b"VALUE = 7\n"):
    """Create one archived source whose original absolute filename is absent."""
    source = root / name
    target = root / "archive/source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    entry = {
        "source": str(source),
        "destination": str(target.relative_to(root)),
        "sha256": _digest(content),
    }
    return source, target, entry


def test_source_requires_original_or_unique_verified_relocation(tmp_path):
    """A missing original cannot silently be found by basename elsewhere."""
    source = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError):
        resolve_preserved_source(source, tmp_path)
    _catalog(tmp_path, [])
    with pytest.raises(FileNotFoundError):
        resolve_preserved_source(source, tmp_path)


def test_original_hash_mismatch_never_falls_back(tmp_path):
    """An altered original must be surfaced even if an archived pristine copy exists."""
    source, target, entry = _relocation(tmp_path)
    _catalog(tmp_path, [entry])
    source.write_bytes(b"VALUE = 999\n")
    with pytest.raises(ValueError, match="hash changed"):
        resolve_preserved_source(source, tmp_path, entry["sha256"])
    assert target.read_bytes() == b"VALUE = 7\n"


def test_verified_relocation_preserves_exact_bytes_and_manifest(tmp_path):
    """The resolver returns the archived file and leaves the migration receipt untouched."""
    source, target, entry = _relocation(tmp_path)
    catalog = _catalog(tmp_path, [entry])
    previous = catalog.read_bytes()
    assert resolve_preserved_source(source, tmp_path, entry["sha256"]) == target.resolve()
    assert catalog.read_bytes() == previous
    assert not source.exists()


@pytest.mark.parametrize("state", [False, None, "planned", True, 1])
def test_planned_or_ambiguous_migration_cannot_authorize_import(tmp_path, state):
    """Only a verified completed migration is a valid source of executable code."""
    source, _, entry = _relocation(tmp_path)
    _catalog(tmp_path, [entry], state=state)
    with pytest.raises(ValueError):
        resolve_preserved_source(source, tmp_path)


def test_missing_migration_state_cannot_authorize_import(tmp_path):
    """A draft catalog lacking execution status cannot make an archive importable."""
    source, _, entry = _relocation(tmp_path)
    catalog = _catalog(tmp_path, [entry])
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload.pop("state")
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_preserved_source(source, tmp_path)


def test_changed_archive_is_rejected(tmp_path):
    """Recorded relocation does not excuse a later edit to preserved source bytes."""
    source, target, entry = _relocation(tmp_path)
    _catalog(tmp_path, [entry])
    target.write_bytes(b"VALUE = 8\n")
    with pytest.raises(ValueError, match="hash changed"):
        resolve_preserved_source(source, tmp_path)


def test_catalog_digest_cannot_replace_frozen_digest(tmp_path):
    """The migration receipt must agree with the earlier experiment's source hash."""
    source, _, entry = _relocation(tmp_path)
    _catalog(tmp_path, [entry])
    with pytest.raises(ValueError, match="conflicts"):
        resolve_preserved_source(source, tmp_path, "0" * 64)


def test_archive_destination_cannot_escape_project(tmp_path):
    """Even correctly hashed files outside archive are not approved relocation targets."""
    source, target, entry = _relocation(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_bytes(target.read_bytes())
    entry["destination"] = "archive/../outside.py"
    _catalog(tmp_path, [entry])
    with pytest.raises(ValueError, match="escapes"):
        resolve_preserved_source(source, tmp_path)


def test_duplicate_source_mapping_is_not_resolved_by_list_order(tmp_path):
    """Ambiguous catalog entries fail rather than silently choosing the first copy."""
    source, _, entry = _relocation(tmp_path)
    _catalog(tmp_path, [entry, dict(entry)])
    with pytest.raises((FileNotFoundError, ValueError), match="unique|duplicate|Duplicate"):
        resolve_preserved_source(source, tmp_path)


@pytest.mark.parametrize("digest", ["short", "z" * 64, None])
def test_malformed_archive_digest_is_rejected(tmp_path, digest):
    """A relocation must provide a real SHA-256 digest before source execution."""
    source, _, entry = _relocation(tmp_path)
    entry["sha256"] = digest
    _catalog(tmp_path, [entry])
    with pytest.raises(ValueError):
        resolve_preserved_source(source, tmp_path)


def _legacy_package(root, *, initializer=b'"""Synthetic preserved package."""\n'):
    """Build a synthetic three-file package with relative imports and catalog hashes."""
    sources = {
        "__init__.py": initializer,
        "helpers.py": b"VALUE = 23\n",
        "tag_events.py": b"from .helpers import VALUE\ndef selected_value():\n    return VALUE\n",
    }
    entries = []
    for name, content in sources.items():
        _, _, entry = _relocation(root, "ai_tagger/" + name, content)
        entries.append(entry)
    _catalog(root, entries)
    return root / "archive/source/ai_tagger", entries


def test_legacy_package_loads_from_verified_archive_with_relative_imports(tmp_path):
    """The relocated validator remains importable without modifying sys.path."""
    import sys

    package, _ = _legacy_package(tmp_path)
    before = list(sys.path)
    module = load_legacy_validator(tmp_path)
    assert module.selected_value() == 23
    assert Path(module.__file__).resolve() == (package / "tag_events.py").resolve()
    assert sys.path == before


def test_legacy_dependency_mismatch_blocks_initializer_execution(tmp_path):
    """Verify all imported package members before even executing __init__.py."""
    marker = tmp_path / "executed.txt"
    initializer = (
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n"
    ).encode("utf-8")
    package, _ = _legacy_package(tmp_path, initializer=initializer)
    (package / "helpers.py").write_bytes(b"VALUE = 999\n")
    with pytest.raises(ValueError):
        load_legacy_validator(tmp_path)
    assert not marker.exists()


def test_legacy_cache_does_not_bypass_archive_hash_verification(tmp_path):
    """A second load still notices source tampering despite Python's module cache."""
    package, _ = _legacy_package(tmp_path)
    assert load_legacy_validator(tmp_path).selected_value() == 23
    (package / "tag_events.py").write_bytes(b"VALUE = 999\n")
    with pytest.raises(ValueError):
        load_legacy_validator(tmp_path)


@pytest.mark.parametrize(
    "entry", [None, "bad", {}, {"source": 3, "destination": "archive/x.py", "sha256": "0" * 64}]
)
def test_malformed_catalog_entries_fail_before_path_resolution(tmp_path, entry):
    """Malformed migration records produce a validation error, not guessed locations."""
    _catalog(tmp_path, [entry])
    with pytest.raises(ValueError):
        resolve_preserved_source(tmp_path / "missing.py", tmp_path)


def test_missing_legacy_dependency_blocks_initializer_execution(tmp_path):
    """A catalogued dependency must exist before any package code is executed."""
    marker = tmp_path / "executed.txt"
    initializer = (
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n"
    ).encode("utf-8")
    package, _ = _legacy_package(tmp_path, initializer=initializer)
    (package / "helpers.py").unlink()
    with pytest.raises(FileNotFoundError, match="MQF_SOURCE_PROJECT") as failure:
        load_legacy_validator(tmp_path)
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert not marker.exists()


def test_unrecorded_legacy_module_is_rejected(tmp_path):
    """Unverified additions cannot become importable through the archived package path."""
    package, _ = _legacy_package(tmp_path)
    (package / "unexpected.py").write_text("VALUE = 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecorded"):
        load_legacy_validator(tmp_path)


def test_relocation_changes_cached_package_identity(tmp_path):
    """Moving an already imported package must not reuse its stale filesystem search path."""
    package, entries = _legacy_package(tmp_path)
    original = tmp_path / "ai_tagger"
    original.mkdir()
    for source in package.glob("*.py"):
        (original / source.name).write_bytes(source.read_bytes())
    before = load_legacy_validator(tmp_path)
    assert Path(before.__file__).parent == original
    for source in original.glob("*.py"):
        source.unlink()
    after = load_legacy_validator(tmp_path)
    assert after is not before
    assert Path(after.__file__).parent == package
    assert after.selected_value() == before.selected_value() == 23
    assert len(entries) == 3

"""Rebuild the public manifest after an intentional, reviewed source change."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TOP_LEVEL = {".env.example", ".gitignore", ".gitattributes", "README.md", "pyproject.toml"}
FOLDERS = {"src", "tests", "tools", "configs", "scripts", "docs", ".github"}
SUFFIXES = {".py", ".md", ".json", ".ps1", ".toml", ".yml", ".yaml", ".txt"}
IGNORED_PARTS = {"__pycache__", ".quality", ".pytest_cache", ".ruff_cache", ".venv"}
PRIVATE_NAMES = {"project_settings.json", "source_relocations.json"}


def distribution_members(root: Path) -> list[Path]:
    """Select code/documentation while excluding private and generated files.

    This allowlist is not a secret scanner. Review contents before publishing.
    Data, model artifacts, archives and environments are never traversed.
    """
    root = root.resolve()
    files = {root / name for name in TOP_LEVEL if (root / name).is_file()}
    for folder in FOLDERS:
        for path in (root / folder).rglob("*"):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            relative = path.relative_to(root)
            if any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.name in PRIVATE_NAMES:
                continue
            if path.resolve() != path or not path.resolve().is_relative_to(root):
                raise ValueError(f"Filesystem links are not distribution members: {relative}")
            files.add(path)
    return sorted(files)


def build(root: Path) -> dict:
    """Create deterministic relative digests without modifying source bytes."""
    root = root.resolve()
    entries = []
    for path in distribution_members(root):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {"schema_version": 1, "scope": "public_source_distribution", "files": entries}


def main() -> int:
    """Preview by default and write only on explicit --write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Replace the reviewed distribution manifest"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    if args.write:
        (root / "DISTRIBUTION_MANIFEST.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"files": len(result["files"]), "written": args.write}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the public distribution's relative-file manifest, without private data.

This is a package-integrity check. It does not certify historical datasets,
predictions, or private original-source provenance. Those are separate audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


def sha256(path: Path) -> str:
    """Hash a distributed file incrementally without importing or executing it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member_path(root: Path, value: object) -> tuple[str, Path]:
    """Resolve one portable relative member and reject traversal and link escape."""
    if not isinstance(value, str) or not value or chr(92) in value:
        raise ValueError("Manifest paths must be nonempty POSIX relative paths")
    relative = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if relative.is_absolute() or windows.drive or ".." in relative.parts or str(relative) != value:
        raise ValueError("Manifest path is absolute, noncanonical, or escapes the repository")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Manifest member escapes the repository")
    return value, path


def verify(root: Path) -> dict:
    """Validate each declared distribution member, reporting missing or changed bytes.

    Ignored local configuration, environments and generated quality receipts are
    deliberately not required in the released manifest. This permits normal use
    after extraction without requiring the private research workspace.
    """
    root = root.resolve()
    manifest = root / "DISTRIBUTION_MANIFEST.json"
    problems = []
    checked = 0
    try:
        record = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("Unsupported distribution manifest")
        entries = record.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Distribution manifest must contain a nonempty files list")
        seen = set()
        for entry in entries:
            checked += 1
            display = "<invalid entry>"
            try:
                if not isinstance(entry, dict):
                    raise ValueError("Manifest member must be an object")
                display, path = _member_path(root, entry.get("path"))
                identity = display.casefold()
                if identity in seen:
                    raise ValueError("Duplicate or case-colliding manifest member")
                seen.add(identity)
                if path == manifest:
                    raise ValueError("The manifest cannot include itself")
                digest = entry.get("sha256")
                size = entry.get("bytes")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Manifest member needs a lowercase SHA-256 digest")
                if type(size) is not int or size < 0:
                    raise ValueError("Manifest member needs a nonnegative integer byte count")
                if not path.is_file():
                    problems.append({"path": display, "reason": "Missing distributed file"})
                elif path.stat().st_size != size or sha256(path) != digest:
                    problems.append({"path": display, "reason": "Distributed file changed"})
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems.append({"path": display, "reason": str(error)})
    except (OSError, ValueError, TypeError) as error:
        problems.append({"path": manifest.name, "reason": str(error)})
    return {
        "scope": "public_distribution_bytes",
        "manifest": manifest.name,
        "files_checked": checked,
        "source_files_checked": checked,
        "private_historical_sources_verified": False,
        "failures": problems,
        "passed": checked > 0 and not problems,
    }


def main(argv: list[str] | None = None) -> int:
    """Write a package-integrity receipt without accessing private research sources."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / ".quality/provenance.json")
    args = parser.parse_args(argv)
    result = verify(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, indent=2) + "\n"
    args.output.write_text(content, encoding="utf-8")
    print(content)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

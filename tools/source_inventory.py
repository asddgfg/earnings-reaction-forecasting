"""Inventory only source and test fixtures included in the public checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inventory(root: Path) -> dict:
    """List repository-relative code members without scanning external research roots."""
    files = []
    for folder in ("src", "tests", "tools"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix != ".py" and not path.name.endswith(".py.txt"):
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "kind": "historical_test_oracle" if path.name.endswith(".py.txt") else "python",
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "scope": "public_checkout_only",
        "files": files,
        "file_count": len(files),
        "private_sources_scanned": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Write a public-checkout inventory without importing model or provider modules."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / ".quality/source_inventory.json")
    args = parser.parse_args(argv)
    result = inventory(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

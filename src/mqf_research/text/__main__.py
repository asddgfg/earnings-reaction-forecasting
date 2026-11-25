"""Command-line read-only validation of existing repaired signal releases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .release import validate_release


def main(argv: list[str] | None = None) -> int:
    """Validate explicit release inputs and print JSON; never alter a release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path, help="Directory containing RELEASE.json")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Check text and index identities without reading vector bytes or target values",
    )
    parser.add_argument(
        "--expected-release-sha256", help="Optional independently frozen manifest hash"
    )
    parser.add_argument(
        "--all-release-files",
        action="store_true",
        help="Also verify every audit/source member in RELEASE.json (incompatible with metadata-only)",
    )
    args = parser.parse_args(argv)
    if args.metadata_only and args.all_release_files:
        parser.error("--metadata-only and --all-release-files are mutually exclusive")
    try:
        summary = validate_release(
            args.release,
            metadata_only=args.metadata_only,
            expected_release_sha256=args.expected_release_sha256,
            verify_all_files=args.all_release_files,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

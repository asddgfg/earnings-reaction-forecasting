"""Optional audit of an explicitly supplied private historical source manifest.

Excluded from the default tests/unit suite. Run this folder explicitly only after
restoring the private research sources and setting MQF_PRIVATE_PROVENANCE_MANIFEST.
A missing manifest fails with instructions; it is not silently skipped.
"""

import hashlib
import json
import os
from pathlib import Path


def test_private_historical_model_sources():
    """Verify original model-source bytes in a separately retained private workspace."""
    configured = os.environ.get("MQF_PRIVATE_PROVENANCE_MANIFEST", "")
    assert configured, (
        "Set MQF_PRIVATE_PROVENANCE_MANIFEST to the original private modeling "
        "PROVENANCE.json before explicitly running tests/private."
    )
    manifest = json.loads(Path(configured).read_text(encoding="utf-8-sig"))
    entries = manifest["files"] + manifest.get("frozen_compatibility_dependencies", [])
    assert entries, "The private manifest must list original source files"
    for entry in entries:
        source = Path(entry["source"])
        assert source.is_file(), "Restore every source named by the private manifest"
        with source.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]

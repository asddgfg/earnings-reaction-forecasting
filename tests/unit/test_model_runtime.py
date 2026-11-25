"""Runtime/provenance checks that require no downloaded observations or API access."""

import subprocess
import sys

import pytest

from mqf_research.modeling._compatibility import activate_legacy_root
from mqf_research.modeling._validation import ensure_new_checkpoints


def test_model_guards_remain_active_under_optimized_python():
    """Maintained invariants survive -O, while assertion-based legacy workflows refuse it."""
    script = """
from mqf_research.modeling._validation import require
from mqf_research.modeling._compatibility import activate_legacy_root
try:
    require(False, 'still checked')
except ValueError:
    print('persistent guard')
else:
    raise SystemExit(2)
try:
    activate_legacy_root('.')
except RuntimeError as error:
    if 'without -O' not in str(error):
        raise
    print('legacy refused')
else:
    raise SystemExit(3)
from mqf_research.modeling.tuning.inputs import feature_lists
try:
    feature_lists()
except RuntimeError as error:
    if 'without -O' not in str(error):
        raise
    print('public loader refused')
else:
    raise SystemExit(4)
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script], capture_output=True, text=True, timeout=30, check=True
    )
    assert "persistent guard" in completed.stdout
    assert "legacy refused" in completed.stdout
    assert "public loader refused" in completed.stdout


def test_model_compatibility_root_must_exist(tmp_path):
    """A wrong configured workspace cannot silently resolve unrelated installed helpers."""
    with pytest.raises(FileNotFoundError, match="frozen helpers"):
        activate_legacy_root(tmp_path)


def test_model_checkpoint_guard_checks_all_seed_files(tmp_path):
    """An existing second-seed checkpoint also blocks replacement of a partial run."""
    (tmp_path / "seed43.pt").write_bytes(b"second-seed")
    with pytest.raises(FileExistsError, match="seed43.pt"):
        ensure_new_checkpoints(tmp_path, [42, 43])


def test_model_new_run_paths_do_not_point_at_archived_outputs():
    """Maintained workflow constants use the new output root, never historical result folders."""
    from mqf_research.config import get_settings
    from mqf_research.modeling.feature_validation import inputs

    settings = get_settings()
    assert inputs.OUT.is_relative_to(settings.output_root)
    for archived in ("nn_tuning_20261123_b032", "nn_feature_validation_20261123_b032"):
        assert not inputs.OUT.is_relative_to(settings.research_root / "outputs" / archived)

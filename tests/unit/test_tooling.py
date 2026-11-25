"""Failure-path tests for provenance and quality receipts, using temporary files."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_tool(name):
    """Import an inert maintenance tool from this checkout, never a same-name package."""
    path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_mqf_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distribution_manifest(root, files):
    """Create only a temporary public distribution fixture."""
    (root / "DISTRIBUTION_MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "files": files}), encoding="utf-8"
    )


def _member(name, content):
    """Describe expected fixture bytes independently of the verifier."""
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def test_missing_distribution_member_fails_integrity(tmp_path):
    """An existing matching file cannot hide another missing distributed member."""
    source = tmp_path / "source.py"
    source.write_bytes(b"pass\n")
    _distribution_manifest(tmp_path, [_member("source.py", b"pass\n"), _member("missing.py", b"x")])
    result = load_tool("verify_provenance").verify(tmp_path)
    assert not result["passed"]
    assert result["files_checked"] == 2
    assert result["failures"][0]["reason"] == "Missing distributed file"
    assert not result["private_historical_sources_verified"]


def test_distribution_integrity_ignores_local_generated_files(tmp_path):
    """Untracked private settings and generated caches are outside the release receipt."""
    (tmp_path / "source.py").write_bytes(b"original")
    _distribution_manifest(tmp_path, [_member("source.py", b"original")])
    (tmp_path / ".quality").mkdir()
    (tmp_path / ".quality/new.json").write_text("{}")
    (tmp_path / ".env").write_text("FAKE_TEST_SETTING=synthetic")
    result = load_tool("verify_provenance").verify(tmp_path)
    assert result["passed"]
    assert result["scope"] == "public_distribution_bytes"


def test_changed_distribution_member_fails_even_with_same_size(tmp_path):
    """The verifier checks content hashes, not just file length."""
    (tmp_path / "source.py").write_bytes(b"changed!")
    _distribution_manifest(tmp_path, [_member("source.py", b"original")])
    result = load_tool("verify_provenance").verify(tmp_path)
    assert not result["passed"]
    assert result["failures"][0]["reason"] == "Distributed file changed"


@pytest.mark.parametrize(
    "name", ["../outside.py", "/absolute.py", "C:/absolute.py", "dir\\file.py"]
)
def test_distribution_manifest_rejects_nonportable_or_escaping_paths(tmp_path, name):
    """External paths cannot cause the public verifier to read private machine files."""
    _distribution_manifest(tmp_path, [_member(name, b"x")])
    result = load_tool("verify_provenance").verify(tmp_path)
    assert not result["passed"]


def test_distribution_manifest_rejects_duplicate_entries(tmp_path):
    """Repeated members cannot inflate the validated-file count deceptively."""
    (tmp_path / "source.py").write_bytes(b"x")
    member = _member("source.py", b"x")
    _distribution_manifest(tmp_path, [member, member])
    result = load_tool("verify_provenance").verify(tmp_path)
    assert not result["passed"]
    assert "Duplicate" in result["failures"][0]["reason"]


@pytest.mark.parametrize(
    "exception,expected_status,exit_code",
    [(OSError("unavailable"), "failed", 1), (KeyboardInterrupt(), "interrupted", 130)],
)
def test_failed_or_interrupted_run_replaces_stale_success(
    tmp_path, monkeypatch, exception, expected_status, exit_code
):
    """A new failed run has its own receipts and cannot reuse old passing artifacts."""
    quality = tmp_path / ".quality"
    quality.mkdir()
    (quality / "checks.json").write_text('{"passed": true, "run_id": "old"}', encoding="utf-8")
    (quality / "coverage.json").write_text('{"old": true}', encoding="utf-8")
    tool = load_tool("run_checks")

    def fail_launch(*args, **kwargs):
        """Raise a process failure before any actual shell or test is invoked."""
        raise exception

    monkeypatch.setattr(tool.subprocess, "run", fail_launch)
    assert tool.run_checks(tmp_path) == exit_code
    receipt = json.loads((quality / "checks.json").read_text(encoding="utf-8"))
    assert not receipt["passed"]
    assert receipt["status"] == expected_status
    assert receipt["run_id"] != "old"
    assert not (Path(receipt["result_directory"]) / "coverage.json").exists()
    assert json.loads((quality / "coverage.json").read_text(encoding="utf-8")) == {"old": True}


def test_second_check_run_cannot_replace_active_receipt(tmp_path):
    """Two invocations cannot interleave latest-status writes or duplicate checks."""
    from filelock import FileLock

    quality = tmp_path / ".quality"
    quality.mkdir()
    active = '{"status": "running", "passed": false, "run_id": "first"}'
    receipt = quality / "checks.json"
    receipt.write_text(active, encoding="utf-8")
    with FileLock(str(quality / "CHECKS.lock"), timeout=0):
        assert load_tool("run_checks").run_checks(tmp_path) == 2
    assert receipt.read_text(encoding="utf-8") == active


def test_code_change_during_checks_invalidates_success(tmp_path, monkeypatch):
    """Passing subprocesses cannot certify a source tree that changed mid-run."""
    import subprocess

    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("before = 1\n", encoding="utf-8")
    tool = load_tool("run_checks")

    def change_source(command, **kwargs):
        """Represent a concurrent edit while every external check reports success."""
        source.write_text("after = 2\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tool.subprocess, "run", change_source)
    assert tool.run_checks(tmp_path) == 1
    receipt = json.loads((tmp_path / ".quality/checks.json").read_text(encoding="utf-8"))
    assert receipt["source_changed_during_checks"] == ["src/example.py"]
    assert not receipt["passed"]

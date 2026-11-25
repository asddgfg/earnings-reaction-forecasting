"""Run the public offline suite and distributed-file integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock, Timeout


def _save_receipt(path: Path, result: dict) -> None:
    """Atomically publish the current run status, including incomplete failures."""
    temporary = path.with_suffix(f".{result['run_id']}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_snapshot(root: Path) -> dict[str, str]:
    """Bind a quality receipt to its actual source, tests, tools and configuration."""
    paths = [
        path
        for folder in ("src", "tests", "tools", "configs", "scripts")
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    paths.extend(
        path
        for name in (
            "pyproject.toml",
            "project_settings.json",
            "project_settings.example.json",
            "Run-Checks.ps1",
            "Run-MQF.ps1",
            "requirements-validated.txt",
            "DISTRIBUTION_MANIFEST.json",
        )
        if (path := root / name).is_file()
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _run_checks(root: Path) -> int:
    """Run each required check once, save logs, and fail on any unsuccessful stage.

    The unit suite uses synthetic fixtures and mocks. This runner never invokes
    dataset building, research training, APIs, or the historical script tree.
    Coverage is reported honestly; no arbitrary project-wide percentage is used
    to hide untested operational paths such as paid extraction or full training.
    """
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    started_utc = datetime.now(timezone.utc)
    run_id = started_utc.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = root / ".quality/runs" / run_id
    destination.mkdir(parents=True)
    receipt = root / ".quality/checks.json"
    result = {
        "run_id": run_id,
        "started_utc": started_utc.isoformat(),
        "status": "running",
        "passed": False,
        "checks": [],
        "result_directory": str(destination),
        "python": sys.executable,
    }
    # Publish a fresh incomplete status before launching anything. A killed run
    # cannot leave an earlier successful receipt looking like its own result.
    _save_receipt(receipt, result)
    result["source_sha256"] = _source_snapshot(root)
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        COVERAGE_FILE=str(destination / ".coverage"),
    )
    checks = [
        ("lint", ["-m", "ruff", "check", "src", "tests", "tools"]),
        ("format", ["-m", "ruff", "format", "--check", "src", "tests", "tools"]),
        (
            "documentation",
            ["tools/audit_code.py", "--output", str(destination / "documentation.json")],
        ),
        (
            "provenance",
            ["tools/verify_provenance.py", "--output", str(destination / "provenance.json")],
        ),
        (
            "unit_tests",
            [
                "-m",
                "coverage",
                "run",
                "-m",
                "pytest",
                "tests/unit",
                "-q",
                f"--junitxml={destination / 'junit.xml'}",
            ],
        ),
        ("coverage_json", ["-m", "coverage", "json", "-o", str(destination / "coverage.json")]),
        ("coverage_report", ["-m", "coverage", "report"]),
    ]
    receipts = result["checks"]
    interrupted = False
    for name, arguments in checks:
        started = time.perf_counter()
        log = destination / f"{name}.log"
        print(f"Running {name}; log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            try:
                completed = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=root,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=900,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                stream.write("Check exceeded its 900-second execution limit.\n")
                returncode = 124
            except OSError as exc:
                stream.write(f"Could not start check: {exc}\n")
                returncode = 125
            except KeyboardInterrupt:
                stream.write("Check interrupted by the operator.\n")
                returncode = 130
                interrupted = True
        receipts.append(
            {
                "check": name,
                "returncode": returncode,
                "seconds": time.perf_counter() - started,
            }
        )
        _save_receipt(receipt, result)
        if interrupted:
            break
    final_sources = _source_snapshot(root)
    result["source_changed_during_checks"] = sorted(
        name
        for name in result["source_sha256"].keys() | final_sources.keys()
        if result["source_sha256"].get(name) != final_sources.get(name)
    )
    result["passed"] = (
        not result["source_changed_during_checks"]
        and len(receipts) == len(checks)
        and all(item["returncode"] == 0 for item in receipts)
    )
    result["status"] = (
        "interrupted" if interrupted else ("passed" if result["passed"] else "failed")
    )
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _save_receipt(receipt, result)
    _save_receipt(destination / "checks.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "source_sha256"}, indent=2
        )
    )
    return 130 if interrupted else (0 if result["passed"] else 1)


def run_checks(root: Path | None = None) -> int:
    """Serialize check runs so an older process cannot replace a newer receipt.

    Lock acquisition is nonblocking. A second invocation exits with status 2 and
    leaves the first run's receipt untouched; interrupted processes release the
    OS-managed lock instead of leaving a permanent hand-written lock sentinel.
    """
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    quality = root / ".quality"
    quality.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(quality / "CHECKS.lock"), timeout=0):
            return _run_checks(root)
    except Timeout:
        print("Another quality-check run is already active; its receipt was not changed.")
        return 2


def main() -> int:
    """Execute checks against this maintained checkout using its current Python."""
    return run_checks()


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline PowerShell launcher contracts using fake Python scripts and temp paths.

No check runner, model command or provider is actually executed. A small .ps1
stand-in records the arguments/environment it receives and returns a chosen exit
code, allowing the launchers themselves to run without recursive test execution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mqf_research.config import get_settings

_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_PS = (
    Path(os.environ.get("SystemRoot", "C:/Windows"))
    / "System32/WindowsPowerShell/v1.0/powershell.exe"
)
_POWERSHELL = (
    str(_NATIVE_PS) if _NATIVE_PS.is_file() else shutil.which("pwsh") or shutil.which("powershell")
)
_NO_CONFIG = object()
pytestmark = pytest.mark.skipif(_POWERSHELL is None, reason="PowerShell executable unavailable")


def ps_literal(value: str | Path) -> str:
    """Quote a literal PowerShell argument, preserving spaces and apostrophes."""
    return "'" + str(value).replace("'", "''") + "'"


def run_fake_launcher(
    tmp_path,
    launcher,
    *,
    config=_NO_CONFIG,
    override=None,
    developer_tools="none",
    prior_path="prior-import-path",
    prior_bytecode="prior-bytecode-setting",
    exit_code=0,
    explicit_python=True,
    arguments=(),
):
    """Run a copied launcher and capture both child and restored-parent environments."""
    maintained = tmp_path / "project with spaces" / "maintained"
    maintained.mkdir(parents=True)
    (maintained / "src").mkdir()
    (maintained / "configs").mkdir()
    (maintained / "scripts").mkdir()
    shutil.copyfile(_ROOT / "scripts" / launcher, maintained / "scripts" / launcher)
    shutil.copyfile(
        _ROOT / "scripts/Launcher-Common.ps1", maintained / "scripts/Launcher-Common.ps1"
    )
    if config is not _NO_CONFIG:
        (maintained / "configs/project_settings.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    if override is not None:
        research = (maintained / "configs" / override).resolve()
    elif isinstance(config, dict) and isinstance(config.get("research_root"), str):
        research = (maintained / "configs" / config["research_root"]).resolve()
    else:
        research = maintained
    if developer_tools == "local":
        (maintained / ".dev_dependencies").mkdir()
    elif developer_tools == "fallback":
        (research / "maintained/.dev_dependencies").mkdir(parents=True)
    capture = tmp_path / "child.json"
    restored = tmp_path / "restored.json"
    fake = tmp_path / "fake python.ps1"
    fake.write_text(
        "[pscustomobject]@{arguments=@($args); pythonpath=$env:PYTHONPATH; "
        "bytecode=$env:PYTHONDONTWRITEBYTECODE; cwd=(Get-Location).Path} | "
        "ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $env:MQF_LAUNCH_CAPTURE -Encoding UTF8\n"
        "$global:LASTEXITCODE = [int]$env:MQF_LAUNCH_EXIT\n",
        encoding="utf-8-sig",
    )
    invocation = "& " + ps_literal(maintained / "scripts" / launcher)
    if explicit_python:
        invocation += " -Python " + ps_literal(fake)
    invocation += " " + " ".join(ps_literal(argument) for argument in arguments)
    wrapper = tmp_path / "invoke.ps1"
    wrapper.write_text(
        "$ErrorActionPreference='Stop'\n"
        + invocation
        + "\n$observedExit=$LASTEXITCODE\n"
        + "[pscustomobject]@{pythonpath=$env:PYTHONPATH; bytecode=$env:PYTHONDONTWRITEBYTECODE; "
        + "cwd=(Get-Location).Path; child_exit=$observedExit} | ConvertTo-Json | "
        + "Set-Content -LiteralPath $env:MQF_LAUNCH_RESTORED -Encoding UTF8\n"
        + "exit $observedExit\n",
        encoding="utf-8-sig",
    )
    caller = tmp_path / "unrelated caller"
    caller.mkdir()
    environment = os.environ.copy()
    environment.pop("MQF_RESEARCH_ROOT", None)
    if override is not None:
        environment["MQF_RESEARCH_ROOT"] = override
    for key, value in (("PYTHONPATH", prior_path), ("PYTHONDONTWRITEBYTECODE", prior_bytecode)):
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    environment.update(
        MQF_LAUNCH_CAPTURE=str(capture),
        MQF_LAUNCH_RESTORED=str(restored),
        MQF_LAUNCH_EXIT=str(exit_code),
    )
    process = subprocess.run(
        [
            _POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
        ],
        cwd=caller,
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    return {
        "process": process,
        "maintained": maintained,
        "research": research,
        "caller": caller,
        "child": json.loads(capture.read_text(encoding="utf-8-sig")) if capture.exists() else None,
        "restored": json.loads(restored.read_text(encoding="utf-8-sig"))
        if restored.exists()
        else None,
    }


@pytest.mark.parametrize("launcher", ["Run-Checks.ps1", "Run-MQF.ps1"])
@pytest.mark.parametrize("config", [_NO_CONFIG, {}], ids=["absent", "empty-object"])
def test_launchers_allow_explicit_python_without_local_settings(tmp_path, launcher, config):
    """A clean checkout can use -Python without requiring machine-specific JSON."""
    result = run_fake_launcher(tmp_path, launcher, config=config)
    assert result["process"].returncode == 0, result["process"].stderr
    paths = result["child"]["pythonpath"].split(os.pathsep)
    assert Path(paths[0]) == result["maintained"] / "src"
    assert Path(paths[1]) == result["research"]
    assert paths[-1] == "prior-import-path"
    assert not any(".dev_dependencies" in path for path in paths)
    assert result["child"]["bytecode"] == "1"
    assert Path(result["child"]["cwd"]) == result["caller"]
    if launcher == "Run-MQF.ps1":
        assert result["child"]["arguments"] == ["-m", "mqf_research", "doctor"]
    else:
        assert result["child"]["arguments"] == [str(result["maintained"] / "tools/run_checks.py")]


@pytest.mark.parametrize("launcher", ["Run-Checks.ps1", "Run-MQF.ps1"])
def test_environment_override_matches_python_and_existing_tools_fallback(tmp_path, launcher):
    """Relative environment overrides win over JSON and resolve at the config directory."""
    result = run_fake_launcher(
        tmp_path,
        launcher,
        config={"research_root": "wrong-root"},
        override="../environment-research",
        developer_tools="fallback",
    )
    assert result["process"].returncode == 0, result["process"].stderr
    settings = get_settings(
        result["maintained"] / "configs/project_settings.json",
        {"MQF_RESEARCH_ROOT": "../environment-research"},
    )
    paths = result["child"]["pythonpath"].split(os.pathsep)
    assert Path(paths[1]) == settings.research_root / "maintained/.dev_dependencies"
    assert Path(paths[2]) == settings.research_root
    assert not any("wrong-root" in path for path in paths)


@pytest.mark.parametrize("launcher", ["Run-Checks.ps1", "Run-MQF.ps1"])
def test_relative_file_root_and_local_development_tools(tmp_path, launcher):
    """File-relative research paths and local tools take priority over caller location."""
    result = run_fake_launcher(
        tmp_path,
        launcher,
        config={"research_root": "../configured-research"},
        developer_tools="local",
    )
    assert result["process"].returncode == 0, result["process"].stderr
    settings = get_settings(result["maintained"] / "configs/project_settings.json", {})
    paths = result["child"]["pythonpath"].split(os.pathsep)
    assert Path(paths[1]) == result["maintained"] / ".dev_dependencies"
    assert Path(paths[2]) == settings.research_root


@pytest.mark.parametrize("launcher", ["Run-Checks.ps1", "Run-MQF.ps1"])
@pytest.mark.parametrize("prior", [None, "existing-value"])
def test_launcher_restores_environment_and_propagates_failure(tmp_path, launcher, prior):
    """A failing child retains its exit code while restoring set or absent parent variables."""
    result = run_fake_launcher(
        tmp_path,
        launcher,
        prior_path=prior,
        prior_bytecode=prior,
        exit_code=7,
    )
    assert result["process"].returncode == 7, result["process"].stderr
    assert result["restored"]["child_exit"] == 7
    assert result["restored"]["pythonpath"] == prior
    assert result["restored"]["bytecode"] == prior
    assert Path(result["restored"]["cwd"]) == result["caller"]


@pytest.mark.parametrize("launcher", ["Run-Checks.ps1", "Run-MQF.ps1"])
def test_default_interpreter_uses_overridden_research_root(tmp_path, launcher):
    """A missing default venv reports the environment-selected location, not stale JSON."""
    result = run_fake_launcher(
        tmp_path,
        launcher,
        config={"research_root": "stale-root"},
        override="../selected-root",
        explicit_python=False,
    )
    assert result["process"].returncode != 0
    assert result["child"] is None
    assert "selected-root" in result["process"].stderr
    assert "stale-root" not in result["process"].stderr


@pytest.mark.parametrize("launcher", ["Run-Checks.ps1", "Run-MQF.ps1"])
def test_launcher_rejects_unknown_configuration_before_dispatch(tmp_path, launcher):
    """A misspelled path key cannot launch a different environment silently."""
    result = run_fake_launcher(tmp_path, launcher, config={"reserch_root": "typo"})
    assert result["process"].returncode != 0
    assert result["child"] is None
    assert "Unknown project path settings" in result["process"].stderr


def test_mqf_launcher_preserves_spaced_cli_arguments(tmp_path):
    """CLI argument boundaries survive PowerShell forwarding without string-built commands."""
    arguments = ("validate-text", str(tmp_path / "input with spaces"), "--metadata-only")
    result = run_fake_launcher(tmp_path, "Run-MQF.ps1", arguments=arguments)
    assert result["process"].returncode == 0, result["process"].stderr
    assert result["child"]["arguments"] == ["-m", "mqf_research", *arguments]

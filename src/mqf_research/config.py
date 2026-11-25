"""Resolve project paths without changing the filesystem or reading API keys.

Environment overrides take precedence over the optional local configuration.
Paths in that file are relative to its directory unless explicitly absolute.
An output directory is deliberately separate from archived experiment outputs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Locations of preserved research inputs, original source, and new outputs.

    Attributes:
        research_root: Workspace containing the frozen research runs and batch_032.
        source_project: Original downloader source and retained data directory.
        output_root: Destination for new maintained runs; completed runs are refused.
        maintained_root: Project root (legacy field name retained for frozen callers).
    """

    research_root: Path
    source_project: Path
    output_root: Path
    maintained_root: Path

    @property
    def project_root(self) -> Path:
        """Return the source checkout root independent of historical folder names."""
        return self.maintained_root

    def output_path(self, *parts: str) -> Path:
        """Resolve a descendant of the new output root, rejecting path traversal.

        This method does not create a directory. A stage must separately refuse
        to overwrite a completed run before it starts writing artifacts.
        """
        destination = self.output_root.joinpath(*parts).resolve()
        if not destination.is_relative_to(self.output_root.resolve()):
            raise ValueError("Output path escapes the configured output root")
        return destination


def _resolve_path(value: str | Path, anchor: Path, name: str) -> Path:
    """Resolve a nonempty local path relative to the configuration directory."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty filesystem path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else anchor / path).resolve()


def _protect_historical_trees(research: Path, source: Path, output: Path) -> None:
    """Keep a broad output override from making historical runs writable.

    The single reserved exception is outputs/maintained_runs, introduced by this
    revision. Custom output roots elsewhere remain supported. Both descendants
    and ancestors of protected trees are rejected to close parent-root bypasses.
    """
    allowed = research / "outputs/maintained_runs"
    if allowed.resolve() != allowed:
        raise ValueError("Maintained output namespace must not redirect through a filesystem link")
    protected = (
        research / "outputs",
        research / "output",
        research / "work",
        research / "releases",
        research / "local_models",
        source / "research_data",
        source / "research_data_cleaned_runs",
        source / "archive",
        *(source / name for name in ("src", "tests", "tools", "configs", "scripts", "docs")),
        *(
            Path(__file__).resolve().parents[2] / name
            for name in ("src", "tests", "tools", "configs", "scripts", "docs")
        ),
    )
    for tree in protected:
        tree = tree.resolve()
        if tree == (research / "outputs").resolve() and output.is_relative_to(allowed):
            continue
        if output.is_relative_to(tree) or tree.is_relative_to(output):
            raise ValueError(f"Output root overlaps a preserved tree: {tree}")


def get_settings(
    config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load non-secret paths with explicit environment-over-file precedence.

    Args:
        config_path: Optional JSON file. An explicitly requested missing file is
            an error; the default local file may be absent on a clean checkout.
        environ: Optional environment mapping for deterministic callers/tests.
            Recognized overrides are MQF_RESEARCH_ROOT, MQF_SOURCE_PROJECT and
            MQF_OUTPUT_ROOT. No .env or credentials file is opened.

    Returns:
        Immutable absolute paths. Input folders need not exist until a stage is
        invoked, so help, static checks and unit tests work without raw data.

    Raises:
        ValueError: Configuration is malformed, contains unknown fields, or sets
            the output root to a preserved source/input root.
        FileNotFoundError: An explicitly requested configuration file is absent.
    """
    environment = os.environ if environ is None else environ
    maintained_root = Path(__file__).resolve().parents[2]
    location = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else default_config_path(maintained_root)
    )
    values: dict[str, str] = {}
    if location.is_file():
        with location.open(encoding="utf-8-sig") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise ValueError("Project configuration must be a JSON object")
        unknown = set(loaded) - {"research_root", "source_project", "output_root"}
        if unknown:
            raise ValueError(f"Unknown project path settings: {sorted(unknown)}")
        values = loaded
    elif config_path is not None:
        raise FileNotFoundError(location)

    research = _resolve_path(
        environment.get("MQF_RESEARCH_ROOT", values.get("research_root", "..")),
        location.parent,
        "research_root",
    )
    source = _resolve_path(
        environment.get("MQF_SOURCE_PROJECT", values.get("source_project", research)),
        location.parent,
        "source_project",
    )
    output = _resolve_path(
        environment.get(
            "MQF_OUTPUT_ROOT", values.get("output_root", research / "outputs/maintained_runs")
        ),
        location.parent,
        "output_root",
    )
    if (
        output in {research, source}
        or research.is_relative_to(output)
        or source.is_relative_to(output)
    ):
        raise ValueError("Output root must not contain or replace a preserved input root")
    _protect_historical_trees(research, source, output)
    if any(
        (output / marker).exists()
        for marker in ("FROZEN_INPUTS.json", "COMPLETE.json", "PROTOCOL.json")
    ):
        raise ValueError("Output root points to an existing experiment; choose a new run root")
    return Settings(research, source, output, maintained_root)


def default_config_path(project_root: Path) -> Path:
    """Prefer the explicit configs directory while supporting pre-migration checkouts.

    Presence of the directory defines the layout even when local settings are
    absent. This prevents an obsolete root-level file from unexpectedly winning.
    """
    if (project_root / "configs").is_dir():
        return project_root / "configs/project_settings.json"
    return project_root / "project_settings.json"

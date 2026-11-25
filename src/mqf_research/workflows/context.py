"""Immutable configuration shared by workflow planning and execution."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mqf_research.config import Settings


@dataclass(frozen=True)
class WorkflowInputs:
    """Explicit existing inputs; no workflow guesses the latest historical batch."""

    release_dir: Path | None = None
    raw_dir: Path | None = None
    batch_dir: Path | None = None
    selected_file: Path | None = None
    structured_csv: Path | None = None

    def require(self, name: str) -> Path:
        """Resolve a named supplied path without reading or changing its contents."""
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"Workflow requires an explicit {name}")
        return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class WorkflowContext:
    """Bind a workflow to one fresh output namespace and existing input roots.

    A run ID identifies output storage, not a scientific recipe. Existing module
    protocols continue to fix the recipes, chronology and selection rules.
    """

    settings: Settings
    run_id: str
    inputs: WorkflowInputs = field(default_factory=WorkflowInputs)
    python: Path = field(default_factory=lambda: Path(sys.executable))
    workers: int = 2

    def __post_init__(self) -> None:
        """Reject traversal, Windows reserved names and invalid concurrency early."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", self.run_id):
            raise ValueError("Run ID must contain 1-80 letters, digits, underscores or hyphens")
        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
        }
        if self.run_id.upper() in reserved:
            raise ValueError("Run ID is a reserved Windows device name")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        object.__setattr__(self, "python", Path(self.python).expanduser().resolve())

    @property
    def run_dir(self) -> Path:
        """Return this run's dedicated receipt, log and artifact directory."""
        return self.settings.output_path("workflows", self.run_id)

    @property
    def artifact_dir(self) -> Path:
        """Return the root passed to existing modules through MQF_OUTPUT_ROOT."""
        return self.run_dir / "artifacts"

    def binding(self) -> dict:
        """Describe non-secret resolved inputs needed to detect changed resumes."""
        return {
            "run_id": self.run_id,
            "python": str(self.python),
            "workers": self.workers,
            "research_root": str(self.settings.research_root.resolve()),
            "source_project": str(self.settings.source_project.resolve()),
            "maintained_root": str(self.settings.maintained_root.resolve()),
            "run_directory": str(self.run_dir),
            "artifact_directory": str(self.artifact_dir),
        }

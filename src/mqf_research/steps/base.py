"""A declarative, inspectable command step rather than a shell command string."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """One module invocation and its dependencies, permissions and output contract.

    Arguments remain separate subprocess arguments, including paths with spaces.
    Expected outputs are deliberately files, so completion can be content-hashed.
    """

    name: str
    description: str
    module: str
    arguments: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    required_inputs: tuple[Path, ...] = ()
    expected_outputs: tuple[Path, ...] = ()
    trains_model: bool = False
    reads_evaluation: bool = False
    network_access: bool = False

    def __post_init__(self) -> None:
        """Reject unsafe identifiers and accidental shell or arbitrary module dispatch."""
        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.name):
            raise ValueError(f"Invalid step name: {self.name}")
        if not re.fullmatch(r"mqf_research(?:\.[a-z][a-z0-9_]*)+", self.module):
            raise ValueError("Steps must use a maintained mqf_research module")
        if any(not isinstance(argument, str) for argument in self.arguments):
            raise ValueError("Step arguments must be strings")

    def command(self, python: Path) -> list[str]:
        """Construct an argument vector executed with shell=False."""
        return [str(python), "-m", self.module, *self.arguments]

    def describe(self, python: Path) -> dict:
        """Serialize the complete public step contract without loading data."""
        return {
            "name": self.name,
            "description": self.description,
            "command": self.command(python),
            "depends_on": list(self.depends_on),
            "required_inputs": [str(path) for path in self.required_inputs],
            "expected_outputs": [str(path) for path in self.expected_outputs],
            "trains_model": self.trains_model,
            "reads_evaluation": self.reads_evaluation,
            "network_access": self.network_access,
        }

"""Ordered workflow graph validation independent of scientific implementations."""

from __future__ import annotations

from dataclasses import dataclass

from mqf_research.steps.base import Step
from mqf_research.workflows.context import WorkflowContext


@dataclass(frozen=True)
class Workflow:
    """An explicit stage graph in the scientifically required execution order."""

    name: str
    description: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        """Require unique names and dependencies already satisfied in declared order."""
        if not self.steps:
            raise ValueError("A workflow must contain at least one step")
        seen: set[str] = set()
        for step in self.steps:
            if step.name in seen:
                raise ValueError(f"Duplicate workflow step: {step.name}")
            missing = set(step.depends_on) - seen
            if missing:
                raise ValueError(
                    f"Missing, cyclic or later dependencies for {step.name}: {missing}"
                )
            seen.add(step.name)

    def plan(self, context: WorkflowContext) -> dict:
        """Return a read-only plan; merely requesting it never starts a subprocess."""
        for step in self.steps:
            for path in step.expected_outputs:
                if not path.resolve().is_relative_to(context.artifact_dir.resolve()):
                    raise ValueError(f"Step output escapes the isolated artifact directory: {path}")
        return {
            "workflow": self.name,
            "description": self.description,
            "binding": context.binding(),
            "steps": [step.describe(context.python) for step in self.steps],
            "llm_calls": 0,
            "execution_requested": False,
        }

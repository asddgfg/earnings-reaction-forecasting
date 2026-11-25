"""Read-only repaired-release validation using the actual text module CLI."""

from __future__ import annotations

from mqf_research.steps.base import Step
from mqf_research.workflows.context import WorkflowContext


def text_audit_steps(context: WorkflowContext) -> tuple[Step, ...]:
    """Check an explicit existing release, including manifest members and vectors."""
    release = context.inputs.require("release_dir")
    return (
        Step(
            name="validate-release",
            description="Validate repaired signal identity, bags, vectors and all release members",
            module="mqf_research.text",
            arguments=(str(release), "--all-release-files"),
            required_inputs=(release / "RELEASE.json",),
        ),
    )

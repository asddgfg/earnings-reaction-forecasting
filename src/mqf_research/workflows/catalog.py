"""Discoverable workflow names and real domain step builders."""

from __future__ import annotations

from mqf_research.steps.data import legacy_data_steps
from mqf_research.steps.modeling import model_steps
from mqf_research.steps.text import text_audit_steps
from mqf_research.workflows.context import WorkflowContext
from mqf_research.workflows.specification import Workflow

DESCRIPTIONS = {
    "text-audit": "Read-only audit of an explicitly selected repaired signal release",
    "model-tuning": "Prespecified neural-network recipes, selected by development MSE",
    "feature-comparison": "Fixed Market / Financial / LLM comparison across three windows",
    "legacy-data-build": "Historical fixed-tag full-day reconstruction from retained raw inputs",
}


def list_workflows() -> dict[str, str]:
    """List workflow purposes without scientific imports or dataset access."""
    return dict(DESCRIPTIONS)


def build_workflow(name: str, context: WorkflowContext) -> Workflow:
    """Build one known plan without executing a stage or changing the filesystem."""
    if name == "text-audit":
        steps = text_audit_steps(context)
    elif name == "model-tuning":
        steps = model_steps(context, "tuning")
    elif name == "feature-comparison":
        steps = model_steps(context, "feature_validation")
    elif name == "legacy-data-build":
        steps = legacy_data_steps(context)
    else:
        raise ValueError(f"Unknown workflow: {name}; choose from {', '.join(DESCRIPTIONS)}")
    return Workflow(name, DESCRIPTIONS[name], steps)

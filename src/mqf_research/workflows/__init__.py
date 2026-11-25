"""Plan and execute explicit workflows; scientific implementations stay in domains."""

from mqf_research.workflows.context import WorkflowContext, WorkflowInputs
from mqf_research.workflows.runner import WorkflowRunner
from mqf_research.workflows.specification import Workflow


def list_workflows() -> dict[str, str]:
    """Load the lightweight catalog only when discovery is requested."""
    from mqf_research.workflows.catalog import list_workflows as catalog_list

    return catalog_list()


def build_workflow(name: str, context: WorkflowContext) -> Workflow:
    """Bind a catalog entry without importing training or downloading modules."""
    from mqf_research.workflows.catalog import build_workflow as catalog_build

    return catalog_build(name, context)


__all__ = [
    "Workflow",
    "WorkflowContext",
    "WorkflowInputs",
    "WorkflowRunner",
    "build_workflow",
    "list_workflows",
]

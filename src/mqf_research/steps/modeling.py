"""Real chronological model stages, preserving the current frozen science."""

from __future__ import annotations

from mqf_research.steps.base import Step
from mqf_research.workflows.context import WorkflowContext


def model_steps(context: WorkflowContext, family: str) -> tuple[Step, ...]:
    """Declare protocol through independent final audit in the required order.

    Selection also fits stopping models, so it requires training authorization.
    Evaluation is never a selection dependency; the development audit comes first.
    """
    if family not in {"tuning", "feature_validation"}:
        raise ValueError(f"Unknown model family: {family}")
    module = f"mqf_research.modeling.{family}"
    output = context.artifact_dir / ("nn_tuning" if family == "tuning" else "nn_feature_validation")
    stages = [
        ("protocol", "study", ("protocol",), "PROTOCOL.json", False, False),
        ("prepare", "study", ("prepare",), "PREPARED.json", family == "feature_validation", False),
    ]
    stages.append(
        (
            "verify-inputs",
            "verify",
            ("inputs",),
            "INDEPENDENT_DEV_INPUT_CHECK.json"
            if family == "tuning"
            else "INDEPENDENT_INPUT_AUDIT.json",
            False,
            False,
        )
    )
    if family == "feature_validation":
        stages.append(
            (
                "verify-engine",
                "verify",
                ("engine",),
                "ENGINE_INDEPENDENT_AUDIT.json",
                False,
                False,
            )
        )
    if family == "tuning":
        stages.append(
            ("smoke", "study", ("smoke",), "original_reproduction/COMPLETE.json", True, False)
        )
    stages.extend(
        [
            (
                "development",
                "study",
                ("development", "--workers", str(context.workers)),
                "DEV_FIT_COMPLETE.json",
                True,
                False,
            ),
            ("select", "study", ("select",), "SELECTION_COMPLETE.json", True, False),
            (
                "verify-development",
                "verify",
                ("development",),
                "INDEPENDENT_DEVELOPMENT_AUDIT.json",
                False,
                False,
            ),
            ("evaluation", "study", ("evaluation",), "EVAL_COMPLETE.json", True, True),
            ("report", "report", (), "REPORT_COMPLETE.json", False, True),
            ("verify-all", "verify", ("all",), "INDEPENDENT_AUDIT.json", False, True),
        ]
    )
    steps = []
    for name, suffix, arguments, marker, training, evaluation in stages:
        steps.append(
            Step(
                name=name,
                description=f"{family}: {name}; retain existing module protocol and checks",
                module=f"{module}.{suffix}",
                arguments=arguments,
                depends_on=(steps[-1].name,) if steps else (),
                expected_outputs=(output / marker,),
                trains_model=training,
                reads_evaluation=evaluation,
            )
        )
    return tuple(steps)

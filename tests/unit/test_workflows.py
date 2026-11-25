"""Workflow contracts use fake executors and tiny artifacts, never research data."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from mqf_research.config import Settings
from mqf_research.steps.base import Step
from mqf_research.workflows import (
    Workflow,
    WorkflowContext,
    WorkflowInputs,
    WorkflowRunner,
    build_workflow,
    list_workflows,
)
from mqf_research.workflows import runner as runner_module


@pytest.fixture
def context(tmp_path, monkeypatch):
    """Isolate outputs and freeze synthetic source identity against unrelated edits."""
    monkeypatch.setattr(runner_module, "_source_snapshot", lambda: {"synthetic.py": "unchanged"})
    settings = Settings(tmp_path / "research", tmp_path / "source", tmp_path / "out", tmp_path)
    return WorkflowContext(settings, "test-run", python=Path(sys.executable))


def toy_workflow(context, **capabilities):
    """Create two dependent commands with independently verifiable output contracts."""
    first = Step(
        "first",
        "Synthetic first stage",
        "mqf_research.text",
        ("first",),
        expected_outputs=(context.artifact_dir / "first.json",),
        **capabilities,
    )
    second = Step(
        "second",
        "Synthetic second stage",
        "mqf_research.text",
        ("second",),
        depends_on=("first",),
        expected_outputs=(context.artifact_dir / "second.json",),
    )
    return Workflow("toy", "No scientific computation", (first, second))


def successful_executor(context, calls):
    """Return a recorder that emits only the synthetic output declared by each step."""

    def execute(command, cwd, environment, log_path, timeout_seconds):
        """Record argv and environment while generating a tiny completion artifact."""
        calls.append(command[-1])
        assert environment["MQF_OUTPUT_ROOT"] == str(context.artifact_dir)
        assert environment["MQF_RESEARCH_ROOT"] == str(context.settings.research_root)
        assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
        assert cwd == context.settings.maintained_root
        log_path.write_text("synthetic stage completed\n", encoding="utf-8")
        (context.artifact_dir / f"{command[-1]}.json").write_text(
            '{"passed": true}', encoding="utf-8"
        )
        return 0

    return execute


def test_plan_is_default_and_creates_nothing(context):
    """Inspecting commands needs no permissions, source hashes or subprocess."""
    calls = []
    runner = WorkflowRunner(context, executor=successful_executor(context, calls))
    plan = runner.run(toy_workflow(context, trains_model=True))
    assert plan["status"] == "planned"
    assert plan["execution_requested"] is False
    assert plan["steps"][0]["command"][-1] == "first"
    assert not context.settings.output_root.exists()
    assert calls == []


def test_success_retains_logs_outputs_and_content_receipt(context):
    """A completed receipt requires actual outputs and stores their hashes."""
    calls = []
    result = WorkflowRunner(context, executor=successful_executor(context, calls)).run(
        toy_workflow(context), execute=True
    )
    assert result["status"] == "complete"
    assert calls == ["first", "second"]
    assert len(result["completed"]) == 2
    assert len(result["completed"][0]["log_sha256"]) == 64
    assert json.loads((context.run_dir / "WORKFLOW.json").read_text()) == result
    with pytest.raises(FileExistsError):
        WorkflowRunner(context).run(toy_workflow(context), execute=True)


def test_clean_pause_resumes_without_replaying_completed_step(context):
    """A declared pause can resume only the remaining dependent stages."""
    calls = []
    runner = WorkflowRunner(context, executor=successful_executor(context, calls))
    workflow = toy_workflow(context)
    paused = runner.run(workflow, execute=True, through="first")
    assert paused["status"] == "paused"
    finished = runner.run(workflow, execute=True, resume=True)
    assert finished["status"] == "complete"
    assert calls == ["first", "second"]
    with pytest.raises(ValueError, match="Only clean paused"):
        runner.run(workflow, execute=True, resume=True)


@pytest.mark.parametrize("changed", ["output", "log", "source", "configuration"])
def test_pause_rejects_changed_binding_or_completed_evidence(context, monkeypatch, changed):
    """Resume cannot silently mix changed source, roots, artifacts or execution logs."""
    calls = []
    runner = WorkflowRunner(context, executor=successful_executor(context, calls))
    runner.run(toy_workflow(context), execute=True, through="first")
    if changed == "output":
        (context.artifact_dir / "first.json").write_text("{}", encoding="utf-8")
    elif changed == "log":
        (context.run_dir / "logs/00_first.log").write_text("changed", encoding="utf-8")
    elif changed == "source":
        monkeypatch.setattr(runner_module, "_source_snapshot", lambda: {"synthetic.py": "changed"})
    else:
        runner = WorkflowRunner(replace(context, workers=3))
    with pytest.raises(ValueError):
        runner.run(toy_workflow(context), execute=True, resume=True)
    assert calls == ["first"]


def test_nonzero_exit_preserves_failure_and_stops_dependencies(context):
    """The second stage never runs after an operational failure."""
    calls = []

    def fail(command, cwd, environment, log_path, timeout_seconds):
        """Produce an inspectable failing log without touching research files."""
        calls.append(command[-1])
        log_path.write_text("failure detail", encoding="utf-8")
        return 7

    runner = WorkflowRunner(context, executor=fail)
    with pytest.raises(RuntimeError, match="exit code 7"):
        runner.run(toy_workflow(context), execute=True)
    state = json.loads((context.run_dir / "WORKFLOW.json").read_text())
    assert state["status"] == "failed"
    assert state["active_step"] == "first"
    assert calls == ["first"]
    with pytest.raises(ValueError, match="Only clean paused"):
        runner.run(toy_workflow(context), execute=True, resume=True)


@pytest.mark.parametrize("mode", ["missing", "false"])
def test_exit_zero_does_not_replace_output_verification(context, mode):
    """Missing outputs and explicit failed audit JSON cannot count as completion."""

    def incomplete(command, cwd, environment, log_path, timeout_seconds):
        """Simulate a buggy CLI that returns success without a valid artifact."""
        log_path.write_text("returned zero", encoding="utf-8")
        if mode == "false":
            (context.artifact_dir / "first.json").write_text('{"passed": false}', encoding="utf-8")
        return 0

    with pytest.raises(RuntimeError):
        WorkflowRunner(context, executor=incomplete).run(toy_workflow(context), execute=True)
    assert json.loads((context.run_dir / "WORKFLOW.json").read_text())["status"] == "failed"


def test_interrupt_records_incomplete_work(context):
    """Keyboard cancellation never leaves a stale successful state."""

    def interrupt(command, cwd, environment, log_path, timeout_seconds):
        """Raise before any fit, as a canceled child process would."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        WorkflowRunner(context, executor=interrupt).run(toy_workflow(context), execute=True)
    assert json.loads((context.run_dir / "WORKFLOW.json").read_text())["status"] == "interrupted"


@pytest.mark.parametrize("flag", ["trains_model", "reads_evaluation", "network_access"])
def test_execution_capabilities_required_before_writes(context, flag):
    """Expensive or held-out operations require explicit caller intent."""
    with pytest.raises(ValueError, match="require explicit"):
        WorkflowRunner(context).run(toy_workflow(context, **{flag: True}), execute=True)
    assert not context.run_dir.exists()


@pytest.mark.parametrize("run_id", ["..", "../escape", "a/b", "a b", "CON", "lpt1", ""])
def test_run_ids_cannot_escape_namespace_or_use_windows_devices(context, run_id):
    """Cross-platform run identifiers stay in one filesystem-safe namespace."""
    with pytest.raises(ValueError):
        replace(context, run_id=run_id)


def test_dependency_validation_rejects_missing_reordered_and_duplicate_steps(context):
    """Declared order must actually satisfy the stage graph."""
    first, second = toy_workflow(context).steps
    with pytest.raises(ValueError, match="dependencies"):
        Workflow("bad", "Wrong order", (second, first))
    with pytest.raises(ValueError, match="Duplicate"):
        Workflow("bad", "Duplicate", (first, first))


def test_output_contract_cannot_point_to_preserved_input(context):
    """A workflow cannot declare preserved source data as its writable output."""
    step = Step(
        "bad",
        "Escape",
        "mqf_research.text",
        expected_outputs=(context.settings.source_project / "old.json",),
    )
    with pytest.raises(ValueError, match="escapes"):
        Workflow("bad", "Escape", (step,)).plan(context)


def test_unknown_stop_and_unknown_workflow_fail_without_writes(context):
    """Typographical command errors never start a partial run."""
    with pytest.raises(ValueError, match="Unknown stopping"):
        WorkflowRunner(context).run(toy_workflow(context), through="typo")
    with pytest.raises(ValueError, match="Unknown workflow"):
        build_workflow("typo", context)
    assert not context.run_dir.exists()


def test_concurrent_resume_is_locked(context):
    """Another process cannot resume the same run while an owner holds its lock."""
    runner = WorkflowRunner(context, executor=successful_executor(context, []))
    workflow = toy_workflow(context)
    runner.run(workflow, execute=True, through="first")
    with FileLock(str(context.run_dir / "WORKFLOW.lock"), timeout=0):
        with pytest.raises(Timeout):
            runner.run(workflow, execute=True, resume=True)


def test_resume_detects_changed_explicit_input(context):
    """A changed release manifest or source file invalidates a clean pause."""
    source = context.settings.maintained_root / "input.json"
    source.write_text("original", encoding="utf-8")
    workflow = toy_workflow(context)
    workflow = replace(
        workflow, steps=(replace(workflow.steps[0], required_inputs=(source,)), workflow.steps[1])
    )
    runner = WorkflowRunner(context, executor=successful_executor(context, []))
    runner.run(workflow, execute=True, through="first")
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit input changed"):
        runner.run(workflow, execute=True, resume=True)


@pytest.mark.parametrize(
    "name,family", [("model-tuning", "tuning"), ("feature-comparison", "feature_validation")]
)
def test_model_plans_use_real_ordered_modules_and_development_before_evaluation(
    context, name, family
):
    """Current workflows preserve stage boundaries and independently gated evaluation."""
    workflow = build_workflow(name, context)
    names = [step.name for step in workflow.steps]
    assert names.index("select") < names.index("verify-development") < names.index("evaluation")
    assert workflow.steps[0].module == f"mqf_research.modeling.{family}.study"
    assert workflow.steps[0].arguments == ("protocol",)
    evaluation = workflow.steps[names.index("evaluation")]
    assert evaluation.reads_evaluation and evaluation.trains_model
    assert all(not step.reads_evaluation for step in workflow.steps[: names.index("evaluation")])
    if family == "tuning":
        smoke = workflow.steps[names.index("smoke")]
        assert str(smoke.expected_outputs[0]).endswith(
            str(Path("original_reproduction/COMPLETE.json"))
        )


def test_data_plan_uses_existing_inputs_and_separate_legacy_outputs(context):
    """Legacy construction is explicit and cannot overwrite the retained warehouse."""
    context = replace(
        context,
        inputs=WorkflowInputs(
            raw_dir=context.settings.source_project / "research_data/raw",
            batch_dir=context.settings.source_project / "completed_batch",
            selected_file=context.settings.source_project / "selected.txt",
        ),
    )
    plan = build_workflow("legacy-data-build", context).plan(context)
    assert [step["name"] for step in plan["steps"]] == [
        "warehouse",
        "market-features",
        "abnormal-returns",
        "legacy-dataset",
    ]
    for step in plan["steps"]:
        assert step["network_access"] is False
        for output in step["expected_outputs"]:
            assert Path(output).is_relative_to(context.artifact_dir)
    assert "batch_032" in plan["steps"][-1]["description"]


def test_text_release_must_be_selected_explicitly(context):
    """Catalog discovery never chooses an arbitrary latest signal release."""
    assert set(list_workflows()) == {
        "text-audit",
        "model-tuning",
        "feature-comparison",
        "legacy-data-build",
    }
    with pytest.raises(ValueError, match="release_dir"):
        build_workflow("text-audit", context)
    release = context.settings.research_root / "batch_032"
    context = replace(context, inputs=WorkflowInputs(release_dir=release))
    step = build_workflow("text-audit", context).steps[0]
    assert step.arguments == (str(release), "--all-release-files")
    assert not step.trains_model and not step.reads_evaluation and not step.network_access


def test_missing_inputs_fail_before_reserving_run(context):
    """Deleted raw files can block execution without stranding a partially started run."""
    workflow = toy_workflow(context)
    workflow = replace(
        workflow,
        steps=(
            replace(
                workflow.steps[0], required_inputs=(context.settings.source_project / "absent",)
            ),
            workflow.steps[1],
        ),
    )
    with pytest.raises(FileNotFoundError, match="Missing input"):
        WorkflowRunner(context).run(workflow, execute=True)
    assert not context.run_dir.exists()


def test_feature_workflow_produces_engine_audit_required_by_development(context):
    """The final development audit has an actual upstream producer for its engine receipt."""
    workflow = build_workflow("feature-comparison", context)
    names = [step.name for step in workflow.steps]
    assert names.index("prepare") < names.index("verify-inputs") < names.index("development")
    assert names.index("verify-engine") < names.index("verify-development")
    step = workflow.steps[names.index("verify-engine")]
    assert step.arguments == ("engine",)
    assert step.expected_outputs[0].name == "ENGINE_INDEPENDENT_AUDIT.json"
    assert not step.trains_model and not step.reads_evaluation


@pytest.mark.parametrize("snapshot", ["_source_snapshot", "_input_snapshot"])
def test_initialization_hash_error_records_failed_reserved_run(context, monkeypatch, snapshot):
    """A sharing error after existence checks leaves a diagnostic failed receipt."""
    calls = []

    def inaccessible(*args):
        """Simulate a file that becomes unreadable during content hashing."""
        raise PermissionError("synthetic source sharing violation")

    monkeypatch.setattr(runner_module, snapshot, inaccessible)
    runner = WorkflowRunner(context, executor=successful_executor(context, calls))
    with pytest.raises(PermissionError, match="sharing violation"):
        runner.run(toy_workflow(context), execute=True)
    state = json.loads((context.run_dir / "WORKFLOW.json").read_text())
    assert state["status"] == "failed"
    assert state["phase"] == "initialization"
    assert state["error_type"] == "PermissionError"
    assert state["completed"] == []
    assert state["active_step"] is None
    assert calls == []
    assert not context.artifact_dir.exists()
    with pytest.raises(ValueError, match="Only clean paused"):
        runner.run(toy_workflow(context), execute=True, resume=True)


def test_resume_hash_error_preserves_existing_clean_pause(context, monkeypatch):
    """A temporarily unreadable input must not replace an otherwise valid paused receipt."""
    calls = []
    runner = WorkflowRunner(context, executor=successful_executor(context, calls))
    workflow = toy_workflow(context)
    runner.run(workflow, execute=True, through="first")
    original = (context.run_dir / "WORKFLOW.json").read_bytes()

    def inaccessible(*args):
        """Simulate a transient read failure before resumed execution begins."""
        raise PermissionError("synthetic resume sharing violation")

    monkeypatch.setattr(runner_module, "_input_snapshot", inaccessible)
    with pytest.raises(PermissionError):
        runner.run(workflow, execute=True, resume=True)
    assert (context.run_dir / "WORKFLOW.json").read_bytes() == original
    assert calls == ["first"]

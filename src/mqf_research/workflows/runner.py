"""Transactional orchestration with explicit execution, isolated outputs and receipts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock

from mqf_research.common.hashing import file_sha256
from mqf_research.common.serialization import read_json, write_json
from mqf_research.common.time import utc_timestamp
from mqf_research.steps.base import Step
from mqf_research.workflows.context import WorkflowContext
from mqf_research.workflows.specification import Workflow

CommandExecutor = Callable[[list[str], Path, dict[str, str], Path, float | None], int]


def execute_command(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: float | None,
) -> int:
    """Run one argument vector without a shell, retaining combined output in a new log."""
    with log_path.open("xb") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    return result.returncode


def _source_snapshot() -> dict[str, str]:
    """Hash the installed maintained Python sources, never historical datasets."""
    package = Path(__file__).resolve().parents[1]
    return {str(path): file_sha256(path) for path in sorted(package.rglob("*.py"))}


def _input_snapshot(steps: tuple[Step, ...]) -> dict[str, str]:
    """Freeze explicit inputs, including directory members, for safe paused resumes.

    Directory hashing can be substantial for a raw rebuild. It occurs only on
    explicit execution or resume, never while listing or planning a workflow.
    Scientific model modules additionally maintain their own full input receipts.
    """
    paths: set[Path] = set()
    for step in steps:
        for required in step.required_inputs:
            if not required.exists():
                raise FileNotFoundError(f"Missing input for {step.name}: {required}")
            if required.is_dir():
                paths.update(path for path in required.rglob("*") if path.is_file())
            else:
                paths.add(required)
    return {str(path.resolve()): file_sha256(path) for path in sorted(paths)}


def _plan_digest(plan: dict) -> str:
    """Hash deterministic commands, dependencies, inputs and resolved root configuration."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()


def _through_steps(workflow: Workflow, through: str | None) -> tuple[Step, ...]:
    """Resolve an inclusive stopping step while preserving declared dependency order."""
    if through is None:
        return workflow.steps
    for index, step in enumerate(workflow.steps):
        if step.name == through:
            return workflow.steps[: index + 1]
    raise ValueError(f"Unknown stopping step: {through}")


class WorkflowRunner:
    """Manage run state while delegating all science to existing module commands.

    Planning is the default. Clean pauses can resume after source, plan, explicit
    input, artifact and log hashes match. Failed or interrupted runs are retained
    for diagnosis and require a new run ID; partial fits are never blindly retried.
    """

    def __init__(
        self,
        context: WorkflowContext,
        *,
        executor: CommandExecutor | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Inject execution for tests; production uses a subprocess with no shell."""
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.context = context
        self.executor = executor or execute_command
        self.timeout_seconds = timeout_seconds

    def _environment(self) -> dict[str, str]:
        """Bind child paths explicitly; retain credentials privately without logging them."""
        context = self.context
        environment = dict(os.environ)
        environment.update(
            MQF_RESEARCH_ROOT=str(context.settings.research_root),
            MQF_SOURCE_PROJECT=str(context.settings.source_project),
            MQF_OUTPUT_ROOT=str(context.artifact_dir),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        # The maintained source wins over frozen helpers with similarly named modules.
        sources = [str(Path(__file__).resolve().parents[2]), str(context.settings.research_root)]
        if environment.get("PYTHONPATH"):
            sources.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(sources)
        return environment

    def _log_path(self, index: int, step: Step) -> Path:
        """Derive a stable log filename rather than trusting paths read from a receipt."""
        return self.context.run_dir / "logs" / f"{index:02d}_{step.name}.log"

    def _verify_completed(self, workflow: Workflow, completed: list[dict]) -> None:
        """Verify a contiguous completed prefix and every retained output and log."""
        if len(completed) > len(workflow.steps):
            raise ValueError("Receipt contains more completed stages than the workflow")
        for index, record in enumerate(completed):
            step = workflow.steps[index]
            if record.get("name") != step.name or record.get("status") != "complete":
                raise ValueError("Completed stages are not an ordered workflow prefix")
            expected = {str(path): file_sha256(path) for path in step.expected_outputs}
            if record.get("output_sha256") != expected:
                raise ValueError(f"Completed artifacts changed: {step.name}")
            if record.get("log_sha256") != file_sha256(self._log_path(index, step)):
                raise ValueError(f"Completed step log changed: {step.name}")

    def _check_permissions(
        self,
        steps: tuple[Step, ...],
        *,
        allow_training: bool,
        allow_evaluation: bool,
        allow_network: bool,
    ) -> None:
        """Require explicit capabilities before any directories or subprocesses are created."""
        checks = (
            ("trains_model", allow_training, "allow_training"),
            ("reads_evaluation", allow_evaluation, "allow_evaluation"),
            ("network_access", allow_network, "allow_network"),
        )
        for attribute, allowed, option in checks:
            if not allowed and any(getattr(step, attribute) for step in steps):
                raise ValueError(f"Selected steps require explicit {option}=True")

    def run(
        self,
        workflow: Workflow,
        *,
        execute: bool = False,
        through: str | None = None,
        resume: bool = False,
        allow_training: bool = False,
        allow_evaluation: bool = False,
        allow_network: bool = False,
    ) -> dict:
        """Plan by default, or explicitly execute to a verified clean pause/completion."""
        plan = workflow.plan(self.context)
        selected = _through_steps(workflow, through)
        if not execute:
            return {**plan, "through": through, "status": "planned"}
        self._check_permissions(
            selected,
            allow_training=allow_training,
            allow_evaluation=allow_evaluation,
            allow_network=allow_network,
        )
        if not self.context.python.is_file():
            raise FileNotFoundError(self.context.python)
        # Check existence before reserving a run ID; a missing deleted raw input
        # should not strand an empty directory or start the first stage.
        for step in workflow.steps:
            for required in step.required_inputs:
                if not required.exists():
                    raise FileNotFoundError(f"Missing input for {step.name}: {required}")
        run_dir = self.context.run_dir
        if resume:
            if not (run_dir / "WORKFLOW.json").is_file():
                raise ValueError("Resume requires an existing clean paused workflow")
        else:
            # An existing directory is not permission to overwrite a prior run.
            run_dir.mkdir(parents=True, exist_ok=False)
        with FileLock(str(run_dir / "WORKFLOW.lock"), timeout=0):
            return self._run_locked(workflow, plan, selected, resume)

    def _run_locked(
        self,
        workflow: Workflow,
        plan: dict,
        selected: tuple[Step, ...],
        resume: bool,
    ) -> dict:
        """Validate the run binding under lock and persist each state transition atomically."""
        run_dir = self.context.run_dir
        receipt_path = run_dir / "WORKFLOW.json"
        digest = _plan_digest(plan)
        if resume:
            state = read_json(receipt_path)
            if state.get("status") != "paused":
                raise ValueError(
                    "Only clean paused workflows resume; failed/complete runs stay intact"
                )
        else:
            state = {
                "schema_version": 1,
                "plan": plan,
                "plan_sha256": digest,
                "started_utc": utc_timestamp(),
                "completed": [],
                "active_step": None,
                "phase": "initialization",
            }
        try:
            source_hashes = _source_snapshot()
            input_hashes = _input_snapshot(workflow.steps)
        except BaseException as error:
            # Existence checks cannot prevent later permission/sharing failures.
            # Preserve a diagnostic receipt for the newly reserved run ID, but
            # never corrupt an earlier clean pause when a resume read fails.
            if not resume:
                self._record_failure(state, error)
            raise
        if resume:
            if (
                state.get("plan_sha256") != digest
                or state.get("source_sha256") != source_hashes
                or state.get("input_sha256") != input_hashes
            ):
                raise ValueError(
                    "Plan, configuration, source or explicit input changed; use a new ID"
                )
            self._verify_completed(workflow, state["completed"])
            if len(selected) <= len(state["completed"]):
                raise ValueError("Resume must advance beyond the last completed step")
        else:
            state.update(source_sha256=source_hashes, input_sha256=input_hashes)
        try:
            (run_dir / "logs").mkdir(exist_ok=True)
            self.context.artifact_dir.mkdir(exist_ok=True)
            state.update(status="running", phase="execution", updated_utc=utc_timestamp())
            write_json(receipt_path, state)
            for index in range(len(state["completed"]), len(selected)):
                step = selected[index]
                state.update(active_step=step.name, updated_utc=utc_timestamp())
                write_json(receipt_path, state)
                log_path = self._log_path(index, step)
                returncode = self.executor(
                    step.command(self.context.python),
                    self.context.settings.maintained_root,
                    self._environment(),
                    log_path,
                    self.timeout_seconds,
                )
                if returncode != 0:
                    raise RuntimeError(
                        f"Step {step.name} failed with exit code {returncode}; see {log_path}"
                    )
                output_hashes = {}
                for path in step.expected_outputs:
                    if not path.is_file():
                        raise RuntimeError(f"Step {step.name} returned zero but omitted {path}")
                    if path.suffix == ".json":
                        output = read_json(path)
                        if isinstance(output, dict) and output.get("passed") is False:
                            raise RuntimeError(
                                f"Step {step.name} produced a failed verification receipt"
                            )
                    output_hashes[str(path)] = file_sha256(path)
                if _source_snapshot() != source_hashes:
                    raise RuntimeError(
                        "Maintained source changed during execution; stop before next stage"
                    )
                state["completed"].append(
                    {
                        "name": step.name,
                        "status": "complete",
                        "finished_utc": utc_timestamp(),
                        "output_sha256": output_hashes,
                        "log_sha256": file_sha256(log_path),
                    }
                )
                write_json(receipt_path, state)
            state.update(
                status="complete" if len(selected) == len(workflow.steps) else "paused",
                active_step=None,
                updated_utc=utc_timestamp(),
            )
            write_json(receipt_path, state)
            return state
        except BaseException as error:
            self._record_failure(state, error)
            raise

    def _record_failure(self, state: dict, error: BaseException) -> None:
        """Persist initialization, execution or interruption failures consistently."""
        state.update(
            status="interrupted"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "failed",
            updated_utc=utc_timestamp(),
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(self.context.run_dir / "WORKFLOW.json", state)

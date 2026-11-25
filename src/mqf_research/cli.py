"""Inspect the environment, validate retained inputs, and run explicit workflows.

Discovery and planning never start training or download data. Workflow execution
uses a distinct command with per-capability flags and a new run namespace.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
from pathlib import Path

from mqf_research.config import get_settings

LOGGER = logging.getLogger(__name__)


def environment_report() -> dict:
    """Report non-secret paths and installed versions without importing models."""
    settings = get_settings()
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "pytest", "ruff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "paths": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in {
                "project_root": settings.project_root,
                "research_root": settings.research_root,
                "source_project": settings.source_project,
                "new_output_root": settings.output_root,
            }.items()
        },
        "packages": packages,
        "historical_sources": "Preserved, provenance-bound compatibility dependencies",
        "network_calls": 0,
        "research_data_loaded": False,
    }


def _add_workflow_arguments(parser: argparse.ArgumentParser, execute: bool) -> None:
    """Expose paths and execution capabilities without duplicating step logic."""
    parser.add_argument(
        "name", choices=("text-audit", "model-tuning", "feature-comparison", "legacy-data-build")
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    for name in ("release-dir", "raw-dir", "batch-dir", "selected-file", "structured-csv"):
        parser.add_argument(f"--{name}", type=Path)
    if execute:
        parser.add_argument("--through", help="Pause after this named step")
        parser.add_argument("--resume", action="store_true", help="Resume a verified clean pause")
        parser.add_argument("--allow-training", action="store_true")
        parser.add_argument("--allow-evaluation", action="store_true")
        parser.add_argument("--allow-network", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Define explicit commands and validate argument structure before dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect runtime and configured roots; no downloads")
    validate = commands.add_parser("validate-text", help="Validate an existing repaired release")
    validate.add_argument("release_dir", type=Path)
    validate.add_argument("--metadata-only", action="store_true")
    validate.add_argument("--expected-release-sha256")
    validate.add_argument("--all-release-files", action="store_true")
    workflow = commands.add_parser("workflow", help="Discover, preview or explicitly run steps")
    actions = workflow.add_subparsers(dest="workflow_action", required=True)
    actions.add_parser("list", help="List the supported workflows without reading datasets")
    _add_workflow_arguments(
        actions.add_parser("plan", help="Preview only; no output writes"), False
    )
    _add_workflow_arguments(actions.add_parser("run", help="Execute in a new isolated run"), True)
    return parser


def dispatch_workflow(args: argparse.Namespace) -> dict:
    """Adapt CLI options to the typed workflow API; keep orchestration out of CLI."""
    from mqf_research.workflows import (
        WorkflowContext,
        WorkflowInputs,
        WorkflowRunner,
        build_workflow,
        list_workflows,
    )

    if args.workflow_action == "list":
        return {"workflows": list_workflows(), "executed": False}
    context = WorkflowContext(
        get_settings(),
        args.run_id,
        WorkflowInputs(
            release_dir=args.release_dir,
            raw_dir=args.raw_dir,
            batch_dir=args.batch_dir,
            selected_file=args.selected_file,
            structured_csv=args.structured_csv,
        ),
        python=Path(sys.executable),
        workers=args.workers,
    )
    workflow = build_workflow(args.name, context)
    if args.workflow_action == "plan":
        return workflow.plan(context)
    return WorkflowRunner(context).run(
        workflow,
        execute=True,
        through=args.through,
        resume=args.resume,
        allow_training=args.allow_training,
        allow_evaluation=args.allow_evaluation,
        allow_network=args.allow_network,
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch one requested command and return its shell-compatible exit status.

    Validation and operational errors are logged without dumping credentials.
    Expensive work is reachable only through explicit workflow execution.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        if args.command == "doctor":
            result = environment_report()
        elif args.command == "workflow":
            result = dispatch_workflow(args)
        else:
            from mqf_research.text import validate_release

            result = validate_release(
                args.release_dir,
                metadata_only=args.metadata_only,
                expected_release_sha256=args.expected_release_sha256,
                verify_all_files=args.all_release_files,
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0

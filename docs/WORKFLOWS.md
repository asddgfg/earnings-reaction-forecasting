# Workflow orchestration

The package separates three responsibilities:

- `steps/` builds small immutable command specifications for each domain. A step states its inputs, expected output files, dependencies, and whether it trains or reads evaluation outcomes.
- `workflows/` defines the execution order and manages run state. `WorkflowContext` holds resolved configuration, `Workflow` validates dependencies, and `WorkflowRunner` owns locking, execution, logs and receipts.
- `data/`, `text/` and `modeling/` implement the scientific work. The orchestration layer does not duplicate preprocessing, fit models itself, modify recipes or regenerate LLM signals.

## Entry points

The common CLI exposes `workflow list`, `workflow plan NAME --run-id ID`, and `workflow run NAME --run-id ID`. The run command is explicit execution; planning and listing never create directories or start subprocesses. Required inputs are supplied explicitly instead of guessing the latest batch.

| Workflow | Purpose |
|---|---|
| `text-audit` | Validate an existing repaired release, including signal and embedding identities and manifest members |
| `model-tuning` | Run the existing numeric neural-network recipe study, selecting by development MSE |
| `feature-comparison` | Run the fixed Market / Financial / LLM comparison across hour, half-day and full-day targets |
| `legacy-data-build` | Reconstruct the historical fixed-tag full-day dataset from existing raw inputs |

The legacy reconstruction is deliberately separate from the batch_032 signal-MIL experiment. It does not replace current cleaned inputs. Deleted raw sources must be restored before that workflow can execute; no workflow downloads them automatically.

## Scientific stage boundaries

Numeric tuning:

```text
protocol -> prepare -> verify-inputs -> smoke -> development -> select
         -> verify-development -> evaluation -> report -> verify-all
```

Feature comparison:

```text
protocol -> prepare -> verify-inputs -> verify-engine -> development -> select
         -> verify-development -> evaluation -> report -> verify-all
```

The independent engine audit is a real prerequisite: the feature development verifier reads `ENGINE_INDEPENDENT_AUDIT.json`. The orchestration explicitly produces it before requesting the development audit. Preparation's `ENGINE_SELF_CHECK.json` is a different artifact.

The feature-comparison workflow retains its archived selected architecture and source dependencies. Running a new tuning workflow does not silently substitute its winner. Evaluation is a separate capability and always follows the development selection and independent development audit. Existing research rules continue to live in each module's protocol.

## State, output and resume policy

Each run writes below the configured output root:

```text
workflows/<run-id>/
  WORKFLOW.json
  WORKFLOW.lock
  logs/
    00_protocol.log
    ...
  artifacts/
    nn_tuning/              # or nn_feature_validation / legacy_data
```

Only explicit execution creates this tree. Arguments are passed as an argument vector with `shell=False`; paths containing spaces remain single arguments. The child process receives explicit research, original-source and isolated output roots.

The runner requires `allow_training`, `allow_evaluation` and `allow_network` capabilities when applicable. No registered workflow generates LLM results or requests network access. These are execution controls, not new research decisions.

A successful exit code is insufficient: each declared output must exist, explicit failed audit JSON is rejected, and completed output and log hashes are recorded. Source changes during execution stop the sequence before the next stage.

Use an explicit stopping step to pause cleanly. A paused run can resume only if the command plan, resolved roots, source files, explicit inputs, completed artifacts and logs still match. Completed stages are skipped. Failed, interrupted and completed runs are preserved; they cannot be overwritten or automatically retried. A failure may have left partial model checkpoints, so recovery uses a new run ID.

Explicit input directories are content-hashed on execution and resume. This may take time for raw reconstructions; no such hashing occurs in a plan. Model modules retain their additional detailed scientific input and checkpoint receipts.

## Verification

The workflow tests use fake executors and tiny temporary files. They cover execution boundaries, dependency order, output isolation, completion evidence, failed and interrupted runs, exclusive locking, changed bindings, and verified pause/resume. They neither train research models nor call APIs.

The workflow's source files, like the domain modules and tests, are included in the project's documentation, formatting and unit-test checks.

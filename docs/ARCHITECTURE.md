# Architecture and ownership

Public export: private archive/workspace references below describe optional historical integration. The public integrity tool verifies distributed files; see [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md).

## Design basis

The source layout follows the [PyPA src-layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
Package code is separate from scripts, documentation, data and tests. pytest uses
[separate tests and importlib import mode](https://docs.pytest.org/en/stable/explanation/goodpractices.html).
Named modules, classes and functions carry [PEP 257 docstrings](https://peps.python.org/pep-0257/);
non-obvious chronology, masking, atomic publication and resume conditions also have
inline explanations. These documents guide design; no claim of external certification
or a universal requirement to use classes is made.

## Responsibilities

| Layer | Owns | Does not own |
|---|---|---|
| cli | Argument validation, lightweight discovery, dispatch, exit status | Model math or step execution internals |
| config / provenance | Typed settings, protected roots, verified archive relocation | Credentials or changing frozen manifests |
| common | Streaming hashes, JSON conversion, atomic writes, UTC timestamps | Domain rules, model selection, global mutable state |
| data | Provider handling, caching, warehouse, market features, IWV targets | Modern free-form signal model selection |
| text | Citations, full source packets, repaired-release integrity and bags | LLM reruns or target-driven edits |
| modeling | Fit/predict engines, preprocessing, calibration, frozen study/report rules | Generic shell orchestration |
| steps | Arguments, explicit inputs/expected outputs, dependencies, resource flags | Implementing a second copy of scientific logic |
| workflows | Plan validation, locks, logs, receipts, ordered execution and resume | Choosing scientific recipes or changing evaluation splits |

Dependency direction is CLI -> workflows -> steps -> domain module entry points.
Domains may use common and config. common never imports domains or workflows.
Studies compose inputs, engines and reports; pure numerical calculations remain functions.

## Where OOP is useful

- Settings, WorkflowInputs and WorkflowContext are immutable configuration objects.
- Step is an immutable contract describing one command and its declared dependencies.
- Workflow validates the stage graph and creates a deterministic plan.
- WorkflowRunner owns process execution, run state, locks, logs and completion receipts.
  Its executor can be replaced with an offline test double.
- Existing HTTPClient, Collector and Warehouse own provider, collection and storage
  state. PyTorch network modules own trainable parameters. Their numeric behavior
  is not rewritten merely to increase the number of classes.

Composition is used instead of a generic multi-level framework. A hash function or
OLS calculation has no persistent lifecycle and does not need a wrapper class.

## Workflow steps

Four discoverable workflows are wired to actual modules: text-audit, model-tuning,
feature-comparison and legacy-data-build. Model workflows include input checks,
preparation, development fitting, selection, independent development verification,
evaluation, reporting and a final independent audit. Feature comparison also runs
the engine verification required by its later independent audit.

Planning has no side effects. Execution reserves a new run ID, acquires a file lock,
checks required inputs and capabilities, hashes the plan/source/explicit inputs,
and logs each process. A nonzero exit, absent/failed output receipt or changed source
stops the run. Only an intentional clean pause can resume, after the completed prefix
and its artifacts/logs are rechecked. This avoids replaying partially completed
training into existing checkpoints. Frozen domain receipts still verify scientific
inputs and chronology; the orchestration receipt does not replace them.

## Archive compatibility

Historical sources are moved byte-for-byte, not rewritten. An executed-and-verified
source relocation catalog maps original absolute names to archived bytes and their
SHA-256 values. In the private checkout, the provenance checker verifies the original digest at the relocated
path. The legacy tag validator verifies archived package members before importing
them under an isolated namespace. Missing or changed sources fail explicitly.

The workspace work/ and outputs/ trees remain in place because current research
artifacts bind their absolute paths and some frozen helpers derive their root from
their source location. They are protected compatibility dependencies, not a second
active source layout. Reorganizing them requires a separate artifact migration.

## Test boundaries

Unit tests use temporary files, fake subprocesses and synthetic data. API credentials
are removed from their environment and network connections are blocked. Tests cover
publication failure, cache time/identity checks, point-in-time behavior, strict
chronology, padding/missing text, forward parity and training lifecycle, relocation,
launcher behavior and orchestration failure/resume cases. Lint, format, documentation
coverage and frozen-source integrity are independent checks.

A passing suite is not proof that every historical experiment or paid provider can
be rerun after approved raw-data deletion. Coverage and remaining compatibility
dependencies are described in REPRODUCIBILITY.md; private migration evidence is not distributed.
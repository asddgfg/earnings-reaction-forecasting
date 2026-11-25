# Earnings Forecasting with Structured and LLM-Derived Features

A Rutgers MQF research project studying whether financial disclosures and LLM-derived
text signals add predictive information beyond market features.

The research considers signed abnormal return (AR) and its absolute magnitude,
with one-hour, half-day and full-day reaction windows. Market adjustment uses IWV
and rolling market-model estimates. Development-period validation selects models;
later-period evaluation is a separate, explicitly gated step.

This repository contains data utilities, text validation and loading, numeric
neural-network and Attention MIL engines, calibration, and workflow code. It is a
**source-code release**, not a distribution of the research dataset or trained
models. It does not include API credentials or launch paid LLM extraction.

## What can run from this repository?

| Operation | Public checkout |
|---|---|
| Environment diagnostics and workflow discovery/planning | Ready after installation |
| Offline unit tests, including synthetic training and frozen-engine parity | Ready after installation |
| Source/citation and release validators | Available; supply your own conforming inputs |
| Historical research studies and statistical reports | Require the retained private research workspace |
| Downloaded data, final LLM signals, embeddings, weights and predictions | Not distributed |
| LLM extraction / new embedding generation | No command provided by this release |

The two historical engine implementations under tests/oracles are small **source-code
test fixtures**, not trained models. They preserve an independent numerical comparison.
Historical source fingerprints use logical roots rather than personal paths.

## Quick start

Use Python 3.12 and a virtual environment.

~~~bash
python -m venv .venv
~~~

Activate it with source .venv/bin/activate on macOS/Linux, or
.\.venv\Scripts\Activate.ps1 in Windows PowerShell. Then:

~~~bash
python -m pip install --upgrade pip
python -m pip install -e ".[model,dev,legacy]"
python -m mqf_research doctor
python -m mqf_research workflow list
python -m mqf_research workflow plan feature-comparison --run-id preview_001
python -m pytest tests/unit -q
python tools/run_checks.py
~~~

Planning does not read a research dataset or create an output directory. Tests use
temporary synthetic fixtures and block network connections; tiny training tests
exercise optimizer/checkpoint behavior without fitting research observations.

On Windows, scripts/Run-MQF.ps1 and scripts/Run-Checks.ps1 are convenience entry
points. They use .venv/Scripts/python.exe when available, or accept -Python.

## Repository structure

~~~text
src/mqf_research/
  common/       Shared hashing, serialization, atomic files and UTC metadata
  data/         Collection, caching, warehouse, market features and target utilities
  text/         Citations, complete-source packets and repaired-release validation
  modeling/     Engines, calibration, input guards, studies and reports
  steps/        Explicit commands, inputs, outputs and dependencies
  workflows/    Planning, execution, locks, logs and verified pause/resume
configs/        Portable example configuration
scripts/        PowerShell entry points
tests/unit/     Offline tests
tests/oracles/  Immutable first-party engine source fixtures
tests/private/  Explicit optional checks requiring the original private workspace
tools/          Formatting, documentation and distribution-integrity checks
docs/           Architecture, data contracts and publication notes
~~~

Workflow state belongs to WorkflowContext, Workflow and WorkflowRunner.
Mathematical transformations and shared utilities remain focused functions.

## Configuration and research inputs

No local configuration is required for diagnostics or unit tests. To connect an
existing private research workspace, copy configs/project_settings.example.json
to configs/project_settings.json and set its three paths. Relative paths resolve
from configs/, not from the shell's current directory.

Environment overrides are MQF_RESEARCH_ROOT, MQF_SOURCE_PROJECT and MQF_OUTPUT_ROOT.
Local settings and outputs are git-ignored.

A full historical run also needs the original frozen helper code, cleaned releases,
embedding matrices and checkpoint/input manifests. Those cannot be replaced by
empty folders or freshly downloaded look-alikes. See
[reproducibility requirements](docs/REPRODUCIBILITY.md).

The legacy-data-build workflow reconstructs the earlier fixed-tag/full-day pipeline.
It is separate from the repaired free-form-signal model experiment. No workflow
silently downloads missing sources, reruns an LLM or changes a frozen cohort.

For explicitly authorized research execution, use a new run ID. Training requires
--allow-training; later-period evaluation also requires --allow-evaluation.
--through verify-development pauses before evaluation. Failed or completed runs
are preserved; only verified clean pauses can resume.

## Text representation

Each event retains its own list of signals. Citation and timestamp checks validate
their source support. Each signal can have an embedding; Attention MIL processes the
variable-length bag, while mean pooling is a comparison representation. Empty eligible
text and unavailable text are separate states. This package loads existing vectors;
it does not generate them. See [text interfaces](docs/TEXT_PIPELINE.md).

## Quality and publication

- [Architecture](docs/ARCHITECTURE.md)
- [Workflow guide](docs/WORKFLOWS.md)
- [Public release scope](docs/PUBLIC_RELEASE.md)
- [Reproducibility requirements](docs/REPRODUCIBILITY.md)

DISTRIBUTION_MANIFEST.json binds the distributed files to SHA-256 hashes.
The public integrity tool checks those bytes; it does **not** claim to revalidate
unpublished research artifacts.

Do not commit credentials, local settings, datasets, signals, embeddings, weights,
predictions, caches or archives. The included .gitignore covers these locations.
No open-source license has been selected for this export; public visibility alone
does not grant a general reuse license.
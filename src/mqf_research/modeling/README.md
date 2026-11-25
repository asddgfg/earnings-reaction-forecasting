Current orchestration: use the project scripts/Run-MQF.ps1 workflow commands described in docs/WORKFLOWS.md. They enforce stage order and isolate each new run. The module commands below remain available for explicit step-level work.

# Maintained model workflows

These modules preserve the 23 November 2026 experiments while making their code
readable, testable, and safer to resume. The twelve original modules under
`work/nn_tuning_20261123_b032` and
`work/nn_feature_validation_20261123_b032` remain unchanged. `PROVENANCE.json`
records their hashes and the read-only historical helper dependencies.

## Scientific contract

- Signed IWV-adjusted abnormal return remains the target. One hour is primary;
  half day and full day remain the declared additional windows.
- Recipe selection uses development MSE. Direction accuracy does not select the
  model, and no events are dropped to change an accuracy number.
- Imputation, scaling, category vocabulary, and target moments use training rows
  only. Inner stopping uses earlier chronological history.
- Network creation order, seeds 42/43, optimizer, clipping, early-stopping rule,
  float32 target inversion, and two-seed aggregation remain unchanged.
- Attention keeps each signal. Empty bags are zero, padding is masked, and there
  is no signal truncation or mean-pooling substitution.
- Calibration candidates and tie order are identity, offset, and affine. Every
  candidate scores the same chronological development cohort. Coefficients are
  frozen before evaluation.

## Running an explicitly requested new experiment

Set `MQF_RESEARCH_ROOT` to the retained research workspace. The project-level
configuration controls `MQF_OUTPUT_ROOT`; it must designate a fresh maintained
run location, not an archived result directory. Install the maintained package
and its model dependencies, or use the project's documented local runner.

The commands below are instructions, not actions performed by the modernization.
They train models when invoked. No command regenerates LLM signals or embeddings.
Use `--help` to inspect a command without loading datasets or fitting a model.

For numeric recipe selection, invoke these stages in order:

```text
python -m mqf_research.modeling.tuning.study protocol
python -m mqf_research.modeling.tuning.study prepare
python -m mqf_research.modeling.tuning.verify inputs
python -m mqf_research.modeling.tuning.study smoke
python -m mqf_research.modeling.tuning.study development --workers 2
python -m mqf_research.modeling.tuning.study select
python -m mqf_research.modeling.tuning.verify development
python -m mqf_research.modeling.tuning.study evaluation
python -m mqf_research.modeling.tuning.report
python -m mqf_research.modeling.tuning.verify all
```

For the fixed three-feature, three-window comparison:

```text
python -m mqf_research.modeling.feature_validation.study protocol
python -m mqf_research.modeling.feature_validation.study prepare
python -m mqf_research.modeling.feature_validation.verify inputs
python -m mqf_research.modeling.feature_validation.verify engine
python -m mqf_research.modeling.feature_validation.study development --workers 2
python -m mqf_research.modeling.feature_validation.study select
python -m mqf_research.modeling.feature_validation.verify development
python -m mqf_research.modeling.feature_validation.study evaluation
python -m mqf_research.modeling.feature_validation.report
python -m mqf_research.modeling.feature_validation.verify all
```

The feature comparison intentionally uses the archived selected architecture
`n32_h64_d20_w5`. It does not silently adopt a different new tuning winner.
`inputs check` provides an explicit development-only data audit; its receipts go
to the new run's `checks` directory, not the installed source package.

## Output and compatibility boundaries

A completed cell resumes only if all receipt hashes match. A partial cell with
existing matrices or checkpoints is rejected before writing replacements. Choose
a fresh output root when recovering from an incomplete fit. Reporting `--out`
is restricted to the maintained output tree.

`RUN_CONFIG.json` records the resolved workspace, original-source, output, and
maintained-package paths plus the exact experiment directory. Resume, evaluation,
and reporting reject changed settings even when the earlier files still exist.
The run's source receipt also includes shared guards, configuration code, and
compatibility code, not only the network implementation.

The pure maintained engines and persistent guards work under Python `-O`. Frozen
input reconstruction and independent statistics still rely on historical helper
modules whose checks use assertions. Their explicit compatibility boundary rejects
`-O`/`-OO`, resolves imports from the configured research root, and verifies the
actual imported helper location. Run the full workflow without optimization.

Existing frozen dataset bindings and receipts contain absolute historical paths.
Changing `MQF_RESEARCH_ROOT` alone does not migrate those bindings. A workspace
move requires a separate, recorded provenance migration; paths are not silently
rewritten. Scientific inputs, outcomes, and completed results were not altered.

## Verification performed for this revision

The unit tests use synthetic frames, signals, and temporary artifacts. They cover
training-only preprocessing, chronology and small-history errors, missing and
invalid signal indices, empty-bag gradients, masking/permutation invariance,
exact archived-engine arithmetic, stopping selection, calibration coefficients
and ties, completed-receipt tampering, partial output protection, and optimized
Python guard behavior. No full experiment or paid/API extraction is run by this
test suite. Full-data verifier commands above remain available for an explicitly
requested new experiment.

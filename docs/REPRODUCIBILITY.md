# Reproducibility boundary

The public repository is self-contained for package import, help, diagnostics,
workflow planning and the offline unit suite. It includes two original engine
source fixtures for numerical/optimizer parity. All test observations are synthetic.

## Full historical experiments

The exported scientific workflow code still refers to dated research artifacts.
To execute it, restore an authorized copy of the original research workspace and
set MQF_RESEARCH_ROOT, MQF_SOURCE_PROJECT and MQF_OUTPUT_ROOT (or local JSON settings).

Required private resources include:

- Frozen helper modules under work/, including input reconstruction and statistical
  inference code. Dated module names in the source are intentional provenance.
- Cleaned event panels and the repaired signal release, including RELEASE.json,
  event/signal files, embedding indexes and existing matrices.
- Prior experiment manifests, locked selections and reference predictions consumed
  by the inherited protocol.
- Any archived original data-validator package and verified relocation catalog
  required by the legacy fixed-tag pipeline.
- Data-provider access and raw records if rebuilding acquisition outputs.

A public clone plus credentials is insufficient to reconstruct the original study.
Some original raw payloads were removed during earlier storage cleanup. Re-downloaded
content may differ from the frozen vintage. The code fails on absent/mismatched
artifacts instead of filling them with current data.

Package help and imports defer these private-resource checks until actual execution.
The public export does not remove outcome masking, chronology guards, frozen-source
checks or training/evaluation capability requirements.

## Provenance files

The module PROVENANCE.json files preserve historical fingerprints with logical
REPOSITORY, RESEARCH_ROOT and SOURCE_PROJECT placeholders. These are metadata labels,
not automatically expanded environment variables. They contain no personal paths.

DISTRIBUTION_MANIFEST.json independently lists the bytes distributed by this export.
The public integrity tool verifies that manifest. The optional tests/private check
requires a separately supplied original, unredacted manifest; it is not part of the
default public test run.

## Data rights and licenses

Provider data, SEC/IR payloads, final LLM responses, embeddings and trained models
are not distributed. No data redistribution rights are implied. Third-party Python
packages are installed from their own distributions and retain their licenses.
No project-wide open-source license is added without the project owners' choice.
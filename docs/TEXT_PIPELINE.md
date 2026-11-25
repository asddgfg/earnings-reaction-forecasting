# Maintained text-input interfaces

The maintained text package validates and loads the existing reviewed signals.
It does not call an LLM, generate embeddings, change a prompt, merge signals, or
replace the current trained model. Every event keeps its variable-length list
of signal vectors. Pooling and prediction belong to the model layer.

## Read-only commands

```powershell
python -m mqf_research.text PATH_TO_BATCH_032 --metadata-only
python -m mqf_research.text PATH_TO_BATCH_032
```

The first command checks the frozen text and index checksums, signal identities,
text hashes, and event mapping. It does not read vector values or target columns.
The second also checks existing vector file integrity, dimensions, float32 dtype,
finiteness and unit norms. Neither command writes to the release. Add
`--expected-release-sha256` to bind the check to an independently frozen manifest
hash; add `--all-release-files` for all original audit-file hashes. The latter
cannot be combined with metadata-only mode.

For model inputs, `load_repaired_inputs(release_dir, horizon)` returns the same
four objects as the frozen batch loader: a DataFrame, financial feature names,
event-to-vector-row lists and read-only vectors. Target columns are excluded by
default; `include_targets=True` is an explicit research/training choice.

## Evidence and empty inputs

Citation checks preserve missing-fragment gaps. They never glue disconnected
text together to manufacture a quote. A short table-cell number needs an exact
complete-line match and accompanying text from the same source. Validation can
verify textual support, but cannot certify the interpretation of a table or the
economic relevance of a signal.

`text_available=False` differs from a completed, eligible extraction with no
signals. The latter has `text_available=True` and `has_signals=False`. A manual
source review may restore text availability without inventing a signal. Distinct
signal IDs are retained even if they happen to have identical text.

Packet serialization measures UTF-8 bytes and fails rather than truncating.
Optional company-context pointers may be omitted to preserve the complete
original text. They only quote evidence already in the same, time-eligible
packet; no remembered company characteristics are introduced.

## Frozen extraction compatibility

The existing `run_codex_text_rerun_v3.py` and
`Continue_Codex_Text_NewPrompt.cmd` remain the historical production entry points.
Their source hashes are bound into continuation policies. Formatting those files
or updating the recorded hashes silently would invalidate provenance or change
which cached results may be reused. They remain private historical dependencies and are not included in this public repository.
The maintained pure validators are adapted from those implementations; original
logical source references and SHA-256 values are in `text/PROVENANCE.json`.

This package deliberately does not offer a new extraction command. No SDK is
imported by the validation/loader path. Resuming extraction remains an explicit
use of the original compatibility entry point and its existing policy guards.

## Unit-test scope

Synthetic tests cover fabricated and gapped citations, short numeric context,
multiple signals, empty versus missing inputs, Unicode byte limits, full-source
preservation, cache identities, relocation, checksum failures, path traversal,
vector corruption and read-only CLI behavior. They do not require downloaded
SEC files, historical production examples, credentials or network access.

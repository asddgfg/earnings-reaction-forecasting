# Engineering standard and preservation policy

Reviewed against the linked primary documentation on 2026-11-24 (America/New_York).
These are documented engineering conventions, not an external certification.

| Practice | Applied decision | Verification |
| --- | --- | --- |
| PEP 8 and Ruff | Consistent imports, whitespace and 100-character team line width | Ruff lint and format check |
| PEP 257 | Contract docstrings for every named maintained function, including private/nested helpers; module documentation | Static AST inventory plus code review |
| pytest | `src` package layout, importlib discovery, isolated synthetic fixtures, temporary outputs | One offline suite; no API credentials or connections |
| scikit-learn | Earlier-history-only transforms and selection; evaluation outcomes cannot select settings | Synthetic leakage and chronology regression cases |
| Explicit data contracts | Raise normal exceptions for malformed external inputs; avoid optimization-disabled assertions | Invalid-input and optimized-Python checks where relevant |
| Safe persistence | Same-directory atomic writes; completed runs protected; separate output roots | Write-failure, concurrency and overwrite cases |
| Research reproducibility | Preserve frozen source/results; exact source hashes and numerical parity where tested | Provenance receipts and regression evidence |
| Honest test reporting | Distinguish line/branch coverage, unit tests, read-only integration and unexecuted operations | Saved pytest, coverage and validation reports |

The detailed rationale is in [PEP 8](https://peps.python.org/pep-0008/),
[PEP 257](https://peps.python.org/pep-0257/),
[pytest good practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html),
[pytest monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html),
[scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html),
[logging](https://docs.python.org/3/library/logging.html), and
[Ruff configuration](https://docs.astral.sh/ruff/configuration/).

## Why historical files are preserved

The project's experimental source is part of its recorded evidence: extraction
resume hashes, source manifests, frozen model inputs and completed verification
receipts bind some source bytes. Formatting those files in place would invalidate
their integrity records without improving their historical scientific result.

The maintained package is a versioned revision, not a rewrite of evidence. It
contains the active data, text-input and current neural-network workflows. Older
CatBoost/quantile/classification/trading explorations, prompt pilots, extraction
runs and document-generation scripts remain historical. They are inventoried or
explicitly listed as compatibility dependencies rather than claimed as modernized.
The latter stay necessary for exact historical input reconstruction/statistics.

The new package does not change which model was selected, sample membership,
date conventions, LLM prompts/results, embeddings or recorded performance. Narrow
robustness fixes affect invalid/incomplete inputs and future execution. Any future
scientific change requires a new protocol and separate results.

## Comment and test policy

Docstrings describe purpose, expected shapes/units, returned values, missing-data
semantics and side effects when relevant. Comments explain the reason for unusual
math, temporal boundaries, masking or compatibility behavior. We do not add a
comment to every assignment merely to increase the comment count.

Tests cover observable contracts and independently expected small examples.
Network calls are blocked. A test that deliberately supplies bad data should
verify the meaningful rejection, not simply reproduce the function body. Real
`batch_032` checks are separately labeled read-only integration evidence. No full
research re-training, paid extraction, downloads or new embeddings are part of
the unit suite.

## Scope of coverage

An AST documentation count is not test coverage. Line and branch coverage are
reported independently. Complete training orchestration, SEC/provider behavior,
report rendering and archived compatibility programs require additional explicit
integration runs if modified later. Their unexecuted branches remain visible in
coverage reports; they are not hidden to obtain a high percentage.

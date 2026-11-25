# Public source release — 2026-11-25

This export was prepared separately from the working research project. The private
checkout, raw/cleaned research data, final LLM results and trained artifacts remain
unchanged on the original machine.

## Included

Application source; focused shared utilities; explicit steps/workflows; offline
unit tests; two data-free historical engine source oracles; portable configuration
and environment examples; PowerShell launchers; English documentation; and a
GitHub Actions workflow for Python 3.12 on Windows.

## Excluded

Credentials and local settings; personal absolute paths; source relocation records;
private audit receipts and inventories; downloaded SEC/IR/provider data; actual
event/signal datasets; final LLM responses; embeddings; checkpoints; forecasts;
old experiment trees; archives; virtual environments; caches; and Git history.

## Export-specific adjustments

- Historical provenance paths are replaced by logical-root labels. Their recorded
  original source digests remain metadata, not a claim that originals are bundled.
- Public integrity checks verify DISTRIBUTION_MANIFEST.json with relative paths.
- Numerical and optimizer parity tests use bundled unchanged first-party engine
  snapshots. No research observations are embedded in those fixtures.
- The original private-source audit is an explicit tests/private operation. It is
  outside the default unit suite and requires a separately supplied private manifest.
- Scientific module imports and --help defer private-workspace checks until actual
  execution. Algorithms, feature definitions and selection rules are retained.
- The PowerShell entry point also recognizes a standard local .venv directory.
- Text files use LF for stable Git checkout hashes; immutable oracle bytes are
  protected separately in .gitattributes.

The release was checked locally; adding a GitHub Actions file does not imply a
remote GitHub workflow has already run. Full scientific reproduction still needs
the resources listed in REPRODUCIBILITY.md. No new research training, LLM tagging
or embedding generation was performed to prepare this export.

## Publishing

Extract the ZIP and use the directory containing pyproject.toml and README.md as
the Git repository root. Review the English README, then push that directory.
The ZIP itself, local quality logs and private data should not be committed.

No project license was invented. Select a license only after the project owners
agree on reuse terms. See GitHub's
[repository licensing guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

After intentional changes, run python tools/update_distribution_manifest.py --write to regenerate the manifest before running integrity
checks; source inventory is not a substitute for the export manifest. The archive's
external SHA-256 checksum identifies this exact prepared ZIP.
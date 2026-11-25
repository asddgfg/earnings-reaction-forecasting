# Historical numerical regression oracles

These two first-party Python source snapshots are bundled only as test fixtures:

- tuning_engine.py.txt: numeric neural network engine from the frozen 2026-11-23 study.
- feature_engine.py.txt: fixed numeric / Attention MIL engine from the same date.

They contain no observations, credentials, trained weights, embeddings or forecasts.
The .py.txt suffix distinguishes preserved historical code from maintained Python
source. Tests load the exact text explicitly with SourceFileLoader and compare
preprocessing, initialization, forward passes and tiny synthetic optimizer loops.
Do not autoformat these fixtures: their unchanged bytes are the independent oracle.

Their source hashes are listed in SHA256.json and the root distribution manifest.
They are not the private research dataset and do not reproduce published forecasts
without the separately retained input artifacts.
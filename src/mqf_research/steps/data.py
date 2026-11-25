"""Legacy reconstruction commands writing exclusively into a fresh run tree.

This pipeline recreates the original fixed-tag, full-day dataset. It is not the
batch_032 signal-MIL dataset consumed by the current model experiments.
"""

from __future__ import annotations

from mqf_research.steps.base import Step
from mqf_research.workflows.context import WorkflowContext


def legacy_data_steps(context: WorkflowContext) -> tuple[Step, ...]:
    """Build a warehouse, features, IWV targets and dataset from retained raw files."""
    raw = context.inputs.require("raw_dir")
    batch = context.inputs.require("batch_dir")
    selected = context.inputs.require("selected_file")
    data = context.artifact_dir / "legacy_data"
    warehouse = data / "warehouse"
    market = data / "features/market_features.csv"
    target = warehouse / "targets_1d_abnormal.csv"
    dataset = data / "model_dataset"
    extra = (
        ("--structured-csv", str(context.inputs.require("structured_csv")))
        if context.inputs.structured_csv is not None
        else ()
    )
    return (
        Step(
            "warehouse",
            "Ingest existing raw files into a new warehouse; no download",
            "mqf_research.data.build_warehouse",
            ("--raw", str(raw), "--warehouse", str(warehouse), "--tickers-file", str(selected)),
            required_inputs=(raw, selected),
            expected_outputs=(warehouse / "global.sqlite", warehouse / "build_report.json"),
        ),
        Step(
            "market-features",
            "Build strictly prior-session market features",
            "mqf_research.data.build_market_features",
            (
                "--data",
                str(data),
                "--batch",
                str(batch),
                "--selected",
                str(selected),
                "--output",
                str(market),
            ),
            depends_on=("warehouse",),
            required_inputs=(batch, selected),
            expected_outputs=(market, market.with_name("market_features_summary.json")),
        ),
        Step(
            "abnormal-returns",
            "Build the original IWV full-day targets with rolling prior-session OLS",
            "mqf_research.data.build_abnormal_returns",
            (
                "--data",
                str(data),
                "--warehouse",
                str(warehouse),
                "--batch",
                str(batch),
                "--selected",
                str(selected),
            ),
            depends_on=("warehouse", "market-features"),
            required_inputs=(batch, selected),
            expected_outputs=(target, warehouse / "targets_1d_abnormal_summary.json"),
            reads_evaluation=True,
        ),
        Step(
            "legacy-dataset",
            "Assemble the legacy fixed-tag dataset; do not substitute for batch_032",
            "mqf_research.data.build_model_dataset",
            (
                "--data",
                str(data),
                "--batch",
                str(batch),
                "--target-csv",
                str(target),
                "--market-csv",
                str(market),
                "--output-dir",
                str(dataset),
                *extra,
            ),
            depends_on=("market-features", "abnormal-returns"),
            required_inputs=(batch, *([context.inputs.require("structured_csv")] if extra else [])),
            expected_outputs=(dataset / "model_dataset.csv", dataset / "data_quality_report.json"),
            reads_evaluation=True,
        ),
    )

"""Read-only CLI behavior, with no dependency on local data or API access."""

import json

from mqf_research.cli import build_parser, main


def test_doctor_does_not_create_output_or_load_data(monkeypatch, tmp_path, capsys):
    """An environment diagnosis is safe even before the dataset is installed."""
    monkeypatch.setenv("MQF_RESEARCH_ROOT", str(tmp_path / "research"))
    monkeypatch.setenv("MQF_OUTPUT_ROOT", str(tmp_path / "new_runs"))
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network_calls"] == 0
    assert not result["research_data_loaded"]
    assert not (tmp_path / "new_runs").exists()


def test_validate_text_flags_are_explicit():
    """The CLI preserves the distinction between metadata and full verification."""
    args = build_parser().parse_args(["validate-text", "batch", "--metadata-only"])
    assert args.metadata_only
    assert not args.all_release_files


def test_malformed_release_returns_failure_without_traceback(monkeypatch, caplog):
    """A missing JSON field produces an actionable CLI error and nonzero status."""
    from mqf_research import text

    def invalid_release(*args, **kwargs):
        """Represent a malformed caller-supplied release without reading disk."""
        raise KeyError("dimensions")

    monkeypatch.setattr(text, "validate_release", invalid_release)
    assert main(["validate-text", "unused"]) == 1
    assert "dimensions" in caplog.text

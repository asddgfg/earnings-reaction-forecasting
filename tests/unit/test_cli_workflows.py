"""CLI workflow contracts: previews are inert and execution is a separate action."""

import json

from mqf_research import cli
from mqf_research.config import Settings


def test_workflow_discovery_does_not_need_local_data(capsys, monkeypatch):
    """Listing cannot resolve paths, read data, or accidentally start work."""

    def forbidden():
        """Make accidental configuration or input access fail this contract."""
        raise AssertionError("discovery accessed configuration")

    monkeypatch.setattr(cli, "get_settings", forbidden)
    assert cli.main(["workflow", "list"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["executed"]
    assert set(result["workflows"]) == {
        "text-audit",
        "model-tuning",
        "feature-comparison",
        "legacy-data-build",
    }


def test_workflow_preview_does_not_create_outputs(tmp_path, monkeypatch, capsys):
    """Preview valid commands against nonexistent inputs without writing receipts."""
    settings = Settings(tmp_path / "research", tmp_path / "source", tmp_path / "outputs", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["workflow", "plan", "model-tuning", "--run-id", "preview"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "steps" in result
    assert not settings.output_root.exists()


def test_workflow_run_needs_training_capability(tmp_path, monkeypatch):
    """A mistaken run command cannot begin model work before a capability check."""
    settings = Settings(tmp_path / "research", tmp_path / "source", tmp_path / "outputs", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["workflow", "run", "model-tuning", "--run-id", "blocked"]) == 1
    assert not (settings.output_root / "workflows/blocked").exists()

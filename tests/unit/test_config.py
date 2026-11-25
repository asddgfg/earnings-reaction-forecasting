"""Path configuration contracts independent of any research data or secrets."""

import json

import pytest

from mqf_research.config import get_settings


def test_environment_overrides_file_without_creating_directories(tmp_path):
    """Explicit environment roots win and resolving them never writes outputs."""
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"research_root": "from_file"}), encoding="utf-8")
    settings = get_settings(config, {"MQF_RESEARCH_ROOT": str(tmp_path / "override")})
    assert settings.research_root == tmp_path / "override"
    assert settings.output_root == tmp_path / "override/outputs/maintained_runs"
    assert not settings.output_root.exists()


@pytest.mark.parametrize("payload", [[], {"reserch_root": "typo"}, {"research_root": ""}])
def test_invalid_configuration_is_not_silently_ignored(tmp_path, payload):
    """Misspelled keys and invalid values fail before an expensive stage begins."""
    config = tmp_path / "settings.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        get_settings(config, {})


def test_explicit_missing_file_is_an_error(tmp_path):
    """A caller-supplied missing configuration must not select another dataset."""
    with pytest.raises(FileNotFoundError):
        get_settings(tmp_path / "missing.json", {})


def test_output_traversal_rejected(tmp_path):
    """No caller can address an archived sibling via an output path helper."""
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    settings = get_settings(config, {"MQF_RESEARCH_ROOT": str(tmp_path / "research")})
    with pytest.raises(ValueError, match="escapes"):
        settings.output_path("..", "old_results")


def test_output_root_cannot_replace_inputs(tmp_path):
    """Input roots and their ancestors cannot be configured as output roots."""
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved"):
        get_settings(config, {"MQF_RESEARCH_ROOT": str(tmp_path), "MQF_OUTPUT_ROOT": str(tmp_path)})


def test_existing_experiment_is_not_a_new_output_root(tmp_path):
    """A completed experiment cannot be chosen as the maintenance output root."""
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    archived = tmp_path / "archived"
    archived.mkdir()
    (archived / "COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="existing experiment"):
        get_settings(config, {"MQF_OUTPUT_ROOT": str(archived)})


@pytest.mark.parametrize(
    "relative", ["outputs", "outputs/old_run", "work", "work/experiment", "releases"]
)
def test_broad_output_override_cannot_expose_archived_runs(tmp_path, relative):
    """Selecting an archive parent must not make its children writable reports."""
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    research = tmp_path / "research"
    with pytest.raises(ValueError, match="preserved tree"):
        get_settings(
            config,
            {"MQF_RESEARCH_ROOT": str(research), "MQF_OUTPUT_ROOT": str(research / relative)},
        )


def test_reserved_maintained_namespace_is_allowed(tmp_path):
    """The new namespace remains usable without weakening archive protection."""
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    research = tmp_path / "research"
    output = research / "outputs/maintained_runs/iteration2"
    assert (
        get_settings(
            config, {"MQF_RESEARCH_ROOT": str(research), "MQF_OUTPUT_ROOT": str(output)}
        ).output_root
        == output
    )


@pytest.mark.parametrize("relative", ["archive", "archive/legacy", "src", "configs", "scripts"])
def test_new_output_cannot_target_archived_or_active_code(tmp_path, relative):
    """A configured run cannot make retained source or configuration writable."""
    config = tmp_path / "settings.json"
    config.write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="preserved tree"):
        get_settings(
            config,
            {
                "MQF_SOURCE_PROJECT": str(source),
                "MQF_RESEARCH_ROOT": str(tmp_path / "research"),
                "MQF_OUTPUT_ROOT": str(source / relative),
            },
        )


def test_configs_directory_wins_over_legacy_root_file(tmp_path):
    """A missing local config in the new layout does not revive an obsolete root file."""
    from mqf_research.config import default_config_path

    (tmp_path / "configs").mkdir()
    (tmp_path / "project_settings.json").write_text('{"research_root":"obsolete"}')
    assert default_config_path(tmp_path) == tmp_path / "configs/project_settings.json"

import pytest
import yaml
from pydantic import ValidationError

from bug2code.config import load_config
from bug2code.paths import PROJECT_ROOT


def test_loads_default_config():
    cfg = load_config()
    assert cfg.seed == 42
    assert set(cfg.projects) == {"spark", "cassandra", "hbase"}


def test_every_selected_project_has_a_spec():
    cfg = load_config()
    for name in cfg.projects:
        spec = cfg.project(name)
        assert spec.jira_key.isupper()
        assert spec.github.startswith("apache/")
        assert spec.source_roots


def test_unknown_project_raises():
    cfg = load_config()
    with pytest.raises(KeyError):
        cfg.project("nosuchproject")


def test_split_fractions_sum_to_one():
    cfg = load_config()
    cfg.split.validate_fractions()


def test_paths_are_absolute_and_under_project_root():
    cfg = load_config()
    for name in type(cfg.paths).model_fields:
        path = getattr(cfg.paths, name)
        assert path.is_absolute()
        assert path.is_relative_to(PROJECT_ROOT)


def test_overrides_are_deep_merged(tmp_path):
    override = tmp_path / "exp.yaml"
    override.write_text(
        yaml.safe_dump({"seed": 7, "localization": {"code_max_length": 128}}),
        encoding="utf-8",
    )
    cfg = load_config(override)
    assert cfg.seed == 7
    assert cfg.localization["code_max_length"] == 128
    # untouched sibling keys survive the merge
    assert "encoder" in cfg.localization


def test_unknown_top_level_key_is_rejected(tmp_path):
    override = tmp_path / "bad.yaml"
    override.write_text(yaml.safe_dump({"sed": 1}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(override)

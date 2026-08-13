"""Typed access to the YAML configuration.

Structural sections that the whole pipeline depends on are validated strictly so
typos fail fast. Model hyperparameter blocks stay free-form dictionaries: they
change every phase, and encoding them in the schema would buy nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bug2code.paths import CONFIG_DIR, resolve

_DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
_PROJECTS_CONFIG = CONFIG_DIR / "projects.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectSpec(_Strict):
    """Repository and Jira coordinates of one Apache project."""

    jira_key: str
    github: str
    default_branch: str
    primary_language: str
    source_roots: list[str]


class SplitConfig(_Strict):
    """Train/validation/test partitioning policy."""

    strategy: str
    train_frac: float
    val_frac: float
    test_frac: float

    def validate_fractions(self) -> None:
        """Raise if the three fractions do not sum to 1."""
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


class PathsConfig(_Strict):
    """Output locations, declared project-relative in YAML and stored absolute."""

    data_raw: Path
    data_interim: Path
    data_processed: Path
    cache: Path
    repos: Path
    experiments: Path
    reports: Path
    figures: Path
    tables: Path

    @field_validator("*", mode="after")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        return resolve(value)


class Config(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    device: str
    projects: list[str]
    split: SplitConfig
    paths: PathsConfig
    # Bug-key subset file under paths.data_processed that every phase reads.
    dev_subset_name: str = "dev_subset.parquet"

    jira: dict[str, Any]
    github: dict[str, Any]
    linking: dict[str, Any]
    candidates: dict[str, Any]
    localization: dict[str, Any]
    evaluation: dict[str, Any]

    project_specs: dict[str, ProjectSpec] = Field(default_factory=dict)

    def project(self, name: str) -> ProjectSpec:
        """Return the spec for one project, by short name."""
        if name not in self.project_specs:
            raise KeyError(f"unknown project {name!r}; known: {sorted(self.project_specs)}")
        return self.project_specs[name]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(overrides: str | Path | None = None) -> Config:
    """Load ``configs/default.yaml`` plus projects, optionally merging an override file.

    Args:
        overrides: Path to an experiment YAML holding a subset of keys to replace.
    """
    raw = _read_yaml(_DEFAULT_CONFIG)
    if overrides is not None:
        raw = _deep_merge(raw, _read_yaml(resolve(overrides)))

    specs = _read_yaml(_PROJECTS_CONFIG)["projects"]
    raw["project_specs"] = specs

    cfg = Config.model_validate(raw)
    cfg.split.validate_fractions()

    unknown = set(cfg.projects) - set(cfg.project_specs)
    if unknown:
        raise ValueError(f"projects not defined in projects.yaml: {sorted(unknown)}")
    return cfg

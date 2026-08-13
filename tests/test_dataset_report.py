"""Tests for the descriptive statistics computed over the collected dataset."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from bug2code.data.dataset_report import (
    component_candidate_coverage,
    component_distribution,
    extension_counts,
    gold_file_stats,
)

FRAME = pd.DataFrame(
    {
        "project": ["spark", "spark", "spark"],
        "key": ["SPARK-1", "SPARK-2", "SPARK-3"],
        "components": [["Core"], ["Core"], ["SQL", "Core"]],
        "n_components": [1, 1, 2],
        "linked": [True, True, False],
        "source_files": [["a/A.scala"], ["b/B.scala", "b/C.java"], []],
        "source_statuses": [["A"], ["M", "M"], []],
        "n_gold": [1, 2, 0],
        "n_gold_new": [1, 0, 0],
        "n_gold_existing": [0, 2, 0],
        "n_candidates": [1.0, 2.0, None],
    }
)


def test_component_distribution_uses_single_component_issues_only():
    out = component_distribution(FRAME)
    assert list(out["component"]) == ["Core"]
    assert out.iloc[0]["issues"] == 2
    assert out.iloc[0]["share"] == 1.0


def test_gold_file_stats_counts_new_files_and_full_losses():
    out = gold_file_stats(FRAME).iloc[0]
    assert out["linked_bugs"] == 2
    assert out["gold_files_total"] == 3
    assert out["gold_files_new"] == 1
    assert out["bugs_with_any_new"] == 1
    assert out["bugs_all_gold_new"] == 1
    assert out["pct_single_file"] == 0.5


def test_extension_counts_ignores_unlinked_bugs():
    out = extension_counts(FRAME)
    assert dict(zip(out["extension"], out["files"], strict=True)) == {".scala": 2, ".java": 1}


CFG = SimpleNamespace(split=SimpleNamespace(train_frac=0.7, val_frac=0.1))


def _coverage_frame(components: list[str], gold: list[list[str]], statuses: list[list[str]]):
    """Ten linked, single-component bugs ordered in time, as the ceiling expects."""
    return pd.DataFrame(
        {
            "project": ["spark"] * len(components),
            "components": [[c] for c in components],
            "n_components": [1] * len(components),
            "linked": [True] * len(components),
            "source_files": gold,
            "source_statuses": statuses,
            "created_dt": pd.date_range("2015-01-01", periods=len(components), freq="D"),
        }
    )


def test_component_candidate_coverage_measures_coverage_and_reduction():
    # Train (rows 0-6) teaches Core -> A.scala and SQL -> S.scala. Of the two test
    # bugs (rows 8-9) only the first is reachable, so coverage is one in two.
    components = ["Core", "Core", "Core", "SQL", "SQL", "Core", "SQL", "Core", "Core", "Core"]
    files = ["A", "A", "A", "S", "S", "A", "S", "A", "A", "Z"]
    gold = [[f"{f}.scala"] for f in files]
    frame = _coverage_frame(components, gold, [["M"]] * 10)

    out = component_candidate_coverage(frame, CFG).iloc[0]

    assert out["test_bugs"] == 2
    assert out["universe_files"] == 3  # A.scala, S.scala, Z.scala
    assert out["mean_candidates"] == 1.0
    assert out["search_space_kept"] == round(1 / 3, 4)
    assert out["candidate_coverage"] == 0.5


def test_component_candidate_coverage_drops_bugs_whose_gold_is_all_new_files():
    # The last bug's only gold file was created by the fix, so it cannot be ranked
    # against a pre-fix snapshot: the bug and its file leave the measurement.
    components = ["Core"] * 10
    gold = [["A.scala"]] * 9 + [["N.scala"]]
    statuses = [["M"]] * 9 + [["A"]]

    out = component_candidate_coverage(_coverage_frame(components, gold, statuses), CFG).iloc[0]

    assert out["universe_files"] == 1  # N.scala never enters the universe
    assert out["candidate_coverage"] == 1.0

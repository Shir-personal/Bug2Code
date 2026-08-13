import pandas as pd
import pytest

from bug2code.config import load_config
from bug2code.data.split import assign_split, build


@pytest.fixture
def cfg():
    return load_config()


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2014-01-01", periods=n, freq="D", tz="UTC"))


def test_assign_split_uses_seventy_ten_twenty(cfg):
    labels = assign_split(_dates(100), cfg)
    assert labels.value_counts().to_dict() == {"train": 70, "val": 10, "test": 20}


def test_assign_split_rounds_the_boundary_instead_of_truncating(cfg):
    # 0.7 + 0.1 is 0.79999... in binary; truncation would give 7 test bugs here.
    labels = assign_split(_dates(10), cfg)
    assert labels.tolist() == ["train"] * 7 + ["val"] + ["test"] * 2


def test_assign_split_is_ordered_in_time(cfg):
    shuffled = _dates(50).sample(frac=1, random_state=0)
    labels = assign_split(shuffled, cfg)
    newest = {"train": shuffled[labels == "train"].max(), "val": shuffled[labels == "val"].max()}
    assert newest["train"] < shuffled[labels == "val"].min()
    assert newest["val"] < shuffled[labels == "test"].min()


def test_build_splits_each_project_separately(cfg):
    bugs = pd.DataFrame(
        {
            "project": ["a"] * 10 + ["b"] * 10,
            "key": [f"A-{i}" for i in range(10)] + [f"B-{i}" for i in range(10)],
            "fix_sha": ["s"] * 20,
            "fix_date": ["2014-01-01"] * 20,
            "parent_sha": ["p"] * 20,
            "gold_files": [["f.java"]] * 20,
            "n_gold": [1] * 20,
            "n_candidates": [100] * 20,
        }
    )
    issues = pd.DataFrame(
        {
            "key": bugs["key"],
            "summary": ["t"] * 20,
            "description": ["d"] * 20,
            "components": [[]] * 20,
            # Project b is entirely newer than project a: a per-project split must
            # still give each of them its own train portion.
            "created": list(_dates(10).astype(str)) + list(_dates(20)[10:].astype(str)),
        }
    )
    out = build(cfg, bugs, issues)
    counts = out.groupby("project")["split"].value_counts().unstack()
    assert counts.loc["a", "train"] == 7
    assert counts.loc["b", "train"] == 7

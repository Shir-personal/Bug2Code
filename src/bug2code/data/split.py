"""Assemble the final bug table and cut it into train / validation / test.

Takes the rows produced by ``build_snapshots``, joins the Jira text onto them and
labels every bug ``train``, ``val`` or ``test`` by issue creation date, per
project: oldest 70% train, next 10% validation, newest 20% test.

The split is temporal, not random, because deployment is temporal — a model is
trained on past bugs and used on future ones — and because near-duplicate reports
filed in the same week would otherwise land on both sides and inflate the scores.

Usage:
    python -m bug2code.data.split [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import Config, load_config
from bug2code.data.build_snapshots import OUTPUT_NAME as BUGS_NAME
from bug2code.data.collect_issues import OUTPUT_NAME as ISSUES_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "localization_dataset.parquet"

COLUMNS = [
    "project",
    "key",
    "title",
    "description",
    "components",
    "created",
    "fix_sha",
    "fix_date",
    "parent_sha",
    "gold_files",
    "n_gold",
    "n_candidates",
    "split",
]


def assign_split(created: pd.Series, cfg: Config) -> pd.Series:
    """Label each bug train/val/test by creation order, oldest first.

    Boundaries are rounded, not truncated: ``0.7 + 0.1`` is ``0.7999…`` in binary
    and would silently move the validation edge by one bug.
    """
    n = len(created)
    train_end = round(cfg.split.train_frac * n)
    val_end = round((cfg.split.train_frac + cfg.split.val_frac) * n)

    order = created.rank(method="first").astype(int) - 1
    labels = pd.Series("test", index=created.index)
    labels[order < val_end] = "val"
    labels[order < train_end] = "train"
    return labels


def build(cfg: Config, bugs: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """The final bug table with a ``split`` column, sorted by date."""
    text = issues[["key", "summary", "description", "components", "created"]]
    df = bugs.merge(text, on="key", how="left").rename(columns={"summary": "title"})
    if df["title"].isna().any():
        raise ValueError("some bugs have no Jira row; rerun collect_issues")

    df["created"] = pd.to_datetime(df["created"], utc=True, format="mixed")
    df = df.sort_values(["project", "created", "key"]).reset_index(drop=True)
    df["split"] = df.groupby("project", group_keys=False)["created"].apply(
        lambda created: assign_split(created, cfg)
    )
    return df[COLUMNS]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Size, date range and Component coverage of each project's three parts."""
    rows = []
    for (project, split), group in df.groupby(["project", "split"]):
        rows.append(
            {
                "project": project,
                "split": split,
                "bugs": len(group),
                "share": round(len(group) / (df["project"] == project).sum(), 3),
                "first_created": group["created"].min().date().isoformat(),
                "last_created": group["created"].max().date().isoformat(),
                "mean_gold": round(group["n_gold"].mean(), 2),
                "mean_candidates": round(group["n_candidates"].mean(), 1),
                "with_component": int(group["components"].apply(len).gt(0).sum()),
            }
        )
    order = {"train": 0, "val": 1, "test": 2}
    out = pd.DataFrame(rows)
    return out.sort_values(["project", "split"], key=lambda s: s.map(order).fillna(s))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    bugs = pd.read_parquet(cfg.paths.data_processed / BUGS_NAME)
    issues = pd.read_parquet(cfg.paths.data_raw / ISSUES_NAME)

    df = build(cfg, bugs, issues)
    summary = summarise(df)

    df.to_parquet(ensure_dir(cfg.paths.data_processed) / OUTPUT_NAME, index=False)
    summary.to_csv(ensure_dir(cfg.paths.tables) / "13_splits.csv", index=False)
    logger.info("wrote %d bugs to %s", len(df), OUTPUT_NAME)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

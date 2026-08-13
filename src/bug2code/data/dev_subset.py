r"""Fixed fractional subsets of the temporal splits.

Phases 2 and 3 (TF-IDF, frozen CodeBERT) are evaluated on a small, reproducible
slice of each split rather than the full dataset, to keep early experiment
iterations fast. The default use (no ``--projects``/``--splits``) reproduces
the original 30% TRAIN+VAL dev subset across all 4 projects; TEST is excluded
by default since that subset is for development only.

``--projects`` and ``--splits`` let a reduced single-project design (e.g. the
Cassandra-30% experiment) sample all three splits, including TEST, the same
deterministic way. The existing temporal split is never re-derived here -
sampling always happens independently *within* each already-fixed split, so
train/val/test membership and their temporal ordering are unchanged.

Sampling is per project and per split, with the project's configured seed, so
the subset is exactly reproducible and each project keeps its own share.

Usage:
    python -m bug2code.data.dev_subset [--config configs/exp.yaml] [--frac 0.3]
    python -m bug2code.data.dev_subset --config configs/cassandra30.yaml \\
        --frac 0.3 --projects cassandra --splits train val test
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import Config, load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "dev_subset.parquet"


def sample_dev_subset(
    df: pd.DataFrame,
    frac: float,
    seed: int,
    splits: tuple[str, ...] = ("train", "val"),
    projects: list[str] | None = None,
) -> pd.DataFrame:
    """Sample ``frac`` of the given splits, per project, with a fixed seed.

    Splits not listed in ``splits`` are dropped entirely. ``projects`` narrows
    to a subset of projects (default: all projects present in ``df``).
    """
    dev = df[df["split"].isin(splits)]
    if projects is not None:
        dev = dev[dev["project"].isin(projects)]
    parts = [
        group.sample(frac=frac, random_state=seed) for _, group in dev.groupby(["project", "split"])
    ]
    return pd.concat(parts).sort_values(["project", "split", "created"])


def summarise(subset: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Per-project, per-split bug counts: sampled vs the full split."""
    rows = []
    for (project, split), group in subset.groupby(["project", "split"]):
        total = ((full["project"] == project) & (full["split"] == split)).sum()
        rows.append(
            {
                "project": project,
                "split": split,
                "sampled_bugs": len(group),
                "full_split_bugs": int(total),
                "fraction": round(len(group) / total, 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--frac", type=float, default=0.3, help="share of each split to keep")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], help="splits to sample from"
    )
    parser.add_argument(
        "--projects", nargs="+", default=None, help="projects to keep (default: all)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg: Config = load_config(args.config)

    df = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    subset = sample_dev_subset(df, args.frac, cfg.seed, tuple(args.splits), args.projects)
    summary = summarise(subset, df)

    out_name = cfg.dev_subset_name
    subset[["project", "key", "split"]].to_parquet(
        ensure_dir(cfg.paths.data_processed) / out_name, index=False
    )
    logger.info("wrote %d dev-subset bug keys to %s", len(subset), out_name)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

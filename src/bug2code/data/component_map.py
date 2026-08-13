"""Learn the Component → files mapping used by RQ4, from training bugs only.

A Jira Component is a coarse area label an issue carries, e.g. ``SQL`` or
``Replication``. Condition B of the Component experiment ranks a bug
only against the files that fixes of *training* bugs with the same Component
touched. The mapping is therefore built here, once, from the train split alone:
using validation or test fixes would leak their answers into the candidate set.

Like every other phase script (``candidates.py``, ``tfidf.py``, ``frozen_codebert.py``,
``train_codebert.py``, ``validate_finetuned.py``), the population is narrowed to
``cfg.dev_subset_name`` before splitting into train/val, so a reduced-scope config
(e.g. ``configs/cassandra30.yaml``) builds the mapping from exactly its own TRAIN
subset (449 Cassandra bugs), never the full 1,497-bug Cassandra train split. The
output filename is derived from the subset name so a non-default subset never
overwrites the historical ``component_files.parquet`` built for the original
4-project design.

The Component is only ever a filter. It is never model input.

Usage:
    python -m bug2code.data.component_map [--config configs/exp.yaml]
    python -m bug2code.data.component_map --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "component_files.parquet"
DEFAULT_SUBSET_NAME = "dev_subset.parquet"

# Cassandra30 Train is fixed at 449 bugs (30% of the 1,497-bug temporal TRAIN
# split); a mismatch here means the wrong population fed the mapping.
CASSANDRA30_EXPECTED_TRAIN_BUGS = 449


def component_map_output_name(dev_subset_name: str) -> str:
    """Historical filename for the default subset; a subset-specific one otherwise.

    ``dev_subset.parquet`` -> ``component_files.parquet`` (unchanged, so the
    already-built 4-project artifact is never silently overwritten with
    different content under the same name). Any other subset, e.g.
    ``cassandra30_subset.parquet`` -> ``component_files_cassandra30.parquet``.
    """
    if dev_subset_name == DEFAULT_SUBSET_NAME:
        return OUTPUT_NAME
    stem = Path(dev_subset_name).stem.removesuffix("_subset")
    return f"component_files_{stem}.parquet"


def build_map(train: pd.DataFrame, min_bugs: int) -> pd.DataFrame:
    """One row per (project, component): the files its training fixes touched.

    A training bug listing several Components contributes its gold files to each
    of them, mirroring the union rule applied at prediction time.
    """
    pairs = train.explode("components").dropna(subset=["components"])
    rows = []
    for (project, component), group in pairs.groupby(["project", "components"]):
        if len(group) < min_bugs:
            continue
        files = sorted(set().union(*group["gold_files"]))
        rows.append(
            {
                "project": project,
                "component": component,
                "train_bugs": len(group),
                "n_files": len(files),
                "files": files,
            }
        )
    return pd.DataFrame(rows)


def candidate_files(mapping: pd.DataFrame, project: str, components: list[str]) -> set[str]:
    """Union of the mapped files of a bug's Components; empty if none are known."""
    rows = mapping[(mapping["project"] == project) & mapping["component"].isin(components)]
    return set().union(*rows["files"]) if len(rows) else set()


def evaluate_coverage(mapping: pd.DataFrame, bugs: pd.DataFrame) -> pd.DataFrame:
    """Per project: how often the mapped candidate set contains a gold file.

    Coverage is the ceiling on Condition B — a gold file outside the candidate
    set can never be retrieved, however good the ranker is. Bugs with no
    Component are ineligible; bugs whose Components are all unseen in training
    get an empty candidate set and count as not covered.

    Sizes here are before intersecting with each bug's snapshot, so they are an
    upper bound on the real candidate set. Coverage itself is unaffected, since
    every gold file exists in its own snapshot by construction.
    """
    rows = []
    for project, group in bugs.groupby("project"):
        eligible = group[group["components"].apply(len) > 0]
        if not len(eligible):
            continue
        sets = [
            candidate_files(mapping, project, list(b.components)) for b in eligible.itertuples()
        ]
        covered = [
            bool(set(bug.gold_files) & files)
            for bug, files in zip(eligible.itertuples(), sets, strict=True)
        ]
        sizes = pd.Series([len(files) for files in sets])
        rows.append(
            {
                "project": project,
                "bugs": len(group),
                "eligible": len(eligible),
                "no_component": len(group) - len(eligible),
                "empty_candidate_set": int((sizes == 0).sum()),
                "mean_candidates": round(sizes.mean(), 1),
                "median_candidates": int(sizes.median()),
                "coverage": round(sum(covered) / len(covered), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    df = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = df.merge(dev_keys[["project", "key"]], on=["project", "key"])
    train = bugs[bugs["split"] == "train"]
    val = bugs[bugs["split"] == "val"]

    if cfg.dev_subset_name == "cassandra30_subset.parquet":
        assert len(train) == CASSANDRA30_EXPECTED_TRAIN_BUGS, (
            f"Cassandra30 Train subset should contain {CASSANDRA30_EXPECTED_TRAIN_BUGS} bugs "
            f"for the Component mapping, got {len(train)} - check cassandra30_subset.parquet"
        )

    mapping = build_map(train, cfg.localization["component_filter"]["min_bugs_per_component"])
    logger.info("%d components mapped from %d training bugs", len(mapping), len(train))

    coverage = evaluate_coverage(mapping, val)

    output_name = component_map_output_name(cfg.dev_subset_name)
    mapping.to_parquet(ensure_dir(cfg.paths.data_processed) / output_name, index=False)
    coverage.to_csv(ensure_dir(cfg.paths.tables) / "14_component_map.csv", index=False)

    n_with_component = int((train["components"].apply(len) > 0).sum())
    n_mapped_files = len(set().union(*mapping["files"])) if len(mapping) else 0
    n_val_eligible = int((val["components"].apply(len) > 0).sum())

    print(f"\n=== component map ({cfg.dev_subset_name}, train bugs only -> {output_name}) ===")
    print(f"train bugs used: {len(train)}")
    print(f"train bugs with >=1 known Component: {n_with_component}")
    print(f"unique Components mapped: {len(mapping)}")
    print(f"mapped source files (union across all components): {n_mapped_files}")
    print(f"validation bugs with >=1 known Component: {n_val_eligible} / {len(val)}")
    print(mapping.groupby("project").agg(components=("component", "size")).to_string())
    print("\n=== validation coverage (ceiling for Condition B) ===")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()

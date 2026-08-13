"""Component experiment: Full vs Component-filtered candidates.

Reuses the saved epoch-3 Fine-tuned CodeBERT Cassandra30 Validation candidate
scores (``save_finetuned_val_scores.py``) - no re-encoding here. Only the
candidate *set* changes between the two conditions; the model scores
themselves are untouched. Evaluated over the paired population: the same
Cassandra30 Validation bugs with at least one known Jira Component, in both
conditions.

- Condition FULL: every candidate in the bug's normal snapshot.
- Condition COMPONENT_FILTERED: intersected with the Cassandra30-TRAIN-derived
  Component -> files mapping (``component_files_cassandra30.parquet``) for the
  bug's known Component(s), via ``component_map.candidate_files`` - the same
  filtering function Condition B was always meant to use, not a new one.

No predicted-Component model, no Component classifier, no soft/partial
Component matching: the Component is read directly from
Jira and only ever narrows the candidate set.

Usage:
    python -m bug2code.localization.component_experiment --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import load_config
from bug2code.data.component_map import candidate_files, component_map_output_name
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import ranked_paths
from bug2code.localization.metrics import score_one
from bug2code.localization.save_finetuned_val_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

RESULTS_NAME = "component_val_results.csv"
PER_BUG_NAME = "component_val_per_bug.csv"


def evaluate_bug(
    project: str,
    key: str,
    bug_scores: pd.DataFrame,
    gold_files: list[str],
    mapping: pd.DataFrame,
    components: list[str],
) -> dict:
    """FULL vs COMPONENT_FILTERED metrics for one bug, same saved scores both times."""
    full_ranked = ranked_paths(bug_scores)
    full_metrics = score_one(full_ranked, gold_files)

    allowed = candidate_files(mapping, project, components)
    filtered_scores = bug_scores[bug_scores["path"].isin(allowed)]
    filtered_ranked = ranked_paths(filtered_scores) if len(filtered_scores) else []
    filtered_metrics = score_one(filtered_ranked, gold_files)

    row = {
        "project": project,
        "key": key,
        "n_candidates_full": len(full_ranked),
        "n_candidates_filtered": len(filtered_ranked),
        "component_candidate_set_empty": len(filtered_ranked) == 0,
    }
    row |= {f"full_{metric}": value for metric, value in full_metrics.items()}
    row |= {f"filtered_{metric}": value for metric, value in filtered_metrics.items()}
    return row


def reduction_stats(per_bug: pd.DataFrame) -> pd.Series:
    """Candidate-set-size reduction stats: mean/median/min/max/% reduction."""
    reduction_frac = 1 - (per_bug["n_candidates_filtered"] / per_bug["n_candidates_full"])
    return pd.Series(
        {
            "mean_n_full": per_bug["n_candidates_full"].mean(),
            "mean_n_filtered": per_bug["n_candidates_filtered"].mean(),
            "median_n_filtered": per_bug["n_candidates_filtered"].median(),
            "min_n_filtered": per_bug["n_candidates_filtered"].min(),
            "max_n_filtered": per_bug["n_candidates_filtered"].max(),
            "mean_pct_reduction": round(100 * reduction_frac.mean(), 2),
            "median_pct_reduction": round(100 * reduction_frac.median(), 2),
            "min_pct_reduction": round(100 * reduction_frac.min(), 2),
            "max_pct_reduction": round(100 * reduction_frac.max(), 2),
        }
    )


def summarise(per_bug: pd.DataFrame) -> pd.DataFrame:
    """One row per condition (FULL / COMPONENT_FILTERED) with mean metrics."""
    metric_names = [c.removeprefix("full_") for c in per_bug.columns if c.startswith("full_")]
    rows = []
    for condition, prefix in (("FULL", "full_"), ("COMPONENT_FILTERED", "filtered_")):
        row = {"condition": condition, "n_bugs": len(per_bug)}
        row |= {metric: per_bug[f"{prefix}{metric}"].mean() for metric in metric_names}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--epoch", type=int, default=3, help="which saved score table to use")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    val_bugs = bugs[bugs["split"] == "val"]

    eligible = val_bugs[val_bugs["components"].apply(len) > 0]
    logger.info(
        "%d/%d Cassandra30 Validation bugs have a known Jira Component (paired population)",
        len(eligible),
        len(val_bugs),
    )

    scores_path = cfg.paths.tables / CODEBERT_SCORES_TEMPLATE.format(epoch=args.epoch)
    scores = pd.read_parquet(scores_path)

    mapping_path = cfg.paths.data_processed / component_map_output_name(cfg.dev_subset_name)
    mapping = pd.read_parquet(mapping_path)

    per_bug_rows = []
    for bug in eligible.itertuples():
        bug_scores = scores[(scores["project"] == bug.project) & (scores["key"] == bug.key)]
        if bug_scores.empty:
            raise ValueError(f"no saved CodeBERT scores for ({bug.project}, {bug.key})")
        per_bug_rows.append(
            evaluate_bug(
                bug.project,
                bug.key,
                bug_scores,
                list(bug.gold_files),
                mapping,
                list(bug.components),
            )
        )
    per_bug = pd.DataFrame(per_bug_rows)

    n_empty = int(per_bug["component_candidate_set_empty"].sum())
    if n_empty:
        logger.warning(
            "%d/%d eligible bugs have an EMPTY Component-filtered candidate set "
            "(all their Component(s) are unseen or too rare in Cassandra30 Train); "
            "scored as 0 on every metric for COMPONENT_FILTERED, not silently dropped or "
            "backfilled with extra candidates",
            n_empty,
            len(per_bug),
        )

    stats = reduction_stats(per_bug)
    print("\n=== candidate-set reduction (FULL -> COMPONENT_FILTERED) ===")
    print(stats.to_string())
    print(f"\nbugs with an empty Component-filtered candidate set: {n_empty}/{len(per_bug)}")

    summary = summarise(per_bug)
    print("\n=== FULL vs COMPONENT_FILTERED (mean over paired eligible bugs) ===")
    print(summary.round(4).to_string(index=False))

    per_bug.to_csv(ensure_dir(cfg.paths.tables) / PER_BUG_NAME, index=False)
    summary.to_csv(ensure_dir(cfg.paths.tables) / RESULTS_NAME, index=False)
    logger.info("wrote %s and %s", RESULTS_NAME, PER_BUG_NAME)


if __name__ == "__main__":
    main()

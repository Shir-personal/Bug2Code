"""Qualitative Cassandra30 Validation error analysis (CPU-only, no CodeBERT).

Reads the already-saved Validation candidate-score tables and the already-
built Component mapping - no model is loaded and no inference runs here. For
every Cassandra30 Validation bug it derives the best-placed gold file's rank
under four conditions:

- ``tfidf_rank``: from ``tfidf_val_candidate_scores.parquet``.
- ``finetuned_rank``: from ``finetuned_epoch{N}_val_candidate_scores.parquet``.
- ``hybrid_rank``: alpha=0.5 combination of the two tables above, the same
  min-max normalization as ``hybrid_experiment.py``.
- ``component_full_rank`` / ``component_filtered_rank``: only for bugs with a
  known Jira Component, reusing ``component_experiment.evaluate_bug``'s FULL
  vs COMPONENT_FILTERED split of the Fine-tuned candidate scores.

A rank is ``None`` when no gold file appears in that condition's candidate
set (e.g. an empty Component-filtered set, or the model simply misses).
Validation only - this script never reads a Test artifact, so it cannot be
used to hand-pick Test qualitative examples.

Usage:
    python -m bug2code.localization.error_analysis --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import load_config
from bug2code.data.component_map import candidate_files, component_map_output_name
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import ranked_paths
from bug2code.localization.hybrid_experiment import normalize_per_bug
from bug2code.localization.metrics import best_gold_rank
from bug2code.localization.save_finetuned_val_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.localization.save_tfidf_val_scores import OUTPUT_NAME as TFIDF_SCORES_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

ALPHA = 0.5
OUTPUT_NAME = "error_analysis_val_per_bug.csv"


def rank_from_bug_scores(bug_scores: pd.DataFrame, gold_files: list[str]) -> int | None:
    """Best gold-file rank for one bug's raw score rows, or None if absent/empty."""
    if bug_scores.empty:
        return None
    return best_gold_rank(ranked_paths(bug_scores), gold_files)


def build_per_bug_table(
    val_bugs: pd.DataFrame,
    tfidf_scores: pd.DataFrame,
    codebert_scores: pd.DataFrame,
    hybrid_scores: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """One row per Validation bug with a rank per condition (None where absent)."""
    tfidf_by_bug = {k: g for k, g in tfidf_scores.groupby(["project", "key"])}
    codebert_by_bug = {k: g for k, g in codebert_scores.groupby(["project", "key"])}
    hybrid_by_bug = {k: g for k, g in hybrid_scores.groupby(["project", "key"])}

    rows = []
    for bug in val_bugs.itertuples():
        key = (bug.project, bug.key)
        gold = list(bug.gold_files)

        row = {
            "project": bug.project,
            "key": bug.key,
            "tfidf_rank": rank_from_bug_scores(tfidf_by_bug.get(key, pd.DataFrame()), gold),
            "finetuned_rank": rank_from_bug_scores(codebert_by_bug.get(key, pd.DataFrame()), gold),
            "hybrid_rank": rank_from_bug_scores(hybrid_by_bug.get(key, pd.DataFrame()), gold),
        }

        components = list(bug.components)
        if components:
            bug_codebert = codebert_by_bug.get(key, pd.DataFrame())
            allowed = candidate_files(mapping, bug.project, components)
            filtered = (
                bug_codebert[bug_codebert["path"].isin(allowed)]
                if len(bug_codebert)
                else bug_codebert
            )
            row["component_full_rank"] = rank_from_bug_scores(bug_codebert, gold)
            row["component_filtered_rank"] = rank_from_bug_scores(filtered, gold)
        else:
            row["component_full_rank"] = None
            row["component_filtered_rank"] = None

        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--epoch", type=int, default=3, help="which saved CodeBERT score table")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    val_bugs = bugs[bugs["split"] == "val"]
    logger.info("%d Cassandra30 Validation bugs (Test is never read here)", len(val_bugs))

    tfidf_scores = pd.read_parquet(cfg.paths.tables / TFIDF_SCORES_NAME)
    codebert_scores = pd.read_parquet(
        cfg.paths.tables / CODEBERT_SCORES_TEMPLATE.format(epoch=args.epoch)
    )

    merged = tfidf_scores.merge(
        codebert_scores, on=["project", "key", "path"], suffixes=("_tfidf", "_codebert")
    )
    merged = normalize_per_bug(merged, "score_tfidf", "tfidf_norm")
    merged = normalize_per_bug(merged, "score_codebert", "codebert_norm")
    hybrid_scores = merged.assign(
        candidate_order=merged["candidate_order_tfidf"],
        score=ALPHA * merged["codebert_norm"] + (1 - ALPHA) * merged["tfidf_norm"],
    )[["project", "key", "path", "candidate_order", "score"]]

    mapping_path = cfg.paths.data_processed / component_map_output_name(cfg.dev_subset_name)
    mapping = pd.read_parquet(mapping_path)

    per_bug = build_per_bug_table(val_bugs, tfidf_scores, codebert_scores, hybrid_scores, mapping)

    out_path = ensure_dir(cfg.paths.tables) / OUTPUT_NAME
    per_bug.to_csv(out_path, index=False)
    logger.info("wrote %s", out_path)

    print("\n=== Cassandra30 Validation error analysis: per-bug ranks ===")
    print(per_bug.head(20).to_string(index=False))

    tfidf_only_hit = per_bug[
        per_bug["tfidf_rank"].notna()
        & (per_bug["tfidf_rank"] <= 10)
        & (per_bug["finetuned_rank"].isna() | (per_bug["finetuned_rank"] > 10))
    ]
    finetuned_only_hit = per_bug[
        per_bug["finetuned_rank"].notna()
        & (per_bug["finetuned_rank"] <= 10)
        & (per_bug["tfidf_rank"].isna() | (per_bug["tfidf_rank"] > 10))
    ]
    print(f"\nTF-IDF hit@10 but Fine-tuned missed (rank>10 or absent): {len(tfidf_only_hit)}")
    print(f"Fine-tuned hit@10 but TF-IDF missed (rank>10 or absent): {len(finetuned_only_hit)}")


if __name__ == "__main__":
    main()

"""Component Test (FINAL): Full vs Component-filtered candidates on Cassandra30 Test.

Reuses the saved epoch-3 Fine-tuned CodeBERT Cassandra30 Test candidate scores
(``save_finetuned_test_scores.py``) - no re-encoding here. Only the candidate
*set* changes between the two conditions; the model scores themselves are
untouched. Evaluated over the paired population: the same Cassandra30 Test
bugs with at least one known Jira Component, in both conditions.

- Condition FULL: every candidate in the bug's normal snapshot.
- Condition COMPONENT_FILTERED: intersected with the Cassandra30-TRAIN-derived
  Component -> files mapping (``component_files_cassandra30.parquet``, built
  once by ``bug2code.data.component_map`` from TRAIN only) for the bug's known
  Component(s), via ``component_map.candidate_files`` - the exact same mapping
  and filtering function the Validation Component experiment used. The mapping
  is never re-learned or updated from Validation or Test.

Empty Component-filtered candidate sets are kept in the evaluation and score
zero on every metric, exactly as in Validation - never dropped or backfilled.

Usage:
    python -m bug2code.localization.component_test_experiment --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import load_config
from bug2code.data.component_map import component_map_output_name
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.component_experiment import evaluate_bug, reduction_stats, summarise
from bug2code.localization.save_finetuned_test_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

RESULTS_NAME = "component_test_results.csv"
PER_BUG_NAME = "component_test_per_bug.csv"


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
    test_bugs = bugs[bugs["split"] == "test"]

    eligible = test_bugs[test_bugs["components"].apply(len) > 0]
    logger.info(
        "%d/%d Cassandra30 Test bugs have a known Jira Component (paired population)",
        len(eligible),
        len(test_bugs),
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
    print("\n=== candidate-set reduction (FULL -> COMPONENT_FILTERED), Cassandra30 TEST ===")
    print(stats.to_string())
    print(f"\nbugs with an empty Component-filtered candidate set: {n_empty}/{len(per_bug)}")

    summary = summarise(per_bug)
    print("\n=== FULL vs COMPONENT_FILTERED (mean over paired eligible Test bugs, FINAL) ===")
    print(summary.round(4).to_string(index=False))

    per_bug.to_csv(ensure_dir(cfg.paths.tables) / PER_BUG_NAME, index=False)
    summary.to_csv(ensure_dir(cfg.paths.tables) / RESULTS_NAME, index=False)
    logger.info("wrote %s and %s", RESULTS_NAME, PER_BUG_NAME)


if __name__ == "__main__":
    main()

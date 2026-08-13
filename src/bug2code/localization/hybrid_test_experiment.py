"""Hybrid Test (FINAL): TF-IDF + Fine-tuned CodeBERT at the Validation-selected alpha.

``hybrid_score = alpha * codebert_norm + (1 - alpha) * tfidf_norm``, exactly the
same combination rule and the same per-bug min-max normalization
(``hybrid_experiment.normalize_per_bug``, reused unchanged) as the Validation
alpha sweep. Reuses the two already-saved Cassandra30 Test candidate-score
tables (``save_tfidf_test_scores.py``, ``save_finetuned_test_scores.py``) - no
re-encoding, no re-fitting TF-IDF, no further CodeBERT inference.

Alpha is NOT a CLI argument here, deliberately: it is fixed at the value
already selected from Cassandra30 Validation MRR (``hybrid_experiment.py``),
so there is no way to invoke this script with a Test-tuned alpha by accident.

Usage:
    python -m bug2code.localization.hybrid_test_experiment --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores
from bug2code.localization.hybrid_experiment import load_and_join, normalize_per_bug
from bug2code.localization.save_finetuned_test_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.localization.save_tfidf_test_scores import OUTPUT_NAME as TFIDF_SCORES_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

# Selected on Cassandra30 Validation MRR only (hybrid_experiment.py); fixed
# here, not re-tuned on Test.
ALPHA = 0.5
RESULTS_NAME = "hybrid_test_results.csv"


def hybrid_scores_at_alpha(merged: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Combine the two normalized score columns at a fixed ``alpha`` (no sweep)."""
    return merged.assign(
        score=alpha * merged["codebert_norm"] + (1 - alpha) * merged["tfidf_norm"]
    )[["project", "key", "path", "candidate_order", "score"]]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--epoch", type=int, default=3, help="which CodeBERT score table to use")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    test_bugs = bugs[bugs["split"] == "test"]
    gold = test_bugs[["project", "key", "gold_files"]]

    tfidf_path = cfg.paths.tables / TFIDF_SCORES_NAME
    codebert_path = cfg.paths.tables / CODEBERT_SCORES_TEMPLATE.format(epoch=args.epoch)
    merged = load_and_join(tfidf_path, codebert_path)
    merged = normalize_per_bug(merged, "score_tfidf", "tfidf_norm")
    merged = normalize_per_bug(merged, "score_codebert", "codebert_norm")

    hybrid_scores = hybrid_scores_at_alpha(merged, ALPHA)
    per_bug = metrics_from_scores(hybrid_scores, gold)
    metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
    mean_metrics = per_bug[metric_cols].mean()

    print(f"\n=== Hybrid alpha={ALPHA} Cassandra30 TEST metrics (FINAL, fixed from Validation) ===")
    print(mean_metrics.round(4).to_string())
    print(f"\nevaluated bugs: {len(per_bug)} / {len(test_bugs)}")

    summary = pd.DataFrame([{"alpha": ALPHA} | mean_metrics.to_dict()])
    summary.to_csv(ensure_dir(cfg.paths.tables) / RESULTS_NAME, index=False)
    logger.info("wrote %s", RESULTS_NAME)


if __name__ == "__main__":
    main()

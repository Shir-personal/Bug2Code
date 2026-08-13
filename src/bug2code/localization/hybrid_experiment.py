"""Hybrid extension: TF-IDF + Fine-tuned CodeBERT, alpha selected on Validation MRR.

``hybrid_score = alpha * codebert_norm + (1 - alpha) * tfidf_norm``, where each
model's raw score is independently min-max normalized *per bug* first (so the
two differently-scaled models are comparable before combining). Reuses the two
already-saved candidate-score tables (``save_tfidf_val_scores.py``,
``save_finetuned_val_scores.py``) - no re-encoding, no re-fitting TF-IDF.
Alpha is chosen only by Cassandra30 Validation MRR; this is an additional
performance comparison alongside Fine-tuned CodeBERT, not a redefinition of
which checkpoint the Component or cross-project experiments use.

Endpoint sanity checks: alpha=0.0 must reproduce the known TF-IDF Validation
metrics exactly (min-max is order-preserving, so ranking is unchanged);
alpha=1.0 must reproduce the known Fine-tuned-CodeBERT epoch-3 Validation
metrics exactly. Either failing stops the script before the sweep is trusted.

Usage:
    python -m bug2code.localization.hybrid_experiment --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores
from bug2code.localization.save_finetuned_val_scores import (
    EXPECTED_METRICS as CODEBERT_EXPECTED_METRICS,
)
from bug2code.localization.save_finetuned_val_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.localization.save_tfidf_val_scores import EXPECTED_METRICS as TFIDF_EXPECTED_METRICS
from bug2code.localization.save_tfidf_val_scores import OUTPUT_NAME as TFIDF_SCORES_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
RESULTS_NAME = "hybrid_val_alpha_sweep.csv"
TOLERANCE = 1e-3  # the EXPECTED_METRICS constants are rounded to 4dp


def normalize_per_bug(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """Per-(project, key) min-max normalization; 0.0 for a bug with a flat score."""
    grouped = df.groupby(["project", "key"])[score_col]
    lo, hi = grouped.transform("min"), grouped.transform("max")
    span = hi - lo
    norm = (df[score_col] - lo) / span
    return df.assign(**{out_col: norm.where(span != 0, 0.0)})


def load_and_join(tfidf_path, codebert_path) -> pd.DataFrame:
    """Join the two saved score tables on (project, key, path)."""
    tfidf = pd.read_parquet(tfidf_path)
    codebert = pd.read_parquet(codebert_path)
    merged = tfidf.merge(
        codebert, on=["project", "key", "path"], suffixes=("_tfidf", "_codebert")
    )
    if len(merged) != len(tfidf) or len(merged) != len(codebert):
        raise ValueError(
            f"TF-IDF ({len(tfidf)} rows) and CodeBERT ({len(codebert)} rows) candidate-score "
            f"tables don't have matching (project, key, path) populations - joined to "
            f"{len(merged)} rows. They must share the exact same per-bug candidate set."
        )
    if not (merged["candidate_order_tfidf"] == merged["candidate_order_codebert"]).all():
        raise ValueError(
            "TF-IDF and CodeBERT candidate orderings disagree for at least one bug - "
            "ranking from the joined table would not be reproducible."
        )
    return merged.rename(columns={"candidate_order_tfidf": "candidate_order"}).drop(
        columns=["candidate_order_codebert"]
    )


def sweep(merged: pd.DataFrame, gold: pd.DataFrame, alphas=ALPHA_GRID) -> pd.DataFrame:
    """Mean Validation metrics at each alpha in the grid."""
    rows = []
    for alpha in alphas:
        hybrid_scores = merged.assign(
            score=alpha * merged["codebert_norm"] + (1 - alpha) * merged["tfidf_norm"]
        )[["project", "key", "path", "candidate_order", "score"]]
        per_bug = metrics_from_scores(hybrid_scores, gold)
        metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
        row = {"alpha": alpha} | per_bug[metric_cols].mean().to_dict()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["is_best_by_val_mrr"] = summary["mrr"] == summary["mrr"].max()
    return summary


def verify_endpoint(summary: pd.DataFrame, alpha: float, expected: dict, label: str) -> None:
    """Raise if the sweep's row at ``alpha`` doesn't match the known standalone metrics."""
    row = summary.loc[summary["alpha"] == alpha].iloc[0]
    mismatches = {
        metric: (float(row[metric]), value)
        for metric, value in expected.items()
        if abs(float(row[metric]) - value) > TOLERANCE
    }
    if mismatches:
        raise AssertionError(
            f"alpha={alpha} does not reproduce the known {label} Validation result "
            f"(tolerance={TOLERANCE}): {mismatches}. STOP and fix before trusting intermediate "
            "alphas."
        )
    logger.info("alpha=%.2f endpoint matches the known %s result exactly", alpha, label)


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
    val_bugs = bugs[bugs["split"] == "val"]
    gold = val_bugs[["project", "key", "gold_files"]]

    tfidf_path = cfg.paths.tables / TFIDF_SCORES_NAME
    codebert_path = cfg.paths.tables / CODEBERT_SCORES_TEMPLATE.format(epoch=args.epoch)
    merged = load_and_join(tfidf_path, codebert_path)
    merged = normalize_per_bug(merged, "score_tfidf", "tfidf_norm")
    merged = normalize_per_bug(merged, "score_codebert", "codebert_norm")

    summary = sweep(merged, gold)

    verify_endpoint(summary, 0.0, TFIDF_EXPECTED_METRICS, "TF-IDF")
    verify_endpoint(summary, 1.0, CODEBERT_EXPECTED_METRICS, "Fine-tuned CodeBERT epoch-3")

    print("\n=== Hybrid alpha sweep (Cassandra30 Validation) ===")
    print(summary.round(4).to_string(index=False))

    best_alpha = summary.loc[summary["is_best_by_val_mrr"], "alpha"].iloc[0]
    print(f"\nSelected alpha = {best_alpha} based only on Validation MRR.")

    summary.to_csv(ensure_dir(cfg.paths.tables) / RESULTS_NAME, index=False)
    logger.info("wrote %s", RESULTS_NAME)


if __name__ == "__main__":
    main()

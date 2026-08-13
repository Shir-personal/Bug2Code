"""Save raw per-(bug, candidate-file) TF-IDF scores on Cassandra30 Validation.

Reuses ``tfidf.py``'s exact vocabulary/IDF fit (``fit_project``, fit on the
Cassandra30 TRAIN candidate files only) and query text (``bug_text``) so the
scores here are identical to a live ``tfidf.py`` run - only the output
differs: every candidate's raw score instead of just the ranking metrics.
Candidates are listed in the same per-bug order as
``candidate_scores.score_codebert_candidates`` (both read it from the same
``snapshot_index.parquet`` via ``file_version_id``), so the two score tables
join safely on (project, key, path) for the Hybrid extension.

Usage:
    python -m bug2code.localization.save_tfidf_val_scores --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME
from bug2code.localization.tfidf import score_candidates
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "tfidf_val_candidate_scores.parquet"

# Known-good TF-IDF Validation result (tfidf.py); the saved table must reproduce it exactly.
EXPECTED_METRICS = {
    "hit_at_1": 0.2969,
    "hit_at_5": 0.5312,
    "hit_at_10": 0.6562,
    "hit_at_20": 0.7344,
    "recall_at_1": 0.1995,
    "recall_at_5": 0.3880,
    "recall_at_10": 0.5123,
    "recall_at_20": 0.5963,
    "mrr": 0.4161,
    "map": 0.3331,
}
TOLERANCE = 1e-3  # EXPECTED_METRICS are rounded to 4dp; allow for that rounding


def verify_against_expected(mean_metrics: pd.Series) -> None:
    """Raise if the recomputed metrics don't match the known TF-IDF result."""
    mismatches = {
        metric: (float(mean_metrics[metric]), expected)
        for metric, expected in EXPECTED_METRICS.items()
        if abs(float(mean_metrics[metric]) - expected) > TOLERANCE
    }
    if mismatches:
        raise AssertionError(
            "Metrics recomputed from the saved TF-IDF candidate-score table do not match "
            f"the known Validation result (tolerance={TOLERANCE}): {mismatches}. "
            "STOP - do not proceed to Hybrid until this is resolved."
        )
    logger.info("saved TF-IDF score metrics match the known Validation result exactly")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    t0 = time.time()
    frames = [
        score_candidates(cfg, project, group, versions, index, split="val")
        for project, group in bugs.groupby("project")
    ]
    scores = pd.concat(frames, ignore_index=True)
    elapsed = time.time() - t0
    logger.info("TF-IDF candidate scoring finished in %.1fs", elapsed)

    out_path = ensure_dir(cfg.paths.tables) / OUTPUT_NAME
    scores.to_parquet(out_path, index=False)
    logger.info("wrote %s", out_path)

    val_bugs = bugs[bugs["split"] == "val"]
    gold = val_bugs[["project", "key", "gold_files"]]
    per_bug = metrics_from_scores(scores, gold)
    metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
    mean_metrics = per_bug[metric_cols].mean()

    print("\n=== metrics recomputed from the saved TF-IDF candidate-score table ===")
    print(mean_metrics.round(4).to_string())

    verify_against_expected(mean_metrics)


if __name__ == "__main__":
    main()

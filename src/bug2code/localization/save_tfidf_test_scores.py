"""Save raw per-(bug, candidate-file) TF-IDF scores on Cassandra30 Test (FINAL).

This is the one and only TF-IDF Test scoring pass: fit on the Cassandra30
TRAIN candidate files only (``fit_project``, unchanged from ``tfidf.py``),
score every Cassandra30 TEST bug's own candidate set, and save every
candidate's raw score - not just the ranking metrics. Hybrid Test
(``hybrid_test_experiment.py``) reads the saved table below instead of
re-fitting or re-scoring TF-IDF. VAL is never read here.

Candidates are listed in the same per-bug order as
``candidate_scores.score_codebert_candidates`` (both read it from the same
``snapshot_index.parquet`` via ``file_version_id``), so this table joins
safely on (project, key, path) with the saved Fine-tuned CodeBERT Test scores.

Usage:
    python -m bug2code.localization.save_tfidf_test_scores --config configs/cassandra30.yaml
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

OUTPUT_NAME = "tfidf_test_candidate_scores.parquet"


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

    train_keys_all = set(bugs.loc[bugs["split"] == "train", "key"])
    test_keys_all = set(bugs.loc[bugs["split"] == "test", "key"])
    assert not (train_keys_all & test_keys_all), "TRAIN/TEST key overlap - leakage, stop"
    logger.info(
        "%d Cassandra30 TEST bugs, TF-IDF fit on %d TRAIN bugs only (VAL is never read)",
        len(test_keys_all),
        len(train_keys_all),
    )

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    t0 = time.time()
    frames = [
        score_candidates(cfg, project, group, versions, index, split="test")
        for project, group in bugs.groupby("project")
    ]
    scores = pd.concat(frames, ignore_index=True)
    elapsed = time.time() - t0
    logger.info("TF-IDF TEST candidate scoring finished in %.1fs", elapsed)

    out_path = ensure_dir(cfg.paths.tables) / OUTPUT_NAME
    scores.to_parquet(out_path, index=False)
    logger.info("wrote %s", out_path)

    test_bugs = bugs[bugs["split"] == "test"]
    gold = test_bugs[["project", "key", "gold_files"]]
    per_bug = metrics_from_scores(scores, gold)
    metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
    mean_metrics = per_bug[metric_cols].mean()

    print("\n=== TF-IDF Cassandra30 TEST metrics (FINAL) ===")
    print(mean_metrics.round(4).to_string())
    print(f"\nevaluated bugs: {len(per_bug)} / {len(test_bugs)}")


if __name__ == "__main__":
    main()

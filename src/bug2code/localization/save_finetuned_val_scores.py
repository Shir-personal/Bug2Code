"""Save raw per-(bug, candidate-file) Fine-tuned CodeBERT scores.

Runs the selected checkpoint (epoch 3, chosen by Cassandra30 Validation MRR in
``validate_finetuned.py``) over Cassandra30 Validation exactly once, and saves
every candidate file's raw score - not just the ranking metrics. Component
filtering and the Hybrid alpha sweep both read this table instead of
re-encoding, so there is exactly one epoch-3 Validation CodeBERT inference
pass across the whole workflow. TEST is never read here.

After saving, metrics recomputed from the saved table are compared against
the already-known epoch-3 Validation result (from
``finetuned_val_summary.csv``); a mismatch stops the script rather than
silently continuing on a bad score table.

Usage:
    python -m bug2code.localization.save_finetuned_val_scores --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from transformers import AutoTokenizer

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores, score_codebert_candidates
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME
from bug2code.localization.frozen_codebert import MODEL_NAME, resolve_device
from bug2code.localization.train_codebert import CHECKPOINT_DIR
from bug2code.localization.validate_finetuned import FineTunedEncoder, load_finetuned_model
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_TEMPLATE = "finetuned_epoch{epoch}_val_candidate_scores.parquet"

# Known-good epoch-3 Validation result (validate_finetuned.py); the saved table must reproduce it exactly.
EXPECTED_METRICS = {
    "hit_at_1": 0.1094,
    "hit_at_5": 0.3281,
    "hit_at_10": 0.3594,
    "hit_at_20": 0.5312,
    "recall_at_1": 0.0510,
    "recall_at_5": 0.1762,
    "recall_at_10": 0.2080,
    "recall_at_20": 0.3461,
    "mrr": 0.2155,
    "map": 0.1344,
}
TOLERANCE = 1e-3  # EXPECTED_METRICS are rounded to 4dp; allow for that rounding


def verify_against_expected(mean_metrics: pd.Series) -> None:
    """Raise if the recomputed metrics don't match the known epoch-3 result."""
    mismatches = {
        metric: (float(mean_metrics[metric]), expected)
        for metric, expected in EXPECTED_METRICS.items()
        if abs(float(mean_metrics[metric]) - expected) > TOLERANCE
    }
    if mismatches:
        raise AssertionError(
            "Metrics recomputed from the saved candidate-score table do not match the "
            f"known epoch-3 Cassandra30 Validation result (tolerance={TOLERANCE}): {mismatches}. "
            "STOP - do not proceed to Component/Hybrid until this is resolved."
        )
    logger.info("saved-score metrics match the known epoch-3 Validation result exactly")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument(
        "--epoch", type=int, default=3, help="checkpoint epoch to score (selected: 3)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    val_bugs = bugs[bugs["split"] == "val"]
    logger.info(
        "%d Cassandra30 Validation bugs, epoch %d checkpoint only (TEST is never read)",
        len(val_bugs),
        args.epoch,
    )

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    device = resolve_device(cfg.device)
    enc_cfg = cfg.localization["encoder"]
    checkpoint_dir = cfg.paths.experiments / CHECKPOINT_DIR
    ckpt_path = checkpoint_dir / f"epoch_{args.epoch}.pt"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("device=%s checkpoint=%s", device, ckpt_path)

    model = load_finetuned_model(ckpt_path, enc_cfg["dropout"], device)
    encoder = FineTunedEncoder(model, tokenizer, device, fp16=False)

    t0 = time.time()
    frames = [
        score_codebert_candidates(cfg, project, encoder, group, versions, index, args.batch_size)
        for project, group in val_bugs.groupby("project")
    ]
    scores = pd.concat(frames, ignore_index=True)
    elapsed = time.time() - t0
    logger.info(
        "scored %d (bug, candidate) pairs across %d bugs in %.1fs",
        len(scores),
        scores.groupby(["project", "key"]).ngroups,
        elapsed,
    )

    out_path = ensure_dir(cfg.paths.tables) / OUTPUT_TEMPLATE.format(epoch=args.epoch)
    scores.to_parquet(out_path, index=False)
    logger.info("wrote %s", out_path)

    gold = val_bugs[["project", "key", "gold_files"]]
    per_bug = metrics_from_scores(scores, gold)
    metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
    mean_metrics = per_bug[metric_cols].mean()

    print("\n=== metrics recomputed from the saved candidate-score table ===")
    print(mean_metrics.round(4).to_string())

    verify_against_expected(mean_metrics)


if __name__ == "__main__":
    main()

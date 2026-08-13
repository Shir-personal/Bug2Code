r"""Save raw per-(bug, candidate-file) Fine-tuned CodeBERT scores on Cassandra30 Test (FINAL).

Runs the already-selected checkpoint (epoch 3, chosen by Cassandra30
Validation MRR in ``validate_finetuned.py``) over Cassandra30 Test exactly
once, and saves every candidate file's raw score - not just the ranking
metrics. Component filtering (``component_test_experiment.py``) and Hybrid
(``hybrid_test_experiment.py``) both read this table instead of re-encoding,
so there is exactly one epoch-3 Test CodeBERT inference pass across the whole
Test pipeline. Loads ONLY the checkpoint's two encoder state dicts (no
optimizer state, no ``backward()``, no training) via
``validate_finetuned.load_finetuned_model``, unchanged from the Validation
scoring script.

``--count-only`` tokenizes every Test candidate file with the CodeBERT
tokenizer and reports the exact encoding workload (bugs, candidate pairs,
distinct files, chunks) plus a GPU-time estimate from the measured Cassandra30
Validation benchmark (80,754 chunks -> 1,429.7s on a Colab T4) - no model
loaded, no checkpoint read, no GPU needed. Use it to size the real run before
paying for it.

Usage:
    # Cheap, CPU-only preflight:
    python -m bug2code.localization.save_finetuned_test_scores \\
        --config configs/cassandra30.yaml --count-only

    # The real Test inference pass (expensive - run only after review):
    python -m bug2code.localization.save_finetuned_test_scores --config configs/cassandra30.yaml
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
from bug2code.localization.cross_project_scores import (
    BENCHMARK_CHUNKS,
    BENCHMARK_SECONDS,
    count_chunks,
    estimate_gpu_seconds,
)
from bug2code.localization.frozen_codebert import MODEL_NAME, resolve_device
from bug2code.localization.train_codebert import CHECKPOINT_DIR
from bug2code.localization.validate_finetuned import FineTunedEncoder, load_finetuned_model
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_TEMPLATE = "finetuned_epoch{epoch}_test_candidate_scores.parquet"

# Benchmark constants live in cross_project_scores.py; reused here, not re-measured.


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument(
        "--epoch", type=int, default=3, help="checkpoint epoch to score (selected: 3)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="tokenize and count the encoding workload plus a GPU-time estimate; "
        "no model, no checkpoint, no GPU",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    test_bugs = bugs[bugs["split"] == "test"]
    project = cfg.projects[0]
    if len(cfg.projects) != 1:
        raise ValueError(f"expected a single-project config, got {cfg.projects}")
    logger.info(
        "%d Cassandra30 TEST bugs, epoch %d checkpoint only (VAL is never read)",
        len(test_bugs),
        args.epoch,
    )

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    if args.count_only:
        stats = count_chunks(cfg, project, versions, index, test_bugs)
        est_seconds = estimate_gpu_seconds(stats["n_chunks"])
        print("\n=== Cassandra30 TEST Fine-tuned CodeBERT encoding workload (count-only) ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        print(
            f"\nestimated GPU time (T4, from {BENCHMARK_CHUNKS} chunks -> {BENCHMARK_SECONDS}s "
            f"Validation benchmark): {est_seconds:.1f}s ({est_seconds / 60:.1f} min)"
        )
        return

    device = resolve_device(cfg.device)
    enc_cfg = cfg.localization["encoder"]
    checkpoint_dir = cfg.paths.experiments / CHECKPOINT_DIR
    ckpt_path = checkpoint_dir / f"epoch_{args.epoch}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found at {ckpt_path} - this script never trains one; "
            "restore the selected epoch-3 checkpoint into experiments_cassandra30/ first."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("device=%s checkpoint=%s", device, ckpt_path)

    model = load_finetuned_model(ckpt_path, enc_cfg["dropout"], device)
    encoder = FineTunedEncoder(model, tokenizer, device, fp16=False)

    t0 = time.time()
    scores = score_codebert_candidates(
        cfg, project, encoder, test_bugs, versions, index, args.batch_size
    )
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

    gold = test_bugs[["project", "key", "gold_files"]]
    per_bug = metrics_from_scores(scores, gold)
    metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
    mean_metrics = per_bug[metric_cols].mean()

    print(f"\n=== Fine-tuned CodeBERT epoch-{args.epoch} Cassandra30 TEST metrics (FINAL) ===")
    print(mean_metrics.round(4).to_string())
    print(f"\nevaluated bugs: {len(per_bug)} / {len(test_bugs)}")


if __name__ == "__main__":
    main()

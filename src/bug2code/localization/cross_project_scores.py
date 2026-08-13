r"""Generalization to other projects: cross-project evaluation.

The already-selected Cassandra30 Fine-tuned CodeBERT checkpoint (epoch 3,
chosen by Cassandra30 Validation MRR - never by anything from the target
project) is applied directly to a target project's own Validation subset -
HBase30 or Spark30 - with no further training, no fine-tuning, and no
target-project-specific checkpoint or hyperparameter selection. Saves raw
per-(bug, candidate-file) scores exactly like ``save_finetuned_val_scores.py``,
so there is exactly one CodeBERT inference pass per target project.

The checkpoint path is independent of ``--config``: ``configs/hbase30.yaml``
and ``configs/spark30.yaml`` each declare their own ``experiments`` dir only
so every phase's outputs stay isolated per project - no target-project
training ever writes anything there. The Cassandra30 checkpoint is passed
explicitly (``--checkpoint``, defaulting to the selected epoch under
``experiments_cassandra30/``), never derived from the target config.

No Jira Component filtering and no Hybrid combination are used here - this
extension evaluates the plain Fine-tuned CodeBERT score only (plan's explicit
scope for the cross-project experiment).

``--split`` selects which of the target project's own bugs to evaluate
against: ``val`` (default, the original H4 experiment) or ``test`` (the
FINAL cross-project Test evaluation - still the same Cassandra30 epoch-3
checkpoint, still no target-project Validation or Test used for any
additional model/config selection).

Usage:
    # Cheap dry run: counts tokenization work without loading a model or GPU.
    python -m bug2code.localization.cross_project_scores \\
        --config configs/hbase30.yaml --split test --count-only

    # The real inference pass (expensive - run only after review):
    python -m bug2code.localization.cross_project_scores \\
        --config configs/hbase30.yaml --split test
    python -m bug2code.localization.cross_project_scores \\
        --config configs/spark30.yaml --split test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from bug2code.config import Config, load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores, score_codebert_candidates
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME, read_versions
from bug2code.localization.frozen_codebert import MODEL_NAME, chunk_token_ids, resolve_device
from bug2code.localization.train_codebert import CHECKPOINT_DIR
from bug2code.localization.validate_finetuned import FineTunedEncoder, load_finetuned_model
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir, resolve

logger = get_logger(__name__)

OUTPUT_TEMPLATE = "cassandra_epoch{epoch}_zeroshot_{project}30_{split}_candidate_scores.parquet"

# The Cassandra30 checkpoint dir, independent of any target-project config -
# HBase30/Spark30 never train, so their own ``cfg.paths.experiments`` is unused.
CASSANDRA30_EXPERIMENTS_DIR = "experiments_cassandra30"

# Measured Cassandra30 Validation benchmark on a Colab T4, reused to estimate runtime for the other projects.
BENCHMARK_CHUNKS = 80_754
BENCHMARK_SECONDS = 1429.7


def estimate_gpu_seconds(n_chunks: int) -> float:
    """Linear extrapolation from the measured Cassandra30 Validation chunk-throughput benchmark."""
    return n_chunks / BENCHMARK_CHUNKS * BENCHMARK_SECONDS


def default_checkpoint_path(epoch: int) -> Path:
    """The already-selected Cassandra30 checkpoint for one epoch, project-root-relative."""
    return resolve(CASSANDRA30_EXPERIMENTS_DIR) / CHECKPOINT_DIR / f"epoch_{epoch}.pt"


def count_chunks(
    cfg: Config,
    project: str,
    versions: pd.DataFrame,
    index: pd.DataFrame,
    bugs: pd.DataFrame,
) -> dict:
    """Tokenize every candidate file and count the exact CodeBERT encoding workload.

    ``bugs`` is whichever split's rows the caller wants counted (Validation or
    Test) - this function is split-agnostic, it only reads ``bugs["key"]``.
    Mirrors ``frozen_codebert.FrozenEncoder.encode_files``'s chunking exactly
    (same tokenizer, same ``chunk_token_ids`` call with the same config values)
    but never loads a model - just the tokenizer, which is CPU-only and cheap.
    Used to estimate GPU runtime before committing to the real inference pass.
    """
    proj_index = index[index["project"] == project]
    keys = set(bugs["key"])
    candidate_ids = proj_index.loc[proj_index["key"].isin(keys), "file_version_id"].unique()
    n_pairs = int(proj_index["key"].isin(keys).sum())

    texts = read_versions(cfg, versions, candidate_ids)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loc = cfg.localization

    n_chunks = 0
    for text in texts.values():
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        n_chunks += len(
            chunk_token_ids(
                token_ids, loc["code_max_length"], loc["chunk_stride"], loc["max_chunks_per_file"]
            )
        )

    return {
        "project": project,
        "n_bugs": len(bugs),
        "n_bug_candidate_pairs": n_pairs,
        "n_distinct_candidate_files": len(candidate_ids),
        "n_chunks": n_chunks,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="target-project config, e.g. configs/hbase30.yaml"
    )
    parser.add_argument(
        "--epoch", type=int, default=3, help="Cassandra30 checkpoint epoch (selected: 3)"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="override the Cassandra30 checkpoint path"
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="which of the target project's own splits to evaluate (default: val)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="tokenize and count the encoding workload; no model, no GPU, no inference",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    project = cfg.projects[0]
    if len(cfg.projects) != 1:
        raise ValueError(
            f"cross-project scoring expects a single-project config, got {cfg.projects}"
        )

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    split_bugs = bugs[bugs["split"] == args.split]
    logger.info(
        "%s30: %d %s bugs (cross-project target, no training here, no other split read)",
        project,
        len(split_bugs),
        args.split.upper(),
    )

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    if args.count_only:
        stats = count_chunks(cfg, project, versions, index, split_bugs)
        est_seconds = estimate_gpu_seconds(stats["n_chunks"])
        print(
            f"\n=== {project}30 {args.split.upper()} cross-project encoding workload "
            "(count-only, no model loaded) ==="
        )
        for k, v in stats.items():
            print(f"{k}: {v}")
        print(
            f"\nestimated GPU time (T4, from {BENCHMARK_CHUNKS} chunks -> {BENCHMARK_SECONDS}s "
            f"Cassandra30 Validation benchmark): {est_seconds:.1f}s ({est_seconds / 60:.1f} min)"
        )
        return

    ckpt_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.epoch)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Cassandra30 checkpoint not found at {ckpt_path} - this script never trains one; "
            "copy the already-selected checkpoint from Colab first."
        )

    device = resolve_device(cfg.device)
    enc_cfg = cfg.localization["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info(
        "device=%s checkpoint=%s (Cassandra30-trained, evaluated on %s30)",
        device,
        ckpt_path,
        project,
    )

    model = load_finetuned_model(ckpt_path, enc_cfg["dropout"], device)
    encoder = FineTunedEncoder(model, tokenizer, device, fp16=False)

    t0 = time.time()
    scores = score_codebert_candidates(
        cfg, project, encoder, split_bugs, versions, index, args.batch_size
    )
    elapsed = time.time() - t0
    logger.info(
        "scored %d (bug, candidate) pairs across %d bugs in %.1fs",
        len(scores),
        scores.groupby(["project", "key"]).ngroups,
        elapsed,
    )

    out_path = ensure_dir(cfg.paths.tables) / OUTPUT_TEMPLATE.format(
        epoch=args.epoch, project=project, split=args.split
    )
    scores.to_parquet(out_path, index=False)
    logger.info("wrote %s", out_path)

    gold = split_bugs[["project", "key", "gold_files"]]
    per_bug = metrics_from_scores(scores, gold)
    metric_cols = [c for c in per_bug.columns if c not in ("project", "key", "n_candidates")]
    mean_metrics = per_bug[metric_cols].mean()

    print(
        f"\n=== cross-project Cassandra30 epoch-{args.epoch} -> {project}30 "
        f"{args.split.upper()} metrics ==="
    )
    print(mean_metrics.round(4).to_string())
    print(f"\nevaluated bugs: {len(per_bug)} / {len(split_bugs)}")


if __name__ == "__main__":
    main()

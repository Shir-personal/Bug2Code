"""Phase 4 — Fine-tuned CodeBERT dual encoder.

Two independently-weighted CodeBERT towers (``tied_weights: false``): one
encodes the bug query (title + description), one encodes a code chunk. Trained
with Multiple Negatives Ranking loss (symmetric in-batch-negative InfoNCE) on
one example per (bug, gold-file) pair from the fixed 30% dev-TRAIN subset
only. No validation runs here — validation is deferred and done separately.
Training is FP32; no gradient accumulation, no LR scheduler (plan only tunes
lr/batch_size/epochs).

Offline execution is opt-in, not forced: set HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1 in the environment before running to require the
local Hugging Face cache (fails instead of downloading). Left unset, the
CodeBERT tower weights download normally on a fresh cache (e.g. a new Colab
runtime). All other data (bugs, candidate cache, code text) is always read
from local artifacts already built by earlier phases, regardless of this
setting.

Resumable at epoch granularity: each epoch's positive-chunk choice and
shuffle order are derived from ``cfg.seed + epoch_index`` alone, so a
checkpoint plus that seed fully determines the next epoch — no RNG state
needs to be saved.

Usage:
    python -m bug2code.localization.train_codebert --target-epoch 1
    python -m bug2code.localization.train_codebert --target-epoch 1 --dry-run
    python -m bug2code.localization.train_codebert --benchmark-steps 30
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from bug2code.config import Config, load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME, read_versions
from bug2code.localization.frozen_codebert import MODEL_NAME, chunk_token_ids, resolve_device
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

CHECKPOINT_DIR = "finetuned_codebert/checkpoints"


class DualEncoder(nn.Module):
    """Two separate CodeBERT towers, both initialised from the same pretrained weights."""

    def __init__(self, hf_name: str, dropout: float):
        super().__init__()
        config = AutoConfig.from_pretrained(hf_name)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        self.query_encoder = AutoModel.from_pretrained(hf_name, config=config)
        self.code_encoder = AutoModel.from_pretrained(hf_name, config=config)

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=1)

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled, L2-normalised embedding of a batch of bug queries."""
        out = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def encode_code(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled, L2-normalised embedding of a batch of code chunks."""
        out = self.code_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)


def mnr_loss(query_vecs: torch.Tensor, code_vecs: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetric in-batch-negative InfoNCE: every other file in the batch is a negative."""
    logits = scale * query_vecs @ code_vecs.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def build_training_pairs(
    train_bugs: pd.DataFrame, versions: pd.DataFrame, index: pd.DataFrame
) -> pd.DataFrame:
    """One (query_text, file_version_id) row per (bug, gold-file) pair — not deduped (D2)."""
    exploded = train_bugs[["project", "key", "title", "description", "gold_files"]].explode(
        "gold_files"
    )
    exploded = exploded.rename(columns={"gold_files": "path"})

    candidates = index.merge(
        versions[["id", "project", "path"]],
        left_on=["project", "file_version_id"],
        right_on=["project", "id"],
    )
    pairs = exploded.merge(
        candidates[["project", "key", "path", "id"]], on=["project", "key", "path"], how="inner"
    )
    missing = len(exploded) - len(pairs)
    if missing:
        raise ValueError(
            f"{missing} gold files were not found in the candidate index — "
            "this indicates a data integrity problem, not an expected case."
        )
    pairs = pairs.rename(columns={"id": "file_version_id"})
    pairs["query_text"] = pairs["title"] + " " + pairs["description"].fillna("")
    return pairs[["project", "key", "query_text", "file_version_id"]].reset_index(drop=True)


def epoch_examples(
    pairs: pd.DataFrame, chunks_by_id: dict[int, list[list[int]]], seed: int
) -> tuple[pd.DataFrame, list[list[int]]]:
    """Pick one random positive chunk per pair and shuffle pair order, both seeded by ``seed``.

    D1: a file's fix is not necessarily in its first chunk (median gold file spans
    5 chunks), so a fixed chunk-0 selection would usually miss the actual fix.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for fid in pairs["file_version_id"]:
        file_chunks = chunks_by_id[fid]
        chosen.append(file_chunks[rng.integers(0, len(file_chunks))])
    order = rng.permutation(len(pairs))
    ordered_pairs = pairs.iloc[order].reset_index(drop=True)
    ordered_chunks = [chosen[i] for i in order]
    return ordered_pairs, ordered_chunks


def find_resume_epoch(checkpoint_dir: Path) -> int:
    """Highest epoch number with a saved checkpoint, or 0 if none exist."""
    existing = [int(p.stem.split("_")[1]) for p in checkpoint_dir.glob("epoch_*.pt")]
    return max(existing) if existing else 0


def wrap_chunks(
    chunks: list[list[int]], max_len: int, cls_id: int, sep_id: int, pad_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Add CLS/SEP and pad a batch of raw token-id chunks to ``max_len``."""
    ids, masks = [], []
    for chunk in chunks:
        wrapped = [cls_id, *chunk, sep_id]
        ids.append(wrapped + [pad_id] * (max_len - len(wrapped)))
        masks.append([1] * len(wrapped) + [0] * (max_len - len(wrapped)))
    return ids, masks


def load_setup(cfg: Config) -> tuple[pd.DataFrame, dict[int, list[list[int]]], AutoTokenizer]:
    """Everything needed before training starts: pairs, precomputed chunks, tokenizer."""
    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    train_bugs = bugs[bugs["split"] == "train"]

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    pairs = build_training_pairs(train_bugs, versions, index)
    logger.info(
        "%d training pairs from %d dev-TRAIN bugs (%d distinct gold files)",
        len(pairs),
        train_bugs["key"].nunique(),
        pairs["file_version_id"].nunique(),
    )

    distinct_ids = pairs["file_version_id"].unique().tolist()
    texts = read_versions(cfg, versions, distinct_ids)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loc = cfg.localization
    chunks_by_id = {
        fid: chunk_token_ids(
            tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"],
            loc["code_max_length"],
            loc["chunk_stride"],
            loc["max_chunks_per_file"],
        )
        for fid, text in texts.items()
    }
    return pairs, chunks_by_id, tokenizer


def run_epoch(
    model: DualEncoder,
    optimizer: torch.optim.Optimizer,
    tokenizer: AutoTokenizer,
    pairs: pd.DataFrame,
    chunks_by_id: dict[int, list[list[int]]],
    cfg: Config,
    epoch: int,
    device: str,
    max_steps: int | None = None,
    on_step: Callable[[int, float, float], None] | None = None,
) -> float:
    """Train one full epoch over ``pairs``; returns the mean batch loss.

    ``max_steps`` and ``on_step`` exist only for the CLI's ``--benchmark-steps``
    mode (stop after N real steps, report per-step timing/loss). Left at their
    defaults, a full epoch runs exactly as before: nothing about the forward,
    loss, backward or optimizer step changes.
    """
    loc = cfg.localization
    enc_cfg = loc["encoder"]
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    code_max = loc["code_max_length"]
    query_max = loc["query_max_length"]
    batch_size = enc_cfg["batch_size"]

    ordered_pairs, ordered_chunks = epoch_examples(pairs, chunks_by_id, cfg.seed + epoch)

    model.train()
    losses: list[float] = []
    step = 0
    for i in range(0, len(ordered_pairs), batch_size):
        batch_pairs = ordered_pairs.iloc[i : i + batch_size]
        batch_chunks = ordered_chunks[i : i + batch_size]
        step_start = time.time() if on_step is not None else None

        # Fixed-length padding, not dynamic: varying shapes forced MPS to recompile
        # every step and blew past RAM. Constant shape reuses one compiled graph.
        q_enc = tokenizer(
            list(batch_pairs["query_text"]),
            padding="max_length",
            truncation=True,
            max_length=query_max,
        )
        q_ids = torch.tensor(q_enc["input_ids"], device=device)
        q_mask = torch.tensor(q_enc["attention_mask"], device=device)

        c_ids_list, c_mask_list = wrap_chunks(batch_chunks, code_max, cls_id, sep_id, pad_id)
        c_ids = torch.tensor(c_ids_list, device=device)
        c_mask = torch.tensor(c_mask_list, device=device)

        q_vecs = model.encode_query(q_ids, q_mask)
        c_vecs = model.encode_code(c_ids, c_mask)
        loss = mnr_loss(q_vecs, c_vecs, enc_cfg["scale"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        losses.append(loss_value)

        if on_step is not None:
            on_step(step, time.time() - step_start, loss_value)
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    return float(np.mean(losses))


WARMUP_STEPS = 5  # excluded from the average s/step so model/CUDA warm-up doesn't skew it


def run_benchmark(
    model: DualEncoder,
    optimizer: torch.optim.Optimizer,
    tokenizer: AutoTokenizer,
    pairs: pd.DataFrame,
    chunks_by_id: dict[int, list[list[int]]],
    cfg: Config,
    device: str,
    n_steps: int,
) -> dict:
    """Run ``n_steps`` real training steps through :func:`run_epoch` and report timing/memory.

    Does not save a checkpoint or touch epoch/resume state: it calls the same
    forward/loss/backward/optimizer-step path as a real epoch, capped early via
    ``run_epoch``'s ``max_steps``.
    """
    enc_cfg = cfg.localization["encoder"]
    batch_size = enc_cfg["batch_size"]
    is_cuda = device == "cuda" and torch.cuda.is_available()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    step_records: list[tuple[float, float]] = []  # (elapsed_s, loss)

    def on_step(_step: int, elapsed_s: float, loss: float) -> None:
        step_records.append((elapsed_s, loss))

    t0 = time.time()
    run_epoch(
        model,
        optimizer,
        tokenizer,
        pairs,
        chunks_by_id,
        cfg,
        epoch=1,  # only seeds this run's chunk/shuffle choice; no epoch state is saved
        device=device,
        max_steps=n_steps,
        on_step=on_step,
    )
    total_s = time.time() - t0

    excluded = WARMUP_STEPS if len(step_records) > WARMUP_STEPS else 0
    timed = step_records[excluded:] if excluded else step_records
    avg_step_s = float(np.mean([elapsed for elapsed, _ in timed]))
    n_batches_per_epoch = math.ceil(len(pairs) / batch_size)

    return {
        "device": device,
        "batch_size": batch_size,
        "steps_completed": len(step_records),
        "warmup_steps_excluded": excluded,
        "avg_s_per_step": avg_step_s,
        "total_benchmark_s": total_s,
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / 1e6 if is_cuda else None,
        "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / 1e6 if is_cuda else None,
        "cuda_peak_memory_allocated_mb": (
            torch.cuda.max_memory_allocated() / 1e6 if is_cuda else None
        ),
        "loss_start": step_records[0][1] if step_records else None,
        "loss_end": step_records[-1][1] if step_records else None,
        "n_batches_per_epoch": n_batches_per_epoch,
        "estimated_epoch_s": avg_step_s * n_batches_per_epoch,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument(
        "--target-epoch",
        type=int,
        default=None,
        help="train up to and including this epoch, then exit (required unless "
        "--dry-run or --benchmark-steps is given)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load data/model and report counts, but run no training step",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=None,
        metavar="N",
        help="run N real training steps (forward/loss/backward/optimizer-step) and "
        "report timing/memory; saves no checkpoint and does not advance epoch state",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.target_epoch is None and not args.dry_run and args.benchmark_steps is None:
        parser.error("--target-epoch is required unless --dry-run or --benchmark-steps is given")

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)
    enc_cfg = cfg.localization["encoder"]

    logger.info(
        "device=%s HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s",
        device,
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("TRANSFORMERS_OFFLINE"),
    )

    pairs, chunks_by_id, tokenizer = load_setup(cfg)

    if args.benchmark_steps is not None:
        model = DualEncoder(MODEL_NAME, enc_cfg["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=enc_cfg["lr"])
        report = run_benchmark(
            model, optimizer, tokenizer, pairs, chunks_by_id, cfg, device, args.benchmark_steps
        )
        logger.info(
            "benchmark: device=%s batch_size=%d steps=%d/%d (warmup excluded=%d) "
            "avg=%.3fs/step total=%.1fs loss %.4f->%.4f "
            "cuda_alloc=%sMB cuda_reserved=%sMB cuda_peak=%sMB "
            "estimated_epoch=%.1fs over %d batches",
            report["device"],
            report["batch_size"],
            report["steps_completed"],
            args.benchmark_steps,
            report["warmup_steps_excluded"],
            report["avg_s_per_step"],
            report["total_benchmark_s"],
            report["loss_start"],
            report["loss_end"],
            report["cuda_memory_allocated_mb"],
            report["cuda_memory_reserved_mb"],
            report["cuda_peak_memory_allocated_mb"],
            report["estimated_epoch_s"],
            report["n_batches_per_epoch"],
        )
        return

    checkpoint_dir = ensure_dir(cfg.paths.experiments / CHECKPOINT_DIR)
    start_epoch = find_resume_epoch(checkpoint_dir) + 1

    if args.dry_run:
        model = DualEncoder(MODEL_NAME, enc_cfg["dropout"]).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "dry-run OK: %d pairs, %d distinct gold files, model loaded (%d params), "
            "next epoch to train would be %d, checkpoints in %s",
            len(pairs),
            len(chunks_by_id),
            n_params,
            start_epoch,
            checkpoint_dir,
        )
        return

    if start_epoch > args.target_epoch:
        logger.info(
            "checkpoint for epoch %d already exists; nothing to do for target-epoch=%d",
            start_epoch - 1,
            args.target_epoch,
        )
        return

    model = DualEncoder(MODEL_NAME, enc_cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=enc_cfg["lr"])

    if start_epoch > 1:
        ckpt = torch.load(checkpoint_dir / f"epoch_{start_epoch - 1}.pt", map_location=device)
        model.query_encoder.load_state_dict(ckpt["query_encoder_state_dict"])
        model.code_encoder.load_state_dict(ckpt["code_encoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("resumed from checkpoint epoch %d", start_epoch - 1)

    for epoch in range(start_epoch, args.target_epoch + 1):
        t0 = time.time()
        mean_loss = run_epoch(model, optimizer, tokenizer, pairs, chunks_by_id, cfg, epoch, device)
        elapsed = time.time() - t0

        ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "query_encoder_state_dict": model.query_encoder.state_dict(),
                "code_encoder_state_dict": model.code_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss_mean": mean_loss,
                "elapsed_s": elapsed,
                "config_snapshot": {
                    "lr": enc_cfg["lr"],
                    "batch_size": enc_cfg["batch_size"],
                    "scale": enc_cfg["scale"],
                    "dropout": enc_cfg["dropout"],
                    "seed": cfg.seed,
                },
            },
            ckpt_path,
        )
        logger.info(
            "epoch %d done: %d pairs, mean loss %.4f, %.1fs, saved %s",
            epoch,
            len(pairs),
            mean_loss,
            elapsed,
            ckpt_path,
        )


if __name__ == "__main__":
    main()

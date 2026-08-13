"""Phase 4 validation — evaluate fine-tuned CodeBERT checkpoints.

Each of the three per-epoch checkpoints from ``train_codebert.py`` is loaded
(both towers) and evaluated on the fixed 30% dev-VALIDATION subset only, with
the exact same per-bug candidate sets, chunking and ranking-metric setup as
Frozen CodeBERT (Phase 3): mean pooling, L2-normalisation, cosine similarity,
file score = max over its own chunk scores. This makes the epoch results
directly comparable to ``frozen_codebert_results.csv`` (RQ2) and to the
TF-IDF baseline (RQ1). TEST is never read here.

FP16 autocast inference is supported for faster evaluation on a Colab CUDA
GPU; it is opt-in via ``--fp16`` and gated to CUDA only. Before running full
validation, a small FP32-vs-FP16 sanity check compares inference under both
precisions on a handful of dev-VAL bugs (using their full candidate sets) so
a precision-induced ranking change would be caught early rather than buried
in the full run.

Usage:
    python -m bug2code.localization.validate_finetuned --config configs/colab.yaml
    python -m bug2code.localization.validate_finetuned --config configs/colab.yaml \
        --epochs 1 2 3 --fp16 --sanity-check-bugs 5
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from bug2code.config import Config, load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME
from bug2code.localization.frozen_codebert import (
    MODEL_NAME,
    chunk_token_ids,
    evaluate_project,
    resolve_device,
)
from bug2code.localization.train_codebert import CHECKPOINT_DIR, DualEncoder, wrap_chunks
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

EPOCH_RESULTS_TEMPLATE = "finetuned_epoch_{epoch}_val_results.csv"
SUMMARY_NAME = "finetuned_val_summary.csv"
SANITY_CHECK_NAME = "finetuned_fp16_sanity_check.csv"


class FineTunedEncoder:
    """Loaded dual-encoder checkpoint, wrapped for the Frozen-CodeBERT-style ranking API.

    Implements the same ``encode_texts``/``encode_files`` interface as
    ``frozen_codebert.FrozenEncoder``, so ``evaluate_project`` and
    ``score_files`` are reused unchanged rather than re-implemented.
    """

    def __init__(
        self, model: DualEncoder, tokenizer: AutoTokenizer, device: str, fp16: bool = False
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        # Autocast fp16 is only meaningful on CUDA; silently no-op elsewhere so
        # a --fp16 flag left on for a local CPU/MPS run doesn't change results.
        self.fp16 = fp16 and device == "cuda"
        self.dim = model.query_encoder.config.hidden_size

    def _autocast(self):
        if self.fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def _pool(
        self, encode_fn, input_ids: list[list[int]], attention_mask: list[list[int]]
    ) -> np.ndarray:
        ids = torch.tensor(input_ids, device=self.device)
        mask = torch.tensor(attention_mask, device=self.device)
        with self._autocast():
            pooled = encode_fn(ids, mask)
        return pooled.float().cpu().numpy()

    def encode_texts(self, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
        """Encode bug queries with the query tower, truncated to ``max_length`` tokens."""
        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True, max_length=max_length
            )
            vecs.append(
                self._pool(self.model.encode_query, enc["input_ids"], enc["attention_mask"])
            )
        return np.concatenate(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    def encode_files(
        self, texts: dict[int, str], cfg: Config, batch_size: int
    ) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
        """Chunk-embed every file version with the code tower (same chunking as Phase 3)."""
        loc = cfg.localization
        cls_id, sep_id, pad_id = (
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        )
        max_len = loc["code_max_length"]

        all_chunks: list[list[int]] = []
        ranges: dict[int, tuple[int, int]] = {}
        for file_id, text in texts.items():
            token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
            file_chunks = chunk_token_ids(
                token_ids, max_len, loc["chunk_stride"], loc["max_chunks_per_file"]
            )
            start = len(all_chunks)
            all_chunks.extend(file_chunks)
            ranges[file_id] = (start, len(all_chunks))

        ids_list, mask_list = wrap_chunks(all_chunks, max_len, cls_id, sep_id, pad_id)
        vecs = np.zeros((len(all_chunks), self.dim), dtype=np.float32)
        for i in range(0, len(all_chunks), batch_size):
            vecs[i : i + batch_size] = self._pool(
                self.model.encode_code, ids_list[i : i + batch_size], mask_list[i : i + batch_size]
            )
        return vecs, ranges


def load_finetuned_model(checkpoint_path: Path, dropout: float, device: str) -> DualEncoder:
    """Load both towers' fine-tuned weights from one epoch checkpoint."""
    model = DualEncoder(MODEL_NAME, dropout)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.query_encoder.load_state_dict(ckpt["query_encoder_state_dict"])
    model.code_encoder.load_state_dict(ckpt["code_encoder_state_dict"])
    return model


def run_sanity_check(
    cfg: Config,
    checkpoint_path: Path,
    dropout: float,
    tokenizer: AutoTokenizer,
    val_bugs: pd.DataFrame,
    versions: pd.DataFrame,
    index: pd.DataFrame,
    device: str,
    batch_size: int,
    n_bugs: int,
) -> pd.DataFrame:
    """Compare FP32 vs FP16-autocast inference on a handful of dev-VAL bugs.

    Loads one checkpoint once and evaluates it under both precisions (same
    weights, only the autocast context differs), over each sampled bug's full
    candidate set. Reports query-embedding cosine similarity (the direct
    numerical-precision signal) alongside any resulting difference in ranking
    metrics (the downstream signal a full run would actually be judged on).
    """
    sample = val_bugs.head(n_bugs).reset_index(drop=True)
    model = load_finetuned_model(checkpoint_path, dropout, device)
    enc_fp32 = FineTunedEncoder(model, tokenizer, device, fp16=False)
    enc_fp16 = FineTunedEncoder(model, tokenizer, device, fp16=True)

    query_max = cfg.localization["query_max_length"]
    queries = [f"{b.title} {b.description or ''}" for b in sample.itertuples()]
    q32 = enc_fp32.encode_texts(queries, query_max, batch_size)
    q16 = enc_fp16.encode_texts(queries, query_max, batch_size)
    cos_df = pd.DataFrame(
        {
            "project": sample["project"],
            "key": sample["key"],
            "query_cosine_sim": (q32 * q16).sum(axis=1),
        }
    )

    rows32, rows16 = [], []
    for project, group in sample.groupby("project"):
        rows32 += evaluate_project(cfg, project, enc_fp32, group, versions, index, batch_size)
        rows16 += evaluate_project(cfg, project, enc_fp16, group, versions, index, batch_size)
    df32 = pd.DataFrame(rows32).set_index(["project", "key"])
    df16 = pd.DataFrame(rows16).set_index(["project", "key"])
    metric_cols = [c for c in df32.columns if c != "n_candidates"]
    diff = (df32[metric_cols] - df16[metric_cols]).abs().add_prefix("diff_").reset_index()

    return diff.merge(cos_df, on=["project", "key"])


def select_best_epoch(summary_df: pd.DataFrame, metric: str = "mrr") -> int:
    """Best epoch by mean dev-VALIDATION ``metric`` only — TEST is never involved."""
    return int(summary_df[metric].idxmax())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=[1, 2, 3], help="checkpoint epochs to validate"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument(
        "--fp16", action="store_true", help="use FP16 autocast for inference (CUDA only)"
    )
    parser.add_argument(
        "--sanity-check-bugs",
        type=int,
        default=5,
        help="dev-VAL bugs used for the FP32-vs-FP16 sanity check before full validation; "
        "0 disables it",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    val_bugs = bugs[bugs["split"] == "val"]
    logger.info("%d dev-VALIDATION bugs (TEST is never read by this script)", len(val_bugs))

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    device = resolve_device(cfg.device)
    enc_cfg = cfg.localization["encoder"]
    checkpoint_dir = cfg.paths.experiments / CHECKPOINT_DIR
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("device=%s fp16=%s", device, args.fp16)

    if args.sanity_check_bugs > 0:
        if device != "cuda":
            logger.warning(
                "FP16 sanity check needs CUDA; device is %s, skipping it", device
            )
        else:
            ckpt_path = checkpoint_dir / f"epoch_{min(args.epochs)}.pt"
            logger.info(
                "FP32-vs-FP16 sanity check on %d dev-VAL bugs using %s",
                args.sanity_check_bugs,
                ckpt_path,
            )
            report = run_sanity_check(
                cfg,
                ckpt_path,
                enc_cfg["dropout"],
                tokenizer,
                val_bugs,
                versions,
                index,
                device,
                args.batch_size,
                args.sanity_check_bugs,
            )
            report.to_csv(ensure_dir(cfg.paths.tables) / SANITY_CHECK_NAME, index=False)
            logger.info(
                "sanity check: min query cosine sim=%.5f, max metric diff=%.4f",
                report["query_cosine_sim"].min(),
                report.filter(like="diff_").to_numpy().max(),
            )

    summaries = []
    for epoch in args.epochs:
        ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
        model = load_finetuned_model(ckpt_path, enc_cfg["dropout"], device)
        encoder = FineTunedEncoder(model, tokenizer, device, fp16=args.fp16)

        t0 = time.time()
        rows = []
        for project, group in val_bugs.groupby("project"):
            rows += evaluate_project(cfg, project, encoder, group, versions, index, args.batch_size)
        elapsed = time.time() - t0

        results = pd.DataFrame(rows)
        results.to_csv(
            ensure_dir(cfg.paths.tables) / EPOCH_RESULTS_TEMPLATE.format(epoch=epoch), index=False
        )

        metric_cols = [c for c in results.columns if c not in ("project", "key", "n_candidates")]
        summary = results[metric_cols].mean()
        summary["epoch"] = epoch
        summaries.append(summary)
        logger.info(
            "epoch %d: %d dev-VAL bugs in %.1fs, mrr=%.4f hit@1=%.4f hit@10=%.4f",
            epoch,
            len(results),
            elapsed,
            summary["mrr"],
            summary["hit_at_1"],
            summary["hit_at_10"],
        )

    summary_df = pd.DataFrame(summaries).set_index("epoch")
    best_epoch = select_best_epoch(summary_df)
    summary_df["is_best"] = summary_df.index == best_epoch
    summary_df.round(4).to_csv(ensure_dir(cfg.paths.tables) / SUMMARY_NAME)

    logger.info("best epoch by dev-VALIDATION mrr: %d", best_epoch)
    print(summary_df.round(4).to_string())


if __name__ == "__main__":
    main()

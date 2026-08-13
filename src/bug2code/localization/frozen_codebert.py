"""Phase 3 — Frozen CodeBERT.

``microsoft/codebert-base`` used exactly as pretrained: no fine-tuning, no
training on any split. Each bug (title + description) and each candidate file
are mean-pooled into vectors and compared by cosine similarity. Long files are
split into overlapping chunks using the fixed strategy (256
tokens, stride 192, at most 12 chunks); a file's score is the max similarity
over its own chunks. The Jira Component is never used.

Evaluated on the exact same dev-validation bugs and exact per-bug candidate
sets as Phase 2, built by ``bug2code.localization.candidates``. Since the model
does not train, the dev-train subset is not used here at all.

Chunk embeddings are cached to disk (fp16) because frozen weights never
change, so a re-run of this script reuses them instead of re-encoding.

Usage:
    python -m bug2code.localization.frozen_codebert [--config configs/exp.yaml]
    python -m bug2code.localization.frozen_codebert --config configs/cassandra30.yaml --split test
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from bug2code.config import Config, load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME, read_versions
from bug2code.localization.metrics import score_one
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "frozen_codebert_results.csv"
TEST_OUTPUT_NAME = "frozen_codebert_test_results.csv"
MODEL_NAME = "microsoft/codebert-base"


def resolve_device(name: str) -> str:
    """Turn the config's "auto" into a concrete torch device string."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chunk_token_ids(
    token_ids: list[int], max_length: int, stride: int, max_chunks: int
) -> list[list[int]]:
    """Sliding-window chunks of a file's tokens, without the CLS/SEP wrapper.

    Empty input yields one empty chunk, so every file gets at least one vector.
    """
    inner = max_length - 2  # room for CLS and SEP
    if not token_ids:
        return [[]]
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids) and len(chunks) < max_chunks:
        chunks.append(token_ids[start : start + inner])
        if start + inner >= len(token_ids):
            break
        start += stride
    return chunks


class FrozenEncoder:
    """Frozen CodeBERT: mean-pool token embeddings, L2-normalise."""

    def __init__(self, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        self.device = device
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def _pool(self, input_ids: list[list[int]], attention_mask: list[list[int]]) -> np.ndarray:
        ids = torch.tensor(input_ids, device=self.device)
        mask = torch.tensor(attention_mask, device=self.device)
        hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        mask_f = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        return pooled.cpu().numpy()

    def encode_texts(self, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
        """Encode whole texts (bug queries), truncated to ``max_length`` tokens."""
        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            vecs.append(self._pool(enc["input_ids"], enc["attention_mask"]))
        return np.concatenate(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    def encode_files(
        self, texts: dict[int, str], cfg: Config, batch_size: int
    ) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
        """Chunk-embed every file version; returns the chunk matrix and each id's row range."""
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
            for chunk in file_chunks:
                wrapped = [cls_id, *chunk, sep_id]
                padded = wrapped + [pad_id] * (max_len - len(wrapped))
                mask = [1] * len(wrapped) + [0] * (max_len - len(wrapped))
                all_chunks.append((padded, mask))
            ranges[file_id] = (start, len(all_chunks))

        vecs = np.zeros((len(all_chunks), self.dim), dtype=np.float32)
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            ids = [b[0] for b in batch]
            masks = [b[1] for b in batch]
            vecs[i : i + len(batch)] = self._pool(ids, masks)
        return vecs, ranges


def score_files(
    chunk_matrix: np.ndarray,
    ranges: dict[int, tuple[int, int]],
    query: np.ndarray,
    candidate_ids: list[int],
) -> np.ndarray:
    """Per-candidate score = max cosine similarity over that file's chunks."""
    scores = np.empty(len(candidate_ids), dtype=np.float32)
    for j, file_id in enumerate(candidate_ids):
        start, end = ranges[file_id]
        scores[j] = (chunk_matrix[start:end] @ query).max()
    return scores


def evaluate_project(
    cfg: Config,
    project: str,
    encoder: FrozenEncoder,
    bugs: pd.DataFrame,
    versions: pd.DataFrame,
    index: pd.DataFrame,
    batch_size: int,
) -> list[dict]:
    """Encode this project's candidate files once, then rank each bug (any single split)."""
    proj_index = index[index["project"] == project]
    bug_keys = set(bugs["key"])
    file_ids = proj_index.loc[proj_index["key"].isin(bug_keys), "file_version_id"].unique()

    logger.info("%s: reading %d distinct candidate file versions", project, len(file_ids))
    texts = read_versions(cfg, versions, file_ids)

    t0 = time.time()
    chunk_matrix, ranges = encoder.encode_files(texts, cfg, batch_size)
    logger.info(
        "%s: encoded %d chunks from %d files in %.1fs",
        project,
        chunk_matrix.shape[0],
        len(texts),
        time.time() - t0,
    )

    path_of_id = versions.set_index("id")["path"]
    query_max = cfg.localization["query_max_length"]
    queries = encoder.encode_texts(
        [f"{b.title} {b.description or ''}" for b in bugs.itertuples()], query_max, batch_size
    )

    rows = []
    for query, bug in zip(queries, bugs.itertuples(), strict=True):
        candidate_ids = proj_index.loc[proj_index["key"] == bug.key, "file_version_id"].tolist()
        scores = score_files(chunk_matrix, ranges, query, candidate_ids)
        order = np.argsort(-scores)
        ranked = [path_of_id[candidate_ids[i]] for i in order]

        row = {"project": project, "key": bug.key, "n_candidates": len(candidate_ids)}
        row |= score_one(ranked, list(bug.gold_files))
        rows.append(row)
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="which fixed split to evaluate (default: val, unchanged historical behaviour)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    split_bugs = bugs[bugs["split"] == args.split]
    logger.info("%d bugs in split=%s", len(split_bugs), args.split)

    cache_dir = cfg.paths.cache / "candidates"
    versions = pd.read_parquet(cache_dir / FILE_VERSIONS_NAME)
    index = pd.read_parquet(cache_dir / SNAPSHOT_INDEX_NAME)

    device = resolve_device(cfg.device)
    logger.info("loading %s on %s", MODEL_NAME, device)
    encoder = FrozenEncoder(device)

    t0 = time.time()
    rows = []
    for project, group in split_bugs.groupby("project"):
        rows += evaluate_project(cfg, project, encoder, group, versions, index, args.batch_size)
    elapsed = time.time() - t0

    results = pd.DataFrame(rows)
    output_name = OUTPUT_NAME if args.split == "val" else TEST_OUTPUT_NAME
    results.to_csv(ensure_dir(cfg.paths.tables) / output_name, index=False)

    metric_cols = [c for c in results.columns if c not in ("project", "key", "n_candidates")]
    summary = results.groupby("project")[metric_cols].mean().round(4)
    logger.info("frozen CodeBERT %s evaluation finished in %.1fs", args.split, elapsed)
    print(summary.to_string())
    print("\noverall:")
    print(results[metric_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()

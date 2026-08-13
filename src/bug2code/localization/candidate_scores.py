"""Raw per-(bug, candidate-file) scores, saved once and reused everywhere else.

Component filtering and the Hybrid extension both need a per-bug,
per-candidate-file score from Fine-tuned CodeBERT and from TF-IDF. Computing
those scores is the expensive part (one CodeBERT inference pass over every
candidate file); ranking them is cheap. This module does the expensive part
once per model and returns a long-format table - one row per (bug, candidate
file) - so everything downstream (Full vs Component-filtered, the alpha
sweep, and later the TEST pass) reuses the same saved scores instead of
re-encoding.

Ranking from a saved table must reproduce the exact ranking a live run would
produce, ties included. Ties are broken by candidate order, not by an
arbitrary re-sort: ``candidate_order`` records each candidate's position in
the same per-bug list the live scoring pass used, so :func:`ranked_paths`
can replay the identical ``argsort(-scores)`` call over the identical array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bug2code.config import Config
from bug2code.localization.candidates import read_versions
from bug2code.localization.frozen_codebert import score_files
from bug2code.localization.metrics import score_one
from bug2code.logging_utils import get_logger

logger = get_logger(__name__)


def score_codebert_candidates(
    cfg: Config,
    project: str,
    encoder,
    bugs: pd.DataFrame,
    versions: pd.DataFrame,
    index: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    """Raw (project, key, path, candidate_order, score) rows, one CodeBERT pass.

    ``encoder`` is anything implementing ``encode_texts``/``encode_files``
    (``FrozenEncoder`` or ``FineTunedEncoder``). Mirrors
    ``frozen_codebert.evaluate_project``'s encoding steps exactly (same
    candidate files read once, same chunk embedding, same ``score_files``
    per-candidate scoring) but returns every candidate's raw score instead of
    collapsing straight to ranking metrics.
    """
    proj_index = index[index["project"] == project]
    bug_keys = set(bugs["key"])
    candidate_ids_all = proj_index.loc[
        proj_index["key"].isin(bug_keys), "file_version_id"
    ].unique()

    logger.info("%s: reading %d distinct candidate file versions", project, len(candidate_ids_all))
    texts = read_versions(cfg, versions, candidate_ids_all)
    chunk_matrix, ranges = encoder.encode_files(texts, cfg, batch_size)

    path_of_id = versions.set_index("id")["path"]
    query_max = cfg.localization["query_max_length"]
    queries = encoder.encode_texts(
        [f"{b.title} {b.description or ''}" for b in bugs.itertuples()], query_max, batch_size
    )

    rows = []
    for query, bug in zip(queries, bugs.itertuples(), strict=True):
        candidate_ids = proj_index.loc[proj_index["key"] == bug.key, "file_version_id"].tolist()
        scores = score_files(chunk_matrix, ranges, query, candidate_ids)
        for order, (file_id, score) in enumerate(zip(candidate_ids, scores, strict=True)):
            rows.append(
                {
                    "project": project,
                    "key": bug.key,
                    "path": path_of_id[file_id],
                    "candidate_order": order,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def ranked_paths(bug_scores: pd.DataFrame) -> list[str]:
    """Ranked paths for one bug, replaying the live scoring pass's tie-break.

    ``bug_scores`` must be one bug's rows only, with ``candidate_order`` and
    ``score`` columns. Restoring ``candidate_order`` before ``argsort(-score)``
    reproduces exactly what ``score_files`` + ``np.argsort(-scores)`` would
    have produced live, ties included.
    """
    ordered = bug_scores.sort_values("candidate_order")
    order = np.argsort(-ordered["score"].to_numpy())
    return ordered["path"].to_numpy()[order].tolist()


def metrics_from_scores(scores: pd.DataFrame, gold_files: pd.DataFrame) -> pd.DataFrame:
    """Per-bug ranking metrics recomputed from a saved raw score table.

    ``gold_files`` needs ``project``, ``key``, ``gold_files`` columns (one row
    per bug). Used to verify a saved score table reproduces a live run's
    metrics exactly before the table is trusted for anything downstream.
    """
    gold_by_bug = gold_files.set_index(["project", "key"])["gold_files"]
    rows = []
    for (project, key), group in scores.groupby(["project", "key"]):
        ranked = ranked_paths(group)
        gold = list(gold_by_bug.loc[(project, key)])
        row = {"project": project, "key": key, "n_candidates": len(ranked)}
        row |= score_one(ranked, gold)
        rows.append(row)
    return pd.DataFrame(rows)

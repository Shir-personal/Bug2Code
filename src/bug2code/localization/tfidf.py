"""Phase 2 — TF-IDF baseline.

TF-IDF is fit per project on the exact candidate files of the **dev-train**
subset only (no validation or test content enters the vocabulary or IDF
weights). Each dev-validation bug is then ranked over its own exact
``parent_sha`` candidate set — the same set the frozen-CodeBERT phase uses —
by cosine similarity between the bug's title+description and each candidate
file's TF-IDF vector. The Jira Component is never used.

Usage:
    python -m bug2code.localization.tfidf [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from bug2code.config import Config, load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores
from bug2code.localization.candidates import FILE_VERSIONS_NAME, SNAPSHOT_INDEX_NAME, read_versions
from bug2code.localization.tokenize import tokenize
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "tfidf_results.csv"


def bug_text(row) -> str:
    """Title + description, the only text the model ever sees."""
    description = row.description or ""
    return f"{row.title} {description}"


def fit_project(cfg: Config, texts_train: list[str]) -> tuple[CountVectorizer, TfidfTransformer]:
    """Fit vocabulary + IDF on the training documents only."""
    tfidf_cfg = cfg.localization["tfidf"]
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        max_features=tfidf_cfg["max_features"],
    )
    counts = vectorizer.fit_transform(texts_train)
    transformer = TfidfTransformer(sublinear_tf=tfidf_cfg["sublinear_tf"])
    transformer.fit(counts)
    return vectorizer, transformer


def score_candidates(
    cfg: Config,
    project: str,
    bugs: pd.DataFrame,
    versions: pd.DataFrame,
    index: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Fit on this project's train bugs, score every candidate of its ``split`` bugs.

    Shared by the standalone baseline (below), and by ``save_tfidf_val_scores.py``
    / ``save_tfidf_test_scores.py``, which save this same raw table instead of
    re-fitting. Same (project, key, path, candidate_order, score) shape as
    ``candidate_scores.score_codebert_candidates``, so the two join safely for
    the Hybrid extension.
    """
    train_keys = set(bugs.loc[bugs["split"] == "train", "key"])
    split_bugs = bugs[bugs["split"] == split]

    proj_index = index[index["project"] == project]
    train_ids = proj_index.loc[proj_index["key"].isin(train_keys), "file_version_id"].unique()
    split_ids = proj_index.loc[
        proj_index["key"].isin(set(split_bugs["key"])), "file_version_id"
    ].unique()

    logger.info(
        "%s: reading %d train + %d %s distinct file versions",
        project,
        len(train_ids),
        len(split_ids),
        split,
    )
    train_text = read_versions(cfg, versions, train_ids)
    split_text = read_versions(cfg, versions, split_ids)

    vectorizer, transformer = fit_project(cfg, list(train_text.values()))

    split_id_list = list(split_text.keys())
    split_matrix = transformer.transform(
        vectorizer.transform([split_text[i] for i in split_id_list])
    )
    row_of_id = {vid: row for row, vid in enumerate(split_id_list)}
    path_of_id = versions.set_index("id")["path"]

    rows = []
    for bug in split_bugs.itertuples():
        candidate_ids = proj_index.loc[proj_index["key"] == bug.key, "file_version_id"].tolist()
        query = transformer.transform(vectorizer.transform([bug_text(bug)]))
        candidate_rows = [row_of_id[vid] for vid in candidate_ids]
        scores = (query @ split_matrix[candidate_rows].T).toarray().ravel()

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


def evaluate_project(
    cfg: Config,
    project: str,
    bugs: pd.DataFrame,
    versions: pd.DataFrame,
    index: pd.DataFrame,
) -> list[dict]:
    """Fit on this project's dev-train bugs, evaluate on its dev-val bugs."""
    scores = score_candidates(cfg, project, bugs, versions, index, split="val")
    val_bugs = bugs[bugs["split"] == "val"]
    gold = val_bugs[["project", "key", "gold_files"]]
    return metrics_from_scores(scores, gold).to_dict("records")


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
    rows = []
    for project, group in bugs.groupby("project"):
        rows += evaluate_project(cfg, project, group, versions, index)
    elapsed = time.time() - t0

    results = pd.DataFrame(rows)
    results.to_csv(ensure_dir(cfg.paths.tables) / OUTPUT_NAME, index=False)

    metric_cols = [c for c in results.columns if c not in ("project", "key", "n_candidates")]
    summary = results.groupby("project")[metric_cols].mean().round(4)
    logger.info("TF-IDF evaluation finished in %.1fs", elapsed)
    print(summary.to_string())
    print("\noverall:")
    print(results[metric_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()

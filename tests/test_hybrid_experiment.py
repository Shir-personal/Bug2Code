import pandas as pd
import pytest

from bug2code.localization.hybrid_experiment import (
    load_and_join,
    normalize_per_bug,
    sweep,
    verify_endpoint,
)

GOLD = pd.DataFrame(
    {
        "project": ["spark", "spark"],
        "key": ["SPARK-1", "SPARK-2"],
        "gold_files": [["A.scala"], ["C.scala"]],
    }
)


def _table(rows):
    return pd.DataFrame(rows, columns=["project", "key", "path", "candidate_order", "score"])


def test_normalize_per_bug_min_max():
    df = _table(
        [
            ["spark", "SPARK-1", "A.scala", 0, 0.0],
            ["spark", "SPARK-1", "B.scala", 1, 1.0],
            ["spark", "SPARK-1", "C.scala", 2, 2.0],
        ]
    )
    out = normalize_per_bug(df, "score", "norm")
    assert out["norm"].tolist() == [0.0, 0.5, 1.0]


def test_normalize_per_bug_flat_scores_are_zero():
    df = _table(
        [
            ["spark", "SPARK-1", "A.scala", 0, 3.0],
            ["spark", "SPARK-1", "B.scala", 1, 3.0],
        ]
    )
    out = normalize_per_bug(df, "score", "norm")
    assert out["norm"].tolist() == [0.0, 0.0]


def test_load_and_join_rejects_mismatched_candidate_order(tmp_path):
    tfidf = _table(
        [["spark", "SPARK-1", "A.scala", 0, 0.1], ["spark", "SPARK-1", "B.scala", 1, 0.2]]
    )
    codebert = _table(
        [["spark", "SPARK-1", "A.scala", 1, 0.5], ["spark", "SPARK-1", "B.scala", 0, 0.6]]
    )
    tfidf_path = tmp_path / "tfidf.parquet"
    codebert_path = tmp_path / "codebert.parquet"
    tfidf.to_parquet(tfidf_path)
    codebert.to_parquet(codebert_path)
    with pytest.raises(ValueError, match="orderings disagree"):
        load_and_join(tfidf_path, codebert_path)


def test_load_and_join_rejects_mismatched_population(tmp_path):
    tfidf = _table([["spark", "SPARK-1", "A.scala", 0, 0.1]])
    codebert = _table(
        [["spark", "SPARK-1", "A.scala", 0, 0.5], ["spark", "SPARK-1", "B.scala", 1, 0.6]]
    )
    tfidf_path = tmp_path / "tfidf.parquet"
    codebert_path = tmp_path / "codebert.parquet"
    tfidf.to_parquet(tfidf_path)
    codebert.to_parquet(codebert_path)
    with pytest.raises(ValueError, match="don't have matching"):
        load_and_join(tfidf_path, codebert_path)


def _merged():
    # SPARK-1: tfidf favors B, codebert favors A (gold=A) -> alpha shifts the winner.
    # SPARK-2: both favor C (gold=C) -> every alpha agrees.
    rows = [
        ["spark", "SPARK-1", "A.scala", 0, 0.1, 0.9],
        ["spark", "SPARK-1", "B.scala", 1, 0.9, 0.1],
        ["spark", "SPARK-2", "C.scala", 0, 0.9, 0.9],
        ["spark", "SPARK-2", "D.scala", 1, 0.1, 0.1],
    ]
    df = pd.DataFrame(
        rows, columns=["project", "key", "path", "candidate_order", "score_tfidf", "score_codebert"]
    )
    df = normalize_per_bug(df, "score_tfidf", "tfidf_norm")
    df = normalize_per_bug(df, "score_codebert", "codebert_norm")
    return df


def test_sweep_endpoints_match_pure_models():
    merged = _merged()
    summary = sweep(merged, GOLD, alphas=(0.0, 0.5, 1.0))
    zero = summary.loc[summary["alpha"] == 0.0].iloc[0]
    one = summary.loc[summary["alpha"] == 1.0].iloc[0]
    # alpha=0 -> pure tfidf: SPARK-1 top is B (wrong), SPARK-2 top is C (right) -> hit@1 = 0.5
    assert zero["hit_at_1"] == 0.5
    # alpha=1 -> pure codebert: SPARK-1 top is A (right), SPARK-2 top is C (right) -> hit@1 = 1.0
    assert one["hit_at_1"] == 1.0


def test_sweep_marks_best_alpha_by_mrr():
    merged = _merged()
    summary = sweep(merged, GOLD, alphas=(0.0, 0.5, 1.0))
    best = summary.loc[summary["is_best_by_val_mrr"]]
    assert len(best) >= 1
    assert best["mrr"].iloc[0] == summary["mrr"].max()


def test_verify_endpoint_raises_on_mismatch():
    summary = pd.DataFrame({"alpha": [0.0], "mrr": [0.1], "hit_at_1": [0.1]})
    with pytest.raises(AssertionError, match="does not reproduce"):
        verify_endpoint(summary, 0.0, {"mrr": 0.9}, "TF-IDF")


def test_verify_endpoint_passes_on_match():
    summary = pd.DataFrame({"alpha": [0.0], "mrr": [0.4161], "hit_at_1": [0.2969]})
    verify_endpoint(summary, 0.0, {"mrr": 0.4161, "hit_at_1": 0.2969}, "TF-IDF")

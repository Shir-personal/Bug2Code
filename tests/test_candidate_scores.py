import numpy as np
import pandas as pd

from bug2code.localization.candidate_scores import metrics_from_scores, ranked_paths


def test_ranked_paths_orders_by_score_descending():
    bug_scores = pd.DataFrame(
        {
            "path": ["a.py", "b.py", "c.py"],
            "candidate_order": [0, 1, 2],
            "score": [0.1, 0.9, 0.5],
        }
    )
    assert ranked_paths(bug_scores) == ["b.py", "c.py", "a.py"]


def test_ranked_paths_tie_break_matches_np_argsort_over_original_order():
    # Two tied scores: whichever np.argsort(-scores) would pick first, in the
    # candidate_order the live scoring pass used, must come first here too.
    scores = np.array([0.5, 0.9, 0.5])
    expected_order = np.argsort(-scores)
    bug_scores = pd.DataFrame(
        {
            "path": ["a.py", "b.py", "c.py"],
            "candidate_order": [0, 1, 2],
            "score": scores,
        }
    )
    assert ranked_paths(bug_scores) == [["a.py", "b.py", "c.py"][i] for i in expected_order]


def test_ranked_paths_restores_order_even_if_rows_are_shuffled():
    shuffled = pd.DataFrame(
        {
            "path": ["c.py", "a.py", "b.py"],
            "candidate_order": [2, 0, 1],
            "score": [0.5, 0.1, 0.9],
        }
    )
    assert ranked_paths(shuffled) == ["b.py", "c.py", "a.py"]


def test_metrics_from_scores_reproduces_score_one():
    scores = pd.DataFrame(
        {
            "project": ["spark", "spark", "spark"],
            "key": ["SPARK-1", "SPARK-1", "SPARK-1"],
            "path": ["a.py", "b.py", "c.py"],
            "candidate_order": [0, 1, 2],
            "score": [0.1, 0.9, 0.5],
        }
    )
    gold = pd.DataFrame({"project": ["spark"], "key": ["SPARK-1"], "gold_files": [["b.py"]]})
    result = metrics_from_scores(scores, gold).iloc[0]
    assert result["hit_at_1"] == 1.0
    assert result["mrr"] == 1.0
    assert result["n_candidates"] == 3

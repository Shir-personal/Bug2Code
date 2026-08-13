import pandas as pd

from bug2code.localization.hybrid_experiment import normalize_per_bug
from bug2code.localization.hybrid_test_experiment import ALPHA, hybrid_scores_at_alpha


def _merged():
    rows = [
        ["cassandra", "CASSANDRA-1", "A.java", 0, 0.1, 0.9],
        ["cassandra", "CASSANDRA-1", "B.java", 1, 0.9, 0.1],
    ]
    df = pd.DataFrame(
        rows, columns=["project", "key", "path", "candidate_order", "score_tfidf", "score_codebert"]
    )
    df = normalize_per_bug(df, "score_tfidf", "tfidf_norm")
    df = normalize_per_bug(df, "score_codebert", "codebert_norm")
    return df


def test_hybrid_scores_at_alpha_matches_pure_tfidf_at_zero():
    merged = _merged()
    out = hybrid_scores_at_alpha(merged, 0.0)
    assert out.set_index("path")["score"].to_dict() == {"A.java": 0.0, "B.java": 1.0}


def test_hybrid_scores_at_alpha_matches_pure_codebert_at_one():
    merged = _merged()
    out = hybrid_scores_at_alpha(merged, 1.0)
    assert out.set_index("path")["score"].to_dict() == {"A.java": 1.0, "B.java": 0.0}


def test_hybrid_scores_at_alpha_averages_at_half():
    merged = _merged()
    out = hybrid_scores_at_alpha(merged, 0.5)
    assert out.set_index("path")["score"].to_dict() == {"A.java": 0.5, "B.java": 0.5}


def test_test_alpha_is_fixed_at_validation_selected_value():
    # Locked by the Final Test design: alpha=0.5, selected on Validation MRR only.
    assert ALPHA == 0.5

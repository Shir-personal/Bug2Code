from bug2code.localization.metrics import (
    average_precision,
    best_gold_rank,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    score_one,
)

RANKED = ["A.java", "B.java", "C.java", "D.java", "E.java"]


def test_hit_at_k_true_when_gold_within_k():
    assert hit_at_k(RANKED, ["C.java"], k=3) == 1.0
    assert hit_at_k(RANKED, ["D.java"], k=3) == 0.0


def test_recall_at_k_is_fraction_of_gold_found():
    assert recall_at_k(RANKED, ["A.java", "D.java"], k=3) == 0.5
    assert recall_at_k(RANKED, [], k=3) == 0.0


def test_reciprocal_rank_of_first_hit():
    assert reciprocal_rank(RANKED, ["C.java", "A.java"]) == 1.0  # A.java is rank 1
    assert reciprocal_rank(RANKED, ["D.java"]) == 0.25
    assert reciprocal_rank(RANKED, ["missing.java"]) == 0.0


def test_average_precision_rewards_ranking_all_gold_highly():
    # Gold at ranks 1 and 3: precision@1 = 1/1, precision@3 = 2/3, mean = 5/6.
    assert average_precision(RANKED, ["A.java", "C.java"]) == (1 + 2 / 3) / 2


def test_average_precision_zero_when_nothing_found():
    assert average_precision(RANKED, ["missing.java"]) == 0.0


def test_score_one_returns_every_planned_metric():
    out = score_one(RANKED, ["A.java"])
    for key in ["hit_at_1", "hit_at_5", "hit_at_10", "recall_at_1", "mrr", "map"]:
        assert key in out
    assert out["hit_at_1"] == 1.0
    assert out["mrr"] == 1.0


def test_best_gold_rank_returns_earliest_position():
    assert best_gold_rank(RANKED, ["C.java", "A.java"]) == 1


def test_best_gold_rank_none_when_no_gold_present():
    assert best_gold_rank(RANKED, ["missing.java"]) is None

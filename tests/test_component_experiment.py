import pandas as pd

from bug2code.data.component_map import build_map
from bug2code.localization.component_experiment import evaluate_bug, reduction_stats, summarise

TRAIN = pd.DataFrame(
    [
        ["spark", ["SQL"], ["Parser.scala"], "train"],
        ["spark", ["SQL"], ["Analyzer.scala"], "train"],
    ],
    columns=["project", "components", "gold_files", "split"],
)
MAPPING = build_map(TRAIN, min_bugs=1)


def _bug_scores(paths_scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": list(paths_scores),
            "candidate_order": range(len(paths_scores)),
            "score": list(paths_scores.values()),
        }
    )


def test_evaluate_bug_full_uses_every_candidate():
    scores = _bug_scores({"Parser.scala": 0.9, "Other.scala": 0.5})
    row = evaluate_bug("spark", "SPARK-1", scores, ["Parser.scala"], MAPPING, ["SQL"])
    assert row["n_candidates_full"] == 2
    assert row["full_hit_at_1"] == 1.0


def test_evaluate_bug_filters_to_mapped_component_files():
    # "Other.scala" is not in SQL's mapped files, so it's excluded from FILTERED only.
    scores = _bug_scores({"Other.scala": 0.9, "Parser.scala": 0.5})
    row = evaluate_bug("spark", "SPARK-1", scores, ["Parser.scala"], MAPPING, ["SQL"])
    assert row["n_candidates_full"] == 2
    assert row["n_candidates_filtered"] == 1
    # FULL ranks Other.scala first (higher score) -> gold not in top-1.
    assert row["full_hit_at_1"] == 0.0
    # FILTERED only has Parser.scala, the gold file -> hit@1.
    assert row["filtered_hit_at_1"] == 1.0


def test_evaluate_bug_empty_filtered_set_scores_zero_not_dropped():
    scores = _bug_scores({"Parser.scala": 0.9})
    row = evaluate_bug("spark", "SPARK-1", scores, ["Parser.scala"], MAPPING, ["Streaming"])
    assert row["component_candidate_set_empty"] is True
    assert row["n_candidates_filtered"] == 0
    assert row["filtered_hit_at_1"] == 0.0
    assert row["filtered_mrr"] == 0.0


def test_reduction_stats_computes_pct_reduction():
    per_bug = pd.DataFrame({"n_candidates_full": [100, 50], "n_candidates_filtered": [10, 25]})
    stats = reduction_stats(per_bug)
    assert stats["mean_pct_reduction"] == round(100 * ((0.9 + 0.5) / 2), 2)


def test_summarise_produces_one_row_per_condition():
    per_bug = pd.DataFrame(
        {
            "full_hit_at_1": [1.0, 0.0],
            "filtered_hit_at_1": [1.0, 1.0],
            "full_mrr": [1.0, 0.5],
            "filtered_mrr": [1.0, 1.0],
        }
    )
    summary = summarise(per_bug)
    assert summary["condition"].tolist() == ["FULL", "COMPONENT_FILTERED"]
    assert summary.loc[summary["condition"] == "COMPONENT_FILTERED", "hit_at_1"].iloc[0] == 1.0

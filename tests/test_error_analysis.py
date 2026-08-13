import pandas as pd

from bug2code.data.component_map import build_map
from bug2code.localization.error_analysis import build_per_bug_table, rank_from_bug_scores

TRAIN = pd.DataFrame(
    [
        ["cassandra", ["SQL"], ["Parser.java"], "train"],
        ["cassandra", ["SQL"], ["Analyzer.java"], "train"],
    ],
    columns=["project", "components", "gold_files", "split"],
)
MAPPING = build_map(TRAIN, min_bugs=1)


def _scores(project, key, paths_scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project": project,
            "key": key,
            "path": list(paths_scores),
            "candidate_order": range(len(paths_scores)),
            "score": list(paths_scores.values()),
        }
    )


def test_rank_from_bug_scores_returns_best_gold_rank():
    scores = _scores("cassandra", "CASSANDRA-1", {"A.java": 0.9, "B.java": 0.1})
    assert rank_from_bug_scores(scores, ["B.java"]) == 2


def test_rank_from_bug_scores_none_when_gold_absent():
    scores = _scores("cassandra", "CASSANDRA-1", {"A.java": 0.9})
    assert rank_from_bug_scores(scores, ["missing.java"]) is None


def test_rank_from_bug_scores_none_on_empty_table():
    assert rank_from_bug_scores(pd.DataFrame(), ["A.java"]) is None


def test_build_per_bug_table_reports_rank_per_condition():
    val_bugs = pd.DataFrame(
        [["cassandra", "CASSANDRA-1", ["Parser.java"], ["SQL"]]],
        columns=["project", "key", "gold_files", "components"],
    )
    tfidf_scores = _scores("cassandra", "CASSANDRA-1", {"Parser.java": 0.1, "Other.java": 0.9})
    codebert_scores = _scores("cassandra", "CASSANDRA-1", {"Parser.java": 0.9, "Other.java": 0.1})
    hybrid_scores = _scores("cassandra", "CASSANDRA-1", {"Parser.java": 0.5, "Other.java": 0.5})

    per_bug = build_per_bug_table(val_bugs, tfidf_scores, codebert_scores, hybrid_scores, MAPPING)

    row = per_bug.iloc[0]
    assert row["tfidf_rank"] == 2  # TF-IDF ranks Other.java first, gold second
    assert row["finetuned_rank"] == 1  # Fine-tuned ranks the gold file first
    assert row["component_full_rank"] == 1  # same Fine-tuned scores, unfiltered
    # Other.java is not in SQL's mapped files, so COMPONENT_FILTERED only has the gold file.
    assert row["component_filtered_rank"] == 1


def test_build_per_bug_table_component_ranks_none_without_known_component():
    val_bugs = pd.DataFrame(
        [["cassandra", "CASSANDRA-2", ["Parser.java"], []]],
        columns=["project", "key", "gold_files", "components"],
    )
    tfidf_scores = _scores("cassandra", "CASSANDRA-2", {"Parser.java": 0.5})
    codebert_scores = _scores("cassandra", "CASSANDRA-2", {"Parser.java": 0.5})
    hybrid_scores = _scores("cassandra", "CASSANDRA-2", {"Parser.java": 0.5})

    per_bug = build_per_bug_table(val_bugs, tfidf_scores, codebert_scores, hybrid_scores, MAPPING)

    row = per_bug.iloc[0]
    assert row["component_full_rank"] is None
    assert row["component_filtered_rank"] is None

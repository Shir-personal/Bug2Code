import pandas as pd

from bug2code.localization.gold_count_error_analysis import (
    build_examples_table,
    build_multigold_diagnostic,
    build_per_bug_table,
    build_summary_table,
    classify_gold_count,
)

# --- classify_gold_count ------------------------------------------------------


def test_single_gold_file_is_single_gold():
    assert classify_gold_count(1) == "single_gold"


def test_two_or_more_gold_files_is_multi_gold():
    assert classify_gold_count(2) == "multi_gold"
    assert classify_gold_count(5) == "multi_gold"


# --- table builders ------------------------------------------------------------


def _fake_test_bugs():
    return pd.DataFrame(
        [
            {
                "project": "cassandra",
                "key": "CASSANDRA-1",
                "title": "NPE in Foo",
                "gold_files": ["Foo.java"],
            },
            {
                "project": "cassandra",
                "key": "CASSANDRA-2",
                "title": "Multiple files broken",
                "gold_files": ["Bar.java", "Baz.java"],
            },
        ]
    )


def _fake_scores(rank_map: dict) -> pd.DataFrame:
    """rank_map: {key: [ranked_path, ...]} -> a raw (project,key,path,order,score) table."""
    rows = []
    for key, paths in rank_map.items():
        n = len(paths)
        for order, path in enumerate(paths):
            rows.append(
                {
                    "project": "cassandra",
                    "key": key,
                    "path": path,
                    "candidate_order": order,
                    "score": float(n - order),
                }
            )
    return pd.DataFrame(rows)


def test_build_per_bug_table_computes_group_and_metrics():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Foo.java", "Other.java"],
            "CASSANDRA-2": ["Bar.java", "Other.java", "Baz.java"],
        }
    )
    codebert_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Other.java", "Foo.java"],
            "CASSANDRA-2": ["Bar.java", "Baz.java", "Other.java"],
        }
    )
    hybrid_scores = codebert_scores

    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)
    per_bug = per_bug.set_index("key")

    assert per_bug.loc["CASSANDRA-1", "group"] == "single_gold"
    assert per_bug.loc["CASSANDRA-1", "n_gold"] == 1
    assert per_bug.loc["CASSANDRA-2", "group"] == "multi_gold"
    assert per_bug.loc["CASSANDRA-2", "n_gold"] == 2

    assert per_bug.loc["CASSANDRA-1", "tfidf_rank"] == 1
    assert per_bug.loc["CASSANDRA-1", "finetuned_rank"] == 2

    # CASSANDRA-2: tfidf ranks both gold files at 1 and 3 -> recall@10 == 1.0
    assert per_bug.loc["CASSANDRA-2", "tfidf_recall_at_10"] == 1.0
    # codebert ranks both gold files at 1 and 2 -> recall@10 == 1.0 too
    assert per_bug.loc["CASSANDRA-2", "finetuned_recall_at_10"] == 1.0
    assert per_bug.loc["CASSANDRA-2", "finetuned_rr"] == 1.0


def test_build_summary_table_has_group_stats_and_deltas():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Foo.java", "Other.java"],
            "CASSANDRA-2": ["Other.java", "Bar.java", "Baz.java"],
        }
    )
    codebert_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Other.java", "Foo.java"],
            "CASSANDRA-2": ["Bar.java", "Baz.java", "Other.java"],
        }
    )
    hybrid_scores = codebert_scores
    gold = test_bugs[["project", "key", "gold_files"]]
    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)
    bug_meta = per_bug[["project", "key", "group", "n_gold"]]

    summary = build_summary_table(tfidf_scores, codebert_scores, hybrid_scores, gold, bug_meta)

    expected_methods = {
        "tfidf",
        "finetuned",
        "hybrid",
        "delta_finetuned_minus_tfidf",
        "delta_hybrid_minus_tfidf",
    }
    assert expected_methods <= set(summary["method"])
    groups_present = set(summary.loc[summary["method"] == "tfidf", "group"])
    assert groups_present == {"single_gold", "multi_gold"}

    is_multi = summary["group"] == "multi_gold"
    tfidf_row = summary[(summary["method"] == "tfidf") & is_multi].iloc[0]
    finetuned_row = summary[(summary["method"] == "finetuned") & is_multi].iloc[0]
    delta_row = summary[(summary["method"] == "delta_finetuned_minus_tfidf") & is_multi].iloc[0]
    assert delta_row["mrr"] == finetuned_row["mrr"] - tfidf_row["mrr"]
    assert delta_row["map"] == finetuned_row["map"] - tfidf_row["map"]
    assert delta_row["recall_at_10"] == finetuned_row["recall_at_10"] - tfidf_row["recall_at_10"]

    multi_row = summary[(summary["method"] == "tfidf") & is_multi].iloc[0]
    assert multi_row["mean_n_gold"] == 2
    assert multi_row["median_n_gold"] == 2


def test_build_multigold_diagnostic_only_covers_multi_gold_bugs():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Foo.java", "Other.java"],
            "CASSANDRA-2": ["Bar.java", "Other.java", "Baz.java"],
        }
    )
    codebert_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Other.java", "Foo.java"],
            "CASSANDRA-2": ["Other.java", "Bar.java", "Baz.java"],
        }
    )
    hybrid_scores = codebert_scores
    gold = test_bugs[["project", "key", "gold_files"]]
    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)
    bug_meta = per_bug[["project", "key", "group", "n_gold"]]

    diagnostic = build_multigold_diagnostic(
        tfidf_scores, codebert_scores, hybrid_scores, gold, bug_meta
    )

    assert set(diagnostic["group"]) == {"multi_gold"}
    assert (diagnostic["n_bugs"] == 1).all()

    tfidf_diag = diagnostic[diagnostic["method"] == "tfidf"].iloc[0]
    # CASSANDRA-2 tfidf: gold at ranks 1,3 - both in top10 -> all-in-top10 True.
    assert tfidf_diag["pct_at_least_one_in_top10"] == 100.0
    assert tfidf_diag["pct_all_in_top10"] == 100.0
    assert tfidf_diag["mean_recall_at_10"] == 1.0

    codebert_diag = diagnostic[diagnostic["method"] == "finetuned"].iloc[0]
    # CASSANDRA-2 finetuned: gold at ranks 2,3 - one gold file missing from top10? No, both present.
    assert codebert_diag["pct_at_least_one_in_top10"] == 100.0


def test_build_examples_table_picks_systematically_not_manually():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Foo.java", "Other.java"],
            "CASSANDRA-2": ["Bar.java", "Baz.java", "Other.java"],
        }
    )
    codebert_scores = _fake_scores(
        {
            "CASSANDRA-1": ["Other.java", "Foo.java"],
            "CASSANDRA-2": ["Other.java", "Bar.java", "Baz.java"],
        }
    )
    hybrid_scores = _fake_scores({"CASSANDRA-2": ["Bar.java", "Baz.java", "Other.java"]})
    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)

    examples = build_examples_table(per_bug, n=1)

    # Only CASSANDRA-2 is multi_gold, so it's the only candidate for every category.
    assert set(examples["key"]) == {"CASSANDRA-2"}
    assert "hybrid_beats_tfidf_recall_at_10" in set(examples["example_category"])
    assert "tfidf_beats_finetuned_recall_at_10" in set(examples["example_category"])

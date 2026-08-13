import pandas as pd

from bug2code.localization.lexical_error_analysis import (
    build_examples_table,
    build_per_bug_table,
    build_summary_table,
    classify,
    extract_features,
)

# --- extract_features / classify --------------------------------------------


def test_plain_prose_bug_report_has_no_signals():
    text = "The application crashes when a user logs in and tries to save their profile."
    features = extract_features(text)
    assert not any(
        features[k]
        for k in (
            "has_stacktrace",
            "has_source_filename",
            "has_method_pattern",
            "has_camelcase_identifier",
            "has_qualified_identifier",
            "has_exception_error",
        )
    )
    assert classify(features) == "plain_prose"


def test_source_filename_is_detected():
    features = extract_features("The bug is in ConsistencyLevel.java around line 120.")
    assert features["has_source_filename"]
    assert classify(features) == "identifier_rich"


def test_stacktrace_is_detected():
    text = (
        "Caused by: java.lang.ClassCastException\n\tat "
        "org.apache.cassandra.db.ConsistencyLevel.blockFor(ConsistencyLevel.java:120)"
    )
    features = extract_features(text)
    assert features["has_stacktrace"]
    assert classify(features) == "identifier_rich"


def test_exception_name_is_detected():
    features = extract_features("This results in a ClassCastException as reported by JAVA-241.")
    assert features["has_exception_error"]
    assert classify(features) == "identifier_rich"


def test_bare_error_and_exception_words_are_not_detected():
    # Ordinary English usage of "error"/"Error"/"exception"/"Exception" alone
    # must never be treated as a code identifier.
    features = extract_features(
        "There was an error in the process. An Exception occurred. This is an Error case."
    )
    assert not features["has_exception_error"]
    assert classify(features) == "plain_prose"


def test_qualified_identifier_is_detected():
    features = extract_features("The bug is in org.apache.cassandra.db.ConsistencyLevel.")
    assert features["has_qualified_identifier"]


def test_two_segment_abbreviation_is_not_a_qualified_identifier():
    features = extract_features("e.g. this happens a lot, and it's not clear why.")
    assert not features["has_qualified_identifier"]


def test_version_number_is_not_a_qualified_identifier():
    features = extract_features("Reproduced on 3.11.2 and 4.0.1 as well.")
    assert not features["has_qualified_identifier"]


def test_method_call_pattern_is_detected():
    features = extract_features("It seems validateForWrite(consistency, keyspace) rejects this.")
    assert features["has_method_pattern"]


def test_parenthetical_english_is_not_a_method_pattern():
    # A space before "(" (ordinary English parenthetical) must not match.
    features = extract_features("This is important (see the linked issue for details).")
    assert not features["has_method_pattern"]


def test_camelcase_identifier_is_detected():
    features = extract_features("It seems validateForWrite calls blockFor incorrectly.")
    assert features["has_camelcase_identifier"]
    assert features["camelcase_count"] >= 2


def test_single_camelcase_mention_alone_is_not_identifier_rich():
    # A single incidental PascalCase-like product name shouldn't tip the group
    # on its own - it needs 2+ camelcase matches, or a strong signal.
    features = extract_features("We integrated with GitHub for our CI pipeline.")
    assert features["camelcase_count"] == 1
    assert classify(features) == "plain_prose"


def test_two_camelcase_mentions_are_identifier_rich():
    text = "We saw issues in ConsistencyLevel and AbstractReplicationStrategy today."
    features = extract_features(text)
    assert features["camelcase_count"] >= 2
    assert classify(features) == "identifier_rich"


def test_ordinary_capitalized_words_are_not_camelcase():
    features = extract_features("The United States Government released a New York Times article.")
    assert not features["has_camelcase_identifier"]
    assert classify(features) == "plain_prose"


def test_identifier_count_sums_matches_across_categories():
    text = (
        "ClassCastException in ConsistencyLevel.java at "
        "org.apache.cassandra.db.ConsistencyLevel.blockFor(x)"
    )
    features = extract_features(text)
    assert features["identifier_count"] >= 4


# --- table builders ----------------------------------------------------------


def _fake_test_bugs():
    return pd.DataFrame(
        [
            {
                "project": "cassandra",
                "key": "CASSANDRA-1",
                "title": "NullPointerException in Foo.java",
                "description": "at org.apache.cassandra.db.Foo.bar(Foo.java:10)",
                "gold_files": ["Foo.java"],
            },
            {
                "project": "cassandra",
                "key": "CASSANDRA-2",
                "title": "App crashes on save",
                "description": "The app crashes whenever I try to save my changes.",
                "gold_files": ["Bar.java"],
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


def test_build_per_bug_table_computes_group_and_ranks():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {"CASSANDRA-1": ["Foo.java", "Other.java"], "CASSANDRA-2": ["Other.java", "Bar.java"]}
    )
    codebert_scores = _fake_scores(
        {"CASSANDRA-1": ["Foo.java", "Other.java"], "CASSANDRA-2": ["Bar.java", "Other.java"]}
    )
    hybrid_scores = codebert_scores

    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)
    per_bug = per_bug.set_index("key")

    assert per_bug.loc["CASSANDRA-1", "group"] == "identifier_rich"
    assert per_bug.loc["CASSANDRA-2", "group"] == "plain_prose"
    assert per_bug.loc["CASSANDRA-1", "tfidf_rank"] == 1
    assert per_bug.loc["CASSANDRA-2", "tfidf_rank"] == 2
    assert per_bug.loc["CASSANDRA-2", "finetuned_rank"] == 1


def test_build_summary_table_has_per_group_metrics_and_deltas():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {"CASSANDRA-1": ["Foo.java", "Other.java"], "CASSANDRA-2": ["Other.java", "Bar.java"]}
    )
    codebert_scores = _fake_scores(
        {"CASSANDRA-1": ["Foo.java", "Other.java"], "CASSANDRA-2": ["Bar.java", "Other.java"]}
    )
    hybrid_scores = codebert_scores
    gold = test_bugs[["project", "key", "gold_files"]]
    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)

    summary = build_summary_table(
        tfidf_scores, codebert_scores, hybrid_scores, gold, per_bug[["project", "key", "group"]]
    )

    methods = set(summary["method"])
    expected_methods = {
        "tfidf",
        "finetuned",
        "hybrid",
        "delta_finetuned_minus_tfidf",
        "delta_hybrid_minus_tfidf",
    }
    assert expected_methods <= methods
    groups_present = set(summary.loc[summary["method"] == "tfidf", "group"])
    assert groups_present == {"identifier_rich", "plain_prose"}

    is_plain_prose = summary["group"] == "plain_prose"
    delta_row = summary[
        (summary["method"] == "delta_finetuned_minus_tfidf") & is_plain_prose
    ].iloc[0]
    tfidf_row = summary[(summary["method"] == "tfidf") & is_plain_prose].iloc[0]
    finetuned_row = summary[(summary["method"] == "finetuned") & is_plain_prose].iloc[0]
    assert delta_row["mrr"] == finetuned_row["mrr"] - tfidf_row["mrr"]


def test_build_examples_table_picks_largest_rr_deltas_not_manually():
    test_bugs = _fake_test_bugs()
    tfidf_scores = _fake_scores(
        {"CASSANDRA-1": ["Foo.java", "Other.java"], "CASSANDRA-2": ["Other.java", "Bar.java"]}
    )
    codebert_scores = _fake_scores(
        {"CASSANDRA-1": ["Foo.java", "Other.java"], "CASSANDRA-2": ["Bar.java", "Other.java"]}
    )
    hybrid_scores = codebert_scores
    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)

    examples = build_examples_table(per_bug, n=1)

    assert set(examples["example_category"]) == {
        "tfidf_beats_finetuned",
        "finetuned_beats_tfidf",
        "hybrid_beats_tfidf",
    }
    finetuned_beats = examples[examples["example_category"] == "finetuned_beats_tfidf"].iloc[0]
    assert finetuned_beats["key"] == "CASSANDRA-2"

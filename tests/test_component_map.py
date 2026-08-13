import pandas as pd

from bug2code.data.component_map import (
    build_map,
    candidate_files,
    component_map_output_name,
    evaluate_coverage,
)


def _bugs(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["project", "components", "gold_files", "split"])


TRAIN = _bugs(
    [
        ["spark", ["SQL"], ["Parser.scala", "Analyzer.scala"], "train"],
        ["spark", ["SQL"], ["Optimizer.scala", "Analyzer.scala"], "train"],
        ["spark", ["Core", "SQL"], ["Executor.scala"], "train"],
    ]
)


def test_build_map_unions_the_files_of_a_components_training_fixes():
    mapping = build_map(TRAIN, min_bugs=1)
    sql = mapping[mapping["component"] == "SQL"].iloc[0]
    assert sql["train_bugs"] == 3
    assert sql["files"] == ["Analyzer.scala", "Executor.scala", "Optimizer.scala", "Parser.scala"]


def test_build_map_counts_a_multi_component_bug_under_each_component():
    mapping = build_map(TRAIN, min_bugs=1)
    assert mapping[mapping["component"] == "Core"].iloc[0]["files"] == ["Executor.scala"]


def test_build_map_skips_components_below_the_threshold():
    mapping = build_map(TRAIN, min_bugs=2)
    assert mapping["component"].tolist() == ["SQL"]


def test_candidate_files_is_empty_for_a_component_unseen_in_training():
    mapping = build_map(TRAIN, min_bugs=1)
    assert candidate_files(mapping, "spark", ["Streaming"]) == set()


def test_candidate_files_unions_several_components():
    mapping = build_map(TRAIN, min_bugs=1)
    assert candidate_files(mapping, "spark", ["Core", "Streaming"]) == {"Executor.scala"}


def test_evaluate_coverage_separates_uncovered_from_ineligible():
    mapping = build_map(TRAIN, min_bugs=1)
    val = _bugs(
        [
            ["spark", ["SQL"], ["Parser.scala"], "val"],  # covered
            ["spark", ["SQL"], ["Unknown.scala"], "val"],  # eligible, not covered
            ["spark", ["Streaming"], ["Kafka.scala"], "val"],  # empty candidate set
            ["spark", [], ["Executor.scala"], "val"],  # no component: ineligible
        ]
    )
    row = evaluate_coverage(mapping, val).iloc[0]
    assert (row["bugs"], row["eligible"], row["no_component"]) == (4, 3, 1)
    assert row["empty_candidate_set"] == 1
    assert row["coverage"] == round(1 / 3, 4)


def test_component_map_output_name_is_unchanged_for_the_default_subset():
    assert component_map_output_name("dev_subset.parquet") == "component_files.parquet"


def test_component_map_output_name_is_subset_specific_otherwise():
    assert (
        component_map_output_name("cassandra30_subset.parquet")
        == "component_files_cassandra30.parquet"
    )

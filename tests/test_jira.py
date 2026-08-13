"""Offline tests for the Jira ingestion module: no network is touched."""

from __future__ import annotations

from bug2code.data.jira import build_bug_jql, clean_text, issues_to_records, parse_issue

RAW_ISSUE = {
    "id": "12345",
    "key": "SPARK-1234",
    "fields": {
        "summary": "NullPointerException   in   Executor",
        "description": "Steps:\n{code:java}\nfoo();\n{code}\n\n\n\ncrashes.",
        "components": [{"name": "Spark Core"}, {"name": "Scheduler"}],
        "status": {"name": "Resolved"},
        "resolution": {"name": "Fixed"},
        "priority": {"name": "Major"},
        "created": "2016-03-02T10:00:00.000+0000",
        "resolutiondate": "2016-03-09T10:00:00.000+0000",
    },
}


def test_clean_text_strips_markup_and_collapses_whitespace():
    out = clean_text("a  b\t\tc {code:java} x {code}")
    assert "{code" not in out
    assert "  " not in out.replace("\n", " ").strip()


def test_clean_text_handles_missing_description():
    assert clean_text(None) == ""


def test_build_bug_jql_quotes_values_and_orders_ascending():
    jql = build_bug_jql(
        jira_key="SPARK",
        issue_types=["Bug"],
        statuses=["Resolved", "Closed"],
        resolutions=["Fixed"],
        created_after="2014-01-01",
        created_before="2021-01-01",
    )
    assert "project = SPARK" in jql
    assert 'status IN ("Resolved", "Closed")' in jql
    assert 'created >= "2014-01-01"' in jql
    assert jql.endswith("ORDER BY created ASC")


def test_parse_issue_extracts_all_fields():
    issue = parse_issue(RAW_ISSUE, project="spark")
    assert issue.key == "SPARK-1234"
    assert issue.project == "spark"
    assert issue.issue_id == "12345"
    assert issue.components == ["Spark Core", "Scheduler"]
    assert issue.n_components == 2
    assert issue.resolution == "Fixed"
    assert "{code" not in issue.description
    assert "\n\n\n" not in issue.description


def test_parse_issue_tolerates_empty_optional_fields():
    issue = parse_issue({"id": 7, "key": "HBASE-1", "fields": {}}, project="hbase")
    assert issue.summary == ""
    assert issue.description == ""
    assert issue.components == []
    assert issue.status == ""


def test_issues_to_records_is_dataframe_ready():
    records = issues_to_records([parse_issue(RAW_ISSUE, "spark")])
    assert records[0]["key"] == "SPARK-1234"
    assert set(records[0]) == {
        "key",
        "project",
        "issue_id",
        "summary",
        "description",
        "components",
        "status",
        "resolution",
        "priority",
        "created",
        "resolved",
    }

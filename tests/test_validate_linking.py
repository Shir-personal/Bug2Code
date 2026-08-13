"""Tests for the independent audits of the collected data."""

from __future__ import annotations

from bug2code.data.validate_linking import agreement_row, commit_urls_to_shas


def test_commit_urls_to_shas_extracts_and_lowercases():
    value = (
        "https://github.com/apache/cassandra/commit/ABC1234DEF5678, "
        "https://github.com/apache/cassandra/commit/0badc0ffee"
    )
    assert commit_urls_to_shas(value) == {"abc1234def5678", "0badc0ffee"}


def test_commit_urls_to_shas_ignores_non_commit_urls():
    assert commit_urls_to_shas("https://github.com/apache/cassandra/pull/42") == set()
    assert commit_urls_to_shas(None) == set()


def test_agreement_row_matches_abbreviated_shas_by_prefix():
    links = {"CASSANDRA-1": {"abc1234"}, "CASSANDRA-2": {"deadbee"}}
    chosen = {"CASSANDRA-1": "abc1234567890", "CASSANDRA-2": "ffffffffffff"}

    row = agreement_row("cassandra", links, chosen)

    assert row["issues_with_field"] == 2
    assert row["also_linked_by_us"] == 2
    assert row["same_commit"] == 1
    assert row["agreement"] == 0.5


def test_agreement_row_counts_only_issues_we_also_linked():
    links = {"CASSANDRA-1": {"abc"}, "CASSANDRA-9": {"xyz"}}
    row = agreement_row("cassandra", links, {"CASSANDRA-1": "abc123"})

    assert row["issues_with_field"] == 2
    assert row["also_linked_by_us"] == 1
    assert row["agreement"] == 1.0


def test_agreement_row_is_none_without_overlap():
    assert agreement_row("spark", {}, {"SPARK-1": "abc"})["agreement"] is None

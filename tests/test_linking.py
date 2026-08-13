"""Tests for commit-record parsing and the fix-commit selection policy."""

from __future__ import annotations

from bug2code.data.linking import (
    is_source_file,
    key_pattern,
    parse_record,
    select_fix_commits,
)

POLICY = {
    "multi_commit_policy": "earliest_on_default_branch",
    "exclude_merge_commits": True,
    "exclude_revert_commits": True,
    "max_files_changed": 30,
    "min_files_changed": 1,
}

RECORD = (
    "\x01abc123\x02par1 par2\x022016-03-09T10:00:00+00:00\x02Dev\x02"
    "[SPARK-1234] fix NPE\n\nlong body\n\x03\n\n"
    "M\tcore/src/main/scala/A.scala\n"
    "A\tcore/src/main/scala/B.scala\n"
    "R100\tcore/src/main/scala/old.scala\tcore/src/main/scala/new.scala\n"
)


def test_parse_record_reads_header_and_files():
    commit = parse_record(RECORD)
    assert commit.sha == "abc123"
    assert commit.parents == ["par1", "par2"]
    assert commit.is_merge
    assert commit.parent == "par1"
    assert commit.subject == "[SPARK-1234] fix NPE"
    assert commit.statuses == ["M", "A", "R100"]
    assert commit.files[-1] == "core/src/main/scala/new.scala"


def test_parse_record_detects_reverts():
    text = '\x01s\x02p\x02d\x02a\x02Revert "[HBASE-1] x"\x03\n'
    assert parse_record(text).is_revert


def test_key_pattern_matches_whole_tokens_only():
    pattern = key_pattern("SPARK")
    assert pattern.findall("fix spark-12 and SPARK-345") == ["spark-12", "SPARK-345"]
    assert pattern.findall("MYSPARK-12") == []


def test_is_source_file_applies_extension_root_and_exclusions():
    kwargs = {
        "extensions": [".java", ".scala"],
        "roots": ["core/src/main"],
        "excludes": ["/test/"],
    }
    assert is_source_file("core/src/main/scala/A.scala", **kwargs)
    assert not is_source_file("core/src/main/scala/A.md", **kwargs)
    assert not is_source_file("docs/A.scala", **kwargs)
    assert not is_source_file("core/src/main/test/A.scala", **kwargs)


def _row(**overrides):
    base = {
        "project": "spark",
        "key": "SPARK-1",
        "sha": "a",
        "commit_date": "2016-01-01T00:00:00+00:00",
        "is_merge": False,
        "is_revert": False,
        "on_default_branch": True,
        "n_files_source": 3,
    }
    return {**base, **overrides}


def test_select_fix_commits_rejects_by_policy():
    rows = [
        _row(key="A", is_merge=True),
        _row(key="B", is_revert=True),
        _row(key="C", on_default_branch=False),
        _row(key="D", n_files_source=0),
        _row(key="E", n_files_source=31),
    ]
    kept, reasons = select_fix_commits(rows, POLICY)
    assert kept == []
    assert reasons["merge_commit"] == 1
    assert reasons["revert_commit"] == 1
    assert reasons["off_default_branch"] == 1
    assert reasons["no_source_files"] == 1
    assert reasons["too_many_files"] == 1


def test_select_fix_commits_takes_the_earliest_candidate():
    rows = [
        _row(sha="late", commit_date="2016-05-01T00:00:00+00:00"),
        _row(sha="early", commit_date="2016-01-01T00:00:00+00:00"),
    ]
    kept, reasons = select_fix_commits(rows, POLICY)
    assert len(kept) == 1
    assert kept[0]["sha"] == "early"
    assert kept[0]["n_candidates"] == 2
    assert reasons["not_earliest"] == 1

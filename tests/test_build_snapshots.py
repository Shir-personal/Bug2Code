import subprocess

import pytest

from bug2code.data.build_snapshots import gold_in_snapshot, snapshot_files


def test_gold_in_snapshot_drops_files_absent_before_the_fix():
    candidates = {"src/A.java", "src/B.java"}
    gold = ["src/A.java", "src/New.java", "src/B.java"]
    assert gold_in_snapshot(gold, candidates) == ["src/A.java", "src/B.java"]


def test_gold_in_snapshot_can_end_up_empty():
    assert gold_in_snapshot(["src/New.java"], {"src/A.java"}) == []


def _commit(repo, files: dict[str, str], message: str) -> str:
    """Write files into a repo and commit them, returning the new sha."""
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True)
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A two-commit repository, so a snapshot can differ from the latest state."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in [("user.email", "t@t"), ("user.name", "t")]:
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    first = _commit(
        tmp_path,
        {
            "src/main/A.java": "class A {}",
            "src/main/B.java": "class B {}",
            "src/test/ATest.java": "class ATest {}",
            "README.md": "docs",
        },
        "first",
    )
    _commit(tmp_path, {"src/main/C.java": "class C {}"}, "second")
    return tmp_path, first


def test_snapshot_files_lists_the_whole_state_not_the_commit_diff(repo):
    path, _ = repo
    # The second commit changed only C.java, yet its snapshot holds every file.
    assert snapshot_files(path, "HEAD", lambda p: p.endswith(".java")) == {
        "src/main/A.java",
        "src/main/B.java",
        "src/main/C.java",
        "src/test/ATest.java",
    }


def test_snapshot_files_reads_the_historical_state(repo):
    path, first = repo
    # C.java exists today but not in the first commit's state.
    assert snapshot_files(path, first, lambda p: p.endswith(".java")) == {
        "src/main/A.java",
        "src/main/B.java",
        "src/test/ATest.java",
    }


def test_snapshot_files_applies_the_predicate(repo):
    path, _ = repo

    def predicate(path_: str) -> bool:
        return path_.endswith(".java") and "/test/" not in path_

    assert snapshot_files(path, "HEAD", predicate) == {
        "src/main/A.java",
        "src/main/B.java",
        "src/main/C.java",
    }

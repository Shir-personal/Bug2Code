import subprocess

import pandas as pd
import pytest

from bug2code.localization.candidates import build, read_blobs, snapshot_blobs


def _commit(repo, files: dict[str, str], message: str) -> str:
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
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in [("user.email", "t@t"), ("user.name", "t")]:
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    first = _commit(tmp_path, {"src/A.java": "class A {}", "src/B.java": "class B {}"}, "first")
    # B.java is unchanged, so it must reuse the same blob sha across commits.
    second = _commit(tmp_path, {"src/A.java": "class A { int x; }"}, "second")
    return tmp_path, first, second


def test_snapshot_blobs_maps_path_to_content_hash(repo):
    path, first, _ = repo
    blobs = snapshot_blobs(path, first, lambda p: p.endswith(".java"))
    assert set(blobs) == {"src/A.java", "src/B.java"}
    assert all(len(sha) == 40 for sha in blobs.values())


def test_snapshot_blobs_unchanged_file_keeps_the_same_blob_across_commits(repo):
    path, first, second = repo
    blobs_first = snapshot_blobs(path, first, lambda p: p.endswith(".java"))
    blobs_second = snapshot_blobs(path, second, lambda p: p.endswith(".java"))
    assert blobs_first["src/B.java"] == blobs_second["src/B.java"]
    assert blobs_first["src/A.java"] != blobs_second["src/A.java"]


def test_read_blobs_returns_the_exact_file_contents(repo):
    path, first, _ = repo
    blobs = snapshot_blobs(path, first, lambda p: p.endswith(".java"))
    texts = read_blobs(path, list(blobs.values()))
    assert texts[blobs["src/A.java"]] == "class A {}"
    assert texts[blobs["src/B.java"]] == "class B {}"


def test_read_blobs_on_empty_list_returns_empty_dict(repo):
    path, _, _ = repo
    assert read_blobs(path, []) == {}


class _FakeSpec:
    source_roots = ["src"]


class _FakeConfig:
    candidates = {
        "source_extensions": [".java"],
        "exclude_path_patterns": [],
    }
    github = {"clone_dir": "unused"}

    def project(self, name):
        return _FakeSpec()


def test_build_deduplicates_the_unchanged_file_across_two_bugs(repo, monkeypatch):
    path, first, second = repo
    project = path.name
    monkeypatch.setattr("bug2code.localization.candidates.resolve", lambda p: path.parent)
    bugs = pd.DataFrame(
        {
            "project": [project, project],
            "key": ["DEMO-1", "DEMO-2"],
            "parent_sha": [first, second],
        }
    )
    versions, index = build(_FakeConfig(), bugs)

    # Two bugs x two files = 4 candidate rows, but B.java is the same blob both times.
    assert len(index) == 4
    b_versions = versions[versions["path"] == "src/B.java"]
    assert len(b_versions) == 1
    a_versions = versions[versions["path"] == "src/A.java"]
    assert len(a_versions) == 2

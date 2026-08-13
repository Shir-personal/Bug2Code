"""Exact per-bug candidate files, deduplicated by git blob content.

Every bug's candidate set is still exactly the in-scope source files present at
its own ``parent_sha`` — nothing approximate or shared across bugs.
The only optimisation is bookkeeping: consecutive bugs' snapshots overlap
heavily, so the same file content recurs across thousands of bugs. A git blob
SHA is a hash of the file's bytes, so grouping by blob SHA merges only byte-
identical files, never merges different ones, and changes no candidate list.

This module builds two small artifacts once:

- ``file_versions.parquet`` — one row per distinct ``(project, path, blob_sha)``.
- ``snapshot_index.parquet`` — for every bug, which file-version rows are in its
  candidate set.

Downstream code (TF-IDF, frozen CodeBERT) processes each distinct file version
once and looks up a bug's candidates via the index, instead of re-reading and
re-processing identical file content for every bug that happens to see it.

Usage:
    python -m bug2code.localization.candidates [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
import subprocess
import threading
from pathlib import Path

import pandas as pd

from bug2code.config import Config, load_config
from bug2code.data.linking import is_source_file
from bug2code.data.repos import run_git
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir, resolve

logger = get_logger(__name__)

FILE_VERSIONS_NAME = "file_versions.parquet"
SNAPSHOT_INDEX_NAME = "snapshot_index.parquet"


def snapshot_blobs(repo: Path, sha: str, predicate) -> dict[str, str]:
    """Map in-scope path -> blob sha for every file present at one commit."""
    listing = run_git(repo, ["ls-tree", "-r", sha])
    out: dict[str, str] = {}
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        if predicate(path):
            out[path] = meta.split()[2]
    return out


def read_blobs(repo: Path, blob_shas: list[str]) -> dict[str, str]:
    """Read many blobs' text content with a single ``git cat-file --batch`` process.

    Writing every requested SHA before reading any output would deadlock once
    the list is large: git fills the stdout pipe buffer and blocks on writing
    to it before we finish writing stdin, so stdin never drains either. Feeding
    stdin from a separate thread lets the two directions proceed concurrently.
    """
    if not blob_shas:
        return {}
    proc = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def feed() -> None:
        assert proc.stdin is not None
        proc.stdin.write(("\n".join(blob_shas) + "\n").encode())
        proc.stdin.close()

    writer = threading.Thread(target=feed)
    writer.start()

    out: dict[str, str] = {}
    for sha in blob_shas:
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size + 1)[:-1]
        out[sha] = data.decode("utf-8", errors="replace")
    writer.join()
    proc.wait()
    return out


def read_versions(cfg: Config, versions: pd.DataFrame, ids) -> dict[int, str]:
    """Read text content for a set of file-version ids, grouped by project repo.

    Distinct blobs are read once even if several ids share a project, since
    ``read_blobs`` is called once per project with the full list it needs.
    """
    wanted = versions[versions["id"].isin(ids)]
    out: dict[int, str] = {}
    for project, group in wanted.groupby("project"):
        repo = resolve(cfg.github["clone_dir"]) / project
        blob_text = read_blobs(repo, group["blob_sha"].tolist())
        for row in group.itertuples():
            out[row.id] = blob_text[row.blob_sha]
    return out


def build(cfg: Config, bugs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the file-version table and the per-bug candidate index.

    ``bugs`` must have ``project``, ``key`` and ``parent_sha`` columns.
    """
    version_id: dict[tuple[str, str, str], int] = {}
    versions: list[dict] = []
    index: list[dict] = []

    for project, group in bugs.groupby("project"):
        spec = cfg.project(project)
        repo = resolve(cfg.github["clone_dir"]) / project

        def predicate(path: str, spec=spec) -> bool:
            return is_source_file(
                path,
                extensions=cfg.candidates["source_extensions"],
                roots=spec.source_roots,
                excludes=cfg.candidates["exclude_path_patterns"],
            )

        logger.info("%s: listing %d snapshots", project, len(group))
        for bug in group.itertuples():
            for path, blob in snapshot_blobs(repo, bug.parent_sha, predicate).items():
                key = (project, path, blob)
                vid = version_id.get(key)
                if vid is None:
                    vid = len(versions)
                    version_id[key] = vid
                    versions.append({"id": vid, "project": project, "path": path, "blob_sha": blob})
                index.append({"project": project, "key": bug.key, "file_version_id": vid})

    return pd.DataFrame(versions), pd.DataFrame(index)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])

    versions, index = build(cfg, bugs)

    cache_dir = ensure_dir(cfg.paths.cache / "candidates")
    versions.to_parquet(cache_dir / FILE_VERSIONS_NAME, index=False)
    index.to_parquet(cache_dir / SNAPSHOT_INDEX_NAME, index=False)

    dedup = len(index) / max(len(versions), 1)
    logger.info(
        "%d bugs, %d (bug,file) pairs, %d distinct file versions (%.1fx dedup)",
        bugs["key"].nunique(),
        len(index),
        len(versions),
        dedup,
    )
    print(versions.groupby("project").size().rename("distinct_file_versions").to_string())


if __name__ == "__main__":
    main()

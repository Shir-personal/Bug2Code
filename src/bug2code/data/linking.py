"""Link Jira issues to their fixing commits by scanning local git history.

The whole history of a project is walked once with ``git log --all --name-status``
and every commit message is searched for issue keys. Nothing is discarded during
the scan: filtering policies (merges, reverts, oversized commits, several
commits per key) are applied afterwards by :func:`select_fix_commits`, which
also reports why each candidate was rejected so the dataset report can account
for every dropped bug.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bug2code.data.repos import run_git, stream_git
from bug2code.logging_utils import get_logger

logger = get_logger(__name__)

# \x01/\x02/\x03 are control chars that never appear in commit messages, so they safely delimit fields.
_LOG_FORMAT = "\x01%H\x02%P\x02%cI\x02%an\x02%B\x03"
_LOG_ARGS = ["log", "--all", "--no-decorate", "--name-status", f"--pretty=format:{_LOG_FORMAT}"]

_REVERT = re.compile(r"^revert\b|this reverts commit", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Commit:
    """One commit, with the paths it touched."""

    sha: str
    parents: list[str]
    date: str
    author: str
    message: str
    statuses: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def is_merge(self) -> bool:
        """Whether the commit has more than one parent."""
        return len(self.parents) > 1

    @property
    def is_revert(self) -> bool:
        """Whether the message marks the commit as a revert."""
        return bool(_REVERT.search(self.message))

    @property
    def parent(self) -> str:
        """First parent: the tree state the fix was applied to."""
        return self.parents[0] if self.parents else ""

    @property
    def subject(self) -> str:
        """First line of the commit message."""
        return self.message.splitlines()[0] if self.message else ""


def parse_record(text: str) -> Commit:
    """Parse one ``git log`` record produced with the module's pretty format."""
    header, _, body = text.partition("\x03")
    sha, parents, date, author, message = header.lstrip("\x01").split("\x02", 4)

    statuses: list[str] = []
    files: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        statuses.append(parts[0])
        files.append(parts[-1])  # renames and copies report the destination path

    return Commit(
        sha=sha,
        parents=parents.split() if parents else [],
        date=date,
        author=author,
        message=message,
        statuses=statuses,
        files=files,
    )


def iter_commits(repo: Path) -> Iterator[Commit]:
    """Stream every reachable commit of ``repo`` together with its changed paths."""
    buffer: list[str] = []
    for line in stream_git(repo, _LOG_ARGS):
        if line.startswith("\x01") and buffer:
            yield parse_record("".join(buffer))
            buffer = []
        buffer.append(line)
    if buffer:
        yield parse_record("".join(buffer))


def default_branch_shas(repo: Path, branch: str) -> set[str]:
    """SHAs reachable from the project's default branch on the origin remote."""
    return set(run_git(repo, ["rev-list", f"origin/{branch}"]).split())


def key_pattern(jira_key: str) -> re.Pattern[str]:
    """Regex matching any issue key of one project as a whole token."""
    return re.compile(rf"\b{re.escape(jira_key)}-\d+\b", re.IGNORECASE)


def is_source_file(
    path: str,
    extensions: Iterable[str],
    roots: Iterable[str],
    excludes: Iterable[str],
) -> bool:
    """Whether a repository path belongs to the product-code candidate universe."""
    if not any(path.endswith(ext) for ext in extensions):
        return False
    if not any(path.startswith(root) for root in roots):
        return False
    return not any(pattern in path for pattern in excludes)


def scan_repo(
    repo: Path,
    project: str,
    jira_key: str,
    known_keys: set[str],
    default_branch: str,
    source_predicate: Any,
) -> list[dict[str, Any]]:
    """Return one row per (issue key, candidate commit) pair found in the history.

    Args:
        repo: Local clone.
        project: Short project name, copied onto every row.
        jira_key: Jira key prefix used to recognise references, e.g. ``SPARK``.
        known_keys: Only these keys are kept; issues outside the study window
            are ignored.
        default_branch: Branch name on the ``origin`` remote.
        source_predicate: ``(path) -> bool`` selecting candidate source files.
    """
    on_branch = default_branch_shas(repo, default_branch)
    pattern = key_pattern(jira_key)
    logger.info("%s: %d commits on origin/%s", project, len(on_branch), default_branch)

    rows: list[dict[str, Any]] = []
    n_commits = 0
    for commit in iter_commits(repo):
        n_commits += 1
        mentioned = {k.upper() for k in pattern.findall(commit.message)}
        matched = mentioned & known_keys
        if not matched:
            continue

        source = [
            (status, path)
            for status, path in zip(commit.statuses, commit.files, strict=True)
            if source_predicate(path)
        ]
        for key in matched:
            rows.append(
                {
                    "project": project,
                    "key": key,
                    "sha": commit.sha,
                    "parent_sha": commit.parent,
                    "commit_date": commit.date,
                    "author": commit.author,
                    "subject": commit.subject[:200],
                    "is_merge": commit.is_merge,
                    "is_revert": commit.is_revert,
                    "on_default_branch": commit.sha in on_branch,
                    "n_keys_in_message": len(mentioned),
                    "n_files_total": len(commit.files),
                    "n_files_source": len(source),
                    "source_files": [p for _, p in source],
                    "source_statuses": [s for s, _ in source],
                }
            )

    logger.info("%s: scanned %d commits, %d key-commit pairs", project, n_commits, len(rows))
    return rows


def _rejection(row: dict[str, Any], policy: dict[str, Any]) -> str | None:
    """Return why a candidate commit is unusable, or None if it is usable."""
    if policy["exclude_merge_commits"] and row["is_merge"]:
        return "merge_commit"
    if policy["exclude_revert_commits"] and row["is_revert"]:
        return "revert_commit"
    if not row["on_default_branch"]:
        return "off_default_branch"
    if row["n_files_source"] < policy["min_files_changed"]:
        return "no_source_files"
    if row["n_files_source"] > policy["max_files_changed"]:
        return "too_many_files"
    return None


def select_fix_commits(
    rows: Sequence[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Choose one fixing commit per issue and count every rejection.

    Args:
        rows: Candidate pairs from :func:`scan_repo`.
        policy: The ``linking`` config block.

    Returns:
        The selected rows (one per linked issue, with ``n_candidates`` added)
        and a counter of rejection reasons at candidate level.
    """
    reasons: Counter[str] = Counter()
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        reason = _rejection(row, policy)
        if reason:
            reasons[reason] += 1
            continue
        by_key[row["key"]].append(row)

    strategy = policy["multi_commit_policy"]
    if strategy != "earliest_on_default_branch":
        raise ValueError(f"unsupported multi_commit_policy: {strategy!r}")

    selected: list[dict[str, Any]] = []
    for candidates in by_key.values():
        chosen = min(candidates, key=lambda r: r["commit_date"])
        selected.append({**chosen, "n_candidates": len(candidates)})
        reasons["not_earliest"] += len(candidates) - 1

    return sorted(selected, key=lambda r: (r["project"], r["commit_date"])), reasons

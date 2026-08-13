"""Local clones of the Apache mirrors.

Commit linking and snapshot extraction both run against local clones rather than
the GitHub API: the API is rate limited to 60 unauthenticated requests per hour
and 10 search requests per hour, while a local ``git log`` scan of the whole
history costs a fraction of a second and can be re-run whenever a linking policy
changes.

Usage:
    python -m bug2code.data.repos [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterator
from pathlib import Path

from bug2code.config import Config, load_config
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

GIT_URL_TEMPLATE = "https://github.com/{slug}.git"

# Disable path quoting so non-ASCII file names come back verbatim.
_BASE_ARGS = ["-c", "core.quotepath=false"]


class GitError(RuntimeError):
    """A git subprocess exited with a non-zero status."""


def run_git(repo: Path, args: list[str], timeout_s: int | None = None) -> str:
    """Run a git command inside ``repo`` and return its stdout."""
    result = subprocess.run(
        ["git", *_BASE_ARGS, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout


def stream_git(repo: Path, args: list[str]) -> Iterator[str]:
    """Yield stdout lines of a git command, without buffering the whole output.

    History dumps are large enough that materialising them as one string wastes
    memory for no benefit.
    """
    process = subprocess.Popen(
        ["git", *_BASE_ARGS, "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr else ""
        if process.stderr:
            process.stderr.close()
        if process.wait() != 0:
            raise GitError(f"git {' '.join(args)} failed in {repo}: {stderr.strip()}")


def is_repo(path: Path) -> bool:
    """Whether ``path`` is an existing git working copy."""
    return (path / ".git").is_dir()


def clone_or_update(slug: str, dest: Path, partial: bool = False) -> Path:
    """Clone ``owner/name`` into ``dest``, or fetch if it is already there.

    Args:
        slug: GitHub ``owner/name`` of the mirror.
        dest: Target directory for the working copy.
        partial: Use a ``blob:none`` partial clone. Smaller, but every later
            file read triggers a network fetch, so snapshot extraction is far
            slower. Off by default.
    """
    if is_repo(dest):
        logger.info("%s: fetching updates", slug)
        run_git(dest, ["fetch", "--all", "--tags", "--prune"])
        return dest

    ensure_dir(dest.parent)
    args = ["clone", "--no-checkout"]
    if partial:
        args.append("--filter=blob:none")
    args += [GIT_URL_TEMPLATE.format(slug=slug), str(dest)]

    logger.info("%s: cloning into %s", slug, dest)
    result = subprocess.run(["git", *_BASE_ARGS, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise GitError(f"clone of {slug} failed: {result.stderr.strip()}")
    return dest


def count_commits(repo: Path) -> int:
    """Number of reachable commits across all refs."""
    return int(run_git(repo, ["rev-list", "--all", "--count"]).strip())


def ensure_repos(cfg: Config) -> dict[str, Path]:
    """Make sure every configured project has a local clone; return their paths."""
    clone_dir = ensure_dir(cfg.github["clone_dir"])
    partial = bool(cfg.github["partial_clone"])

    paths: dict[str, Path] = {}
    for project in cfg.projects:
        spec = cfg.project(project)
        dest = clone_dir / project
        paths[project] = clone_or_update(spec.github, dest, partial=partial)
        logger.info("%s: %d commits", project, count_commits(dest))
    return paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    paths = ensure_repos(load_config(args.config))
    for project, path in paths.items():
        print(f"{project:10s} {path}")


if __name__ == "__main__":
    main()

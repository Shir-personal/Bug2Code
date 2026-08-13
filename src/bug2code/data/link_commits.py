"""Link collected Jira issues to fixing commits and report the link rate.

Usage:
    python -m bug2code.data.link_commits [--config configs/exp.yaml]

Reads ``data/raw/issues.parquet``, writes ``data/interim/commit_candidates.parquet``
(every key-commit pair found) and ``data/interim/fix_commits.parquet`` (the one
commit selected per issue), plus a filtering audit table under reports/tables.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import partial

import pandas as pd

from bug2code.config import Config, load_config
from bug2code.data.collect_issues import OUTPUT_NAME as ISSUES_NAME
from bug2code.data.linking import is_source_file, scan_repo, select_fix_commits
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir, resolve

logger = get_logger(__name__)

CANDIDATES_NAME = "commit_candidates.parquet"
FIX_COMMITS_NAME = "fix_commits.parquet"


def link(cfg: Config, issues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan every clone and select fixing commits.

    Returns:
        Candidate pairs, selected fix commits, and a per-project audit table.
    """
    candidates: list[dict] = []
    audit_rows: list[dict] = []
    selected: list[dict] = []

    for project in cfg.projects:
        spec = cfg.project(project)
        keys = set(issues.loc[issues["project"] == project, "key"])
        predicate = partial(
            is_source_file,
            extensions=cfg.candidates["source_extensions"],
            roots=spec.source_roots,
            excludes=cfg.candidates["exclude_path_patterns"],
        )
        rows = scan_repo(
            repo=resolve(cfg.github["clone_dir"]) / project,
            project=project,
            jira_key=spec.jira_key,
            known_keys=keys,
            default_branch=spec.default_branch,
            source_predicate=predicate,
        )
        kept, reasons = select_fix_commits(rows, cfg.linking)

        candidates.extend(rows)
        selected.extend(kept)
        audit_rows.append(_audit_row(project, keys, rows, kept, reasons))

    return (
        pd.DataFrame(candidates),
        pd.DataFrame(selected),
        pd.DataFrame(audit_rows),
    )


def _audit_row(
    project: str,
    keys: set[str],
    rows: list[dict],
    kept: list[dict],
    reasons: Counter[str],
) -> dict:
    """One row accounting for every issue of a project: linked, or why not."""
    with_any_commit = {r["key"] for r in rows}
    linked = {r["key"] for r in kept}
    return {
        "project": project,
        "issues": len(keys),
        "issues_with_any_commit": len(with_any_commit),
        "issues_linked": len(linked),
        "link_rate": round(len(linked) / len(keys), 4) if keys else 0.0,
        "issues_all_candidates_rejected": len(with_any_commit - linked),
        "candidate_pairs": len(rows),
        "rejected_merge_commit": reasons["merge_commit"],
        "rejected_revert_commit": reasons["revert_commit"],
        "rejected_off_default_branch": reasons["off_default_branch"],
        "rejected_no_source_files": reasons["no_source_files"],
        "rejected_too_many_files": reasons["too_many_files"],
        "dropped_not_earliest": reasons["not_earliest"],
        "issues_multi_commit": sum(1 for r in kept if r["n_candidates"] > 1),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    issues = pd.read_parquet(cfg.paths.data_raw / ISSUES_NAME)
    candidates, fixes, audit = link(cfg, issues)

    interim = ensure_dir(cfg.paths.data_interim)
    candidates.to_parquet(interim / CANDIDATES_NAME, index=False)
    fixes.to_parquet(interim / FIX_COMMITS_NAME, index=False)
    audit.to_csv(ensure_dir(cfg.paths.tables) / "02_commit_linking.csv", index=False)

    logger.info("linked %d/%d issues", len(fixes), len(issues))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

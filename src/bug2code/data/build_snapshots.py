"""Build each bug's candidate set from the code state before its fixing commit.

For every linked bug the parent of the fixing commit is the repository exactly as
it was before the fix, so it cannot contain the fix. Every in-scope
source file present in that state is a candidate the model may rank; the gold
files are the handful of them the fixing commit went on to change.

Candidate paths are not stored: thousands per bug would dwarf the dataset and are
reproducible from ``parent_sha`` at any time. Only the count is kept.

Usage:
    python -m bug2code.data.build_snapshots [--limit N] [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from bug2code.config import Config, load_config
from bug2code.data.collect_issues import OUTPUT_NAME as ISSUES_NAME
from bug2code.data.link_commits import FIX_COMMITS_NAME
from bug2code.data.linking import is_source_file
from bug2code.data.repos import run_git
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir, resolve

logger = get_logger(__name__)

OUTPUT_NAME = "localization_bugs.parquet"


def snapshot_files(repo: Path, sha: str, predicate: Callable[[str], bool]) -> set[str]:
    """Every in-scope source file existing in the repository at one commit.

    This is the whole codebase at that point, not the files that commit changed.
    """
    listing = run_git(repo, ["ls-tree", "-r", "--name-only", sha])
    return {path for path in listing.splitlines() if predicate(path)}


def gold_in_snapshot(gold: Sequence[str], candidates: set[str]) -> list[str]:
    """Gold files that are present in the candidate set, keeping their order.

    A file the fixing commit created — or renamed into place — does not exist in
    the pre-fix state, so it can never be ranked and is dropped from the gold set
    (``new_file_policy: drop_from_gold``).
    """
    return [path for path in gold if path in candidates]


def build(cfg: Config, fixes: pd.DataFrame) -> pd.DataFrame:
    """One row per bug whose gold set survives, with its candidate-set size."""
    rows: list[dict] = []
    for project, group in fixes.groupby("project"):
        spec = cfg.project(project)
        repo = resolve(cfg.github["clone_dir"]) / project

        def predicate(path: str, spec=spec) -> bool:
            return is_source_file(
                path,
                extensions=cfg.candidates["source_extensions"],
                roots=spec.source_roots,
                excludes=cfg.candidates["exclude_path_patterns"],
            )

        logger.info("%s: reading %d snapshots", project, len(group))
        for bug in group.itertuples():
            candidates = snapshot_files(repo, bug.parent_sha, predicate)
            gold = gold_in_snapshot(bug.source_files, candidates)
            if not gold and cfg.candidates["drop_bug_if_gold_empty"]:
                continue
            rows.append(
                {
                    "project": project,
                    "key": bug.key,
                    "fix_sha": bug.sha,
                    "parent_sha": bug.parent_sha,
                    "fix_date": bug.commit_date,
                    "n_candidates": len(candidates),
                    "gold_files": gold,
                    "n_gold": len(gold),
                    "n_gold_dropped": len(bug.source_files) - len(gold),
                }
            )
    return pd.DataFrame(rows)


def for_inspection(built: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Join the Jira text onto the built rows and flatten lists, for reading by eye."""
    out = built.merge(
        issues[["key", "summary", "description", "components"]], on="key", how="left"
    ).rename(columns={"summary": "title"})
    out["components"] = out["components"].apply(lambda values: "; ".join(values))
    out["gold_files"] = out["gold_files"].apply("; ".join)
    return out[
        [
            "project",
            "key",
            "title",
            "description",
            "components",
            "fix_sha",
            "parent_sha",
            "gold_files",
            "n_gold",
            "n_gold_dropped",
            "n_candidates",
        ]
    ]


def summarise(fixes: pd.DataFrame, built: pd.DataFrame) -> pd.DataFrame:
    """Per-project account of what the snapshot step kept and lost."""
    rows = []
    for project, group in built.groupby("project"):
        rows.append(
            {
                "project": project,
                "linked_bugs": int((fixes["project"] == project).sum()),
                "usable_bugs": len(group),
                "mean_candidates": round(group["n_candidates"].mean(), 1),
                "median_candidates": int(group["n_candidates"].median()),
                "mean_gold": round(group["n_gold"].mean(), 2),
                "bugs_with_gold_dropped": int((group["n_gold_dropped"] > 0).sum()),
                "gold_files_dropped": int(group["n_gold_dropped"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--limit", type=int, default=None, help="first N bugs per project")
    parser.add_argument(
        "--sample-csv",
        default=None,
        help="write a readable CSV here instead of the dataset, for manual inspection",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    fixes = pd.read_parquet(cfg.paths.data_interim / FIX_COMMITS_NAME)
    if args.limit:
        fixes = fixes.groupby("project").head(args.limit)

    built = build(cfg, fixes)
    summary = summarise(fixes, built)

    if args.sample_csv:
        issues = pd.read_parquet(cfg.paths.data_raw / ISSUES_NAME)
        path = Path(args.sample_csv)
        for_inspection(built, issues).to_csv(ensure_dir(path.parent) / path.name, index=False)
        logger.info("wrote %d rows to %s; dataset not touched", len(built), path)
    elif args.limit:
        logger.info("sample run of %d bugs per project, nothing written", args.limit)
    else:
        built.to_parquet(ensure_dir(cfg.paths.data_processed) / OUTPUT_NAME, index=False)
        summary.to_csv(ensure_dir(cfg.paths.tables) / "12_candidate_sets.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

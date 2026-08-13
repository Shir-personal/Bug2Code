"""Collect Jira bug reports for every configured project.

Usage:
    python -m bug2code.data.collect_issues [--config configs/exp.yaml] [--limit N]

Writes ``data/raw/issues.parquet`` and a per-project summary of what was kept.
"""

from __future__ import annotations

import argparse
from itertools import islice

import pandas as pd

from bug2code.config import Config, load_config
from bug2code.data.jira import JiraClient, build_bug_jql, issues_to_records
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

OUTPUT_NAME = "issues.parquet"


def collect(cfg: Config, limit: int | None = None) -> pd.DataFrame:
    """Fetch issues for all configured projects into one DataFrame."""
    jira_cfg = cfg.jira
    client = JiraClient(
        base_url=jira_cfg["base_url"],
        cache_dir=cfg.paths.cache,
        timeout_s=jira_cfg["request_timeout_s"],
        max_retries=jira_cfg["max_retries"],
    )

    frames = []
    for project in cfg.projects:
        spec = cfg.project(project)
        jql = build_bug_jql(
            jira_key=spec.jira_key,
            issue_types=jira_cfg["issue_types"],
            statuses=jira_cfg["statuses"],
            resolutions=jira_cfg["resolutions"],
            created_after=jira_cfg["created_after"],
            created_before=jira_cfg["created_before"],
        )
        total = client.count(jql)
        cap = limit if limit is not None else jira_cfg["max_issues_per_project"]
        logger.info("%s: %d issues match; cap=%s", project, total, cap)

        stream = client.search(jql, project=project, page_size=jira_cfg["page_size"])
        issues = list(islice(stream, cap) if cap else stream)
        frames.append(pd.DataFrame(issues_to_records(issues)))

    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-project counts used as the first sanity check on the collection."""
    return (
        df.assign(
            has_description=df["description"].str.len() > 0,
            has_component=df["components"].str.len() > 0,
            single_component=df["components"].str.len() == 1,
        )
        .groupby("project")
        .agg(
            issues=("key", "size"),
            with_description=("has_description", "sum"),
            with_component=("has_component", "sum"),
            single_component=("single_component", "sum"),
            first_created=("created", "min"),
            last_created=("created", "max"),
        )
        .reset_index()
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--limit", type=int, default=None, help="max issues per project")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    df = collect(cfg, limit=args.limit)
    out_dir = ensure_dir(cfg.paths.data_raw)
    out_path = out_dir / OUTPUT_NAME
    df.to_parquet(out_path, index=False)

    summary = summarise(df)
    summary.to_csv(ensure_dir(cfg.paths.tables) / "01_issue_collection.csv", index=False)

    logger.info("wrote %d issues -> %s", len(df), out_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Cross-check our Jira collection against the public GitBugs dataset.

GitBugs is not a data source here: its CSVs carry no Component field (needed by
RQ4) and identify issues by internal numeric id, which would cost one API call
each to resolve. It is still useful as an independent sample of the same
projects, so we measure how much of it our JQL window reproduces. A high overlap
is evidence that the collection filters are not silently losing bugs.

Usage:
    python -m bug2code.data.gitbugs [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
import io
import random
from collections import Counter

import pandas as pd
import requests

from bug2code.config import Config, load_config
from bug2code.data.collect_issues import OUTPUT_NAME as ISSUES_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir
from bug2code.utils.cache import JsonCache, cache_key

logger = get_logger(__name__)

CSV_URL = "https://raw.githubusercontent.com/av9ash/gitbugs/main/{project}/{project}_bugs.csv"


def fetch_csv(project: str, cache: JsonCache, timeout_s: int = 90) -> pd.DataFrame:
    """Download one GitBugs CSV, caching the raw text."""
    key = cache_key("gitbugs", project)
    text = cache.get(key)
    if text is None:
        url = CSV_URL.format(project=project)
        logger.info("downloading %s", url)
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        text = response.text
        cache.set(key, text)
    return pd.read_csv(io.StringIO(text))


def compare(cfg: Config, ours: pd.DataFrame) -> pd.DataFrame:
    """Compare GitBugs coverage against our collection, project by project.

    GitBugs starts in 2020 while our JQL window ends in 2021, so the two are
    only comparable on their intersection. Within that intersection the GitBugs
    rows are also restricted to the same status and resolution filters our JQL
    applies, otherwise unfixed bugs would count as misses.
    """
    cache = JsonCache(cfg.paths.cache, "gitbugs")
    our_start = pd.Timestamp(cfg.jira["created_after"], tz="UTC")
    our_end = pd.Timestamp(cfg.jira["created_before"], tz="UTC")
    statuses = set(cfg.jira["statuses"])
    resolutions = set(cfg.jira["resolutions"])

    our_created = pd.to_datetime(ours["created"], errors="coerce", utc=True, format="mixed")

    rows = []
    for project in cfg.projects:
        theirs = fetch_csv(project, cache)
        their_created = pd.to_datetime(theirs["Created"], errors="coerce", utc=True, format="mixed")

        start = max(our_start, their_created.min())
        end = min(our_end, their_created.max())

        comparable = theirs[
            (their_created >= start)
            & (their_created < end)
            & theirs["Status"].isin(statuses)
            & theirs["Resolution"].isin(resolutions)
        ]
        mine = ours[(ours["project"] == project) & (our_created >= start) & (our_created < end)]

        their_ids = set(comparable["Issue id"].astype(str))
        our_ids = set(mine["issue_id"].astype(str))
        shared = their_ids & our_ids

        rows.append(
            {
                "project": project,
                "gitbugs_rows": len(theirs),
                "gitbugs_first_year": int(their_created.min().year),
                "overlap_from": str(start.date()),
                "overlap_to": str(end.date()),
                "gitbugs_comparable": len(their_ids),
                "ours_comparable": len(our_ids),
                "shared": len(shared),
                "gitbugs_only": len(their_ids - our_ids),
                "ours_only": len(our_ids - their_ids),
                "gitbugs_recovered_frac": (
                    round(len(shared) / len(their_ids), 4) if their_ids else 0.0
                ),
                "has_component_column": "Component/s" in theirs.columns,
            }
        )
    return pd.DataFrame(rows)


def audit_missing(
    cfg: Config,
    ours: pd.DataFrame,
    sample_size: int = 40,
) -> pd.DataFrame:
    """Ask Jira what the issues GitBugs has and we do not actually are.

    GitBugs labels every row a "bug", but the CSVs carry no issue-type column.
    A random sample of the difference is looked up through one JQL ``id IN (...)``
    call per project, so the gap can be attributed rather than assumed.
    """
    cache = JsonCache(cfg.paths.cache, "gitbugs")
    rng = random.Random(cfg.seed)
    our_start = pd.Timestamp(cfg.jira["created_after"], tz="UTC")
    our_end = pd.Timestamp(cfg.jira["created_before"], tz="UTC")
    statuses = set(cfg.jira["statuses"])
    resolutions = set(cfg.jira["resolutions"])
    our_created = pd.to_datetime(ours["created"], errors="coerce", utc=True, format="mixed")

    rows = []
    for project in cfg.projects:
        theirs = fetch_csv(project, cache)
        their_created = pd.to_datetime(theirs["Created"], errors="coerce", utc=True, format="mixed")
        start = max(our_start, their_created.min())
        end = min(our_end, their_created.max())

        comparable = theirs[
            (their_created >= start)
            & (their_created < end)
            & theirs["Status"].isin(statuses)
            & theirs["Resolution"].isin(resolutions)
        ]
        mine = ours[(ours["project"] == project) & (our_created >= start) & (our_created < end)]
        missing = sorted(
            set(comparable["Issue id"].astype(str)) - set(mine["issue_id"].astype(str))
        )
        if not missing:
            continue

        sample = rng.sample(missing, min(sample_size, len(missing)))
        for issue_type, count in _issue_types(cfg, sample).items():
            rows.append(
                {
                    "project": project,
                    "missing_total": len(missing),
                    "sampled": len(sample),
                    "issue_type": issue_type,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def _issue_types(cfg: Config, issue_ids: list[str]) -> Counter[str]:
    """Issue-type distribution of a set of internal Jira ids, in one request."""
    response = requests.get(
        f"{cfg.jira['base_url'].rstrip('/')}/rest/api/2/search",
        params={
            "jql": f"id IN ({','.join(issue_ids)})",
            "fields": "issuetype",
            "maxResults": len(issue_ids),
        },
        timeout=cfg.jira["request_timeout_s"],
    )
    response.raise_for_status()
    return Counter(i["fields"]["issuetype"]["name"] for i in response.json()["issues"])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    ours = pd.read_parquet(cfg.paths.data_raw / ISSUES_NAME)
    tables = ensure_dir(cfg.paths.tables)

    table = compare(cfg, ours)
    table.to_csv(tables / "03_gitbugs_crosscheck.csv", index=False)
    print(table.to_string(index=False))

    audit = audit_missing(cfg, ours)
    audit.to_csv(tables / "04_gitbugs_missing_types.csv", index=False)
    print()
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

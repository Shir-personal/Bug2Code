"""Independent check that the automatic Jira-to-commit linker picks the right commit.

*Source Control Link* is a rarely-used Apache Jira custom field where a human pasted the
fixing commit URL. It is far too sparse to link with, but where it exists it is a
human-curated answer key for the linker — the part of the pipeline most able to fail
silently, since a wrong commit yields a plausible-looking but wrong gold set.

The field is never used as a model input, only to validate phase 1.

Usage:
    python -m bug2code.data.validate_linking [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
import re

import pandas as pd
import requests

from bug2code.config import Config, load_config
from bug2code.data.jira import build_bug_jql
from bug2code.data.link_commits import FIX_COMMITS_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

SOURCE_CONTROL_FIELD = "customfield_12313924"
_COMMIT_URL = re.compile(r"/commit/([0-9a-f]{7,40})", re.IGNORECASE)


def _bug_jql(cfg: Config, project: str) -> str:
    """The project's collection JQL, without the ORDER BY clause."""
    spec = cfg.project(project)
    j = cfg.jira
    jql = build_bug_jql(
        spec.jira_key,
        j["issue_types"],
        j["statuses"],
        j["resolutions"],
        j["created_after"],
        j["created_before"],
    )
    return jql.replace(" ORDER BY created ASC", "")


def _search(cfg: Config, **params: object) -> dict:
    """One call to the Jira search endpoint."""
    response = requests.get(
        f"{cfg.jira['base_url'].rstrip('/')}/rest/api/2/search",
        params=params,
        timeout=cfg.jira["request_timeout_s"],
    )
    response.raise_for_status()
    return response.json()


def fetch_source_control_links(cfg: Config, project: str) -> dict[str, set[str]]:
    """Return ``{issue key: {commit sha, ...}}`` for issues carrying the field."""
    jql = f'{_bug_jql(cfg, project)} AND "Source Control Link" IS NOT EMPTY'
    links: dict[str, set[str]] = {}
    start = 0
    while True:
        page = _search(cfg, jql=jql, startAt=start, maxResults=100, fields=SOURCE_CONTROL_FIELD)
        issues = page.get("issues", [])
        for issue in issues:
            shas = commit_urls_to_shas(issue["fields"].get(SOURCE_CONTROL_FIELD))
            if shas:
                links[issue["key"]] = shas
        start += len(issues)
        if not issues or start >= page.get("total", 0):
            return links


def commit_urls_to_shas(value: object) -> set[str]:
    """Extract commit shas from a Source Control Link field value."""
    return {sha.lower() for sha in _COMMIT_URL.findall(str(value))}


def agreement_row(project: str, links: dict[str, set[str]], chosen: dict[str, str]) -> dict:
    """Count how often our chosen commit matches the human-recorded one."""
    overlap = [key for key in links if key in chosen]
    # Jira URLs sometimes carry an abbreviated sha, so compare by prefix.
    agree = [key for key in overlap if any(chosen[key].startswith(sha) for sha in links[key])]
    return {
        "project": project,
        "issues_with_field": len(links),
        "also_linked_by_us": len(overlap),
        "same_commit": len(agree),
        "agreement": round(len(agree) / len(overlap), 4) if overlap else None,
    }


def validate_against_source_control(cfg: Config, fixes: pd.DataFrame) -> pd.DataFrame:
    """Compare our selected fix commit with the commit a human recorded in Jira."""
    chosen = dict(zip(fixes["key"], fixes["sha"], strict=True))

    rows = []
    for project in cfg.projects:
        row = agreement_row(project, fetch_source_control_links(cfg, project), chosen)
        logger.info("%s: %d/%d agree", project, row["same_commit"], row["also_linked_by_us"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    tables = ensure_dir(cfg.paths.tables)

    fixes = pd.read_parquet(cfg.paths.data_interim / FIX_COMMITS_NAME)

    agreement = validate_against_source_control(cfg, fixes)
    agreement.to_csv(tables / "11_linker_validation.csv", index=False)
    print("\n=== linker vs Jira Source Control Link ===")
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()

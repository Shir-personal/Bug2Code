"""Apache Jira ingestion.

Issues are fetched through the public REST search endpoint, 100 per request,
which also returns the Component field used by the RQ4 experiment. Every page is
cached on disk, so an interrupted run resumes instead of refetching.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from bug2code.logging_utils import get_logger
from bug2code.utils.cache import JsonCache, cache_key

logger = get_logger(__name__)

FIELDS = [
    "key",
    "summary",
    "description",
    "components",
    "issuetype",
    "status",
    "resolution",
    "priority",
    "created",
    "resolutiondate",
]

# Jira renders descriptions in wiki markup; these carry no signal for the models
# but do add tokens, so they are stripped at ingestion time.
_CODE_BLOCK = re.compile(r"\{code(:[^}]*)?\}|\{noformat\}|\{quote\}", re.IGNORECASE)
_WS = re.compile(r"[ \t]+")
_LINE_PADDING = re.compile(r" *\n *")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class JiraIssue:
    """One Jira bug report, flattened to the fields the pipeline uses."""

    key: str
    project: str
    issue_id: str
    summary: str
    description: str
    components: list[str]
    status: str
    resolution: str
    priority: str
    created: str
    resolved: str

    @property
    def n_components(self) -> int:
        """Number of components assigned to this issue."""
        return len(self.components)


def clean_text(text: str | None) -> str:
    """Strip Jira wiki markers and collapse whitespace."""
    if not text:
        return ""
    out = _CODE_BLOCK.sub("\n", text)
    out = _WS.sub(" ", out)
    out = _LINE_PADDING.sub("\n", out)
    return _BLANK_LINES.sub("\n\n", out).strip()


def build_bug_jql(
    jira_key: str,
    issue_types: list[str],
    statuses: list[str],
    resolutions: list[str],
    created_after: str,
    created_before: str,
) -> str:
    """Build the JQL selecting fixed bug reports in the evaluation window."""

    def _in(values: list[str]) -> str:
        return ", ".join(f'"{v}"' for v in values)

    return (
        f"project = {jira_key} "
        f"AND issuetype IN ({_in(issue_types)}) "
        f"AND status IN ({_in(statuses)}) "
        f"AND resolution IN ({_in(resolutions)}) "
        f'AND created >= "{created_after}" AND created < "{created_before}" '
        f"ORDER BY created ASC"
    )


def parse_issue(raw: dict[str, Any], project: str) -> JiraIssue:
    """Convert one raw Jira API issue into a :class:`JiraIssue`."""
    fields = raw.get("fields") or {}

    def _name(key: str) -> str:
        value = fields.get(key)
        return (value or {}).get("name", "") if isinstance(value, dict) else ""

    return JiraIssue(
        key=raw["key"],
        project=project,
        issue_id=str(raw.get("id", "")),
        summary=clean_text(fields.get("summary")),
        description=clean_text(fields.get("description")),
        components=[c["name"] for c in (fields.get("components") or [])],
        status=_name("status"),
        resolution=_name("resolution"),
        priority=_name("priority"),
        created=fields.get("created") or "",
        resolved=fields.get("resolutiondate") or "",
    )


class JiraClient:
    """Cached, retrying client for the Apache Jira REST search endpoint."""

    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        timeout_s: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache = JsonCache(cache_dir, "jira")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _fetch_page(self, jql: str, start_at: int, page_size: int) -> dict[str, Any]:
        """Fetch one page, from cache when available."""
        key = cache_key("jira-search", self.base_url, jql, start_at, page_size)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        payload = self._request(jql, start_at, page_size)
        self.cache.set(key, payload)
        return payload

    def _request(self, jql: str, start_at: int, page_size: int) -> dict[str, Any]:
        @retry(
            retry=retry_if_exception_type(requests.RequestException),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            reraise=True,
        )
        def _call() -> dict[str, Any]:
            response = self.session.get(
                f"{self.base_url}/rest/api/2/search",
                params={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": page_size,
                    "fields": ",".join(FIELDS),
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            return response.json()

        return _call()

    def count(self, jql: str) -> int:
        """Return how many issues the query matches, without fetching them."""
        return int(self._fetch_page(jql, 0, 0).get("total", 0))

    def search(self, jql: str, project: str, page_size: int = 100) -> Iterator[JiraIssue]:
        """Yield every issue matching ``jql``, paging until exhausted."""
        start_at = 0
        total: int | None = None

        while total is None or start_at < total:
            page = self._fetch_page(jql, start_at, page_size)
            total = int(page.get("total", 0))
            issues = page.get("issues", [])
            if not issues:
                break
            for raw in issues:
                yield parse_issue(raw, project)
            start_at += len(issues)
            logger.info("%s: %d/%d issues", project, min(start_at, total), total)


def issues_to_records(issues: list[JiraIssue]) -> list[dict[str, Any]]:
    """Flatten issues for DataFrame construction."""
    return [asdict(issue) for issue in issues]

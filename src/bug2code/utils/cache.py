"""On-disk JSON cache so collection can resume without repeating requests."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from bug2code.logging_utils import get_logger
from bug2code.paths import ensure_dir

logger = get_logger(__name__)


def cache_key(*parts: Any) -> str:
    """Build a stable filename-safe key from arbitrary parts."""
    raw = "|".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class JsonCache:
    """Namespaced gzip-JSON cache rooted at a directory.

    Each entry is one file, so a partial run leaves everything already fetched
    intact and the next run picks up where it stopped.
    """

    def __init__(self, root: Path, namespace: str) -> None:
        self.dir = ensure_dir(Path(root) / namespace)

    def path(self, key: str) -> Path:
        """Return the file backing ``key``."""
        return self.dir / f"{key}.json.gz"

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None if absent or unreadable."""
        path = self.path(key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.warning("corrupt cache entry %s; refetching", path.name)
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, written atomically."""
        path = self.path(key)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(value, fh)
        tmp.replace(path)

    def __len__(self) -> int:
        """Number of entries currently cached."""
        return sum(1 for _ in self.dir.glob("*.json.gz"))

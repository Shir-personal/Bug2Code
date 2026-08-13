"""Tests for the resumable on-disk cache."""

from __future__ import annotations

import gzip

from bug2code.utils.cache import JsonCache, cache_key


def test_cache_key_is_stable_and_order_sensitive():
    assert cache_key("a", 1) == cache_key("a", 1)
    assert cache_key("a", 1) != cache_key(1, "a")


def test_cache_key_ignores_dict_ordering():
    assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})


def test_roundtrip(tmp_path):
    cache = JsonCache(tmp_path, "jira")
    assert cache.get("missing") is None
    cache.set("k", {"total": 3})
    assert cache.get("k") == {"total": 3}
    assert len(cache) == 1


def test_corrupt_entry_is_dropped(tmp_path):
    cache = JsonCache(tmp_path, "jira")
    cache.set("k", {"total": 3})
    with gzip.open(cache.path("k"), "wt", encoding="utf-8") as fh:
        fh.write("{ not json")
    assert cache.get("k") is None
    assert not cache.path("k").exists()

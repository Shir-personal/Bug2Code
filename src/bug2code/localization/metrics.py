"""Ranking metrics for bug localization.

Every function takes a ranked list of candidate files (best first) and the set
of gold files for that bug. This is retrieval, not classification, so there is
no notion of a fixed positive/negative threshold — only how early and how
completely the gold files surface in the ranking.
"""

from __future__ import annotations

from collections.abc import Sequence


def _ranks_of_gold(ranked: Sequence[str], gold: Sequence[str]) -> list[int]:
    """1-based positions of every gold file that appears in ``ranked``."""
    gold_set = set(gold)
    return [i for i, path in enumerate(ranked, start=1) if path in gold_set]


def best_gold_rank(ranked: Sequence[str], gold: Sequence[str]) -> int | None:
    """1-based rank of the best-placed gold file, or ``None`` if none appear."""
    ranks = _ranks_of_gold(ranked, gold)
    return min(ranks) if ranks else None


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if any gold file is in the top ``k``, else 0.0."""
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold files that appear in the top ``k``."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first gold file found, or 0.0 if none appear."""
    ranks = _ranks_of_gold(ranked, gold)
    return 1.0 / min(ranks) if ranks else 0.0


def average_precision(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """Mean of precision@rank over every rank at which a gold file appears."""
    ranks = _ranks_of_gold(ranked, gold)
    if not ranks or not gold:
        return 0.0
    precisions = [(i + 1) / rank for i, rank in enumerate(ranks)]
    return sum(precisions) / len(gold)


K_VALUES = (1, 5, 10, 20)


def score_one(ranked: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    """All ranking metrics for a single bug's ranking."""
    out = {f"hit_at_{k}": hit_at_k(ranked, gold, k) for k in K_VALUES}
    out |= {f"recall_at_{k}": recall_at_k(ranked, gold, k) for k in K_VALUES}
    out["mrr"] = reciprocal_rank(ranked, gold)
    out["map"] = average_precision(ranked, gold)
    return out

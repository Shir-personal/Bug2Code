"""Identifier-aware tokenizer for the TF-IDF baseline.

Splits text on word boundaries, then further splits each word on camelCase and
snake_case boundaries, so ``NullPointerException`` also contributes the terms
``null``, ``pointer`` and ``exception`` — the terms a bug report is likely to
use in prose even when it quotes a different identifier.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words plus their camelCase/snake_case sub-parts."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())
        parts = _CAMEL_PART.findall(word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens

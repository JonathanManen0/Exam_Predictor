"""Shared stopwords and cluster-evidence normalization helpers."""

from __future__ import annotations

import re
from typing import Iterable


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "have", "in", "into", "is", "it", "of", "on",
    "or", "other", "such", "than", "that", "the", "their", "these",
    "this", "those", "through", "to", "was", "were", "which", "with",
    "within", "without",
}

GENERIC_TERMS = {
    "bacteria", "bacterial", "diagnosis", "diagnostic", "disease",
    "infection", "infections", "principles", "process", "rule",
    "theory", "topic", "topics", "treatment",
}


def meaningful_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9][a-z0-9/-]{2,}", text.lower()))
    return {token for token in tokens if token not in STOPWORDS and token not in GENERIC_TERMS}


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def keyword_phrases(keywords: Iterable[str]) -> set[str]:
    phrases: set[str] = set()
    for keyword in keywords:
        phrase = normalize_phrase(str(keyword))
        if not phrase or phrase in STOPWORDS or phrase in GENERIC_TERMS:
            continue
        if len(phrase) >= 3:
            phrases.add(phrase)
    return phrases

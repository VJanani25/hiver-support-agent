from __future__ import annotations

import re


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("message must be a string")
    cleaned = URL_RE.sub(" ", text.lower().replace("\n", " "))
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    return [token for token in tokens if len(token) > 1]


def ngrams(tokens: list[str], max_n: int = 2) -> list[str]:
    features = list(tokens)
    for n in range(2, max_n + 1):
        features.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return features


def validate_message(message: str) -> str:
    normalized = normalize_text(message)
    if not normalized:
        raise ValueError("customer message cannot be empty")
    if len(normalized) > 2_000:
        raise ValueError("customer message is too long; limit is 2,000 characters")
    return normalized
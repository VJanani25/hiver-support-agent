from __future__ import annotations

import math
from collections import Counter

from .preprocessing import ngrams, tokenize, validate_message
from .types import ConversationExample, RetrievedExample


class RetrievalIndex:
    def __init__(self, examples: list[ConversationExample]):
        self.examples = examples
        self.document_frequency = Counter()
        self.vectors: list[dict[str, float]] = []
        for example in examples:
            tokens = set(ngrams(tokenize(example.customer_message)))
            self.document_frequency.update(tokens)
        self.vectors = [self._vector(example.customer_message) for example in examples]

    def _vector(self, message: str) -> dict[str, float]:
        counts = Counter(ngrams(tokenize(message)))
        vector = {
            token: (1.0 + math.log(count)) * math.log((1 + len(self.examples)) / (1 + self.document_frequency[token])) + 1.0
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    def search(self, message: str, limit: int = 3, min_similarity: float = 0.0) -> list[RetrievedExample]:
        query = self._vector(validate_message(message))
        scored = [
            (self._cosine(query, vector), example)
            for vector, example in zip(self.vectors, self.examples)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedExample(
                example_id=example.example_id,
                customer_message=example.customer_message,
                support_response=example.support_response,
                similarity=round(score, 4),
                intent=example.intent,
            )
            for score, example in scored[:limit]
            if score >= min_similarity
        ]
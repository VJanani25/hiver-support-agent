from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .preprocessing import ngrams, tokenize, validate_message
from .types import ConversationExample, Prediction


DEFAULT_INTENTS = [
    "account_access",
    "billing_payment",
    "refund_cancellation",
    "delivery_status",
    "technical_issue",
    "product_information",
    "complaint",
    "other",
]


KEYWORDS = {
    "account_access": ["login", "log in", "password", "locked", "verification", "sign in", "access"],
    "billing_payment": ["charged", "charge", "payment", "billing", "card", "invoice", "duplicate", "fee"],
    "refund_cancellation": ["refund", "cancel", "cancellation", "money back", "reverse", "unsubscribe"],
    "delivery_status": ["delivery", "delivered", "shipping", "tracking", "order", "late", "missing", "package"],
    "technical_issue": ["error", "bug", "broken", "not working", "crash", "technical", "app", "website"],
    "product_information": ["how", "where", "when", "available", "feature", "information", "price", "plan"],
    "complaint": ["angry", "terrible", "unhappy", "complaint", "awful", "disappointed", "frustrated"],
    "other": [],
}


class MajorityClassifier:
    def __init__(self, intents: list[str] | None = None):
        self.intents = intents or DEFAULT_INTENTS
        self.label = "other"

    def fit(self, examples: list[ConversationExample]) -> "MajorityClassifier":
        counts = Counter(example.intent for example in examples)
        self.label = counts.most_common(1)[0][0] if counts else "other"
        return self

    def predict(self, message: str) -> Prediction:
        validate_message(message)
        return Prediction(self.label, 1.0 if self.label != "other" else 0.2, {self.label: 1.0})


class KeywordClassifier:
    def __init__(self, keywords: dict[str, list[str]] | None = None):
        self.keywords = keywords or KEYWORDS

    def fit(self, examples: list[ConversationExample]) -> "KeywordClassifier":
        return self

    def predict(self, message: str) -> Prediction:
        normalized = validate_message(message)
        scores: dict[str, float] = {}
        for intent, terms in self.keywords.items():
            hits = sum(1 for term in terms if term in normalized)
            scores[intent] = float(hits)
        best_intent = max(scores, key=scores.get, default="other")
        total = sum(scores.values())
        if total == 0:
            return Prediction("other", 0.18, scores)
        confidence = min(0.98, 0.45 + scores[best_intent] / max(1.0, total) * 0.5)
        return Prediction(best_intent, confidence, scores)


class TfidfLogisticClassifier:
    """Small deterministic multiclass logistic regression without heavy dependencies."""

    def __init__(self, intents: list[str] | None = None, epochs: int = 120, learning_rate: float = 0.12):
        self.intents = intents or DEFAULT_INTENTS
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.weights: list[list[float]] = []
        self.bias: list[float] = []
        self.fitted = False

    def _features(self, message: str) -> list[str]:
        return ngrams(tokenize(message), max_n=2)

    def _vectorize(self, message: str) -> dict[int, float]:
        counts = Counter(self._features(message))
        vector: dict[int, float] = {}
        for token, count in counts.items():
            if token in self.vocabulary:
                index = self.vocabulary[token]
                vector[index] = (1.0 + math.log(count)) * self.idf[index]
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {index: value / norm for index, value in vector.items()}

    def fit(self, examples: list[ConversationExample]) -> "TfidfLogisticClassifier":
        if not examples:
            raise ValueError("at least one example is required")
        docs = [self._features(example.customer_message) for example in examples]
        document_frequency = Counter(token for doc in docs for token in set(doc))
        vocabulary = sorted(token for token, count in document_frequency.items() if count >= 1)
        self.vocabulary = {token: index for index, token in enumerate(vocabulary)}
        self.idf = [
            math.log((1 + len(docs)) / (1 + document_frequency[token])) + 1.0
            for token in vocabulary
        ]
        class_counts = Counter(example.intent for example in examples)
        self.intents = sorted(set(self.intents) | set(class_counts))
        class_index = {label: index for index, label in enumerate(self.intents)}
        vectors = [self._vectorize(example.customer_message) for example in examples]
        labels = [class_index.get(example.intent, class_index.get("other", 0)) for example in examples]
        self.weights = [[0.0] * len(self.vocabulary) for _ in self.intents]
        self.bias = [math.log((class_counts.get(label, 0) + 1) / (len(examples) + len(self.intents))) for label in self.intents]
        rng = random.Random(17)
        for row in self.weights:
            for index in range(len(row)):
                row[index] = rng.uniform(-0.01, 0.01)
        for _ in range(self.epochs):
            for vector, target in zip(vectors, labels):
                logits = [
                    self.bias[class_id] + sum(self.weights[class_id][i] * value for i, value in vector.items())
                    for class_id in range(len(self.intents))
                ]
                maximum = max(logits)
                exp_values = [math.exp(value - maximum) for value in logits]
                denominator = sum(exp_values) or 1.0
                probabilities = [value / denominator for value in exp_values]
                for class_id, probability in enumerate(probabilities):
                    error = probability - (1.0 if class_id == target else 0.0)
                    self.bias[class_id] -= self.learning_rate * error
                    for index, value in vector.items():
                        self.weights[class_id][index] -= self.learning_rate * error * value
        self.fitted = True
        return self

    def predict(self, message: str) -> Prediction:
        if not self.fitted:
            raise RuntimeError("classifier must be fitted before prediction")
        vector = self._vectorize(validate_message(message))
        logits = [
            self.bias[class_id] + sum(self.weights[class_id][i] * value for i, value in vector.items())
            for class_id in range(len(self.intents))
        ]
        maximum = max(logits)
        exp_values = [math.exp(value - maximum) for value in logits]
        denominator = sum(exp_values) or 1.0
        probabilities = [value / denominator for value in exp_values]
        scores = {label: probabilities[index] for index, label in enumerate(self.intents)}
        best = max(scores, key=scores.get)
        return Prediction(best, scores[best], scores)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationExample:
    example_id: str
    customer_message: str
    support_response: str
    context: str = ""
    intent: str = "other"


@dataclass(frozen=True)
class RetrievedExample:
    example_id: str
    customer_message: str
    support_response: str
    similarity: float
    intent: str


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    intent: str
    confidence: float
    drafted_reply: str
    decision: str
    escalation_reason: str
    evidence: list[RetrievedExample]


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }
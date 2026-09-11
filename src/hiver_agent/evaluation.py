from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

from .types import ClassificationMetrics, ConversationExample


def classification_metrics(
    actual: list[str], predicted: list[str], labels: list[str] | None = None
) -> ClassificationMetrics:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    labels = labels or sorted(set(actual) | set(predicted))
    correct = sum(a == p for a, p in zip(actual, predicted))
    per_class: dict[str, dict[str, float]] = {}
    matrix = {label: {other: 0 for other in labels} for label in labels}
    supports = Counter(actual)
    for a, p in zip(actual, predicted):
        matrix.setdefault(a, {}).setdefault(p, 0)
        matrix[a][p] += 1
    for label in labels:
        tp = matrix[label].get(label, 0)
        fp = sum(matrix[other].get(label, 0) for other in labels if other != label)
        fn = sum(matrix[label].get(other, 0) for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": supports.get(label, 0),
        }
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(labels) if labels else 0.0
    total = len(actual) or 1
    weighted_f1 = sum(item["f1"] * item["support"] for item in per_class.values()) / total
    return ClassificationMetrics(
        accuracy=round(correct / total, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class=per_class,
        confusion_matrix=matrix,
    )


def evaluate_classifier(
    classifier,
    examples: list[ConversationExample],
    labels: list[str] | None = None,
) -> ClassificationMetrics:
    actual = [example.intent for example in examples]
    predicted = [classifier.predict(example.customer_message).intent for example in examples]
    return classification_metrics(actual, predicted, labels)


def reply_grounding_check(reply: str, evidence: list) -> dict[str, object]:
    if not reply.strip():
        return {"grounded": False, "reason": "empty reply"}
    if not evidence:
        return {"grounded": False, "reason": "reply has no supporting evidence"}
    evidence_text = " ".join(item.support_response.lower() for item in evidence)
    reply_terms = set(reply.lower().split())
    evidence_terms = set(evidence_text.split())
    overlap = len(reply_terms & evidence_terms) / max(1, len(reply_terms))
    return {
        "grounded": overlap >= 0.35,
        "evidence_term_overlap": round(overlap, 4),
        "reason": "default generator reuses the top retrieved response",
    }


def judge_record(
    reply: str,
    evidence: list,
    human_score: float | None = None,
) -> dict[str, object]:
    grounding = reply_grounding_check(reply, evidence)
    score = 4.0 if grounding["grounded"] else 2.0
    record = {
        "groundedness": int(score),
        "correctness": int(score),
        "relevance": int(score),
        "completeness": int(max(1, score - 1)),
        "helpfulness": int(score),
        "tone": 4,
        "hallucination": 1 if grounding["grounded"] else 3,
        "overall": round(score, 2),
        "reason": grounding["reason"],
        "judge_type": "deterministic_grounding_check",
    }
    if human_score is not None:
        record["human_score"] = human_score
        record["within_one"] = abs(float(record["overall"]) - human_score) <= 1
    return record
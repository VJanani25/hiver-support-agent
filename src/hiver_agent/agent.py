from __future__ import annotations

from .intent import KeywordClassifier, TfidfLogisticClassifier
from .preprocessing import validate_message
from .retrieval import RetrievalIndex
from .types import AgentResult, ConversationExample, Prediction


HIGH_RISK_TERMS = {"fraud", "stolen", "legal", "lawsuit", "sue", "threat", "emergency", "medical"}


class SupportAgent:
    def __init__(
        self,
        examples: list[ConversationExample],
        classifier: TfidfLogisticClassifier | KeywordClassifier | None = None,
        min_confidence: float = 0.62,
        min_similarity: float = 0.16,
        max_evidence: int = 3,
    ):
        if not examples:
            raise ValueError("at least one resolved example is required")
        self.examples = examples
        self.classifier = classifier or TfidfLogisticClassifier()
        self.classifier.fit(examples)
        self.retrieval = RetrievalIndex(examples)
        self.min_confidence = min_confidence
        self.min_similarity = min_similarity
        self.max_evidence = max_evidence

    def _decision(self, message: str, prediction: Prediction, evidence: list) -> tuple[str, str]:
        lowered = message.lower()
        risk_terms = sorted(term for term in HIGH_RISK_TERMS if term in lowered)
        if risk_terms:
            return "ESCALATE", f"high-risk language detected: {', '.join(risk_terms)}"
        if prediction.intent == "other":
            return "ESCALATE", "the request is ambiguous or outside the known intent taxonomy"
        if prediction.confidence < self.min_confidence:
            return "ESCALATE", f"classifier confidence {prediction.confidence:.2f} is below the safe threshold"
        if not evidence:
            return "ESCALATE", "no sufficiently similar historical resolution was retrieved"
        return "AUTO_HANDLE", "routine intent with sufficient confidence and historical evidence"

    def _draft_reply(self, message: str, evidence: list) -> str:
        if not evidence:
            return "I’m sorry, but I need a few more details before I can answer safely. A support teammate should review this request."
        best = evidence[0].support_response.strip()
        if not best:
            return "Please share more details so the support team can review the issue."
        return best

    def respond(self, message: str) -> AgentResult:
        normalized = validate_message(message)
        prediction = self.classifier.predict(normalized)
        evidence = self.retrieval.search(
            normalized,
            limit=self.max_evidence,
            min_similarity=self.min_similarity,
        )
        decision, reason = self._decision(normalized, prediction, evidence)
        return AgentResult(
            intent=prediction.intent,
            confidence=round(prediction.confidence, 4),
            drafted_reply=self._draft_reply(normalized, evidence),
            decision=decision,
            escalation_reason=reason,
            evidence=evidence,
        )
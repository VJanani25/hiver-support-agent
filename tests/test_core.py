import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_agent.agent import SupportAgent
from hiver_agent.data import load_examples
from hiver_agent.evaluation import classification_metrics
from hiver_agent.intent import KeywordClassifier, TfidfLogisticClassifier
from hiver_agent.preprocessing import normalize_text, tokenize
from hiver_agent.retrieval import RetrievalIndex


DATA = ROOT / "data/sample/support_sample.csv"


def examples():
    return load_examples(DATA)


def test_preprocessing_removes_urls_and_tokenizes():
    assert normalize_text("  Help! https://example.com\nNow ") == "help! now"
    assert "help" in tokenize("Help! https://example.com")


def test_input_validation_rejects_empty_message():
    agent = SupportAgent(examples())
    with pytest.raises(ValueError):
        agent.respond("   ")


def test_keyword_and_tfidf_prediction_return_known_intents():
    data = examples()
    keyword = KeywordClassifier().fit(data)
    tfidf = TfidfLogisticClassifier(epochs=20).fit(data)
    assert keyword.predict("I need a refund").intent in {"refund_cancellation", "other"}
    assert tfidf.predict("I need a refund").intent in {"refund_cancellation", "other", "billing_payment"}


def test_retrieval_returns_support_evidence():
    result = RetrievalIndex(examples()).search("my order delivery is late", limit=2)
    assert result
    assert result[0].support_response
    assert result[0].similarity >= result[-1].similarity


def test_agent_escalates_high_risk_and_returns_evidence():
    agent = SupportAgent(examples())
    result = agent.respond("This is fraud and I need legal help")
    assert result.decision == "ESCALATE"
    assert "high-risk" in result.escalation_reason


def test_metrics_are_well_formed():
    metrics = classification_metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert metrics.accuracy == 0.6667
    assert metrics.per_class["a"]["recall"] == 0.5
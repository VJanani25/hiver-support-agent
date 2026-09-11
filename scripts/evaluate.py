from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_agent.agent import SupportAgent
from hiver_agent.data import load_examples
from hiver_agent.evaluation import evaluate_classifier, judge_record
from hiver_agent.intent import TfidfLogisticClassifier
from hiver_agent.types import ConversationExample


def read_processed(path: Path) -> list[ConversationExample]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ConversationExample(**row) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transparent intent and reply evaluation.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/examples.csv")
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/generated")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--demo"], check=True)
        examples = read_processed(ROOT / "data/processed/examples.csv")
        output = args.output
    else:
        examples = read_processed(args.data)
        output = args.output
    classifier = TfidfLogisticClassifier().fit(examples)
    agent = SupportAgent(examples, classifier=classifier)
    metrics = evaluate_classifier(classifier, examples)
    replies = []
    for example in examples:
        result = agent.respond(example.customer_message)
        replies.append(judge_record(result.drafted_reply, result.evidence))
    result = {
        "status": "demo_smoke_only" if args.demo else "data_run",
        "sample_size": len(examples),
        "intent_metrics": metrics.as_dict(),
        "reply_checks": replies,
        "golden_set_status": "not supplied" if not args.golden else "requires reviewed labels",
        "headline_claim_allowed": False if args.demo else bool(args.golden),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
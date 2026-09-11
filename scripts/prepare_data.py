from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_agent.data import inspect_rows, load_examples, read_rows, write_examples
from hiver_agent.intent import KEYWORDS
from hiver_agent.preprocessing import normalize_text
from hiver_agent.types import ConversationExample


def infer_intent(message: str) -> str:
    normalized = normalize_text(message)
    scores = {intent: sum(term in normalized for term in terms) for intent, terms in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "other"


def print_inspection(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, default=lambda value: list(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and prepare support conversation pairs.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/sample/support_sample.csv")
    parser.add_argument("--brand-account")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    source = ROOT / "data/sample/support_sample.csv" if args.demo else args.input
    rows = read_rows(source)
    summary = inspect_rows(rows)
    print_inspection(summary)
    if args.inspect_only:
        return
    brand = args.brand_account
    if not args.demo and not brand:
        raise SystemExit("Choose --brand-account after reviewing --inspect-only output.")
    examples = load_examples(source, brand_account=brand)
    if not examples:
        raise SystemExit("No resolved customer/support pairs found. Check the brand account and input columns.")
    labelled = [
        ConversationExample(
            example_id=example.example_id,
            customer_message=example.customer_message,
            support_response=example.support_response,
            context=example.context,
            intent=infer_intent(example.customer_message),
        )
        for example in examples
    ]
    output_file = args.output / "examples.csv"
    write_examples(output_file, labelled)
    counts = Counter(example.intent for example in labelled)
    print(f"Wrote {len(labelled)} paired examples to {output_file}")
    print(f"Suggested taxonomy counts: {dict(counts)}")


if __name__ == "__main__":
    main()
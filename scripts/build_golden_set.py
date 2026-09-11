from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_agent.intent import KeywordClassifier
from hiver_agent.data import read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create candidate rows for human golden-set review.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/examples.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/golden_set.csv")
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input contains no examples.")
    classifier = KeywordClassifier()
    rng = random.Random(17)
    sampled = [rows[index % len(rows)] for index in rng.sample(range(max(args.size, len(rows))), args.size)] if len(rows) >= args.size else [rows[index % len(rows)] for index in range(args.size)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "customer_message", "context", "intent", "expected_action", "notes", "suggested_intent", "suggested_action"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(sampled, 1):
            suggestion = classifier.predict(row["customer_message"])
            writer.writerow({
                "id": f"golden-{index:03d}",
                "customer_message": row["customer_message"],
                "context": row.get("context", ""),
                "intent": "",
                "expected_action": "",
                "notes": "Candidate only; human review required before evaluation.",
                "suggested_intent": suggestion.intent,
                "suggested_action": "ESCALATE" if suggestion.intent == "other" else "AUTO_HANDLE",
            })
    print(f"Wrote {args.size} unreviewed candidate rows to {args.output}")


if __name__ == "__main__":
    main()
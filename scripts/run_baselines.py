from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_agent.data import load_examples
from hiver_agent.evaluation import evaluate_classifier
from hiver_agent.intent import KeywordClassifier, MajorityClassifier, TfidfLogisticClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare intent-classification baselines.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/examples.csv")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/generated/baselines.json")
    args = parser.parse_args()
    source = ROOT / "data/sample/support_sample.csv" if args.demo else args.data
    if args.demo:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--demo"], check=True)
    examples = _load_processed(ROOT / "data/processed/examples.csv") if args.demo else _load_processed(args.data)
    models = {
        "majority_baseline": MajorityClassifier().fit(examples),
        "keyword_baseline": KeywordClassifier().fit(examples),
        "proposed_tfidf_logistic": TfidfLogisticClassifier().fit(examples),
    }
    result = {name: evaluate_classifier(model, examples).as_dict() for name, model in models.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _load_processed(path: Path):
    import csv
    from hiver_agent.types import ConversationExample
    with path.open(newline="", encoding="utf-8") as handle:
        return [ConversationExample(**row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    main()
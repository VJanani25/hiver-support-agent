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
from hiver_agent.types import ConversationExample


def load_processed(path: Path) -> list[ConversationExample]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ConversationExample(**row) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval-grounded support agent.")
    parser.add_argument("--message", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/examples.csv")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--demo"], check=True)
        examples = load_processed(ROOT / "data/processed/examples.csv")
    else:
        examples = load_processed(args.data)
    result = SupportAgent(examples).respond(args.message)
    print(json.dumps({
        "intent": result.intent,
        "confidence": result.confidence,
        "drafted_reply": result.drafted_reply,
        "decision": result.decision,
        "escalation_reason": result.escalation_reason,
        "evidence": [item.__dict__ for item in result.evidence],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
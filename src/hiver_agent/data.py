from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from .types import ConversationExample


REQUIRED_COLUMNS = {"tweet_id", "author_id", "inbound", "text"}


def _is_inbound(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "inbound"}


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def inspect_rows(rows: Iterable[dict[str, str]]) -> dict[str, object]:
    rows = list(rows)
    inbound = [row for row in rows if _is_inbound(row.get("inbound", ""))]
    outbound = [row for row in rows if not _is_inbound(row.get("inbound", ""))]
    linked = sum(bool(row.get("in_response_to_tweet_id")) for row in rows)
    support_authors = Counter(row.get("author_id", "") for row in outbound)
    return {
        "rows": len(rows),
        "customer_messages": len(inbound),
        "support_responses": len(outbound),
        "linked_rows": linked,
        "support_authors": support_authors.most_common(20),
        "empty_text_rows": sum(not str(row.get("text", "")).strip() for row in rows),
    }


def pair_tweets(rows: Iterable[dict[str, str]], brand_account: str | None = None) -> list[ConversationExample]:
    rows = list(rows)
    by_id = {row.get("tweet_id", ""): row for row in rows}
    examples: list[ConversationExample] = []
    for response in rows:
        if _is_inbound(response.get("inbound", "")):
            continue
        if brand_account and response.get("author_id") != brand_account:
            continue
        parent_id = response.get("in_response_to_tweet_id", "")
        parent = by_id.get(parent_id)
        if not parent or not _is_inbound(parent.get("inbound", "")):
            continue
        customer = str(parent.get("text", "")).strip()
        support = str(response.get("text", "")).strip()
        if not customer or not support:
            continue
        examples.append(
            ConversationExample(
                example_id=str(parent.get("tweet_id", "")),
                customer_message=customer,
                support_response=support,
                context=f"support_author={response.get('author_id', '')}",
            )
        )
    return examples


def load_examples(path: str | Path, brand_account: str | None = None) -> list[ConversationExample]:
    return pair_tweets(read_rows(path), brand_account=brand_account)


def write_examples(path: str | Path, examples: Iterable[ConversationExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["example_id", "customer_message", "support_response", "context", "intent"],
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(example.__dict__)
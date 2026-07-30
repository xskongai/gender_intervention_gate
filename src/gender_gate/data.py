from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schema import DatasetItem


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_items(path: str | Path) -> list[DatasetItem]:
    return [DatasetItem.from_dict(row) for row in read_jsonl(path)]


def validate_items(items: list[DatasetItem]) -> dict:
    ids = [item.id for item in items]
    texts = [item.text for item in items]
    labels = [item.label for item in items]

    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("Duplicate IDs detected.")
    if len(texts) != len(set(texts)):
        errors.append("Duplicate texts detected.")
    if any(not text.strip() for text in texts):
        errors.append("Empty text detected.")
    invalid_labels = sorted(set(labels) - {"POSITIVE", "NEGATIVE"})
    if invalid_labels:
        errors.append(f"Invalid labels: {invalid_labels}")

    return {
        "count": len(items),
        "label_counts": dict(Counter(labels)),
        "errors": errors,
        "valid": not errors,
    }

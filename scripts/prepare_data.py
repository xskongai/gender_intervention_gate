#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def convert(path: Path, label: str, source: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = clean(row.get("输入句子"))
            if not text:
                continue
            rows.append(
                {
                    "id": clean(row.get("编号")),
                    "text": text,
                    "label": label,
                    "meta": {
                        "l1": clean(row.get("L1类别")),
                        "l2": clean(row.get("L2子类")),
                        "register_group": clean(row.get("语体大类")),
                        "register": clean(row.get("语体")),
                        "noise": clean(row.get("噪声类型")),
                        "difficulty": clean(row.get("难度")),
                        "controversial": clean(row.get("是否争议")),
                        "source": source,
                        "original_split": clean(row.get("切分")),
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-csv", default="data/raw/positive_main.csv")
    parser.add_argument("--negative-csv", default="data/raw/negative_main.csv")
    parser.add_argument("--output", default="data/processed/main.jsonl")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    records = convert(root / args.positive_csv, "POSITIVE", "positive.xlsx")
    records += convert(root / args.negative_csv, "NEGATIVE", "negative.xlsx")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TEMPLATE_GROUP_RE = re.compile(r"模板组\s*(\d+)")


def clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def template_group(row: dict[str, str]) -> str | None:
    note = clean(row.get("备注")) or ""
    match = TEMPLATE_GROUP_RE.search(note)
    if not match:
        return None
    return f"cn_gi_template_{int(match.group(1)):02d}"


def convert(path: Path, label: str, source_workbook: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = clean(row.get("编号"))
            text = clean(row.get("输入句子"))
            if not item_id or not text:
                continue

            group = template_group(row)
            reference_output = clean(
                row.get("参考改写") if label == "POSITIVE" else row.get("期望输出")
            )
            rows.append(
                {
                    "id": item_id,
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
                        "source": clean(row.get("来源版本")) or source_workbook,
                        "source_workbook": source_workbook,
                        "dataset_version": "v2.3",
                        "original_split": clean(row.get("切分")),
                        "reference_output": reference_output,
                        "template_group": group,
                        "split_group": group or item_id,
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
    records = convert(root / args.positive_csv, "POSITIVE", "positive_v2.3")
    records += convert(root / args.negative_csv, "NEGATIVE", "negative_v2.3")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    positive = sum(row["label"] == "POSITIVE" for row in records)
    negative = sum(row["label"] == "NEGATIVE" for row in records)
    print(
        f"Wrote {len(records)} records to {output} "
        f"(POSITIVE={positive}, NEGATIVE={negative})"
    )


if __name__ == "__main__":
    main()

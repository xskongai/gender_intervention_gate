#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

from gender_gate.data import read_jsonl, write_jsonl


def proportional_quotas(group_sizes: dict, total: int) -> dict:
    total_size = sum(group_sizes.values())
    raw = {key: total * size / total_size for key, size in group_sizes.items()}
    quotas = {
        key: min(group_sizes[key], math.floor(raw[key]))
        for key in group_sizes
    }
    remaining = total - sum(quotas.values())
    order = sorted(
        group_sizes,
        key=lambda key: (
            raw[key] - math.floor(raw[key]),
            group_sizes[key],
            str(key),
        ),
        reverse=True,
    )
    index = 0
    while remaining > 0:
        key = order[index % len(order)]
        if quotas[key] < group_sizes[key]:
            quotas[key] += 1
            remaining -= 1
        index += 1
    return quotas


def stratified_take(
    items: list[dict], total: int, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        groups[(item["label"], item["meta"]["l1"])].append(item)
    for group in groups.values():
        rng.shuffle(group)

    quotas = proportional_quotas(
        {key: len(value) for key, value in groups.items()},
        total,
    )
    selected: list[dict] = []
    remaining: list[dict] = []
    for key in sorted(groups, key=str):
        selected.extend(groups[key][: quotas[key]])
        remaining.extend(groups[key][quotas[key] :])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/main.jsonl")
    parser.add_argument("--output-dir", default="data/splits/iid_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exemplar-size", type=int, default=80)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--pilot-size", type=int, default=60)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    items = read_jsonl(root / args.input)
    exemplar, remainder = stratified_take(
        items, args.exemplar_size, args.seed
    )
    dev, test = stratified_take(
        remainder, args.dev_size, args.seed + 1
    )
    pilot, _ = stratified_take(
        dev, args.pilot_size, args.seed + 2
    )

    output = root / args.output_dir
    write_jsonl(output / "exemplar_pool.jsonl", exemplar)
    write_jsonl(output / "dev.jsonl", dev)
    write_jsonl(output / "dev_pilot_60.jsonl", pilot)
    write_jsonl(output / "test.jsonl", test)

    print(
        f"exemplar={len(exemplar)} dev={len(dev)} "
        f"pilot={len(pilot)} test={len(test)}"
    )


if __name__ == "__main__":
    main()

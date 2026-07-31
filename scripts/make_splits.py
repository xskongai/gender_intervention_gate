#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from gender_gate.data import read_jsonl, write_jsonl


def proportional_quotas(group_sizes: dict, total: int) -> dict:
    total_size = sum(group_sizes.values())
    raw = {key: total * size / total_size for key, size in group_sizes.items()}
    quotas = {key: min(group_sizes[key], math.floor(raw[key])) for key in group_sizes}
    remaining = total - sum(quotas.values())
    order = sorted(
        group_sizes,
        key=lambda key: (raw[key] - math.floor(raw[key]), group_sizes[key], str(key)),
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


def strata_key(item: dict) -> tuple:
    return item["label"], item["meta"].get("l1")


def stratified_take(items: list[dict], total: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        groups[strata_key(item)].append(item)
    for group in groups.values():
        rng.shuffle(group)

    quotas = proportional_quotas({key: len(value) for key, value in groups.items()}, total)
    selected: list[dict] = []
    remaining: list[dict] = []
    for key in sorted(groups, key=str):
        selected.extend(groups[key][: quotas[key]])
        remaining.extend(groups[key][quotas[key] :])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def group_aware_take(items: list[dict], total: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Take an exact-size stratified sample without splitting template groups."""
    rng = random.Random(seed)
    item_groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        group_id = item["meta"].get("split_group") or item["id"]
        item_groups[str(group_id)].append(item)

    # Template groups can legitimately contain several L1/L2 categories.
    # Keep each group intact and stratify the group-aware split by label only.
    strata_groups: dict[str, list[list[dict]]] = defaultdict(list)
    for group in item_groups.values():
        labels = {item["label"] for item in group}
        if len(labels) != 1:
            raise ValueError(f"Split group crosses labels: {[item['id'] for item in group]}")
        strata_groups[next(iter(labels))].append(group)

    quotas = proportional_quotas(
        {key: sum(len(group) for group in groups) for key, groups in strata_groups.items()},
        total,
    )

    chosen_ids: set[str] = set()
    for key in sorted(strata_groups, key=str):
        groups = list(strata_groups[key])
        rng.shuffle(groups)
        multi = [group for group in groups if len(group) > 1]
        singles = [group for group in groups if len(group) == 1]
        rng.shuffle(multi)
        rng.shuffle(singles)

        remaining_quota = quotas[key]
        chosen: list[list[dict]] = []
        skipped: list[list[dict]] = []

        for group in multi:
            if len(group) <= remaining_quota:
                chosen.append(group)
                remaining_quota -= len(group)
            else:
                skipped.append(group)

        if remaining_quota > len(singles):
            # Add one skipped group only when it improves the distance to target.
            skipped.sort(key=lambda group: abs(len(group) - remaining_quota))
            if skipped:
                best = skipped[0]
                if len(best) <= remaining_quota + len(singles):
                    chosen.append(best)
                    remaining_quota -= len(best)

        if remaining_quota < 0:
            # Remove singletons later cannot repair an overshoot from a multi-item group.
            raise ValueError(f"Could not satisfy exact group-aware quota for {key}")
        if remaining_quota > len(singles):
            raise ValueError(f"Not enough singleton groups to satisfy quota for {key}")

        chosen.extend(singles[:remaining_quota])
        for group in chosen:
            chosen_ids.update(item["id"] for item in group)

    selected = [item for item in items if item["id"] in chosen_ids]
    remaining = [item for item in items if item["id"] not in chosen_ids]
    if len(selected) != total:
        raise AssertionError(f"Expected {total} selected items, got {len(selected)}")
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def split_group_ids(items: list[dict]) -> set[str]:
    return {
        str(item["meta"].get("split_group") or item["id"])
        for item in items
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/main.jsonl")
    parser.add_argument("--output-dir", default="data/splits/group_aware_v2.3")
    parser.add_argument("--mode", choices=["iid", "group-aware"], default="group-aware")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exemplar-size", type=int, default=80)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--pilot-size", type=int, default=60)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    items = read_jsonl(root / args.input)
    take = group_aware_take if args.mode == "group-aware" else stratified_take

    exemplar, remainder = take(items, args.exemplar_size, args.seed)
    dev, test = take(remainder, args.dev_size, args.seed + 1)
    pilot, _ = stratified_take(dev, args.pilot_size, args.seed + 2)

    output = root / args.output_dir
    write_jsonl(output / "exemplar_pool.jsonl", exemplar)
    write_jsonl(output / "dev.jsonl", dev)
    write_jsonl(output / "dev_pilot_60.jsonl", pilot)
    write_jsonl(output / "test.jsonl", test)

    if args.mode == "group-aware":
        assert not split_group_ids(exemplar) & split_group_ids(dev)
        assert not split_group_ids(exemplar) & split_group_ids(test)
        assert not split_group_ids(dev) & split_group_ids(test)

    manifest = {
        "dataset_version": "v2.3",
        "mode": args.mode,
        "seed": {"exemplar": args.seed, "dev": args.seed + 1, "pilot": args.seed + 2},
        "counts": {
            "main": len(items),
            "labels": dict(Counter(item["label"] for item in items)),
            "exemplar_pool": len(exemplar),
            "dev": len(dev),
            "pilot": len(pilot),
            "test": len(test),
        },
        "template_groups": len(
            {
                item["meta"].get("template_group")
                for item in items
                if item["meta"].get("template_group")
            }
        ),
        "note": (
            "Template-derived samples sharing a template_group never cross exemplar/dev/test."
            if args.mode == "group-aware"
            else "Row-level stratified random split retained for baseline comparability."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"mode={args.mode} exemplar={len(exemplar)} dev={len(dev)} "
        f"pilot={len(pilot)} test={len(test)}"
    )


if __name__ == "__main__":
    main()

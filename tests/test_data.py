from pathlib import Path

from gender_gate.data import load_items, validate_items


ROOT = Path(__file__).resolve().parents[1]


def test_main_dataset_counts() -> None:
    items = load_items(ROOT / "data/processed/main.jsonl")
    report = validate_items(items)
    assert report["valid"], report["errors"]
    assert report["count"] == 1588
    assert report["label_counts"] == {
        "POSITIVE": 871,
        "NEGATIVE": 717,
    }


def test_all_rows_are_v23_and_have_split_groups() -> None:
    items = load_items(ROOT / "data/processed/main.jsonl")
    assert all(item.meta.get("dataset_version") == "v2.3" for item in items)
    assert all(item.meta.get("split_group") for item in items)

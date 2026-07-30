from pathlib import Path

from gender_gate.data import load_items, validate_items


ROOT = Path(__file__).resolve().parents[1]


def test_main_dataset_counts() -> None:
    items = load_items(ROOT / "data/processed/main.jsonl")
    report = validate_items(items)
    assert report["valid"], report["errors"]
    assert report["count"] == 1532
    assert report["label_counts"] == {
        "POSITIVE": 734,
        "NEGATIVE": 798,
    }

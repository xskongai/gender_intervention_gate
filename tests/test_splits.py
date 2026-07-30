from pathlib import Path

from gender_gate.data import load_items


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/splits/iid_v1"


def ids(name: str) -> set[str]:
    return {
        item.id
        for item in load_items(SPLIT_DIR / f"{name}.jsonl")
    }


def test_split_sizes_and_disjointness() -> None:
    exemplar = ids("exemplar_pool")
    dev = ids("dev")
    test = ids("test")
    pilot = ids("dev_pilot_60")

    assert len(exemplar) == 80
    assert len(dev) == 400
    assert len(test) == 1052
    assert len(pilot) == 60
    assert not exemplar & dev
    assert not exemplar & test
    assert not dev & test
    assert pilot <= dev

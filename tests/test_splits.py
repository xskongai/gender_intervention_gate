from pathlib import Path

from gender_gate.data import load_items


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/splits/group_aware_v2.3"


def load(name: str):
    return load_items(SPLIT_DIR / f"{name}.jsonl")


def ids(name: str) -> set[str]:
    return {item.id for item in load(name)}


def split_groups(name: str) -> set[str]:
    return {
        str(item.meta.get("split_group") or item.id)
        for item in load(name)
    }


def test_split_sizes_and_disjointness() -> None:
    exemplar = ids("exemplar_pool")
    dev = ids("dev")
    test = ids("test")
    pilot = ids("dev_pilot_60")

    assert len(exemplar) == 80
    assert len(dev) == 400
    assert len(test) == 1108
    assert len(pilot) == 60
    assert not exemplar & dev
    assert not exemplar & test
    assert not dev & test
    assert pilot <= dev


def test_template_groups_do_not_cross_main_splits() -> None:
    exemplar = split_groups("exemplar_pool")
    dev = split_groups("dev")
    test = split_groups("test")

    assert not exemplar & dev
    assert not exemplar & test
    assert not dev & test

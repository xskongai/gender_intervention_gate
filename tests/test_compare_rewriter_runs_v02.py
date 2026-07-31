import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_rewriter_runs.py"
spec = importlib.util.spec_from_file_location("compare_rewriter_runs_v02", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(item_id: str, gold: str, text: str = "句子") -> dict:
    return {
        "id": item_id,
        "gold": gold,
        "text": text,
        "final_output": text + "改",
        "changed": True,
        "error": None,
        "cache_hit": False,
        "latency_ms": 1,
        "reference_output": text + "改",
    }


def test_align_allows_mixed_baseline_and_positive_candidate() -> None:
    baseline = [row("P1", "POSITIVE"), row("N1", "NEGATIVE")]
    candidate = [row("P1", "POSITIVE")]
    b, c = module.align_positive_predictions(baseline, candidate)
    assert [x["id"] for x in b] == ["P1"]
    assert [x["id"] for x in c] == ["P1"]


def test_align_rejects_missing_candidate_id() -> None:
    with pytest.raises(ValueError, match="missing"):
        module.align_positive_predictions([row("P1", "POSITIVE")], [row("P2", "POSITIVE")])

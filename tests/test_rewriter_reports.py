import pytest

from gender_gate.rewriter_reports import calculate_rewriter_metrics


def test_rewriter_metrics_positive_only() -> None:
    predictions = [
        {
            "gold": "POSITIVE",
            "changed": True,
            "error": None,
            "cache_hit": False,
            "latency_ms": 100,
            "final_output": "她跑步很快。",
            "reference_output": "她跑步很快。",
        },
        {
            "gold": "POSITIVE",
            "changed": False,
            "error": None,
            "cache_hit": True,
            "latency_ms": 0,
            "final_output": "男人不能怕。",
            "reference_output": "害怕是正常情绪。",
        },
    ]
    metrics = calculate_rewriter_metrics(predictions)
    assert metrics["count"] == 2
    assert metrics["intervention_rate"] == 0.5
    assert metrics["under_edit_rate"] == 0.5
    assert metrics["exact_reference_match_rate"] == 0.5
    assert metrics["mean_latency_ms_non_cached"] == 100


def test_rewriter_metrics_reject_negative() -> None:
    with pytest.raises(ValueError, match="POSITIVE-only"):
        calculate_rewriter_metrics([{"gold": "NEGATIVE"}])

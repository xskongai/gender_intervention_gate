from gender_gate.rewrite_metrics import (
    calculate_rewrite_metrics,
    semantic_review_reasons,
)


def test_rewrite_metrics() -> None:
    predictions = [
        {
            "gold": "POSITIVE",
            "changed": True,
            "rewrite_called": True,
            "error": None,
            "cache_hit": False,
            "final_output": "改写一",
            "reference_output": "改写一",
        },
        {
            "gold": "POSITIVE",
            "changed": False,
            "rewrite_called": False,
            "error": None,
            "cache_hit": False,
            "final_output": "原文二",
            "reference_output": "改写二",
        },
        {
            "gold": "NEGATIVE",
            "changed": False,
            "rewrite_called": False,
            "error": None,
            "cache_hit": False,
            "final_output": "原文三",
            "reference_output": "原文三",
        },
        {
            "gold": "NEGATIVE",
            "changed": True,
            "rewrite_called": True,
            "error": None,
            "cache_hit": True,
            "final_output": "误改四",
            "reference_output": "原文四",
        },
    ]
    metrics = calculate_rewrite_metrics(predictions)
    assert metrics["positive_intervention_rate"] == 0.5
    assert metrics["under_edit_rate"] == 0.5
    assert metrics["negative_preservation"] == 0.5
    assert metrics["over_edit_rate"] == 0.5
    assert metrics["rewrite_calls"] == 2
    assert metrics["rewrite_calls_saved"] == 2
    assert metrics["exact_reference_match_rate"] == 0.5


def test_semantic_review_flags_are_candidates_only() -> None:
    prediction = {
        "rewrite_called": True,
        "error": None,
        "text": "女人都是路痴。",
        "final_output": "一个人的方向感由多种复杂因素共同决定，与性别没有必然关系。",
    }
    reasons = semantic_review_reasons(prediction)
    assert "OUTPUT_MUCH_LONGER" in reasons

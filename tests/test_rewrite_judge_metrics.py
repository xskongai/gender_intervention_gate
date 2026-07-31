from gender_gate.rewrite_judge_metrics import (
    calculate_judge_metrics,
    normalize_dimension_score,
    score_judgment,
)


def test_normalize_dimension_score() -> None:
    assert normalize_dimension_score(1) == 0
    assert normalize_dimension_score(2) == 50
    assert normalize_dimension_score(3) == 100


def test_score_local_judgment() -> None:
    row = {
        "id": "P1",
        "rewrite_type": "LOCAL_REPAIR",
        "debiasing_score": 3,
        "naturalness_score": 2,
        "no_added_facts_score": 3,
        "relevance_score": None,
        "error": None,
    }
    scored = score_judgment(row)
    assert scored["quality_score"] == 87.5
    assert scored["verdict"] == "PARTIAL"
    assert scored["type_specific_metric"] == "no_added_facts"


def test_score_reconstruction_fail() -> None:
    row = {
        "id": "P2",
        "rewrite_type": "PROPOSITION_RECONSTRUCTION",
        "debiasing_score": 3,
        "naturalness_score": 3,
        "no_added_facts_score": None,
        "relevance_score": 1,
        "error": None,
    }
    scored = score_judgment(row)
    assert scored["quality_score"] == 75.0
    assert scored["verdict"] == "FAIL"


def test_calculate_macro_quality() -> None:
    rows = [
        score_judgment(
            {
                "rewrite_type": "LOCAL_REPAIR",
                "debiasing_score": 3,
                "naturalness_score": 3,
                "no_added_facts_score": 3,
                "relevance_score": None,
                "error": None,
            }
        ),
        score_judgment(
            {
                "rewrite_type": "PROPOSITION_RECONSTRUCTION",
                "debiasing_score": 2,
                "naturalness_score": 2,
                "no_added_facts_score": None,
                "relevance_score": 2,
                "error": None,
            }
        ),
    ]
    metrics = calculate_judge_metrics(rows)
    assert metrics["local_repair"]["quality_score"] == 100
    assert metrics["proposition_reconstruction"]["quality_score"] == 50
    assert metrics["macro_quality_score"] == 75

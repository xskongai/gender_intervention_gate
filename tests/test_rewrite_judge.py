import pytest

from gender_gate.rewrite_judge import (
    normalize_rewrite_type,
    parse_judge_output,
)


def test_normalize_rewrite_type_aliases() -> None:
    assert normalize_rewrite_type("local") == "LOCAL_REPAIR"
    assert (
        normalize_rewrite_type("reconstruction")
        == "PROPOSITION_RECONSTRUCTION"
    )


def test_parse_local_judge_v02_json() -> None:
    raw = """```json
    {
      "debiasing": {"score": 3, "reason": "偏见已消除"},
      "naturalness": {"score": 2, "reason": "略生硬"},
      "fidelity": {"score": 3, "reason": "其余内容忠实"},
      "relevance": null
    }
    ```"""
    parsed = parse_judge_output(raw, "LOCAL_REPAIR")
    assert parsed.debiasing_score == 3
    assert parsed.naturalness_score == 2
    assert parsed.fidelity_score == 3
    assert parsed.no_added_facts_score is None
    assert parsed.relevance_score is None


def test_parse_local_judge_v01_legacy_json() -> None:
    raw = """
    {
      "debiasing": {"score": 3, "reason": "偏见已消除"},
      "naturalness": {"score": 3, "reason": "自然"},
      "no_added_facts": {"score": 3, "reason": "未增加事实"},
      "relevance": null
    }
    """
    parsed = parse_judge_output(raw, "LOCAL_REPAIR")
    assert parsed.fidelity_score is None
    assert parsed.no_added_facts_score == 3


def test_parse_reconstruction_judge_v02_json() -> None:
    raw = """
    {
      "debiasing": {"score": 3, "reason": "偏见已消除"},
      "naturalness": {"score": 3, "reason": "自然"},
      "fidelity": null,
      "relevance": {"score": 2, "reason": "略宽泛"}
    }
    """
    parsed = parse_judge_output(raw, "PROPOSITION_RECONSTRUCTION")
    assert parsed.relevance_score == 2
    assert parsed.fidelity_score is None
    assert parsed.no_added_facts_score is None


def test_parse_rejects_both_local_specific_dimensions() -> None:
    raw = """
    {
      "debiasing": {"score": 3, "reason": "ok"},
      "naturalness": {"score": 3, "reason": "ok"},
      "fidelity": {"score": 3, "reason": "ok"},
      "no_added_facts": {"score": 3, "reason": "legacy"},
      "relevance": null
    }
    """
    with pytest.raises(ValueError, match="not both"):
        parse_judge_output(raw, "LOCAL_REPAIR")


def test_parse_rejects_wrong_type_specific_dimension() -> None:
    raw = """
    {
      "debiasing": {"score": 3, "reason": "ok"},
      "naturalness": {"score": 3, "reason": "ok"},
      "fidelity": {"score": 3, "reason": "ok"},
      "relevance": {"score": 3, "reason": "wrong"}
    }
    """
    with pytest.raises(ValueError, match="relevance to null"):
        parse_judge_output(raw, "LOCAL_REPAIR")

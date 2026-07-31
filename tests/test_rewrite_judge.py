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


def test_parse_single_dimension_output() -> None:
    from gender_gate.rewrite_judge import parse_single_dimension_output

    score, reason = parse_single_dimension_output(
        '```json\n{"score": "2", "reason": "部分满足"}\n```'
    )
    assert score == 2
    assert reason == "部分满足"


def test_split_judge_merges_local_dimension_outputs() -> None:
    import json
    from types import MethodType

    from gender_gate.rewrite_judge import (
        SplitRewriteQualityJudge,
        parse_judge_output,
    )

    judge = SplitRewriteQualityJudge.__new__(SplitRewriteQualityJudge)
    responses = {
        "debiasing": '{"score": 3, "reason": "偏见已消除"}',
        "naturalness": '{"score": 3, "reason": "表达自然"}',
        "type_specific": '{"score": 2, "reason": "略有扩展"}',
    }

    def fake_call(self, *, dimension, item_id, rewrite_type, text, output):
        return responses[dimension], 1, False, None

    judge._call_dimension = MethodType(fake_call, judge)
    raw, latency, cache_hit, error = judge.judge(
        item_id="P1",
        rewrite_type="LOCAL_REPAIR",
        text="原句",
        output="改写",
    )
    assert error is None
    assert latency == 3
    assert cache_hit is False
    payload = json.loads(raw)
    assert "_dimension_raw_outputs" in payload
    parsed = parse_judge_output(raw, "LOCAL_REPAIR")
    assert parsed.debiasing_score == 3
    assert parsed.naturalness_score == 3
    assert parsed.fidelity_score == 2
    assert parsed.relevance_score is None


def test_split_judge_merges_reconstruction_dimension_outputs() -> None:
    from types import MethodType

    from gender_gate.rewrite_judge import (
        SplitRewriteQualityJudge,
        parse_judge_output,
    )

    judge = SplitRewriteQualityJudge.__new__(SplitRewriteQualityJudge)
    responses = {
        "debiasing": '{"score": 3, "reason": "偏见已消除"}',
        "naturalness": '{"score": 2, "reason": "略生硬"}',
        "type_specific": '{"score": 3, "reason": "直接回应核心命题"}',
    }

    def fake_call(self, *, dimension, item_id, rewrite_type, text, output):
        return responses[dimension], 2, True, None

    judge._call_dimension = MethodType(fake_call, judge)
    raw, latency, cache_hit, error = judge.judge(
        item_id="P2",
        rewrite_type="PROPOSITION_RECONSTRUCTION",
        text="原句",
        output="改写",
    )
    assert error is None
    assert latency == 6
    assert cache_hit is True
    parsed = parse_judge_output(raw, "PROPOSITION_RECONSTRUCTION")
    assert parsed.debiasing_score == 3
    assert parsed.naturalness_score == 2
    assert parsed.relevance_score == 3
    assert parsed.fidelity_score is None


def test_split_judge_reports_dimension_parse_error() -> None:
    from types import MethodType

    from gender_gate.rewrite_judge import SplitRewriteQualityJudge

    judge = SplitRewriteQualityJudge.__new__(SplitRewriteQualityJudge)
    responses = {
        "debiasing": '{"score": 3, "reason": "ok"}',
        "naturalness": "not json",
        "type_specific": '{"score": 3, "reason": "ok"}',
    }

    def fake_call(self, *, dimension, item_id, rewrite_type, text, output):
        return responses[dimension], 0, False, None

    judge._call_dimension = MethodType(fake_call, judge)
    _, _, _, error = judge.judge(
        item_id="P3",
        rewrite_type="LOCAL_REPAIR",
        text="原句",
        output="改写",
    )
    assert error is not None
    assert "naturalness" in error

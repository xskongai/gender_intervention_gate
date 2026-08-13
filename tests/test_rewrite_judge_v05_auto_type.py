from pathlib import Path

import pytest
import yaml

from gender_gate.rewrite_judge import parse_auto_type_judge_output

ROOT = Path(__file__).resolve().parents[1]


def test_v05_auto_type_config_is_isolated_from_v04() -> None:
    v04 = yaml.safe_load(
        (ROOT / "configs/judge/rewrite_judge_v04_balanced_gpt4o.yaml").read_text(
            encoding="utf-8"
        )
    )
    v05 = yaml.safe_load(
        (ROOT / "configs/judge/rewrite_judge_v05_auto_type_gpt4o.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert v04["prompt"] == "prompts/rewrite_judge_v04_balanced.txt"
    assert v05["prompt"] == "prompts/rewrite_judge_v05_auto_type.txt"
    assert v04["name"] == "rewrite_judge_v04_balanced"
    assert v05["name"] == "rewrite_judge_v05_auto_type"


def test_v05_prompt_preserves_v04_quality_principles() -> None:
    prompt = (ROOT / "prompts/rewrite_judge_v05_auto_type.txt").read_text(
        encoding="utf-8"
    )
    for marker in [
        "3 分表示改写已经合理、正确、可接受",
        "同一个问题不要在多个维度重复扣分",
        "不得被视为信息损失",
        "不要为了严格而寻找细小缺点",
        "不要求再次明确提到性别",
    ]:
        assert marker in prompt
    assert "不要根据候选改写实际上改了多少" in prompt
    assert "fidelity 仍然是合理且必要的评价标准" in prompt
    assert "应以 relevance 而不是 fidelity" in prompt


def test_parse_auto_type_local_reuses_v04_dimension_rules() -> None:
    raw = """
    {
      "rewrite_type": "LOCAL_REPAIR",
      "debiasing": {"score": 3, "reason": "偏见已消除"},
      "naturalness": {"score": 3, "reason": "自然"},
      "fidelity": {"score": 2, "reason": "略有扩展"},
      "relevance": null
    }
    """
    rewrite_type, parsed = parse_auto_type_judge_output(raw)
    assert rewrite_type == "LOCAL_REPAIR"
    assert parsed.debiasing_score == 3
    assert parsed.naturalness_score == 3
    assert parsed.fidelity_score == 2
    assert parsed.relevance_score is None


def test_parse_auto_type_reconstruction_reuses_v04_dimension_rules() -> None:
    raw = """
    {
      "rewrite_type": "PROPOSITION_RECONSTRUCTION",
      "debiasing": {"score": 3, "reason": "偏见已消除"},
      "naturalness": {"score": 2, "reason": "略生硬"},
      "fidelity": null,
      "relevance": {"score": 3, "reason": "回应核心命题"}
    }
    """
    rewrite_type, parsed = parse_auto_type_judge_output(raw)
    assert rewrite_type == "PROPOSITION_RECONSTRUCTION"
    assert parsed.debiasing_score == 3
    assert parsed.naturalness_score == 2
    assert parsed.fidelity_score is None
    assert parsed.relevance_score == 3


def test_parse_auto_type_requires_rewrite_type() -> None:
    raw = """
    {
      "debiasing": {"score": 3, "reason": "ok"},
      "naturalness": {"score": 3, "reason": "ok"},
      "fidelity": {"score": 3, "reason": "ok"},
      "relevance": null
    }
    """
    with pytest.raises(ValueError, match="missing rewrite_type"):
        parse_auto_type_judge_output(raw)

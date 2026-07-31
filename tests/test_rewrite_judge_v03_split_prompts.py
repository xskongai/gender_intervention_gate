from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_split_judge_config_and_prompts_exist() -> None:
    config_path = ROOT / "configs/judge/rewrite_judge_v03_split_gpt4o.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["judge_mode"] == "split_dimensions"
    assert set(config["prompts"]) == {
        "debiasing",
        "naturalness",
        "type_specific",
    }
    for value in config["prompts"].values():
        assert (ROOT / value).exists()


def test_split_prompts_keep_dimensions_separate() -> None:
    debiasing = (ROOT / "prompts/rewrite_judge_v03_debiasing.txt").read_text(
        encoding="utf-8"
    )
    naturalness = (ROOT / "prompts/rewrite_judge_v03_naturalness.txt").read_text(
        encoding="utf-8"
    )
    type_specific = (
        ROOT / "prompts/rewrite_judge_v03_type_specific.txt"
    ).read_text(encoding="utf-8")

    assert "不要评价中文是否自然" in debiasing
    assert "不要评价它是否消除了性别偏见" in naturalness
    assert "不要评价性别偏见是否被消除" in type_specific
    assert "LOCAL_REPAIR" in type_specific
    assert "PROPOSITION_RECONSTRUCTION" in type_specific

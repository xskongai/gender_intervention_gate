from pathlib import Path

from gender_gate.rewrite_type_judge import (
    parse_rewrite_type_output,
    render_rewrite_type_prompt,
)


def test_parse_local_json() -> None:
    label, reason = parse_rewrite_type_output(
        '{"rewrite_type":"LOCAL_REPAIR","reason":"保留非偏见核心命题"}'
    )
    assert label == "LOCAL_REPAIR"
    assert reason == "保留非偏见核心命题"


def test_parse_reconstruction_json_fence() -> None:
    label, _ = parse_rewrite_type_output(
        '```json\n{"rewrite_type":"PROPOSITION_RECONSTRUCTION","reason":"核心命题本身需重构"}\n```'
    )
    assert label == "PROPOSITION_RECONSTRUCTION"


def test_prompt_uses_original_only() -> None:
    template = Path("prompts/rewrite_type_judge_v01.txt").read_text(encoding="utf-8")
    rendered = render_rewrite_type_prompt(
        template,
        item_id="POS-X",
        text="原句文本",
    )
    assert "POS-X" in rendered
    assert "原句文本" in rendered
    assert "{{ID}}" not in rendered
    assert "{{TEXT}}" not in rendered
    assert "Candidate rewrite" not in rendered
    assert "{{OUTPUT}}" not in rendered

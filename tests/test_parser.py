from gender_gate.parser import parse_label


def test_exact_labels() -> None:
    assert parse_label("POSITIVE") == "POSITIVE"
    assert parse_label(" NEGATIVE\n") == "NEGATIVE"


def test_single_label_inside_extra_text() -> None:
    assert parse_label("判断结果：POSITIVE") == "POSITIVE"
    assert parse_label("The final answer is NEGATIVE.") == "NEGATIVE"


def test_edit_keep_aliases() -> None:
    assert parse_label("EDIT") == "POSITIVE"
    assert parse_label("最终判断：KEEP") == "NEGATIVE"


def test_json_and_code_fence_outputs() -> None:
    assert parse_label('{"label": "POSITIVE"}') == "POSITIVE"
    assert parse_label('```json\n{"decision": "KEEP"}\n```') == "NEGATIVE"


def test_thinking_block_is_ignored() -> None:
    raw = "<think>POSITIVE and NEGATIVE are both possible.</think>\nNEGATIVE"
    assert parse_label(raw) == "NEGATIVE"


def test_final_standalone_line_wins() -> None:
    raw = "I considered POSITIVE and NEGATIVE.\n\nPOSITIVE"
    assert parse_label(raw) == "POSITIVE"


def test_ambiguous_or_invalid_output() -> None:
    assert parse_label("POSITIVE or NEGATIVE") is None
    assert parse_label("CLEAR") is None
    assert parse_label("") is None

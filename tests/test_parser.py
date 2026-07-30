from gender_gate.parser import parse_label


def test_exact_labels() -> None:
    assert parse_label("POSITIVE") == "POSITIVE"
    assert parse_label(" NEGATIVE\n") == "NEGATIVE"


def test_single_label_inside_extra_text() -> None:
    assert parse_label("判断结果：POSITIVE") == "POSITIVE"


def test_ambiguous_or_invalid_output() -> None:
    assert parse_label("POSITIVE or NEGATIVE") is None
    assert parse_label("CLEAR") is None

from gender_gate.deterministic_rules import deterministic_label


def test_lexicalized_term_is_negative() -> None:
    result = deterministic_label("女儿墙完成了防水施工。")
    assert result is not None
    assert result["label"] == "NEGATIVE"
    assert result["rule"] == "KEEP_LEXICALIZED_TERM"


def test_lexicalized_term_does_not_hide_other_gender_claim() -> None:
    assert deterministic_label("女儿墙施工队认为女人不适合高空作业。") is None


def test_explicit_generic_masculine_is_positive() -> None:
    result = deterministic_label("每位员工都应保管好他的工牌。")
    assert result is not None
    assert result["label"] == "POSITIVE"
    assert result["rule"] == "EDIT_GENERIC_MASCULINE"


def test_specific_pronoun_is_not_routed() -> None:
    assert deterministic_label("王先生到了以后，他把材料交给了前台。") is None


def test_gender_comparison_is_positive() -> None:
    result = deterministic_label("她比男人还果断。")
    assert result is not None
    assert result["label"] == "POSITIVE"
    assert result["rule"] == "EDIT_GENDER_COMPARISON"


def test_metalinguistic_comparison_falls_back() -> None:
    assert deterministic_label("‘她比男人还果断’这种说法属于刻板印象。") is None


def test_direct_slur_is_positive() -> None:
    result = deterministic_label("都三十五了还不结婚，典型的剩女。")
    assert result is not None
    assert result["label"] == "POSITIVE"
    assert result["rule"] == "EDIT_GENDER_SLUR"


def test_slur_discussion_falls_back() -> None:
    assert deterministic_label("‘剩女’是一个贬义词。") is None


def test_reported_quote_is_negative() -> None:
    result = deterministic_label(
        "该主播说：“女生就别学编程了。”平台已对其作出处罚。"
    )
    assert result is not None
    assert result["label"] == "NEGATIVE"
    assert result["rule"] == "KEEP_QUOTED_REPORT"

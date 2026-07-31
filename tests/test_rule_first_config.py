from scripts.run_experiment import apply_rule_first_override


def test_rule_first_override_enables_default_ruleset() -> None:
    config: dict = {}
    apply_rule_first_override(config, True)
    assert config["rule_first"] == {
        "enabled": True,
        "ruleset": "deterministic_v01",
    }


def test_rule_first_override_can_disable() -> None:
    config = {
        "rule_first": {
            "enabled": True,
            "ruleset": "deterministic_v01",
        }
    }
    apply_rule_first_override(config, False)
    assert config["rule_first"]["enabled"] is False


def test_no_override_preserves_config() -> None:
    config = {"rule_first": {"enabled": True}}
    apply_rule_first_override(config, None)
    assert config == {"rule_first": {"enabled": True}}

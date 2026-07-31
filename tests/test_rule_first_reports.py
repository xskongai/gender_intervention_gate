from gender_gate.reports import route_metrics


def test_route_metrics() -> None:
    predictions = [
        {"route": "RULE", "rule": "R1", "predicted": "POSITIVE", "gold": "POSITIVE"},
        {"route": "RULE", "rule": "R1", "predicted": "NEGATIVE", "gold": "NEGATIVE"},
        {"route": "LLM", "rule": None, "predicted": "POSITIVE", "gold": "POSITIVE"},
    ]
    metrics = route_metrics(predictions)
    assert metrics["rule_routed"] == 2
    assert metrics["llm_routed"] == 1
    assert metrics["rule_coverage"] == 2 / 3
    assert metrics["llm_call_rate"] == 1 / 3
    assert metrics["rule_observed_accuracy"] == 1.0
    assert metrics["rule_counts"] == {"R1": 2}

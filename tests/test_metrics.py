from gender_gate.metrics import calculate_metrics


def test_metrics() -> None:
    predictions = [
        {"gold": "POSITIVE", "predicted": "POSITIVE"},
        {"gold": "POSITIVE", "predicted": "NEGATIVE"},
        {"gold": "NEGATIVE", "predicted": "NEGATIVE"},
        {"gold": "NEGATIVE", "predicted": None},
    ]
    metrics = calculate_metrics(predictions)
    assert metrics["positive_recall"] == 0.5
    assert metrics["negative_recall"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["format_error_rate"] == 0.25
    assert metrics["passes_90_target"] is False

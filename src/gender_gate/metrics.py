from __future__ import annotations

from typing import Any


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(
        p["gold"] == "POSITIVE" and p.get("predicted") == "POSITIVE"
        for p in predictions
    )
    fn = sum(
        p["gold"] == "POSITIVE" and p.get("predicted") != "POSITIVE"
        for p in predictions
    )
    tn = sum(
        p["gold"] == "NEGATIVE" and p.get("predicted") == "NEGATIVE"
        for p in predictions
    )
    fp = sum(
        p["gold"] == "NEGATIVE" and p.get("predicted") != "NEGATIVE"
        for p in predictions
    )

    predicted_positive = sum(p.get("predicted") == "POSITIVE" for p in predictions)
    predicted_negative = sum(p.get("predicted") == "NEGATIVE" for p in predictions)
    format_errors = sum(p.get("predicted") is None for p in predictions)

    positive_recall = safe_div(tp, tp + fn)
    negative_recall = safe_div(tn, tn + fp)
    positive_precision = safe_div(tp, predicted_positive)
    negative_precision = safe_div(tn, predicted_negative)
    positive_f1 = safe_div(
        2 * positive_precision * positive_recall,
        positive_precision + positive_recall,
    )
    negative_f1 = safe_div(
        2 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )
    accuracy = safe_div(tp + tn, len(predictions))

    return {
        "count": len(predictions),
        "confusion": {
            "true_positive": tp,
            "false_negative": fn,
            "true_negative": tn,
            "false_positive": fp,
            "format_errors": format_errors,
        },
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "positive_precision": positive_precision,
        "negative_precision": negative_precision,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "accuracy": accuracy,
        "format_error_rate": safe_div(format_errors, len(predictions)),
        "passes_90_target": positive_recall >= 0.90 and negative_recall >= 0.90,
        "passes_94_dev_target": positive_recall >= 0.94 and negative_recall >= 0.94,
    }

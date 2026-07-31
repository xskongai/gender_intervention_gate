from __future__ import annotations

import json
import re
from typing import Any

from .schema import Label

_LABEL_MAP: dict[str, Label] = {
    "POSITIVE": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "EDIT": "POSITIVE",
    "KEEP": "NEGATIVE",
}

_LABEL_RE = re.compile(r"\b(POSITIVE|NEGATIVE|EDIT|KEEP)\b", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_MARKER_RE = re.compile(
    r"(?:"
    r"final\s+(?:answer|label|decision|result)"
    r"|answer|label|decision|result"
    r"|最终(?:答案|判断|标签|结论)"
    r"|判断结果|分类结果|答案|标签|结论"
    r")"
    r"\s*(?:is|为|是|[:：=\-])?\s*"
    r"[`*\[\(]*\s*(POSITIVE|NEGATIVE|EDIT|KEEP)\b",
    re.IGNORECASE,
)


def _map_label(value: str) -> Label | None:
    return _LABEL_MAP.get(value.strip().upper())


def _strip_wrappers(raw_output: str) -> str:
    text = raw_output.strip()
    text = _THINK_BLOCK_RE.sub("\n", text)
    text = re.sub(r"^\s*```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _parse_json_label(text: str) -> Label | None:
    candidates = [text]
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match and object_match.group(0) != text:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            value: Any = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, str):
            label = _map_label(value)
            if label is not None:
                return label
        if isinstance(value, dict):
            for key in ("label", "decision", "prediction", "predicted", "result"):
                field = value.get(key)
                if isinstance(field, str):
                    label = _map_label(field)
                    if label is not None:
                        return label
    return None


def _parse_exact_fragment(text: str) -> Label | None:
    cleaned = text.strip().strip("`*_[](){}<> \t\r\n.,;:：。；！!?\"'")
    return _map_label(cleaned)


def parse_label(raw_output: str) -> Label | None:
    """Parse a conservative binary Gate label from heterogeneous model output.

    The parser accepts exact labels, EDIT/KEEP aliases, JSON fields, explicit
    final-answer markers, and a final standalone label line. It deliberately
    rejects genuinely ambiguous prose containing both classes without a clear
    final decision.
    """

    if not raw_output or not raw_output.strip():
        return None

    text = _strip_wrappers(raw_output)
    if not text:
        return None

    exact = _parse_exact_fragment(text)
    if exact is not None:
        return exact

    json_label = _parse_json_label(text)
    if json_label is not None:
        return json_label

    marker_matches = _FINAL_MARKER_RE.findall(text)
    if marker_matches:
        return _map_label(marker_matches[-1])

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        final_line = _parse_exact_fragment(nonempty_lines[-1])
        if final_line is not None:
            return final_line

    mapped_matches = [
        label
        for token in _LABEL_RE.findall(text)
        if (label := _map_label(token)) is not None
    ]
    unique = set(mapped_matches)
    if len(unique) == 1:
        return mapped_matches[-1]
    return None

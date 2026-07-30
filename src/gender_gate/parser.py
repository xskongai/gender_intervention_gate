from __future__ import annotations

import re

from .schema import Label

_LABEL_RE = re.compile(r"\b(POSITIVE|NEGATIVE)\b", re.IGNORECASE)


def parse_label(raw_output: str) -> Label | None:
    normalized = raw_output.strip().upper()
    if normalized in {"POSITIVE", "NEGATIVE"}:
        return normalized  # type: ignore[return-value]

    matches = {match.upper() for match in _LABEL_RE.findall(raw_output)}
    if len(matches) == 1:
        return next(iter(matches))  # type: ignore[return-value]
    return None

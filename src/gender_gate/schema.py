from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Label = Literal["POSITIVE", "NEGATIVE"]


@dataclass(frozen=True)
class DatasetItem:
    id: str
    text: str
    label: Label
    meta: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetItem":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            label=value["label"],
            meta=dict(value.get("meta") or {}),
        )


@dataclass
class Prediction:
    id: str
    text: str
    gold: Label
    predicted: Label | None
    raw_output: str
    model: str
    prompt_version: str
    latency_ms: int
    cache_hit: bool
    error: str | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

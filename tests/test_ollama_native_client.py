from __future__ import annotations

import json
import sys
import types
from io import BytesIO

from gender_gate.clients import OpenAICompatibleClient


class FakeResponse:
    def __init__(self, payload: dict):
        self._bytes = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._bytes


def test_ollama_native_sends_think_false_and_label_schema(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "message": {"role": "assistant", "content": '{"label":"NEGATIVE"}'},
                "done": True,
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        {
            "provider": "ollama_native",
            "base_url": "http://127.0.0.1:11434",
            "model": "deepseek-r1:8b",
            "allow_missing_key": True,
            "timeout_seconds": 30,
            "think": False,
            "structured_output": True,
        },
        {"temperature": 0.0, "max_output_tokens": 128, "retries": 1},
    )

    assert client.complete([{"role": "user", "content": "classify"}]) == (
        '{"label":"NEGATIVE"}'
    )
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["think"] is False
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_predict"] == 128
    assert captured["body"]["format"]["properties"]["label"]["enum"] == [
        "POSITIVE",
        "NEGATIVE",
    ]


def test_openai_mode_is_unchanged(monkeypatch) -> None:
    class DummyCompletions:
        def create(self, **request):
            message = types.SimpleNamespace(content="POSITIVE", model_extra={})
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            return types.SimpleNamespace(choices=[choice])

    class DummyOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=DummyCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=DummyOpenAI))
    client = OpenAICompatibleClient(
        {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "glm4:9b",
            "allow_missing_key": True,
        },
        {"max_output_tokens": 16, "retries": 1},
    )
    assert client.complete([{"role": "user", "content": "classify"}]) == "POSITIVE"


def test_ollama_native_plain_output_omits_schema(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "message": {"role": "assistant", "content": "NEGATIVE"},
                "done": True,
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        {
            "provider": "ollama_native",
            "base_url": "http://127.0.0.1:11434",
            "model": "deepseek-r1:8b",
            "allow_missing_key": True,
            "timeout_seconds": 30,
            "think": False,
            "structured_output": False,
        },
        {"temperature": 0.0, "max_output_tokens": 256, "retries": 1},
    )

    assert client.complete([{"role": "user", "content": "classify"}]) == "NEGATIVE"
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_predict"] == 256
    assert "format" not in captured["body"]

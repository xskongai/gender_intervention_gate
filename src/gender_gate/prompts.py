from __future__ import annotations

import json
from pathlib import Path


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_examples(path: str | Path | None) -> str:
    if path is None:
        return ""
    blocks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            blocks.append(
                "\n".join(
                    [
                        f"边界：{pair['boundary']}",
                        f"POSITIVE 示例：{pair['positive']['text']}",
                        f"NEGATIVE 示例：{pair['negative']['text']}",
                    ]
                )
            )
    return "\n\n".join(blocks)


def render_prompt(template: str, text: str, examples: str = "") -> str:
    return template.replace("{{EXAMPLES}}", examples).replace("{{TEXT}}", text)

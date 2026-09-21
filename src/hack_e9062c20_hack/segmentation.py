"""Сегментация обращений и подготовка черновиков ответов."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


CATEGORIES = ("справка", "жалоба", "другое")


def load_dotenv(path: Path) -> None:
    """Загрузить простые KEY=VALUE из .env, не перезаписывая окружение."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def read_messages(path: Path) -> list[str]:
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        message = line.strip()
        if not message:
            continue
        if ")" in message and message.split(")", 1)[0].isdigit():
            message = message.split(")", 1)[1].strip()
        messages.append(message)
    return messages


def classify_messages(client: OpenAI, messages: list[str], model: str) -> list[dict[str, str]]:
    prompt = "\n".join(f"{i}. {message}" for i, message in enumerate(messages, 1))
    response = client.responses.create(
        model=model,
        instructions=(
            "Ты оператор службы поддержки университета. Для каждого обращения "
            "выбери ровно одну категорию: справка, жалоба или другое. "
            "Составь один короткий вежливый черновик ответа на русском языке. "
            "Не выдумывай факты и сохрани порядок обращений."
        ),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "segmented_messages",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string", "enum": list(CATEGORIES)},
                                    "draft": {"type": "string"},
                                },
                                "required": ["category", "draft"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
            }
        },
    )
    data: Any = json.loads(response.output_text)
    items = data["items"]
    if len(items) != len(messages):
        raise RuntimeError("Модель вернула не то количество результатов, что и обращений")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Классификация обращений и черновики ответов")
    parser.add_argument("messages", nargs="?", type=Path, help="путь к файлу с обращениями")
    parser.add_argument("--model", default="gpt-4o-mini", help="модель OpenAI")
    args = parser.parse_args()

    # segmentation.py находится в <project>/src/hack_e9062c20_hack/.
    # Поэтому .env и messages.txt лежат на два уровня выше каталога пакета.
    project_root = Path(__file__).resolve().parents[2]
    messages_path = args.messages or project_root / "messages.txt"
    if not messages_path.is_absolute():
        messages_path = Path.cwd() / messages_path
    load_dotenv(project_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Не найден OPENAI_API_KEY: добавьте его в .env или окружение")
    messages = read_messages(messages_path)
    if not messages:
        raise SystemExit(f"В файле {messages_path} нет обращений")

    results = classify_messages(OpenAI(), messages, args.model)
    for index, (message, result) in enumerate(zip(messages, results), 1):
        print(f"{index}. {message}")
        print(f"   Категория: {result['category']}")
        print(f"   Черновик ответа: {result['draft']}\n")


if __name__ == "__main__":
    main()

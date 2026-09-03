"""Validación compartida de datasets conversacionales JSONL."""
from __future__ import annotations

import json
from pathlib import Path

ROLES = {"system", "user", "assistant"}
MAX_EXAMPLES = 50_000
MAX_MESSAGES_PER_EXAMPLE = 64
MAX_MESSAGE_CHARACTERS = 32_000


def load_jsonl(path: Path) -> tuple[list[dict], dict[str, int | bool]]:
    examples: list[dict] = []
    message_count = 0
    character_count = 0

    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if len(examples) >= MAX_EXAMPLES:
                raise ValueError(f"Línea {number}: el dataset supera {MAX_EXAMPLES} ejemplos")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Línea {number}: JSON inválido ({error.msg})") from error

            messages = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(messages, list) or not 2 <= len(messages) <= MAX_MESSAGES_PER_EXAMPLE:
                raise ValueError(f"Línea {number}: messages debe contener entre 2 y {MAX_MESSAGES_PER_EXAMPLE} elementos")

            validated_messages: list[dict[str, str]] = []
            for index, message in enumerate(messages, 1):
                if not isinstance(message, dict):
                    raise ValueError(f"Línea {number}, mensaje {index}: se esperaba un objeto")
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or role not in ROLES or not isinstance(content, str):
                    raise ValueError(f"Línea {number}, mensaje {index}: role o content no es válido")
                content = content.strip()
                if not content or len(content) > MAX_MESSAGE_CHARACTERS:
                    raise ValueError(f"Línea {number}, mensaje {index}: content está vacío o supera {MAX_MESSAGE_CHARACTERS} caracteres")
                validated_messages.append({"role": role, "content": content})
                character_count += len(content)

            if not any(message["role"] == "user" for message in validated_messages):
                raise ValueError(f"Línea {number}: falta al menos un mensaje user")
            if validated_messages[-1]["role"] != "assistant":
                raise ValueError(f"Línea {number}: el último mensaje debe ser assistant")

            examples.append({"messages": validated_messages})
            message_count += len(validated_messages)

    if not examples:
        raise ValueError("El dataset está vacío")

    return examples, {
        "valid": True,
        "examples": len(examples),
        "messages": message_count,
        "characters": character_count,
    }

"""Controles deterministas y sin dependencias para auditar la calidad del SFT."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        serialized = repr(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _last_user_prompt(messages: list[dict]) -> str | None:
    prompts = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not prompts or not isinstance(prompts[-1], str):
        return None
    normalized = unicodedata.normalize("NFKD", prompts[-1].casefold())
    without_accents = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    without_punctuation = re.sub(r"[^\w\s]", " ", without_accents)
    return _WHITESPACE.sub(" ", without_punctuation.strip())


def count_prompt_overlaps(first: list[dict], second: list[dict]) -> int:
    """Cuenta preguntas finales repetidas entre dos splits sin exponer su contenido."""
    first_prompts = {
        prompt
        for prompt in (_last_user_prompt(item.get("messages", [])) for item in first)
        if prompt
    }
    second_prompts = {
        prompt
        for prompt in (_last_user_prompt(item.get("messages", [])) for item in second)
        if prompt
    }
    return len(first_prompts & second_prompts)


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 2)


def _distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "sum": sum(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def summarize_token_lengths(
    total_tokens: list[int],
    assistant_tokens: list[int],
    max_length: int,
    truncated_examples: list[bool],
    target_truncated: list[bool],
    empty_assistant_after_limit: list[bool],
) -> dict[str, Any]:
    if not total_tokens or len(total_tokens) != len(assistant_tokens):
        raise ValueError("No se pueden resumir longitudes de un dataset vacío o inconsistente.")
    return {
        "examples": len(total_tokens),
        "maxLength": max_length,
        "totalTokens": _distribution(total_tokens),
        "assistantTokens": _distribution(assistant_tokens),
        "truncatedExamples": sum(truncated_examples),
        "targetTruncatedExamples": sum(target_truncated),
        "emptyAssistantAfterLimit": sum(empty_assistant_after_limit),
    }

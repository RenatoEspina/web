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

# Qwen3.5 es híbrido: de sus 24 capas solo 6 son atención completa y 18 son
# atención lineal de estado recurrente. En esas capas `in_proj_a` e `in_proj_b`
# no son proyecciones corrientes: producen el decaimiento y la puerta que
# gobiernan el estado recurrente. Entrenar LoRA sobre ellas desestabiliza esa
# recurrencia y la generación degenera en bucles de repetición, aunque el
# eval_loss con teacher forcing mejore.
#
# `attention-mlp` es el conjunto clásico de LoRA: cubre el MLP de las 24 capas y
# la atención de las 6 completas, sin tocar ninguna puerta de estado.
#
# Vive aquí, sin dependencias de ML, para que train.py y train_axolotl.py
# compartan exactamente la misma lista.
TARGET_MODULE_PRESETS = {
    "attention-mlp": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "all-linear": "all-linear",
}


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


def _as_flat_list(value: object) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value if isinstance(value, list) else []


def audit_token_lengths(
    tokenizer,
    examples: list[dict],
    max_length: int,
    label: str,
    allow_truncation: bool,
) -> dict:
    """Mide el texto con la misma plantilla que usará SFTTrainer."""
    total_tokens: list[int] = []
    assistant_tokens: list[int] = []
    truncated_examples: list[bool] = []
    target_truncated: list[bool] = []
    empty_assistant_after_limit: list[bool] = []

    for number, example in enumerate(examples, 1):
        encoded = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = _as_flat_list(encoded.get("input_ids"))
        assistant_mask = _as_flat_list(encoded.get("assistant_masks"))
        if not input_ids or len(input_ids) != len(assistant_mask):
            raise RuntimeError(
                f"La auditoría de tokens no pudo validar el ejemplo {number} de {label}."
            )
        if not any(bool(value) for value in assistant_mask):
            raise RuntimeError(
                f"El ejemplo {number} de {label} no contiene tokens assistant entrenables."
            )

        total = len(input_ids)
        assistant = sum(int(bool(value)) for value in assistant_mask)
        truncated = total > max_length
        target_would_be_truncated = any(
            bool(value) for value in assistant_mask[max_length:]
        )
        empty_after_limit = not any(
            bool(value) for value in assistant_mask[:max_length]
        )

        total_tokens.append(total)
        assistant_tokens.append(assistant)
        truncated_examples.append(truncated)
        target_truncated.append(target_would_be_truncated)
        empty_assistant_after_limit.append(empty_after_limit)

    summary = summarize_token_lengths(
        total_tokens,
        assistant_tokens,
        max_length,
        truncated_examples,
        target_truncated,
        empty_assistant_after_limit,
    )
    print(json.dumps({"dataset": label, **summary}, ensure_ascii=False))

    if summary["emptyAssistantAfterLimit"]:
        raise RuntimeError(
            f"{summary['emptyAssistantAfterLimit']} ejemplos de {label} quedarían sin "
            "tokens assistant dentro de --max-length."
        )
    if summary["targetTruncatedExamples"] and not allow_truncation:
        raise RuntimeError(
            f"{summary['targetTruncatedExamples']} respuestas de {label} quedarían "
            f"truncadas por --max-length={max_length}. Aumenta --max-length o usa "
            "--allow-truncation de forma explícita."
        )
    if summary["targetTruncatedExamples"]:
        print(
            f"Aviso: {summary['targetTruncatedExamples']} respuestas de {label} "
            "serán truncadas porque se solicitó --allow-truncation."
        )
    return summary

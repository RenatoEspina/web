#!/usr/bin/env python3
"""Evalúa conversiones por igualdad exacta canónica, sin dependencias externas.

Modo offline: --dataset evaluation.jsonl --predictions respuestas.jsonl --output reporte.json
Predicciones: una línea {"id": "...", "output": "respuesta textual del modelo"} por caso.
Comparación pareada: --compare-predictions respuestas_finetuned.jsonl; el primer archivo
es la base y el segundo el candidato. Alternativamente use --model base --compare-model
adaptador con un servidor OpenAI-compatible. Nunca se envían las referencias al modelo.
Las tasas usan escala 0..1; los fallos del servidor se conservan como respuestas fallidas.
Exactitud exige el valor decimal canónico esperado. La métrica de cálculo acepta
decimales equivalentes (por ejemplo, 10.0 y 1e1 equivalen numéricamente a 10).
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

STATES = ("ok", "incompatible", "falta_dato", "no_soportada")
ANSWER_KEYS = {"estado", "valor", "unidad"}
DECIMAL_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
CANONICAL_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]{0,5}[1-9])?\Z")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Clave JSON duplicada: {key}")
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise ValueError(f"Constante no permitida en JSON: {value}")


def strict_json(text: str) -> object:
    """Rechaza claves duplicadas, NaN/Infinity y contenido fuera del documento JSON."""
    return json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_constant)


def decimal_value(value: object) -> Decimal | None:
    """Solo cadenas decimales finitas; bool y números JSON no cumplen el contrato."""
    if not isinstance(value, str) or not DECIMAL_TEXT.fullmatch(value):
        return None
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def valid_answer(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != ANSWER_KEYS:
        return False
    state = value["estado"]
    if not isinstance(state, str) or state not in STATES:
        return False
    if state == "ok":
        return (
            decimal_value(value["valor"]) is not None
            and isinstance(value["unidad"], str)
            and bool(value["unidad"])
        )
    return value["valor"] is None and value["unidad"] is None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = strict_json(line)
            except (ValueError, RecursionError) as error:
                raise ValueError(f"{path}, línea {line_number}: JSON inválido ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}, línea {line_number}: se esperaba un objeto JSON")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: archivo vacío")
    return rows


def load_cases(path: Path) -> list[dict]:
    """Valida IDs, metadatos y mensajes sin respuestas antes de inferir."""
    cases = _read_jsonl(path)
    seen = set()
    for index, case in enumerate(cases, 1):
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"Caso {index}: id vacío, inválido o duplicado")
        seen.add(identifier)
        for field in ("category", "group_id", "template_id"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f"Caso {identifier}: falta {field}")
        if not valid_answer(case.get("expected")):
            raise ValueError(f"Caso {identifier}: expected no cumple el esquema")
        expected = case["expected"]
        if expected["estado"] == "ok" and (
            not CANONICAL_DECIMAL.fullmatch(expected["valor"])
            or expected["valor"] == "-0"
        ):
            raise ValueError(f"Caso {identifier}: valor de referencia no canónico")
        messages = case.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"Caso {identifier}: messages debe contener solo system y user")
        for message, role in zip(messages, ("system", "user")):
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError(f"Caso {identifier}: mensaje {role} inválido")
    return cases


def load_predictions(path: Path, cases: list[dict]) -> dict[str, str]:
    predictions = {}
    for row in _read_jsonl(path):
        if set(row) != {"id", "output"}:
            raise ValueError("Cada predicción debe tener exactamente id y output")
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in predictions:
            raise ValueError("Predicción con id vacío, inválido o duplicado")
        if not isinstance(row["output"], str):
            raise ValueError(f"Predicción {identifier}: output debe ser texto")
        predictions[identifier] = row["output"]
    expected_ids = {case["id"] for case in cases}
    missing = expected_ids - predictions.keys()
    extra = predictions.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"IDs desalineados: faltan {sorted(missing)}, sobran {sorted(extra)}")
    return predictions


def assess_output(output: str, expected: dict) -> dict:
    try:
        answer = strict_json(output)
        json_valid = True
    except (ValueError, TypeError, RecursionError):
        answer = None
        json_valid = False
    schema_valid = valid_answer(answer)
    predicted_state = answer["estado"] if schema_valid else None
    numeric_correct = (
        schema_valid
        and expected["estado"] == "ok"
        and predicted_state == "ok"
        and decimal_value(answer["valor"]) == decimal_value(expected["valor"])
        and answer["unidad"] == expected["unidad"]
    )
    exact_correct = schema_valid and predicted_state == expected["estado"] and (
        numeric_correct and answer["valor"] == expected["valor"]
        if expected["estado"] == "ok" else True
    )
    # Detecta un valor numérico incluso si el modelo rompe el esquema usando un
    # número JSON. Solo mide campos valor de JSON legible; no interpreta prosa.
    returned_numeric = False
    if isinstance(answer, dict):
        value = answer.get("valor")
        returned_numeric = decimal_value(value) is not None or (
            type(value) is int or (type(value) is float and math.isfinite(value))
        )
    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "predicted_status": predicted_state,
        "exact_correct": bool(exact_correct),
        "numeric_unit_correct": bool(numeric_correct),
        "canonical_value_format": bool(
            schema_valid and predicted_state == "ok"
            and CANONICAL_DECIMAL.fullmatch(answer["valor"])
            and answer["valor"] != "-0"
        ),
        "invalid_request_numeric_response": expected["estado"] != "ok" and returned_numeric,
    }


def wilson95(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _rate(successes: int, total: int) -> float | None:
    return successes / total if total else None


def evaluate_predictions(
    cases: list[dict], predictions: dict[str, str], label: str = "predictions",
    failures: dict[str, str] | None = None, latencies: dict[str, float] | None = None,
) -> dict:
    if set(predictions) != {case["id"] for case in cases}:
        raise ValueError("Las predicciones no coinciden exactamente con los IDs del dataset")
    if not cases:
        raise ValueError("No hay casos para evaluar")
    results = []
    failures = failures or {}
    latencies = latencies or {}
    for case in cases:
        identifier = case["id"]
        result = {
            "id": identifier, "category": case["category"],
            "group_id": case["group_id"], "template_id": case["template_id"],
            "expected": case["expected"], "output": predictions[identifier],
            **assess_output(predictions[identifier], case["expected"]),
        }
        if identifier in failures:
            result["request_error"] = failures[identifier]
        if identifier in latencies:
            result["latency_seconds"] = latencies[identifier]
        results.append(result)
    total = len(results)
    correct = sum(item["exact_correct"] for item in results)
    valid_requests = sum(item["expected"]["estado"] == "ok" for item in results)
    invalid_requests = total - valid_requests
    by_status = {}
    # Macro F1 usa las clases presentes en la referencia o predicción; no agrega
    # ceros artificiales por estados ausentes en un subconjunto de evaluación.
    active_states = set(item["expected"]["estado"] for item in results)
    active_states.update(item["predicted_status"] for item in results if item["predicted_status"])
    for state in STATES:
        tp = sum(item["expected"]["estado"] == state and item["predicted_status"] == state for item in results)
        fp = sum(item["expected"]["estado"] != state and item["predicted_status"] == state for item in results)
        fn = sum(item["expected"]["estado"] == state and item["predicted_status"] != state for item in results)
        denominator = 2 * tp + fp + fn
        by_status[state] = {
            "support": tp + fn, "precision": _rate(tp, tp + fp),
            "recall": _rate(tp, tp + fn), "f1": 2 * tp / denominator if denominator else None,
        }
    by_category = {}
    for category in sorted({item["category"] for item in results}):
        members = [item for item in results if item["category"] == category]
        passed = sum(item["exact_correct"] for item in members)
        by_category[category] = {
            "total": len(members), "correct": passed,
            "exact_accuracy": passed / len(members), "wilson95": wilson95(passed, len(members)),
        }
    return {
        "label": label, "total": total, "correct": correct,
        "exact_accuracy": correct / total, "exact_accuracy_wilson95": wilson95(correct, total),
        "expected_ok_total": valid_requests,
        "numeric_unit_accuracy": _rate(sum(item["numeric_unit_correct"] for item in results), valid_requests),
        "json_validity": sum(item["json_valid"] for item in results) / total,
        "schema_validity": sum(item["schema_valid"] for item in results) / total,
        "status_macro_f1": sum(by_status[state]["f1"] or 0 for state in active_states) / len(active_states),
        "status_macro_f1_classes": sorted(active_states), "per_status": by_status,
        "invalid_request_total": invalid_requests,
        "invalid_request_numeric_response_rate": _rate(sum(item["invalid_request_numeric_response"] for item in results), invalid_requests),
        "invalid_request_correct_rejection_rate": _rate(sum(item["exact_correct"] for item in results if item["expected"]["estado"] != "ok"), invalid_requests),
        "invalid_request_error_rate": _rate(sum(not item["exact_correct"] for item in results if item["expected"]["estado"] != "ok"), invalid_requests),
        "request_failures": len(failures), "per_category": by_category, "results": results,
    }


def compare_reports(base: dict, candidate: dict, seed: int = 20260909, resamples: int = 10000) -> dict:
    """Delta candidato-base: bootstrap pareado por grupo semántico, reproducible.

    Muestrea group_id con reemplazo y conserva todas sus filas juntas. Cada
    réplica calcula la tasa por fila, incluyendo el tamaño de cada grupo.
    """
    if resamples < 1:
        raise ValueError("resamples debe ser positivo")
    base_by_id = {item["id"]: item for item in base["results"]}
    candidate_by_id = {item["id"]: item for item in candidate["results"]}
    if not base_by_id or set(base_by_id) != set(candidate_by_id):
        raise ValueError("La comparación requiere los mismos IDs no vacíos")
    differences = []
    grouped = {}
    pairs = Counter()
    for identifier, first in base_by_id.items():
        second = candidate_by_id[identifier]
        if first["expected"] != second["expected"]:
            raise ValueError("La comparación requiere referencias idénticas")
        if first["group_id"] != second["group_id"]:
            raise ValueError("La comparación requiere group_id idénticos")
        a, b = int(first["exact_correct"]), int(second["exact_correct"])
        differences.append(b - a)
        group = grouped.setdefault(first["group_id"], [0, 0])
        group[0] += b - a
        group[1] += 1
        pairs[(a, b)] += 1
    rng = random.Random(seed)
    n = len(differences)
    groups = [grouped[key] for key in sorted(grouped)]
    deltas = []
    for _ in range(resamples):
        sampled = rng.choices(groups, k=len(groups))
        deltas.append(sum(item[0] for item in sampled) / sum(item[1] for item in sampled))
    deltas.sort()
    def percentile(q: float) -> float:
        location = q * (resamples - 1)
        lower = math.floor(location)
        upper = math.ceil(location)
        return deltas[lower] + (deltas[upper] - deltas[lower]) * (location - lower)
    return {
        "base": base["label"], "candidate": candidate["label"],
        "exact_accuracy_delta": sum(differences) / n,
        "exact_accuracy_delta_bootstrap95": [percentile(.025), percentile(.975)],
        "bootstrap_seed": seed, "bootstrap_resamples": resamples,
        "bootstrap_unit": "group_id", "bootstrap_groups": len(groups),
        "both_correct": pairs[(1, 1)],
        "both_wrong": pairs[(0, 0)], "improved": pairs[(0, 1)], "regressed": pairs[(1, 0)],
    }


def complete(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int, timeout: float) -> str:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    data = json.dumps({
        "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
    }, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint + "/chat/completions", data=data, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    try:
        output = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Respuesta del servidor sin choices[0].message.content") from error
    if not isinstance(output, str):
        raise ValueError("message.content no es texto")
    return output


def evaluate_model(cases: list[dict], model: str, base_url: str, api_key: str, max_tokens: int = 128, timeout: float = 120) -> dict:
    predictions, failures, latencies = {}, {}, {}
    for case in cases:
        identifier = case["id"]
        start = time.perf_counter()
        try:
            predictions[identifier] = complete(base_url, api_key, model, case["messages"], max_tokens, timeout)
        except (URLError, OSError, ValueError, TypeError) as error:
            predictions[identifier] = ""
            failures[identifier] = f"{type(error).__name__}: {error}"
        latencies[identifier] = round(time.perf_counter() - start, 6)
    return evaluate_predictions(cases, predictions, model, failures, latencies)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL con referencias de evaluación.")
    parser.add_argument("--output", type=Path, required=True, help="Informe JSON con métricas y respuestas crudas.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="JSONL de predicciones del modelo base.")
    source.add_argument("--model", help="Modelo base servido en la API OpenAI-compatible.")
    parser.add_argument("--compare-predictions", type=Path, help="JSONL del candidato para comparación pareada.")
    parser.add_argument("--compare-model", help="Modelo candidato; recibe los mismos mensajes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Nombre de variable de entorno para la clave opcional.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.compare_predictions and not args.predictions:
        parser.error("--compare-predictions requiere --predictions")
    if args.compare_model and not args.model:
        parser.error("--compare-model requiere --model")
    if args.compare_model and args.compare_model == args.model:
        parser.error("Los modelos base y candidato deben ser distintos")
    if args.max_tokens <= 0 or args.timeout <= 0 or args.bootstrap_resamples <= 0:
        parser.error("max-tokens, timeout y bootstrap-resamples deben ser positivos")
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    cases = load_cases(args.dataset)
    if args.predictions:
        # Validar ambos archivos antes de evaluar o escribir resultados.
        base_predictions = load_predictions(args.predictions, cases)
        candidate_predictions = load_predictions(args.compare_predictions, cases) if args.compare_predictions else None
        base = evaluate_predictions(cases, base_predictions, str(args.predictions))
        candidate = evaluate_predictions(cases, candidate_predictions, str(args.compare_predictions)) if candidate_predictions is not None else None
        mode = "offline"
    else:
        api_key = os.environ.get(args.api_key_env, "")
        base = evaluate_model(cases, args.model, args.base_url, api_key, args.max_tokens, args.timeout)
        candidate = evaluate_model(cases, args.compare_model, args.base_url, api_key, args.max_tokens, args.timeout) if args.compare_model else None
        mode = "remote"
    report = {
        "format_version": 1, "mode": mode, "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "rate_scale": "0..1; null si el denominador es cero", "base": base,
        "definitions": {
            "exact_accuracy": "Esquema válido, estado correcto y, si ok, cadena decimal canónica y unidad idénticas a la referencia.",
            "numeric_unit_accuracy": "Aciertos de valor decimal semántico/unidad sobre TODOS los casos cuyo expected.estado es ok; acepta formatos decimales equivalentes.",
            "invalid_request_numeric_response_rate": "Casos esperados no-ok con valor numérico finito en un objeto JSON legible, incluso si rompe el esquema; no analiza números en prosa.",
            "invalid_request_error_rate": "Uno menos exactitud en casos expected.estado distinto de ok; incluye prosa, JSON inválido, valores inventados, estado incorrecto y fallos de servidor.",
            "status_macro_f1": "Media F1 de estados presentes en referencia o predicción; esquema inválido cuenta como estado ausente.",
            "confidence_intervals": "Wilson nominal por caso supone independencia y puede ser optimista con paráfrasis; la comparación usa bootstrap pareado por group_id para conservar la dependencia intragrupo.",
        },
    }
    if mode == "remote":
        report["generation"] = {"base_url": args.base_url, "temperature": 0, "max_tokens": args.max_tokens, "timeout_seconds": args.timeout}
    if candidate is not None:
        report["candidate"] = candidate
        report["comparison"] = compare_reports(base, candidate, args.seed, args.bootstrap_resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"output": str(args.output), "total": base["total"], "base_exact_accuracy": base["exact_accuracy"]}
    if candidate is not None:
        summary.update(candidate_exact_accuracy=candidate["exact_accuracy"], comparison=report["comparison"])
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

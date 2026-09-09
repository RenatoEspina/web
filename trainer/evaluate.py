#!/usr/bin/env python3
"""Evalúa uno o dos modelos OpenAI-compatible con el mismo conjunto de casos."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def load_cases(path: Path) -> list[tuple[int, dict]]:
    """Valida todo el archivo antes de hacer cualquier request de inferencia."""
    cases = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Línea {number}: JSON inválido") from error
            if not isinstance(case, dict):
                raise ValueError(f"Línea {number}: se esperaba un caso de evaluación")

            expected = case.get("contains")
            if not isinstance(expected, list) or not expected or any(
                not isinstance(fragment, str) or not fragment.strip()
                for fragment in expected
            ):
                raise ValueError(
                    f"Línea {number}: contains debe incluir fragmentos no vacíos"
                )

            messages = case.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Línea {number}: messages debe ser una conversación no vacía"
                )

            next_role = "user"
            for index, message in enumerate(messages):
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                ):
                    raise ValueError(
                        f"Línea {number}, mensaje {index + 1}: contenido inválido"
                    )
                role = message.get("role")
                if index == 0 and role == "system":
                    continue
                if role != next_role:
                    raise ValueError(
                        f"Línea {number}, mensaje {index + 1}: "
                        f"se esperaba el rol {next_role}"
                    )
                next_role = "assistant" if role == "user" else "user"

            if messages[-1].get("role") != "user":
                raise ValueError(
                    f"Línea {number}: la evaluación debe terminar en user, "
                    "sin la respuesta de referencia"
                )
            cases.append((number, case))

    if not cases:
        raise ValueError("El dataset de evaluación está vacío")
    return cases


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--compare-model",
        help="Segundo modelo servido que recibirá exactamente los mismos casos.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def complete(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        payload = json.load(response)
    try:
        answer = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("La respuesta OpenAI-compatible no contiene message.content") from error
    if not isinstance(answer, str):
        raise ValueError("message.content no es texto")
    return answer


def evaluate_model(
    base_url: str,
    api_key: str,
    model: str,
    cases: list[tuple[int, dict]],
    max_tokens: int,
) -> dict:
    results = []
    passed = 0
    for number, case in cases:
        expected = case["contains"]
        started = time.perf_counter()
        answer = complete(
            base_url,
            api_key,
            model,
            case["messages"],
            max_tokens,
        )
        normalized_answer = answer.casefold()
        ok = all(
            fragment.strip().casefold() in normalized_answer
            for fragment in expected
        )
        passed += int(ok)
        results.append(
            {
                "line": number,
                "passed": ok,
                "latencySeconds": round(time.perf_counter() - started, 3),
                "responseCharacters": len(answer),
                "contains": expected,
                "answer": answer,
            }
        )

    average_latency = (
        sum(item["latencySeconds"] for item in results) / len(results)
        if results
        else 0
    )
    return {
        "model": model,
        "passed": passed,
        "total": len(results),
        "passRate": passed / len(results) if results else 0,
        "averageLatencySeconds": round(average_latency, 3),
        "results": results,
    }


def main() -> None:
    args = arguments()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens debe ser mayor que cero")

    # La validación completa ocurre antes de la primera llamada a cualquier modelo.
    cases = load_cases(args.dataset)
    comparison_model = getattr(args, "compare_model", None)
    if comparison_model and comparison_model == args.model:
        raise ValueError("--compare-model debe ser distinto de --model")

    reports = {
        args.model: evaluate_model(
            args.base_url,
            args.api_key,
            args.model,
            cases,
            args.max_tokens,
        )
    }
    if comparison_model:
        reports[comparison_model] = evaluate_model(
            args.base_url,
            args.api_key,
            comparison_model,
            cases,
            args.max_tokens,
        )

    primary = reports[args.model]
    report = dict(primary)
    if comparison_model:
        comparison = reports[comparison_model]
        report["models"] = reports
        report["comparison"] = {
            "primaryModel": args.model,
            "comparisonModel": comparison_model,
            "passedDelta": comparison["passed"] - primary["passed"],
            "passRateDelta": round(
                comparison["passRate"] - primary["passRate"],
                4,
            ),
            "averageLatencyDeltaSeconds": round(
                comparison["averageLatencySeconds"]
                - primary["averageLatencySeconds"],
                3,
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: report[key]
        for key in ("model", "passed", "total", "passRate")
    }
    if comparison_model:
        summary["comparison"] = report["comparison"]
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

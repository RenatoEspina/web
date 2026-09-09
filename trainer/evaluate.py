#!/usr/bin/env python3
"""Evalúa un modelo OpenAI-compatible sin reutilizar ejemplos de entrenamiento."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from urllib.request import Request, urlopen


def load_cases(path: Path) -> list[tuple[int, dict]]:
    """Validate the whole evaluation set before making any inference requests."""
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
                not isinstance(fragment, str) or not fragment.strip() for fragment in expected
            ):
                raise ValueError(f"Línea {number}: contains debe incluir fragmentos de texto no vacíos")
            messages = case.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Línea {number}: messages debe ser una conversación no vacía")
            next_role = "user"
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError(f"Línea {number}, mensaje {index + 1}: contenido inválido")
                role = message.get("role")
                if index == 0 and role == "system":
                    continue
                if role != next_role:
                    raise ValueError(f"Línea {number}, mensaje {index + 1}: se esperaba el rol {next_role}")
                next_role = "assistant" if role == "user" else "user"
            if messages[-1].get("role") != "user":
                raise ValueError(f"Línea {number}: la evaluación debe terminar en user, sin la respuesta de referencia")
            cases.append((number, case))
    if not cases:
        raise ValueError("El dataset de evaluación está vacío")
    return cases


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()

def complete(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int) -> str:
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=payload, headers=headers, method="POST")
    with urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]["content"]

def main() -> None:
    args = arguments(); results = []; passed = 0
    for number, case in load_cases(args.dataset):
        expected = case["contains"]
        started = time.perf_counter()
        answer = complete(args.base_url, args.api_key, args.model, case["messages"], args.max_tokens)
        ok = all(fragment.casefold() in answer.casefold() for fragment in expected)
        passed += int(ok)
        results.append({"line": number, "passed": ok, "latencySeconds": round(time.perf_counter() - started, 3), "contains": expected, "answer": answer})
    report = {"model": args.model, "passed": passed, "total": len(results), "passRate": passed / len(results) if results else 0, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "passed", "total", "passRate")}))

if __name__ == "__main__": main()

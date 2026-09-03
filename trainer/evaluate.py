#!/usr/bin/env python3
"""Evalúa un modelo OpenAI-compatible sin reutilizar ejemplos de entrenamiento."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from urllib.request import Request, urlopen

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
    with args.dataset.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip(): continue
            case = json.loads(line)
            expected = case.get("contains", [])
            if not isinstance(case.get("messages"), list) or not isinstance(expected, list): raise ValueError(f"Línea {number}: caso inválido")
            started = time.perf_counter()
            answer = complete(args.base_url, args.api_key, args.model, case["messages"], args.max_tokens)
            ok = all(str(fragment).casefold() in answer.casefold() for fragment in expected)
            passed += int(ok)
            results.append({"line": number, "passed": ok, "latencySeconds": round(time.perf_counter() - started, 3), "contains": expected, "answer": answer})
    report = {"model": args.model, "passed": passed, "total": len(results), "passRate": passed / len(results) if results else 0, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "passed", "total", "passRate")}))

if __name__ == "__main__": main()

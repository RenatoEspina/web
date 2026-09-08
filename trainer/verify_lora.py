#!/usr/bin/env python3
"""A/B diagnostic for LoRA registration, effect and early-stop behaviour."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = (
    "¿Qué ocurre si uso get con una clave inexistente en un HashMap? Responde brevemente en español.",
    "¿Cuándo conviene usar RAG en vez de fine-tuning? Responde en una frase.",
    "Explica brevemente cómo empezar una partida de Terraria.",
)
GENERATION_MAX_TOKENS = 256


def chat_request(base_url: str, model: str, prompt: str, *, max_tokens: int,
                 logprobs: bool = False) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload.update({"logprobs": True, "top_logprobs": 5})
    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode(),
    )
    started = time.perf_counter()
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def distribution_probe(base_url: str, model: str, prompt: str) -> dict:
    result, _ = chat_request(base_url, model, prompt, max_tokens=1, logprobs=True)
    choice = result["choices"][0]
    entries = (choice.get("logprobs") or {}).get("content") or []
    if not entries or not entries[0].get("top_logprobs"):
        raise RuntimeError("El runtime no devolvió logprobs; no se puede comprobar el efecto del LoRA.")
    return {
        "text": choice["message"].get("content"),
        "top": {item["token"]: item["logprob"] for item in entries[0]["top_logprobs"]},
    }


def generation_probe(base_url: str, model: str, prompt: str) -> dict:
    result, latency = chat_request(base_url, model, prompt, max_tokens=GENERATION_MAX_TOKENS)
    choice = result["choices"][0]
    usage = result.get("usage") or {}
    text = choice.get("message", {}).get("content") or ""
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    return {
        "text": text,
        "characters": len(text),
        "latencySeconds": round(latency, 3),
        "finishReason": choice.get("finish_reason"),
        "promptTokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "completionTokens": completion_tokens if isinstance(completion_tokens, int) else None,
        "tokensPerSecond": round(completion_tokens / latency, 2)
        if isinstance(completion_tokens, int) and latency > 0 else None,
    }


def difference(first: dict, second: dict) -> float:
    common = first["top"].keys() & second["top"].keys()
    values = [abs(first["top"][key] - second["top"][key]) for key in common]
    return max(values + [1.0 if first["top"].keys() != second["top"].keys() else 0.0])


def markedly_shorter(base: dict, adapter: dict) -> bool:
    base_tokens = base.get("completionTokens")
    adapter_tokens = adapter.get("completionTokens")
    if isinstance(base_tokens, int) and isinstance(adapter_tokens, int) and base_tokens >= 16:
        return adapter_tokens < base_tokens * 0.5
    base_chars = base.get("characters", 0)
    adapter_chars = adapter.get("characters", 0)
    return base_chars >= 80 and adapter_chars < base_chars * 0.5


def verify(base_url: str, base_model: str, adapter: str) -> dict:
    if base_model == adapter:
        raise ValueError("El modelo base y el adaptador deben ser distintos.")

    distribution_cases = []
    generation_cases = []
    for prompt in PROMPTS:
        baseline = distribution_probe(base_url, base_model, prompt)
        repeated = distribution_probe(base_url, base_model, prompt)
        adapted = distribution_probe(base_url, adapter, prompt)
        noise = difference(baseline, repeated)
        delta = difference(baseline, adapted)
        distribution_cases.append({
            "prompt": prompt,
            "baseRepeatDelta": noise,
            "adapterDelta": delta,
            "effectDetected": delta > max(1e-4, 10 * noise),
            "base": baseline,
            "adapter": adapted,
        })

        base_generation = generation_probe(base_url, base_model, prompt)
        adapter_generation = generation_probe(base_url, adapter, prompt)
        generation_cases.append({
            "prompt": prompt,
            "base": base_generation,
            "adapter": adapter_generation,
            "adapterMarkedlyShorter": markedly_shorter(base_generation, adapter_generation),
        })

    detected = sum(case["effectDetected"] for case in distribution_cases) >= 2
    shorter_cases = sum(case["adapterMarkedlyShorter"] for case in generation_cases)
    early_stop_suspected = shorter_cases >= 2
    return {
        "baseModel": base_model,
        "adapter": adapter,
        "status": "effect_detected" if detected else "inconclusive",
        "note": "Un cambio de probabilidades demuestra efecto del LoRA, no mejor calidad ni cobertura completa de tensores.",
        "distributionCases": distribution_cases,
        "generationCases": generation_cases,
        "diagnostics": {
            "markedlyShorterCases": shorter_cases,
            "earlyStopSuspected": early_stop_suspected,
            "interpretation": (
                "El adaptador termina mucho antes que el modelo base en varias pruebas; revisa EOS, longitud y dataset."
                if early_stop_suspected else
                "No se detectó un patrón consistente de terminación prematura en estas pruebas."
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.base_url, args.base_model, args.adapter)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    if report["status"] == "inconclusive":
        raise SystemExit("INCONCLUSO: no se detectó un efecto consistente; no asumas que el LoRA se está aplicando.")

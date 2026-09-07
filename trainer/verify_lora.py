#!/usr/bin/env python3
"""A/B diagnostic: registration in /v1/models alone does not prove LoRA use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = (
    "¿Qué ocurre si uso get con una clave inexistente en un HashMap? Responde brevemente en español.",
    "¿Cuándo conviene usar RAG en vez de fine-tuning? Responde en una frase.",
    "Explica brevemente cómo empezar una partida de Terraria.",
)


def probe(base_url: str, model: str, prompt: str) -> dict:
    request = Request(base_url.rstrip("/") + "/v1/chat/completions", method="POST",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                                       "temperature": 0, "seed": 42, "max_tokens": 1,
                                       "logprobs": True, "top_logprobs": 5}).encode())
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    choice = result["choices"][0]
    entries = (choice.get("logprobs") or {}).get("content") or []
    if not entries or not entries[0].get("top_logprobs"):
        raise RuntimeError("El runtime no devolvió logprobs; no se puede comprobar el efecto del LoRA.")
    return {"text": choice["message"].get("content"),
            "top": {item["token"]: item["logprob"] for item in entries[0]["top_logprobs"]}}


def difference(first: dict, second: dict) -> float:
    common = first["top"].keys() & second["top"].keys()
    # A changed candidate set is also a distribution change, even if no tokens overlap.
    values = [abs(first["top"][key] - second["top"][key]) for key in common]
    return max(values + [1.0 if first["top"].keys() != second["top"].keys() else 0.0])


def verify(base_url: str, base_model: str, adapter: str) -> dict:
    if base_model == adapter:
        raise ValueError("El modelo base y el adaptador deben ser distintos.")
    cases = []
    for prompt in PROMPTS:
        baseline = probe(base_url, base_model, prompt)
        repeated = probe(base_url, base_model, prompt)
        adapted = probe(base_url, adapter, prompt)
        noise = difference(baseline, repeated)
        delta = difference(baseline, adapted)
        cases.append({"prompt": prompt, "baseRepeatDelta": noise, "adapterDelta": delta,
                      "effectDetected": delta > max(1e-4, 10 * noise),
                      "base": baseline, "adapter": adapted})
    detected = sum(case["effectDetected"] for case in cases) >= 2
    return {"baseModel": base_model, "adapter": adapter,
            "status": "effect_detected" if detected else "inconclusive",
            "note": "Un cambio de probabilidades no demuestra mejor calidad ni que se apliquen todos los tensores.",
            "cases": cases}


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

#!/usr/bin/env python3
"""Comprueba que el entorno de fine-tuning coincide con requirements.txt."""
from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).with_name("requirements.txt")
PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)$")


def expected_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PINNED_REQUIREMENT.fullmatch(line)
        if not match:
            raise RuntimeError(f"Requirement no soportado por el verificador: {line}")
        result[match.group(1)] = match.group(2)
    return result


def main() -> None:
    mismatches: list[str] = []
    for package, expected in expected_versions().items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            mismatches.append(f"{package}: no instalado (esperado {expected})")
            continue
        if installed != expected:
            mismatches.append(f"{package}: {installed} (esperado {expected})")

    if mismatches:
        print("Dependencias del trainer desactualizadas:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"- {mismatch}", file=sys.stderr)
        raise SystemExit(1)

    # Smoke test del Dataset para Python 3.14.
    from datasets import Dataset

    smoke = Dataset.from_list([{"text": "compatibility-check"}])
    if len(smoke) != 1:
        raise RuntimeError("datasets no pudo crear el Dataset mínimo de comprobación")

    # Qwen3.5 necesita Transformers >=5.2.0. Comprobar el registro local evita
    # iniciar una descarga/carga de modelo con una versión que no lo reconoce.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    if "qwen3_5" not in CONFIG_MAPPING:
        raise RuntimeError(
            f"Transformers {version('transformers')} no registra la arquitectura qwen3_5"
        )
    if MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get("qwen3_5") != "Qwen3_5ForCausalLM":
        raise RuntimeError("AutoModelForCausalLM no tiene soporte esperado para qwen3_5")

    print(f"Python: {sys.version.split()[0]}")
    print(f"datasets: {version('datasets')}")
    print(f"transformers: {version('transformers')} (Qwen3.5 OK)")
    print(f"trl: {version('trl')}")
    print(f"peft: {version('peft')}")
    print("Dependencias del trainer: OK")


if __name__ == "__main__":
    main()

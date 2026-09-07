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

    # Smoke test deliberado: reproduce la ruta de fingerprint que fallaba con
    # datasets 4.0.0 + Python 3.14 antes de cargar el modelo en GPU.
    from datasets import Dataset

    smoke = Dataset.from_list([{"text": "compatibility-check"}])
    if len(smoke) != 1:
        raise RuntimeError("datasets no pudo crear el Dataset mínimo de comprobación")

    print(f"Python: {sys.version.split()[0]}")
    print(f"datasets: {version('datasets')}")
    print("Dependencias del trainer: OK")


if __name__ == "__main__":
    main()

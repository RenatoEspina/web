#!/usr/bin/env python3
"""Validate the isolated Axolotl QLoRA environment used by the local GUI."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import sys

from check_environment import main as check_qlora_environment

REQUIRED_AXOLOTL_VERSION = "0.18.0"
REQUIRED_PYTHON = (3, 12)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "no instalado"


def main() -> None:
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise RuntimeError(
            "El entorno Axolotl debe usar Python "
            f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}; "
            f"se detectó {sys.version_info.major}.{sys.version_info.minor}. "
            "Pulsa 'Preparar / actualizar entorno' para reconstruirlo con uv."
        )

    check_qlora_environment()
    installed = package_version("axolotl")
    if installed != REQUIRED_AXOLOTL_VERSION:
        raise RuntimeError(
            f"Axolotl incompatible: se requiere {REQUIRED_AXOLOTL_VERSION} y está instalado {installed}."
        )
    cli = Path(sys.executable).with_name("axolotl")
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise RuntimeError(f"No se encontró el CLI de Axolotl en {cli}")
    print(f"Python Axolotl: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"Axolotl: {installed}")
    print(f"Axolotl CLI: {cli}")
    print("DeepSpeed: no requerido para el perfil QLoRA de una GPU")
    print("Backend Axolotl: OK")


if __name__ == "__main__":
    main()

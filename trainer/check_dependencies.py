#!/usr/bin/env python3
"""Comprueba que el entorno de fine-tuning coincide con requirements.txt.

También informa la versión de torch y su build CUDA, y falla si esa build no
incluye la arquitectura de la GPU visible. Sin GPU CUDA visible solo avisa:
la disponibilidad de la GPU la comprueba check_environment.py.
"""
from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).with_name("requirements.txt")
PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)$")
CUDA_ARCH = re.compile(r"^(sm|compute)_(\d+)([a-z]?)$")


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


def architecture_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Aplica la compatibilidad binaria de CUDA a la lista de `torch.cuda.get_arch_list()`.

    Un cubin `sm_XY` corre en la misma versión mayor X con menor >= Y (sm_89 usa
    los de sm_86). Un PTX `compute_XY` se compila al vuelo para cualquier
    arquitectura >= XY. Las variantes con sufijo (`sm_90a`) solo corren en la
    arquitectura exacta.
    """
    major, minor = capability
    for arch in arch_list:
        match = CUDA_ARCH.fullmatch(arch)
        if not match:
            continue
        kind, number, suffix = match.groups()
        arch_major, arch_minor = divmod(int(number), 10)
        if suffix:
            if (arch_major, arch_minor) == (major, minor):
                return True
        elif kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        elif kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def torch_report() -> tuple[list[str], list[str], list[str]]:
    """Devuelve (líneas de estado, avisos, errores) sobre torch y la GPU visible."""
    try:
        import torch
    except ImportError:
        return [], [], [
            "torch: no instalado (lo instalan las dependencias de trainer/requirements.txt)"
        ]

    cuda_build = torch.version.cuda
    if not cuda_build:
        return (
            [f"torch: {torch.__version__} (sin soporte CUDA)"],
            ["Aviso: esta build de torch no tiene soporte CUDA; no se comprobó la arquitectura de la GPU."],
            [],
        )

    build = f"torch {torch.__version__} (CUDA {cuda_build})"
    info = [f"torch: {torch.__version__} (CUDA {cuda_build})"]
    try:
        devices = (
            [
                (index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        )
        arch_list = list(torch.cuda.get_arch_list()) if devices else []
    except (RuntimeError, AssertionError) as error:
        return info, [f"Aviso: no se pudo consultar la GPU ({error}); no se comprobó la arquitectura de la GPU."], []
    if not devices:
        return info, [
            "Aviso: CUDA no está disponible (no hay una GPU visible o el driver no soporta "
            f"CUDA {cuda_build}); no se comprobó la arquitectura de la GPU."
        ], []
    if not arch_list:
        return info, ["Aviso: torch no informa sus arquitecturas CUDA; no se comprobó la arquitectura de la GPU."], []

    unsupported: list[str] = []
    for index, name, (major, minor) in devices:
        if architecture_supported((major, minor), arch_list):
            info.append(f"GPU {index}: {name} (sm_{major}{minor}, compatible)")
        else:
            unsupported.append(f"- GPU {index}: {name} (sm_{major}{minor})")
    if unsupported:
        return info, [], [
            "La build de torch no soporta la GPU visible:",
            *unsupported,
            f"{build} incluye: {' '.join(arch_list)}",
            "Instala una build de torch compilada para esa arquitectura desde el índice CUDA "
            "de PyTorch (https://pytorch.org/get-started/locally/).",
        ]
    return info, [], []


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

    # requirements.txt no fija torch (llega por bitsandbytes y accelerate): se
    # informa la build instalada y se comprueba que soporte la GPU visible.
    torch_lines, torch_warnings, torch_errors = torch_report()
    if torch_errors:
        for line in torch_errors:
            print(line, file=sys.stderr)
        raise SystemExit(1)

    print(f"Python: {sys.version.split()[0]}")
    print(f"datasets: {version('datasets')}")
    print(f"transformers: {version('transformers')} (Qwen3.5 OK)")
    print(f"trl: {version('trl')}")
    print(f"peft: {version('peft')}")
    for line in torch_lines:
        print(line)
    for warning in torch_warnings:
        print(warning, file=sys.stderr)
    print("Dependencias del trainer: OK")


if __name__ == "__main__":
    main()

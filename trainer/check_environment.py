#!/usr/bin/env python3
"""Comprueba CUDA/NF4 y muestra las versiones relevantes del entorno QLoRA."""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

import bitsandbytes
import torch
from bitsandbytes.functional import quantize_4bit


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "no instalado"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible")

    sample = torch.ones((2, 2), device="cuda")
    quantize_4bit(sample, quant_type="nf4")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA de PyTorch: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")
    print(f"datasets: {package_version('datasets')}")
    print(f"transformers: {package_version('transformers')}")
    print(f"trl: {package_version('trl')}")
    print("bitsandbytes NF4: OK")


if __name__ == "__main__":
    main()

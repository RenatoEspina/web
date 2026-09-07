#!/usr/bin/env python3
"""Comprueba que el entorno QLoRA dispone de CUDA y bitsandbytes NF4."""
from __future__ import annotations

import bitsandbytes
import torch
from bitsandbytes.functional import quantize_4bit


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible")

    sample = torch.ones((2, 2), device="cuda")
    quantize_4bit(sample, quant_type="nf4")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA de PyTorch: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")
    print("bitsandbytes NF4: OK")


if __name__ == "__main__":
    main()

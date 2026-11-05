#!/usr/bin/env python3
"""Reconcilia LLM_ADAPTER_MODELS después de (re)iniciar vLLM (FT-03).

Reiniciar vLLM descarta los LoRA cargados dinámicamente. ``./comandos.fish vllm``
y ``scripts/train-adapter*.sh`` ejecutan este comando después de levantar vLLM:
recarga los adaptadores de la allowlist con validación de base o los retira.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path

import gui_server as core


class _EchoLog(deque):
    """Registro del job que además muestra cada línea en la terminal."""

    def append(self, line: str) -> None:
        print(line, flush=True)
        super().append(line)


def export_python(default: Callable[[], Path]) -> Path:
    """Intérprete para exportar Qwen3.5 fuera de las GUIs.

    Respeta FINE_TUNE_PYTHON (las GUIs y los scripts lo propagan) y el venv básico;
    si solo existe el entorno Axolotl, lo usa en lugar de un intérprete inexistente.
    """
    preferred = default()
    if preferred.is_file():
        return preferred
    axolotl = core.TRAINER_DIR / ".venv-axolotl" / "bin" / "python"
    if axolotl.is_file() and os.access(axolotl, os.X_OK):
        return axolotl
    return preferred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recarga o retira de LLM_ADAPTER_MODELS los LoRA que vLLM perdió al reiniciarse.",
    )
    parser.add_argument(
        "--wait-vllm",
        type=int,
        default=0,
        metavar="SEGUNDOS",
        help="espera hasta SEGUNDOS a que vLLM responda /health antes de reconciliar",
    )
    args = parser.parse_args(argv)
    job = core.Job(id="reconcile", action="reconcile-adapters", logs=_EchoLog(maxlen=core.MAX_LOG_LINES))
    default_python = core.trainer_python
    core.trainer_python = lambda: export_python(default_python)
    try:
        if args.wait_vllm > 0:
            core.wait_for_vllm(job, args.wait_vllm)
        core.reconcile_adapter_allowlist(job)
    except KeyboardInterrupt:
        print("Aviso: reconciliación de adaptadores interrumpida; queda pendiente.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Aviso: no se pudo reconciliar LLM_ADAPTER_MODELS: {error}", file=sys.stderr)
        return 1
    finally:
        core.trainer_python = default_python
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

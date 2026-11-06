#!/usr/bin/env python3
"""Reconcilia LLM_ADAPTER_MODELS después de (re)iniciar vLLM (FT-03).

Reiniciar vLLM descarta los LoRA cargados dinámicamente. ``./comandos.fish vllm``
y ``scripts/train-adapter*.sh`` ejecutan este comando después de levantar vLLM:
recarga los adaptadores de la allowlist con validación de base o los retira.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

import gui_server as core


class _EchoLog(deque):
    """Registro del job que además muestra cada línea en la terminal."""

    def append(self, line: str) -> None:
        print(line, flush=True)
        super().append(line)


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
    try:
        if args.wait_vllm > 0:
            core.wait_for_vllm(job, args.wait_vllm)
        core.reconcile_adapter_allowlist(job)
    except KeyboardInterrupt:
        print("Aviso: reconciliación de adaptadores interrumpida.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Aviso: no se pudo reconciliar LLM_ADAPTER_MODELS: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

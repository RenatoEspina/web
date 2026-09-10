#!/usr/bin/env python3
"""Axolotl backend for the existing local fine-tuning GUI.

The basic TRL/PEFT GUI remains untouched. This module reuses its HTTP/UI and
adapter-management code, but redirects environment setup and training to an
isolated Axolotl installation.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import gui_server as core

AXOLOTL_VERSION = "0.18.0"
AXOLOTL_VENV = core.TRAINER_DIR / ".venv-axolotl"
AXOLOTL_CHECK = core.TRAINER_DIR / "check_axolotl_environment.py"
AXOLOTL_TRAIN_SCRIPT = core.ROOT / "scripts" / "train-adapter-axolotl.sh"


def axolotl_python() -> Path:
    configured = os.environ.get("FINE_TUNE_AXOLOTL_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return AXOLOTL_VENV / "bin" / "python"


def _host_python() -> str:
    for name in ("python3.13", "python3.12", "python3.11", "python3.14", "python3", "python"):
        executable = shutil.which(name)
        if executable:
            return executable
    raise RuntimeError("No se encontró Python 3 para crear trainer/.venv-axolotl.")


def setup_axolotl_environment(job: core.Job, *, automatic: bool = False) -> None:
    python_command = _host_python()
    python_path = AXOLOTL_VENV / "bin" / "python"
    if not python_path.is_file():
        core.run_command(
            job,
            [python_command, "-m", "venv", str(AXOLOTL_VENV)],
            label="Creando entorno Axolotl automáticamente" if automatic else "Creando entorno Axolotl",
        )
    python_path = axolotl_python()
    core.run_command(
        job,
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip", "packaging", "setuptools", "wheel", "ninja"],
        label="Actualizando herramientas de instalación de Axolotl",
    )
    # Axolotl's pip installation requires PyTorch to exist before the
    # --no-build-isolation install. 2.12.1 is the release recommended by the
    # current Axolotl installation guide and supports the project's CUDA path.
    core.run_command(
        job,
        [str(python_path), "-m", "pip", "install", "torch==2.12.1", "torchvision"],
        label="Instalando PyTorch para Axolotl",
    )
    core.run_command(
        job,
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            f"axolotl[deepspeed]=={AXOLOTL_VERSION}",
        ],
        label=f"Instalando Axolotl {AXOLOTL_VERSION}",
    )


def check_axolotl_environment(job: core.Job, label: str) -> None:
    core.run_command(job, [str(axolotl_python()), str(AXOLOTL_CHECK)], label=label)


def run_axolotl_train(job: core.Job, payload: dict[str, Any]) -> None:
    dataset = core.dataset_path(payload.get("dataset"))
    _, summary = core.load_jsonl(dataset)
    core.ensure_adapter_dir_writable(job)
    name = core.validate_adapter_name(payload.get("name"))
    model = core.validate_model(payload.get("model"))
    hf_token = core.validate_hf_token(payload.pop("hfToken", ""))
    rank = core.int_arg(payload, "rank", 16, 8, 32)
    if rank not in {8, 16, 32}:
        raise ValueError("rank debe ser 8, 16 o 32.")
    alpha = core.int_arg(payload, "alpha", 32, 1, 256)
    epochs = core.number_arg(payload, "epochs", 2.0, 0.1, 20.0)
    dropout = core.number_arg(payload, "dropout", 0.05, 0.0, 0.5)
    learning_rate = core.number_arg(payload, "learningRate", 1e-4, 1e-7, 0.1)
    batch_size = core.int_arg(payload, "batchSize", 1, 1, 32)
    gradient_accumulation = core.int_arg(payload, "gradientAccumulation", 8, 1, 256)
    max_length = core.int_arg(payload, "maxLength", 1024, 128, 16384)
    seed = core.int_arg(payload, "seed", 42, 0, 2_147_483_647)

    if not axolotl_python().is_file():
        setup_axolotl_environment(job, automatic=True)
    check_axolotl_environment(job, "Verificando entorno Axolotl")

    command = [
        "bash",
        str(AXOLOTL_TRAIN_SCRIPT),
        str(dataset),
        name,
        "--model",
        model,
        "--rank",
        str(rank),
        "--alpha",
        str(alpha),
        "--dropout",
        str(dropout),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--max-length",
        str(max_length),
        "--seed",
        str(seed),
    ]
    env = core.environment_with_hf_token(hf_token)
    env["FINE_TUNE_AXOLOTL_PYTHON"] = str(axolotl_python())
    env["COMPOSE_PROJECT_NAME"] = os.environ.get("COMPOSE_PROJECT_NAME", "llm-bridge")
    if hf_token:
        core.append_log(job, "Hugging Face: autenticación efímera habilitada para Axolotl.")
    core.run_command(job, command, env=env, label=f"Entrenando {name} con Axolotl")
    job.result = {
        "adapter": name,
        "dataset": dataset.name,
        "examples": summary["examples"],
        "backend": "axolotl",
    }


_original_run_action = core.run_action
_original_status_payload = core.status_payload


def run_action(job: core.Job, payload: dict[str, Any]) -> None:
    if job.action == "train":
        run_axolotl_train(job, payload)
        return
    _original_run_action(job, payload)


def status_payload() -> dict[str, Any]:
    payload = _original_status_payload()
    payload["trainingBackend"] = "axolotl"
    payload["axolotlVersion"] = AXOLOTL_VERSION
    payload["environmentReady"] = axolotl_python().is_file()
    return payload


class AxolotlHandler(core.Handler):
    server_version = "LLMBridgeAxolotlFineTune/1.0"

    def _index(self) -> None:
        path = core.GUI_DIR / "index.html"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            self.send_error(core.HTTPStatus.NOT_FOUND)
            return
        text = text.replace("QLoRA · SFT · asistente local", "AXOLOTL · QLoRA · SFT")
        text = text.replace("Fine-tuning local", "Fine-tuning local · Axolotl")
        text = text.replace("Python 3.14 compatible", f"Backend: Axolotl {AXOLOTL_VERSION}")
        text = text.replace("trainer/.venv", "trainer/.venv-axolotl")
        body = text.encode("utf-8")
        self.send_response(core.HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        self.wfile.write(body)


# Patch only this process. Importing/running trainer/gui_server.py directly keeps
# the existing TRL/PEFT backend unchanged.
core.trainer_python = axolotl_python
core.setup_environment = setup_axolotl_environment
core.check_training_environment = check_axolotl_environment
core.ENVIRONMENT_CHECK = AXOLOTL_CHECK
core.run_action = run_action
core.status_payload = status_payload
core.Handler = AxolotlHandler


if __name__ == "__main__":
    print(f"Backend de entrenamiento: Axolotl {AXOLOTL_VERSION}")
    core.main()

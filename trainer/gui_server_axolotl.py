#!/usr/bin/env python3
"""Axolotl backend for the existing local fine-tuning GUI.

The basic TRL/PEFT GUI remains untouched. This module reuses its HTTP/UI and
adapter-management code, but redirects environment setup and training to an
isolated Axolotl installation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gui_server as core

AXOLOTL_VERSION = "0.18.0"
AXOLOTL_PYTHON_VERSION = "3.12"
AXOLOTL_TORCH_VERSION = "2.12.1"
AXOLOTL_TORCH_BACKEND = os.environ.get("AXOLOTL_TORCH_BACKEND", "cu130").strip() or "cu130"
AXOLOTL_VENV = core.TRAINER_DIR / ".venv-axolotl"
UV_BOOTSTRAP_VENV = core.TRAINER_DIR / ".venv-uv-bootstrap"
AXOLOTL_CHECK = core.TRAINER_DIR / "check_axolotl_environment.py"
AXOLOTL_TRAIN_SCRIPT = core.ROOT / "scripts" / "train-adapter-axolotl.sh"


def axolotl_python() -> Path:
    configured = os.environ.get("FINE_TUNE_AXOLOTL_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return AXOLOTL_VENV / "bin" / "python"


def _bootstrap_python() -> str:
    """Return any host Python capable of bootstrapping uv.

    Axolotl itself does not run in this interpreter. uv creates the real
    training environment with Python 3.12.
    """
    for name in ("python3.14", "python3.13", "python3.12", "python3.11", "python3", "python"):
        executable = shutil.which(name)
        if executable:
            return executable
    raise RuntimeError("No se encontró Python 3 para preparar uv.")


def _configured_uv() -> Path | None:
    configured = os.environ.get("AXOLOTL_UV", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    system_uv = shutil.which("uv")
    if system_uv:
        return Path(system_uv)
    candidate = UV_BOOTSTRAP_VENV / "bin" / "uv"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _ensure_uv(job: core.Job) -> Path:
    uv = _configured_uv()
    if uv is not None:
        return uv

    bootstrap_python = _bootstrap_python()
    bootstrap_venv_python = UV_BOOTSTRAP_VENV / "bin" / "python"
    if not bootstrap_venv_python.is_file():
        core.run_command(
            job,
            [bootstrap_python, "-m", "venv", str(UV_BOOTSTRAP_VENV)],
            label="Creando bootstrap local para uv",
        )
    core.run_command(
        job,
        [
            str(bootstrap_venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "uv",
        ],
        label="Instalando uv para administrar Python 3.12",
    )
    uv = _configured_uv()
    if uv is None:
        raise RuntimeError("uv se instaló, pero no se encontró su ejecutable.")
    return uv


def _venv_python_minor() -> str | None:
    python_path = AXOLOTL_VENV / "bin" / "python"
    if not python_path.is_file():
        return None
    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def axolotl_environment_ready() -> bool:
    return (
        _venv_python_minor() == AXOLOTL_PYTHON_VERSION
        and (AXOLOTL_VENV / "bin" / "axolotl").is_file()
        and (AXOLOTL_VENV / "bin" / "python").is_file()
    )


def _reset_incompatible_axolotl_venv(job: core.Job) -> None:
    if not AXOLOTL_VENV.exists():
        return
    if AXOLOTL_VENV.is_symlink() or not AXOLOTL_VENV.is_dir():
        raise RuntimeError(
            f"{AXOLOTL_VENV} no es un directorio local seguro; no se eliminará automáticamente."
        )
    current = _venv_python_minor()
    if current == AXOLOTL_PYTHON_VERSION:
        return
    core.append_log(
        job,
        f"El entorno Axolotl existente usa Python {current or 'desconocido'}; "
        f"se reconstruirá con Python {AXOLOTL_PYTHON_VERSION}.",
    )
    shutil.rmtree(AXOLOTL_VENV)


def setup_axolotl_environment(job: core.Job, *, automatic: bool = False) -> None:
    _reset_incompatible_axolotl_venv(job)
    uv = _ensure_uv(job)
    python_path = AXOLOTL_VENV / "bin" / "python"

    install_env = os.environ.copy()
    install_env["UV_TORCH_BACKEND"] = AXOLOTL_TORCH_BACKEND

    if not python_path.is_file():
        core.run_command(
            job,
            [
                str(uv),
                "venv",
                "--python",
                AXOLOTL_PYTHON_VERSION,
                str(AXOLOTL_VENV),
            ],
            env=install_env,
            label=(
                f"Creando entorno Axolotl con Python {AXOLOTL_PYTHON_VERSION} automáticamente"
                if automatic
                else f"Creando entorno Axolotl con Python {AXOLOTL_PYTHON_VERSION}"
            ),
        )

    if _venv_python_minor() != AXOLOTL_PYTHON_VERSION:
        raise RuntimeError(
            f"Axolotl requiere el entorno del proyecto en Python {AXOLOTL_PYTHON_VERSION}; "
            f"se detectó {_venv_python_minor() or 'una versión desconocida'}."
        )

    # This project uses single-GPU QLoRA and the generated config has no
    # `deepspeed:` section. Installing the optional deepspeed extra would force
    # a local CUDA-toolkit build (CUDA_HOME/nvcc) that this workflow does not use.
    core.run_command(
        job,
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python_path),
            "--upgrade",
            "packaging",
            "setuptools",
            "wheel",
            "ninja",
        ],
        env=install_env,
        label="Actualizando herramientas de instalación de Axolotl",
    )
    core.run_command(
        job,
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python_path),
            f"torch=={AXOLOTL_TORCH_VERSION}",
            "torchvision",
        ],
        env=install_env,
        label=f"Instalando PyTorch {AXOLOTL_TORCH_VERSION} para Axolotl",
    )
    core.run_command(
        job,
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python_path),
            "--no-build-isolation",
            f"axolotl=={AXOLOTL_VERSION}",
        ],
        env=install_env,
        label=f"Instalando Axolotl {AXOLOTL_VERSION} sin DeepSpeed",
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

    if not axolotl_environment_ready():
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
    payload["axolotlPythonVersion"] = AXOLOTL_PYTHON_VERSION
    payload["environmentReady"] = axolotl_environment_ready()
    return payload


class AxolotlHandler(core.Handler):
    server_version = "LLMBridgeAxolotlFineTune/1.1"

    def _index(self) -> None:
        path = core.GUI_DIR / "index.html"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            self.send_error(core.HTTPStatus.NOT_FOUND)
            return
        text = text.replace("QLoRA · SFT · asistente local", "AXOLOTL · QLoRA · SFT")
        text = text.replace("Fine-tuning local", "Fine-tuning local · Axolotl")
        text = text.replace(
            "Python 3.14 compatible",
            f"Axolotl {AXOLOTL_VERSION} · Python {AXOLOTL_PYTHON_VERSION}",
        )
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
    print(
        f"Backend de entrenamiento: Axolotl {AXOLOTL_VERSION} "
        f"(Python {AXOLOTL_PYTHON_VERSION})"
    )
    core.main()

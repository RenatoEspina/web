#!/usr/bin/env python3
"""Panel web local para automatizar el flujo de fine-tuning QLoRA."""
from __future__ import annotations

import argparse
import grp
import hmac
import json
import os
import pwd
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import runtime_lock
from dataset_validation import load_jsonl
from vllm_export import export_destination, needs_export, read_config, validate_export

ROOT = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT / "trainer"
GUI_DIR = TRAINER_DIR / "gui"
DATASET_DIR = TRAINER_DIR / "datasets"
EXAMPLE_DIR = TRAINER_DIR / "examples"
ADAPTER_DIR = ROOT / "adapters"
RUNTIME_DIR = ROOT / ".runtime"
ENVIRONMENT_CHECK = TRAINER_DIR / "check_environment.py"
MAX_BODY_BYTES = 40 * 1024 * 1024
# json.loads only fails on deep nesting when the interpreter runs out of stack,
# which varies between machines; this fixed limit makes the rejection
# deterministic. Real payloads and dataset lines nest a handful of levels.
MAX_JSON_DEPTH = 512
JSON_DEPTH_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')
MAX_LOG_LINES = 2500
MAX_HF_TOKEN_LENGTH = 4096
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.jsonl$")
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
EVALUATION_DATASETS = {"evaluation.jsonl"}
# Un split de validación nunca debe poder seleccionarse como dataset SFT:
# entrenarlo destruye la señal con la que se elige el checkpoint. Cubre la
# convención hermana <nombre>-training.jsonl / <nombre>-validation.jsonl que
# usan scripts/train-adapter.sh y trainer/mix_replay.py.
VALIDATION_SUFFIX = "-validation.jsonl"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.29.0"
# Cancelación escalonada sobre el grupo de procesos del job: SIGINT, espera,
# SIGTERM, espera y SIGKILL. Si un proceso desacoplado conserva el stdout, el
# job se da por cancelado tras CANCEL_OUTPUT_DRAIN_SECONDS sin esperar su EOF.
CANCEL_SIGINT_GRACE_SECONDS = 10.0
CANCEL_SIGTERM_GRACE_SECONDS = 5.0
CANCEL_OUTPUT_DRAIN_SECONDS = 2.0
# Una reconciliación de la allowlist interrumpida (por ejemplo, al cancelar un
# entrenamiento mientras vLLM se reiniciaba) deja esta marca en .runtime/. El
# panel la completa automáticamente cuando vLLM vuelve a responder.
RECONCILE_MARKER_FILENAME = "adapters-reconcile-pending"
RECONCILE_POLL_SECONDS = 5.0
RECONCILE_RETRY_SECONDS = 60.0
_LAST_AUTO_RECONCILE_FAILURE: float | None = None


@dataclass
class Job:
    id: str
    action: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    process: subprocess.Popen[str] | None = None
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    gpu_lock: runtime_lock.GpuLock | None = None


STATE_LOCK = threading.Lock()
CURRENT_JOB: Job | None = None
CSRF_TOKEN = secrets.token_urlsafe(32)


class JobCancelled(RuntimeError):
    pass


class AdapterBaseMismatch(RuntimeError):
    """vLLM sirve una base conocida distinta de la usada para entrenar el adaptador."""


class InvalidAllowlistEntry(ValueError):
    """La entrada de LLM_ADAPTER_MODELS no corresponde a un adaptador cargable."""


def gpu_lock_path() -> Path:
    return RUNTIME_DIR / runtime_lock.LOCK_FILENAME


def job_record_path() -> Path:
    return RUNTIME_DIR / runtime_lock.JOB_RECORD_FILENAME


def external_job_message(record: dict[str, Any]) -> str:
    return (
        "Hay una operación de GPU/vLLM en curso fuera de este panel "
        f"(acción {record.get('action') or 'desconocida'}, PGID {record['pgid']}). "
        "Espera a que termine o detenla antes de iniciar otra."
    )


def trainer_python() -> Path:
    configured = os.environ.get("FINE_TUNE_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    preferred = TRAINER_DIR / ".venv" / "bin" / "python"
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return preferred
    fallback = ROOT / ".venv" / "bin" / "python"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return fallback
    return preferred


def validate_adapter_name(value: Any) -> str:
    name = str(value or "").strip()
    if not SAFE_NAME.fullmatch(name):
        raise ValueError("Nombre de adaptador inválido.")
    return name


def validate_model(value: Any) -> str:
    model = str(value or "Qwen/Qwen3.5-0.8B").strip()
    if not SAFE_MODEL.fullmatch(model) or ".." in model:
        raise ValueError("Nombre de modelo inválido.")
    return model


def validate_hf_token(value: Any) -> str:
    """Valida un token efímero sin asumir un prefijo concreto de Hugging Face."""
    token = str(value or "").strip()
    if not token:
        return ""
    if len(token) > MAX_HF_TOKEN_LENGTH or any(character.isspace() for character in token):
        raise ValueError("HF token inválido.")
    return token


def environment_with_hf_token(token: str) -> dict[str, str]:
    """Crea un entorno hijo con credenciales HF sin modificar os.environ ni persistirlas."""
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    return env


def number_arg(payload: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} debe ser numérico.") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} debe estar entre {minimum} y {maximum}.")
    return value


def int_arg(payload: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{key} debe ser entero.") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} debe estar entre {minimum} y {maximum}.")
    return value


@dataclass(frozen=True)
class TrainingRequest:
    """Payload de entrenamiento ya validado; compartido por ambas GUIs."""

    dataset: Path
    examples: int
    name: str
    model: str
    hf_token: str
    rank: int
    alpha: int
    epochs: float
    dropout: float
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    max_length: int
    seed: int

    def script_arguments(self) -> list[str]:
        """Argumentos comunes de scripts/train-adapter.sh y train-adapter-axolotl.sh."""
        return [
            str(self.dataset),
            self.name,
            "--model",
            self.model,
            "--rank",
            str(self.rank),
            "--alpha",
            str(self.alpha),
            "--dropout",
            str(self.dropout),
            "--epochs",
            str(self.epochs),
            "--learning-rate",
            str(self.learning_rate),
            "--batch-size",
            str(self.batch_size),
            "--gradient-accumulation",
            str(self.gradient_accumulation),
            "--max-length",
            str(self.max_length),
            "--seed",
            str(self.seed),
        ]


def validate_training_payload(payload: dict[str, Any]) -> TrainingRequest:
    """Valida el payload completo antes de cualquier efecto (FT-11).

    Ninguna operación con efectos, como ensure_adapter_dir_writable (que puede
    lanzar un contenedor root), debe ejecutarse antes de esta función.
    """
    dataset = dataset_path(payload.get("dataset"))
    _, summary = load_jsonl(dataset)
    name = validate_adapter_name(payload.get("name"))
    model = validate_model(payload.get("model"))
    hf_token = validate_hf_token(payload.pop("hfToken", ""))
    rank = int_arg(payload, "rank", 16, 8, 32)
    if rank not in {8, 16, 32}:
        raise ValueError("rank debe ser 8, 16 o 32.")
    return TrainingRequest(
        dataset=dataset,
        examples=int(summary["examples"]),
        name=name,
        model=model,
        hf_token=hf_token,
        rank=rank,
        alpha=int_arg(payload, "alpha", 32, 1, 256),
        epochs=number_arg(payload, "epochs", 2.0, 0.1, 20.0),
        dropout=number_arg(payload, "dropout", 0.05, 0.0, 0.5),
        learning_rate=number_arg(payload, "learningRate", 1e-4, 1e-7, 0.1),
        batch_size=int_arg(payload, "batchSize", 1, 1, 32),
        gradient_accumulation=int_arg(payload, "gradientAccumulation", 8, 1, 256),
        max_length=int_arg(payload, "maxLength", 1024, 128, 16384),
        seed=int_arg(payload, "seed", 42, 0, 2_147_483_647),
    )


def dataset_path(dataset_id: Any) -> Path:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        raise ValueError("Selecciona un dataset.")
    candidate = (TRAINER_DIR / dataset_id).resolve()
    if candidate.parent not in {DATASET_DIR.resolve(), EXAMPLE_DIR.resolve()} or candidate.suffix != ".jsonl":
        raise ValueError("Dataset fuera de las carpetas permitidas.")
    if candidate.name.casefold() in EVALUATION_DATASETS:
        raise ValueError("evaluation.jsonl es un dataset de evaluación y no se puede usar para SFT.")
    if candidate.name.casefold().endswith(VALIDATION_SUFFIX):
        raise ValueError("Es un dataset de validación; se conecta solo y no se puede entrenar.")
    if not candidate.is_file():
        raise ValueError("El dataset seleccionado ya no existe.")
    return candidate


def json_too_deep(text: str) -> bool:
    """Return True when arrays/objects nest deeper than MAX_JSON_DEPTH.

    Brackets inside JSON strings do not count. Malformed JSON is left for
    json.loads to report with its usual error.
    """
    depth = 0
    for match in JSON_DEPTH_TOKEN.finditer(text):
        token = match.group()
        if token == "[" or token == "{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return True
        elif token == "]" or token == "}":
            depth -= 1
    return False


def dataset_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", errors="replace") as lines:
            if any(json_too_deep(line) for line in lines):
                raise RecursionError
        _, summary = load_jsonl(path)
        return dict(summary)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "error": str(error)}
    except RecursionError:
        return {"valid": False, "error": "No se pudo analizar el dataset: el JSON está demasiado anidado."}
    except Exception as error:
        # Cualquier otro fallo del análisis también es un dataset inválido (FT-10).
        return {"valid": False, "error": f"No se pudo analizar el dataset ({type(error).__name__}): {error}"}


def list_datasets() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, directory in (("Subido", DATASET_DIR), ("Ejemplo", EXAMPLE_DIR)):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            if path.name.casefold() in EVALUATION_DATASETS:
                continue
            if path.name.casefold().endswith(VALIDATION_SUFFIX):
                continue
            result.append(
                {
                    "id": path.relative_to(TRAINER_DIR).as_posix(),
                    "name": path.name,
                    "kind": label,
                    "bytes": path.stat().st_size,
                }
            )
    return result


def adapter_dir_diagnostic() -> str:
    target = ADAPTER_DIR if ADAPTER_DIR.exists() else ADAPTER_DIR.parent
    try:
        info = target.stat()
        owner = pwd.getpwuid(info.st_uid).pw_name
        group = grp.getgrgid(info.st_gid).gr_name
        mode = f"{info.st_mode & 0o777:03o}"
        return f"{target}: owner={owner}:{group} ({info.st_uid}:{info.st_gid}), mode={mode}"
    except (KeyError, OSError) as error:
        return f"{target}: no se pudo obtener owner/mode ({error})"


def adapter_dir_fallback_command() -> str:
    path = shlex.quote(str(ADAPTER_DIR))
    return f"sudo chown -R {os.getuid()}:{os.getgid()} -- {path} && sudo chmod -R u+rwX -- {path}"


def adapter_dir_is_writable() -> bool:
    probe = ADAPTER_DIR / f".gui-write-test-{uuid.uuid4().hex}"
    try:
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        return True
    except OSError:
        return False
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass


def ensure_adapter_dir_writable(job: Job) -> None:
    """Create or safely repair the adapters bind mount without deleting contents."""
    if adapter_dir_is_writable():
        return

    append_log(job, f"adapters/ no es escribible: {adapter_dir_diagnostic()}")
    docker = shutil.which("docker")
    if docker and ADAPTER_DIR.exists() and ADAPTER_DIR.is_dir():
        append_log(job, "Intentando reparar ownership/permisos con un contenedor efímero…")
        helper_image = os.environ.get("VLLM_IMAGE", DEFAULT_VLLM_IMAGE).strip() or DEFAULT_VLLM_IMAGE
        try:
            run_command(
                job,
                [
                    docker,
                    "run",
                    "--rm",
                    "--user",
                    "0:0",
                    "-v",
                    f"{ADAPTER_DIR}:/adapters",
                    "--entrypoint",
                    "/bin/sh",
                    helper_image,
                    "-c",
                    f"chown -R {os.getuid()}:{os.getgid()} /adapters && chmod -R u+rwX /adapters",
                ],
                label="Reparando permisos de adapters/",
            )
        except RuntimeError as error:
            append_log(job, f"La reparación automática con Docker falló: {error}")

    if adapter_dir_is_writable():
        append_log(job, "adapters/ ya es escribible por el usuario local; los adaptadores existentes se conservaron.")
        return

    raise RuntimeError(
        "No fue posible preparar adapters/ para escritura. "
        f"Diagnóstico: {adapter_dir_diagnostic()}. "
        f"Ejecuta manualmente: {adapter_dir_fallback_command()}"
    )


def list_adapters() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not ADAPTER_DIR.exists():
        return result
    for directory in sorted(ADAPTER_DIR.iterdir()):
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.is_file() or not SAFE_NAME.fullmatch(directory.name):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        result.append(
            {
                "name": directory.name,
                "baseModel": manifest.get("baseModel"),
                "createdAt": manifest.get("createdAt"),
                "examples": manifest.get("examples"),
                "rank": (manifest.get("parameters") or {}).get("rank"),
                "metrics": manifest.get("metrics") or {},
            }
        )
    return result


def fetch_vllm_models() -> tuple[bool, list[str]]:
    try:
        with urlopen("http://127.0.0.1:8000/v1/models", timeout=0.8) as response:
            payload = json.load(response)
        models = [
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        return True, models
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False, []


def job_snapshot(job: Job | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "id": job.id,
        "action": job.action,
        "status": job.status,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
        "returncode": job.returncode,
        "logs": list(job.logs),
        "result": job.result,
        "cancelRequested": job.cancel_requested,
    }


def environment_check_path(check_script: Path | None = None) -> Path:
    """Resultado persistido de la última comprobación, uno por backend."""
    return RUNTIME_DIR / f"environment-check.{(check_script or ENVIRONMENT_CHECK).stem}.json"


def _venv_config_mtime(python: Path) -> int | None:
    try:
        return (python.parent.parent / "pyvenv.cfg").stat().st_mtime_ns
    except OSError:
        return None


def clear_environment_check(check_script: Path | None = None) -> None:
    try:
        environment_check_path(check_script).unlink(missing_ok=True)
    except OSError:
        pass


def _record_environment_check(job: Job, check_script: Path | None, python: Path, ok: bool) -> None:
    try:
        runtime_lock.write_private_json(environment_check_path(check_script), {
            "ok": ok,
            "checkedAt": time.time(),
            "python": str(python),
            "venvConfigMtimeNs": _venv_config_mtime(python),
        })
    except OSError as error:
        append_log(job, f"Aviso: no se pudo guardar el resultado de la comprobación: {error}")


def verify_environment(job: Job, label: str, *, check: Any = None, check_script: Path | None = None) -> None:
    """Ejecuta la comprobación real (CUDA, NF4) y persiste su resultado (FT-07).

    Una cancelación no deja resultado: no prueba ni descarta el entorno.
    """
    python = trainer_python()
    try:
        (check or check_training_environment)(job, label)
    except JobCancelled:
        raise
    except Exception:
        _record_environment_check(job, check_script, python, ok=False)
        raise
    _record_environment_check(job, check_script, python, ok=True)


def environment_status(venv_present: bool, *, check_script: Path | None = None) -> dict[str, Any]:
    """Separa "existe el venv" de "la última comprobación real fue correcta"."""
    record = runtime_lock.read_json_record(environment_check_path(check_script)) if venv_present else None
    python = trainer_python()
    verified = bool(
        record
        and record.get("ok") is True
        and record.get("python") == str(python)
        and record.get("venvConfigMtimeNs") == _venv_config_mtime(python)
    )
    return {
        # Se conserva por compatibilidad: significa solo que el venv existe.
        "environmentReady": venv_present,
        "venvPresent": venv_present,
        "environmentVerified": verified,
        "environmentCheckedAt": record.get("checkedAt") if verified and record else None,
    }


def status_payload() -> dict[str, Any]:
    running, models = fetch_vllm_models()
    with STATE_LOCK:
        job = job_snapshot(CURRENT_JOB)
    return {
        "localOnly": True,
        **environment_status(trainer_python().is_file()),
        "dockerCli": shutil.which("docker") is not None,
        "bashCli": shutil.which("bash") is not None,
        "vllmRunning": running,
        "vllmModels": models,
        "datasets": list_datasets(),
        "adapters": list_adapters(),
        "job": job,
        "externalJob": external_job_snapshot(),
        "allowlistedAdapters": allowlisted_adapters(),
    }


def external_job_snapshot() -> dict[str, Any] | None:
    """Job vivo registrado por otro proceso, sin modificar el registro."""
    record = runtime_lock.live_job_record(job_record_path(), remove_stale=False)
    with STATE_LOCK:
        own_process = CURRENT_JOB.process if CURRENT_JOB is not None else None
    if record is None or (own_process is not None and record.get("pgid") == own_process.pid):
        return None
    return {"action": record.get("action"), "pgid": record["pgid"], "startedAt": record.get("startedAt")}


def append_log(job: Job, line: str) -> None:
    clean = line.rstrip("\n")
    if clean:
        with STATE_LOCK:
            job.logs.append(clean)


def _pump_output(job: Job, stream: Any) -> None:
    for line in stream:
        append_log(job, line)


def _wait_for_output(job: Job, reader: threading.Thread) -> None:
    """Espera el EOF del stdout; tras una cancelación, solo un margen acotado."""
    while True:
        reader.join(0.2)
        if not reader.is_alive():
            return
        with STATE_LOCK:
            cancelled = job.cancel_requested
        if cancelled:
            reader.join(CANCEL_OUTPUT_DRAIN_SECONDS)
            if reader.is_alive():
                append_log(
                    job,
                    "Un subproceso desacoplado mantiene abierta la salida; "
                    "la operación se da por cancelada sin esperar su cierre.",
                )
            return


def _wait_for_process_group(job: Job, pgid: int) -> None:
    """El job no termina mientras quede vivo algún proceso de su grupo.

    Tras una cancelación la escalada acaba en SIGKILL; sin cancelación se espera a
    los procesos que el comando dejó en su grupo y el job sigue siendo cancelable.
    """
    if not runtime_lock.process_group_alive(pgid):
        return
    append_log(job, f"Esperando a que terminen los procesos restantes del grupo {pgid}…")
    while runtime_lock.process_group_alive(pgid):
        time.sleep(0.1)


def run_command(
    job: Job,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    label: str | None = None,
) -> None:
    if job.cancel_requested:
        raise JobCancelled()
    if label:
        append_log(job, f"\n== {label} ==")
    child_env = dict(os.environ if env is None else env)
    pass_fds: tuple[int, ...] = ()
    if job.gpu_lock is not None and job.gpu_lock.fd is not None:
        # El hijo hereda el lock de GPU/vLLM: los scripts de entrenamiento lo
        # reconocen y un proceso que sobreviva al panel lo sigue reteniendo.
        child_env[runtime_lock.LOCK_FD_ENV] = str(job.gpu_lock.fd)
        pass_fds = (job.gpu_lock.fd,)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
        start_new_session=True,
        pass_fds=pass_fds,
    )
    if job.gpu_lock is not None:
        try:
            runtime_lock.write_private_json(job_record_path(), {
                "pgid": process.pid,
                "pid": process.pid,
                "action": job.action,
                "jobId": job.id,
                "guiPid": os.getpid(),
                "bootId": runtime_lock.boot_id(),
                "startedAt": time.time(),
            })
        except OSError as error:
            append_log(job, f"Aviso: no se pudo registrar el PGID del job en {RUNTIME_DIR}: {error}")
    with STATE_LOCK:
        job.process = process
        cancel_pending = job.cancel_requested
    if cancel_pending:
        start_cancel_escalation(job, process)
    assert process.stdout is not None
    reader = threading.Thread(target=_pump_output, args=(job, process.stdout), daemon=True)
    reader.start()
    try:
        returncode = process.wait()
        _wait_for_output(job, reader)
        _wait_for_process_group(job, process.pid)
    finally:
        # Con procesos vivos el registro se conserva: el próximo arranque los detecta.
        if job.gpu_lock is not None and not runtime_lock.process_group_alive(process.pid):
            runtime_lock.remove_job_record(job_record_path(), process.pid)
    with STATE_LOCK:
        job.process = None
        job.returncode = returncode
    if job.cancel_requested:
        raise JobCancelled()
    if returncode != 0:
        raise RuntimeError(f"El proceso terminó con código {returncode}.")


def check_training_environment(job: Job, label: str) -> None:
    run_command(job, [str(trainer_python()), str(ENVIRONMENT_CHECK)], label=label)


def wait_for_vllm(job: Job, timeout: int) -> None:
    append_log(job, "Esperando a que vLLM quede saludable…")
    started = time.monotonic()
    last_notice = -10
    while time.monotonic() - started < timeout:
        if job.cancel_requested:
            raise JobCancelled()
        try:
            with urlopen("http://127.0.0.1:8000/health", timeout=1.0):
                append_log(job, "vLLM disponible en http://127.0.0.1:8000")
                return
        except (OSError, URLError):
            pass
        elapsed = int(time.monotonic() - started)
        if elapsed - last_notice >= 10:
            append_log(job, f"vLLM aún está iniciando… {elapsed}s")
            last_notice = elapsed
        time.sleep(1)
    raise RuntimeError(f"Timeout esperando vLLM después de {timeout} segundos.")


def post_vllm(path: str, payload: dict[str, str]) -> str:
    request = Request(
        f"http://127.0.0.1:8000{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace").strip()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"vLLM respondió HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError("vLLM no está disponible en 127.0.0.1:8000.") from error


def adapter_training_base(name: str) -> str:
    """Base declarada por los metadatos del adaptador; ValueError si son inválidos."""
    config, manifest = read_config(ADAPTER_DIR / name)
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise ValueError("Los metadatos del adaptador deben ser objetos JSON.")
    base = config.get("base_model_name_or_path")
    if not isinstance(base, str) or not base.strip():
        raise ValueError("El adaptador no declara base_model_name_or_path; no se puede verificar su base.")
    base = base.strip()
    if "baseModel" in manifest and manifest["baseModel"] != base:
        raise ValueError("baseModel del manifest no coincide con adapter_config.json.")
    return base


def validate_adapter_base(name: str) -> str:
    """Check training metadata against the served root, never a LoRA's parent."""
    base = adapter_training_base(name)

    with urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        payload = json.load(response)
    models = payload.get("data", []) if isinstance(payload, dict) else []
    bases = [item for item in models if isinstance(item, dict) and not item.get("parent")
             and isinstance(item.get("id"), str) and item["id"].strip()
             and isinstance(item.get("root"), str) and item["root"].strip()] if isinstance(models, list) else []
    # Different served names may be aliases for one root. An alias alone is not
    # evidence that the weights match the base used to train this adapter.
    roots = {item["root"].strip() for item in bases}
    if roots != {base}:
        served = ", ".join(sorted(roots)) or "base desconocida"
        message = f"El adaptador '{name}' fue entrenado sobre '{base}', pero vLLM sirve '{served}'. No se cargará un LoRA incompatible."
        # Sin una base identificable no hay prueba de incompatibilidad: no es un mismatch.
        raise AdapterBaseMismatch(message) if roots else RuntimeError(message)
    return bases[0]["id"]


def prepare_runtime_adapter(job: Job, name: str) -> str:
    directory = ADAPTER_DIR / name
    if directory.is_symlink():
        raise ValueError("El adaptador no puede ser un enlace simbólico.")
    config, manifest = read_config(directory)
    if not needs_export(config, manifest):
        return f"/adapters/{name}"
    ensure_adapter_dir_writable(job)
    run_command(job, [str(trainer_python()), str(TRAINER_DIR / "vllm_export.py"), str(directory)],
                label="Preparando namespace LoRA para vLLM (originales conservados)")
    destination = export_destination(directory)
    metadata = validate_export(destination)
    append_log(job, f"Exportación verificada: {metadata['tensors']} tensores, {metadata['renamed']} nombres adaptados.")
    return "/adapters/" + destination.relative_to(ADAPTER_DIR).as_posix()


def loaded_adapter_path(name: str) -> str:
    """Capture the actual runtime path BEFORE unloading, including legacy paths."""
    with urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        models = json.load(response).get("data", [])
    for model in models:
        if model.get("id") == name and model.get("parent") and isinstance(model.get("root"), str):
            return model["root"]
    raise RuntimeError(f"El adaptador {name} no aparece cargado en vLLM.")


def adapter_is_loaded(name: str) -> bool:
    running, models = fetch_vllm_models()
    return running and name in models


def delete_adapter(job: Job, name: str) -> None:
    """Descarga, retira de la allowlist y borra un adaptador persistido."""
    directory = ADAPTER_DIR / name
    if ADAPTER_DIR.is_symlink() or directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"No existe un adaptador válido llamado '{name}'.")
    if directory.parent.resolve() != ADAPTER_DIR.resolve():
        raise RuntimeError("La ruta del adaptador está fuera de adapters/.")

    export_base = ADAPTER_DIR / ".vllm-exports"
    if export_base.is_symlink() or (export_base.exists() and not export_base.is_dir()):
        raise RuntimeError("La caché de exportación no es una carpeta local válida.")
    export_root = export_base / name
    if export_root.is_symlink():
        raise RuntimeError("La caché de exportación del adaptador no puede ser un enlace simbólico.")
    if export_root.exists() and not export_root.is_dir():
        raise RuntimeError("La caché de exportación del adaptador no es una carpeta válida.")

    ensure_adapter_dir_writable(job)
    was_loaded = adapter_is_loaded(name)
    runtime_path: str | None = None
    allowlist_removed = False

    if was_loaded:
        runtime_path = loaded_adapter_path(name)
        append_log(job, f"Descargando {name} de vLLM antes de borrarlo...")
        response = post_vllm("/v1/unload_lora_adapter", {"lora_name": name})
        append_log(job, response or "Adaptador descargado.")

    try:
        # Primero actualiza la allowlist mientras los archivos siguen intactos;
        # si esto falla, todavía es posible restaurar el runtime sin reconstruirlo.
        allowlist_removed = set_adapter_allowed_in_env(name, False)
        if export_root.exists():
            shutil.rmtree(export_root)
        shutil.rmtree(directory)
    except (OSError, RuntimeError) as error:
        if allowlist_removed:
            try:
                set_adapter_allowed_in_env(name, True)
            except (OSError, RuntimeError):
                append_log(job, f"No se pudo restaurar la allowlist tras el borrado incompleto de {name}.")
        if was_loaded and directory.exists():
            try:
                restored_path = prepare_runtime_adapter(job, name)
                post_vllm(
                    "/v1/load_lora_adapter",
                    {"lora_name": name, "lora_path": restored_path or runtime_path or f"/adapters/{name}"},
                )
                append_log(job, f"Se restauró {name} en vLLM tras fallar el borrado.")
            except Exception as rollback_error:
                append_log(job, f"No se pudo restaurar {name} en vLLM: {rollback_error}")
        raise RuntimeError(f"No se pudo borrar el adaptador '{name}': {error}") from error

    append_log(job, f"Adaptador {name} borrado de adapters/, allowlist y caché de exportación.")


def rollback_adapter_runtime(name: str, allowed: bool, runtime_path: str | None = None) -> None:
    """Revierte en vLLM el cambio que ocurrió antes de actualizar la allowlist."""
    if allowed:
        post_vllm("/v1/unload_lora_adapter", {"lora_name": name})
    else:
        post_vllm(
            "/v1/load_lora_adapter",
            {"lora_name": name, "lora_path": runtime_path or f"/adapters/{name}"},
        )


def _env_file_target(env_file: Path) -> Path:
    """Un .env.local enlazado se conserva: se reescribe el archivo apuntado."""
    if env_file.is_symlink() and env_file.exists():
        return env_file.resolve()
    return env_file


def _replace_env_file(target: Path, content: str) -> None:
    """Reemplazo atómico sin ampliar permisos (FT-04).

    mkstemp crea el temporal con O_EXCL y modo 0600; si el archivo ya existía
    se copia su modo, de modo que uno privado sigue privado y uno nuevo nace 0600.
    """
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        if target.exists():
            shutil.copymode(target, temporary)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _allowlist_entry(lines: list[str]) -> tuple[int | None, list[str]]:
    """Índice de la línea LLM_ADAPTER_MODELS= y los nombres que declara."""
    entry_index = next(
        (index for index, line in enumerate(lines) if line.startswith("LLM_ADAPTER_MODELS=")),
        None,
    )
    if entry_index is None:
        return None, []
    names = [item.strip() for item in lines[entry_index].split("=", 1)[1].split(",") if item.strip()]
    return entry_index, names


def allowlisted_adapters() -> list[str]:
    """Adaptadores que el gateway ofrecerá según LLM_ADAPTER_MODELS en .env.local."""
    env_file = ROOT / ".env.local"
    try:
        text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    except OSError:
        return []
    return _allowlist_entry(text.splitlines())[1]


def set_adapter_allowed_in_env(name: str, allowed: bool, *, rollback_runtime: bool = False,
                               runtime_path: str | None = None) -> bool:
    env_file = ROOT / ".env.local"
    try:
        text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
        lines = text.splitlines()
        entry_index, current = _allowlist_entry(lines)
        if entry_index is None:
            if not allowed:
                return False
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(["# Adaptadores LoRA permitidos por el gateway.", f"LLM_ADAPTER_MODELS={name}"])
            changed = True
        else:
            updated = list(current)
            if allowed and name not in current:
                updated.append(name)
            if not allowed:
                updated = [item for item in updated if item != name]
            changed = updated != current
            if changed:
                lines[entry_index] = "LLM_ADAPTER_MODELS=" + ",".join(updated)

        if not changed:
            return False

        _replace_env_file(_env_file_target(env_file), "\n".join(lines).rstrip() + "\n")
        return True
    except OSError as error:
        if not rollback_runtime:
            raise RuntimeError(f"No fue posible actualizar {env_file.name}: {error}") from error
        try:
            rollback_adapter_runtime(name, allowed, runtime_path)
        except Exception as rollback_error:
            raise RuntimeError(
                f"No fue posible actualizar {env_file.name} y también falló el rollback de vLLM. "
                f"El estado del adaptador '{name}' puede ser inconsistente y requiere revisión manual. "
                f"Error de allowlist: {error}. Error de rollback: {rollback_error}"
            ) from error
        desired = "cargado" if allowed else "descargado"
        restored = "descargado" if allowed else "cargado"
        raise RuntimeError(
            f"vLLM había dejado el adaptador '{name}' {desired}, pero no fue posible actualizar {env_file.name}. "
            f"El cambio de vLLM fue revertido y el adaptador quedó {restored}; no se aplicó la operación."
        ) from error


def load_adapter_runtime(job: Job, name: str) -> str:
    """Valida la base, prepara la exportación y carga el LoRA en vLLM."""
    directory = ADAPTER_DIR / name
    if not (directory / "adapter_config.json").is_file() or not (directory / "adapter_model.safetensors").is_file():
        raise RuntimeError("El adaptador no contiene los archivos PEFT esperados.")
    validate_adapter_base(name)
    runtime_path = prepare_runtime_adapter(job, name)
    append_log(job, f"Cargando {name} en vLLM...")
    response = post_vllm(
        "/v1/load_lora_adapter",
        {"lora_name": name, "lora_path": runtime_path},
    )
    append_log(job, response or "Adaptador cargado.")
    return runtime_path


def reconcile_marker_path() -> Path:
    return RUNTIME_DIR / RECONCILE_MARKER_FILENAME


def _clear_reconcile_marker() -> None:
    try:
        reconcile_marker_path().unlink(missing_ok=True)
    except OSError:
        pass


def _check_allowlist_entry(name: str) -> str:
    """Errores propios de la entrada; solo estos permiten retirarla de la allowlist."""
    try:
        name = validate_adapter_name(name)
    except ValueError as error:
        raise InvalidAllowlistEntry(str(error)) from error
    directory = ADAPTER_DIR / name
    if directory.is_symlink() or not directory.is_dir():
        raise InvalidAllowlistEntry("el adaptador ya no existe en adapters/")
    if not (directory / "adapter_config.json").is_file() or not (directory / "adapter_model.safetensors").is_file():
        raise InvalidAllowlistEntry("El adaptador no contiene los archivos PEFT esperados.")
    try:
        adapter_training_base(name)
    except ValueError as error:
        raise InvalidAllowlistEntry(str(error)) from error
    # Aquí un ValueError vendría de la respuesta de vLLM: es infraestructura y se propaga.
    try:
        validate_adapter_base(name)
    except AdapterBaseMismatch as error:
        raise InvalidAllowlistEntry(str(error)) from error
    return name


def reconcile_adapter_allowlist(job: Job) -> dict[str, list[str]]:
    """Alinea LLM_ADAPTER_MODELS con vLLM después de (re)iniciarlo (FT-03).

    Reiniciar vLLM descarta los LoRA cargados dinámicamente. Cada adaptador de la
    allowlist que ya no está servido se recarga con la misma validación de base
    que "Cargar en vLLM". Solo se retira una entrada inválida en sí misma (el
    adaptador ya no existe, sus metadatos son inválidos o vLLM sirve otra base).
    Un fallo de infraestructura (intérprete o exportación, error de vLLM) nunca
    retira un adaptador válido: se conserva y la reconciliación termina en error.
    """
    summary: dict[str, list[str]] = {"kept": [], "reloaded": [], "removed": []}
    allowlisted = allowlisted_adapters()
    if not allowlisted:
        append_log(job, "LLM_ADAPTER_MODELS está vacía; no hay adaptadores que reconciliar.")
        _clear_reconcile_marker()
        return summary
    running, served = fetch_vllm_models()
    if not running:
        raise RuntimeError("vLLM no está disponible; no se puede reconciliar LLM_ADAPTER_MODELS.")
    append_log(job, f"Reconciliando LLM_ADAPTER_MODELS con vLLM: {', '.join(allowlisted)}")
    failed: list[str] = []
    for name in allowlisted:
        if name in served:
            summary["kept"].append(name)
            append_log(job, f"{name}: sigue cargado en vLLM.")
            continue
        try:
            load_adapter_runtime(job, _check_allowlist_entry(name))
        except JobCancelled:
            raise
        except InvalidAllowlistEntry as error:
            summary["removed"].append(name)
            append_log(job, f"{name}: se retira de LLM_ADAPTER_MODELS ({error}).")
            continue
        except Exception as error:
            failed.append(f"{name} ({error})")
            append_log(job, f"{name}: se conserva en LLM_ADAPTER_MODELS; no se pudo recargar ({error}).")
            continue
        summary["reloaded"].append(name)
        append_log(job, f"{name}: recargado en vLLM tras el reinicio.")
    for name in summary["removed"]:
        set_adapter_allowed_in_env(name, False)
    if summary["removed"]:
        append_log(job, "Si la web ya estaba activa, reiníciala para que relea .env.local.")
    if failed:
        raise RuntimeError(
            "No se pudo recargar en vLLM: " + "; ".join(failed)
            + "; se conservan en LLM_ADAPTER_MODELS hasta poder verificarlos. Vuelve a intentarlo cuando se resuelva."
        )
    _clear_reconcile_marker()
    return summary


def _run_reconcile_job(job: Job) -> None:
    global _LAST_AUTO_RECONCILE_FAILURE
    try:
        job.result = reconcile_adapter_allowlist(job)
    except JobCancelled:
        raise
    except Exception:
        _LAST_AUTO_RECONCILE_FAILURE = time.monotonic()
        raise
    _LAST_AUTO_RECONCILE_FAILURE = None


def maybe_start_pending_reconcile() -> Job | None:
    """Completa una reconciliación pendiente en cuanto vLLM vuelve a responder.

    La marca la dejan scripts/train-adapter*.sh al detener vLLM y la borra la
    reconciliación completa; persiste si una cancelación la interrumpió.
    """
    if not reconcile_marker_path().exists():
        return None
    if (_LAST_AUTO_RECONCILE_FAILURE is not None
            and time.monotonic() - _LAST_AUTO_RECONCILE_FAILURE < RECONCILE_RETRY_SECONDS):
        return None
    with STATE_LOCK:
        if CURRENT_JOB is not None and CURRENT_JOB.status == "running":
            return None
    running, _ = fetch_vllm_models()
    if not running:
        return None
    try:
        return start_job("reconcile-adapters", {})
    except RuntimeError:
        # Otra operación (otra GUI o CLI) tiene la GPU: se reintenta más tarde.
        return None


def watch_pending_reconcile(stop: threading.Event) -> None:
    while not stop.wait(RECONCILE_POLL_SECONDS):
        try:
            maybe_start_pending_reconcile()
        except Exception as error:
            print(f"Aviso: no se pudo iniciar la reconciliación pendiente: {error}", flush=True)


def setup_environment(job: Job, *, automatic: bool = False) -> None:
    python_command = shutil.which("python3.13") or shutil.which("python3") or shutil.which("python")
    if not python_command:
        raise RuntimeError("No se encontró Python 3 para crear trainer/.venv.")
    preferred = TRAINER_DIR / ".venv" / "bin" / "python"
    if not preferred.is_file():
        run_command(
            job,
            [python_command, "-m", "venv", str(TRAINER_DIR / ".venv")],
            label="Creando entorno virtual automáticamente" if automatic else "Creando entorno virtual",
        )
    python_path = str(trainer_python())
    run_command(job, [python_path, "-m", "pip", "install", "--upgrade", "pip"], label="Actualizando pip")
    run_command(
        job,
        [python_path, "-m", "pip", "install", "-r", str(TRAINER_DIR / "requirements.txt")],
        label="Instalando dependencias",
    )


def run_action(job: Job, payload: dict[str, Any]) -> None:
    action = job.action
    if action == "setup":
        # El entorno va a cambiar: la comprobación anterior deja de valer.
        clear_environment_check()
        setup_environment(job)
        verify_environment(job, "Comprobando CUDA y bitsandbytes")
        job.result = {"environmentReady": True}
        return
    if action == "check":
        if not trainer_python().is_file():
            raise RuntimeError("Primero prepara el entorno de fine-tuning.")
        verify_environment(job, "Comprobando CUDA y bitsandbytes")
        job.result = {"check": "ok"}
        return
    if action == "train":
        request = validate_training_payload(payload)
        ensure_adapter_dir_writable(job)
        if not trainer_python().is_file():
            setup_environment(job, automatic=True)
        verify_environment(job, "Verificando entorno")
        command = ["bash", str(ROOT / "scripts" / "train-adapter.sh"), *request.script_arguments()]
        training_env = environment_with_hf_token(request.hf_token)
        if request.hf_token:
            append_log(job, "Hugging Face: autenticación efímera habilitada para las descargas del entrenamiento.")
        run_command(job, command, env=training_env, label=f"Entrenando {request.name}")
        job.result = {"adapter": request.name, "dataset": request.dataset.name, "examples": request.examples}
        return
    if action == "start-vllm":
        model = validate_model(payload.get("model"))
        hf_token = validate_hf_token(payload.pop("hfToken", ""))
        ensure_adapter_dir_writable(job)
        fish = shutil.which("fish")
        if not fish:
            raise RuntimeError("No se encontró fish; la GUI necesita ./comandos.fish para iniciar vLLM y embeddings.")
        env = environment_with_hf_token(hf_token)
        env["VLLM_MODEL"] = model
        env["LLM_BRIDGE_NONINTERACTIVE"] = "1"
        python = trainer_python()
        if python.is_file():
            # La reconciliación posterior exporta con el intérprete de este backend.
            env["FINE_TUNE_PYTHON"] = str(python)
        run_command(
            job,
            [fish, str(ROOT / "comandos.fish"), "vllm", model],
            env=env,
            label=f"Iniciando vLLM + Ollama embeddings con {model}",
        )
        job.result = {
            "vllm": "started",
            "ollama": "started",
            "embeddings": "ready",
            "model": model,
        }
        return
    if action == "stop-vllm":
        run_command(
            job,
            ["docker", "compose", "-p", "llm-bridge", "stop", "vllm"],
            label="Deteniendo vLLM",
        )
        job.result = {"vllm": "stopped"}
        return
    if action == "verify-adapter":
        name = validate_adapter_name(payload.get("name"))
        loaded_adapter_path(name)
        base_model = validate_adapter_base(name)
        run_command(job, [str(trainer_python()), str(TRAINER_DIR / "verify_lora.py"),
                          "--base-model", base_model, "--adapter", name],
                    label="Comparando probabilidades: base repetida vs LoRA")
        append_log(job, "Efecto del LoRA detectado. Esto no mide la calidad de las respuestas.")
        job.result = {"adapter": name, "inferenceVerified": True}
        return
    if action == "reconcile-adapters":
        _run_reconcile_job(job)
        return
    if action == "delete-adapter":
        name = validate_adapter_name(payload.get("name"))
        delete_adapter(job, name)
        job.result = {"adapter": name, "deleted": True, "allowlistUpdated": True}
        return
    if action in {"load-adapter", "unload-adapter"}:
        name = validate_adapter_name(payload.get("name"))
        if action == "load-adapter":
            runtime_path = load_adapter_runtime(job, name)
            set_adapter_allowed_in_env(name, True, rollback_runtime=True)
            append_log(
                job,
                "Adaptador agregado a LLM_ADAPTER_MODELS en .env.local. Si la web ya estaba activa, reiníciala para que relea el entorno.",
            )
            job.result = {
                "adapter": name,
                "runtimeLoaded": True,
                "runtimePath": runtime_path,
                "inferenceVerified": False,
                "allowlistUpdated": True,
                "gatewayRestartRequired": True,
            }
        elif name in allowlisted_adapters() and not adapter_is_loaded(name):
            # Tras un reinicio de vLLM el adaptador puede seguir en la allowlist
            # sin estar servido: se permite retirarlo sin tocar vLLM (FT-03).
            append_log(job, f"{name} no está cargado en vLLM; solo se retira de LLM_ADAPTER_MODELS.")
            set_adapter_allowed_in_env(name, False)
            append_log(
                job,
                "Adaptador retirado de LLM_ADAPTER_MODELS. Si la web ya estaba activa, reiníciala para que relea el entorno.",
            )
            job.result = {
                "adapter": name,
                "runtimeLoaded": False,
                "allowlistUpdated": True,
                "gatewayRestartRequired": True,
            }
        else:
            runtime_path = loaded_adapter_path(name)
            append_log(job, f"Descargando {name} de vLLM...")
            response = post_vllm("/v1/unload_lora_adapter", {"lora_name": name})
            append_log(job, response or "Adaptador descargado.")
            set_adapter_allowed_in_env(name, False, rollback_runtime=True, runtime_path=runtime_path)
            append_log(
                job,
                "Adaptador retirado de LLM_ADAPTER_MODELS. Si la web ya estaba activa, reiníciala para que relea el entorno.",
            )
            job.result = {
                "adapter": name,
                "runtimeLoaded": False,
                "allowlistUpdated": True,
                "gatewayRestartRequired": True,
            }
        return
    raise ValueError("Acción no permitida.")


def job_worker(job: Job, payload: dict[str, Any]) -> None:
    try:
        run_action(job, payload)
        with STATE_LOCK:
            job.status = "succeeded"
    except JobCancelled:
        append_log(job, "Operación cancelada por el usuario.")
        with STATE_LOCK:
            job.status = "cancelled"
    except Exception as error:
        append_log(job, f"ERROR: {error}")
        with STATE_LOCK:
            job.status = "failed"
    finally:
        with STATE_LOCK:
            job.finished_at = time.time()
            job.process = None
            gpu_lock = job.gpu_lock
        if gpu_lock is not None:
            gpu_lock.release()


def start_job(action: str, payload: dict[str, Any]) -> Job:
    allowed = {"setup", "check", "train", "start-vllm", "stop-vllm", "load-adapter", "unload-adapter", "verify-adapter", "delete-adapter", "reconcile-adapters"}
    if action not in allowed:
        raise ValueError("Acción no permitida.")
    global CURRENT_JOB
    with STATE_LOCK:
        if CURRENT_JOB is not None and CURRENT_JOB.status == "running":
            raise RuntimeError("Ya hay una operación en curso.")
        external = runtime_lock.live_job_record(job_record_path())
        if external is not None:
            raise RuntimeError(external_job_message(external))
        # GpuLockBusy es un RuntimeError: el handler lo responde como 409.
        gpu_lock = runtime_lock.GpuLock.acquire(gpu_lock_path())
        job = Job(id=uuid.uuid4().hex, action=action, gpu_lock=gpu_lock)
        CURRENT_JOB = job
    threading.Thread(target=job_worker, args=(job, payload), daemon=True).start()
    return job


def escalate_cancellation(job: Job, process: subprocess.Popen[str]) -> None:
    """SIGINT, SIGTERM y SIGKILL al grupo del job hasta que no quede ningún proceso."""
    pgid = process.pid  # start_new_session=True: el hijo lidera su propio grupo.
    steps = (
        (signal.SIGINT, CANCEL_SIGINT_GRACE_SECONDS),
        (signal.SIGTERM, CANCEL_SIGTERM_GRACE_SECONDS),
        (signal.SIGKILL, 0.0),
    )
    for index, (signum, grace) in enumerate(steps):
        if not runtime_lock.process_group_alive(pgid):
            return
        if index == 0:
            append_log(job, f"Cancelación solicitada: enviando SIGINT al grupo de procesos {pgid}.")
        else:
            append_log(
                job,
                f"El grupo {pgid} sigue activo tras {steps[index - 1][1]:g} s; enviando {signum.name}.",
            )
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            return
        except PermissionError as error:
            append_log(job, f"No se pudo enviar {signum.name} al grupo {pgid}: {error}")
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not runtime_lock.process_group_alive(pgid):
                return
            time.sleep(0.05)


def start_cancel_escalation(job: Job, process: subprocess.Popen[str]) -> None:
    threading.Thread(target=escalate_cancellation, args=(job, process), daemon=True).start()


def cancel_job() -> bool:
    with STATE_LOCK:
        job = CURRENT_JOB
        if job is None or job.status != "running":
            return False
        already_requested = job.cancel_requested
        job.cancel_requested = True
        process = job.process
    if process is not None and not already_requested:
        start_cancel_escalation(job, process)
    return True


def stop_running_job() -> None:
    """Cancela y espera el job activo antes de que el panel termine."""
    with STATE_LOCK:
        job = CURRENT_JOB
        running = job is not None and job.status == "running"
    if not running or job is None:
        return
    print("Cancelando la operación en curso antes de cerrar el panel…", flush=True)
    cancel_job()
    limit = CANCEL_SIGINT_GRACE_SECONDS + CANCEL_SIGTERM_GRACE_SECONDS + CANCEL_OUTPUT_DRAIN_SECONDS + 5.0
    if _wait_job_finished(job, limit):
        return
    # La escalada corre en un hilo daemon que muere con el panel: antes de salir
    # se envía SIGKILL directamente al grupo para no dejar procesos huérfanos.
    with STATE_LOCK:
        process = job.process
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if _wait_job_finished(job, 5.0):
            return
    print("Aviso: la operación no terminó a tiempo; revisa los procesos manualmente.", flush=True)


def _wait_job_finished(job: Job, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with STATE_LOCK:
            if job.status != "running":
                return True
        time.sleep(0.1)
    return False


def report_external_job() -> None:
    """Comprueba al arrancar si quedó vivo el job de una sesión anterior."""
    record = runtime_lock.live_job_record(job_record_path())
    if record is not None:
        print(
            f"Aviso: {external_job_message(record)} Para detenerla: kill -INT -- -{record['pgid']}",
            flush=True,
        )


def _exit_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMBridgeFineTune/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _host_allowed(self) -> bool:
        host = (self.headers.get("host") or "").strip().lower()
        if host.startswith("["):
            hostname = host.split("]", 1)[0] + "]" if "]" in host else host
        else:
            hostname = host.split(":", 1)[0]
        return hostname in {"127.0.0.1", "localhost", "[::1]"}

    def _token_allowed(self) -> bool:
        return hmac.compare_digest(self.headers.get("x-fine-tune-token") or "", CSRF_TOKEN)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("content-length") or "0")
        except ValueError as error:
            raise ValueError("Content-Length inválido.") from error
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Solicitud vacía o demasiado grande.")
        raw = self.rfile.read(length)
        try:
            if json_too_deep(raw.decode("utf-8", errors="replace")):
                raise RecursionError
            payload = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("JSON inválido.") from error
        if not isinstance(payload, dict):
            raise ValueError("Se esperaba un objeto JSON.")
        return payload

    def _index(self) -> None:
        path = GUI_DIR / "index.html"
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
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

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._json({"error": "Host no permitido."}, 403)
            return
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            self._index()
        elif path == "/api/bootstrap":
            self._json({"token": CSRF_TOKEN, "status": status_payload()})
        elif path == "/api/status":
            self._json(status_payload())
        elif path == "/api/job":
            with STATE_LOCK:
                snapshot = job_snapshot(CURRENT_JOB)
            self._json({"job": snapshot})
        elif path == "/api/health":
            self._json({"ok": True, "localOnly": True})
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._json({"error": "Ruta no encontrada."}, 404)

    def do_POST(self) -> None:
        if not self._host_allowed():
            self._json({"error": "Host no permitido."}, 403)
            return
        if not self._token_allowed():
            self._json({"error": "Token local inválido."}, 403)
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/datasets":
                payload = self._read_json()
                filename = Path(str(payload.get("filename") or "")).name
                content = payload.get("content")
                if not SAFE_DATASET.fullmatch(filename):
                    raise ValueError("El archivo debe tener un nombre simple y extensión .jsonl.")
                if filename.casefold() in EVALUATION_DATASETS:
                    raise ValueError("evaluation.jsonl está reservado para evaluación y no es un dataset SFT.")
                if not isinstance(content, str):
                    raise ValueError("Contenido de dataset inválido.")
                encoded = content.encode("utf-8")
                if len(encoded) > 25 * 1024 * 1024:
                    self._json({"error": "El dataset supera 25 MB."}, 413)
                    return
                DATASET_DIR.mkdir(parents=True, exist_ok=True)
                destination = DATASET_DIR / filename
                temporary = DATASET_DIR / f".{filename}.{uuid.uuid4().hex}.tmp"
                try:
                    temporary.write_bytes(encoded)
                    summary = dataset_summary(temporary)
                    if summary.get("valid"):
                        os.replace(temporary, destination)
                finally:
                    # El temporal nunca sobrevive a la subida, falle o no (FT-10).
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                if not summary.get("valid"):
                    self._json({"error": summary.get("error", "Dataset inválido.")}, 422)
                    return
                self._json(
                    {
                        "dataset": {
                            "id": destination.relative_to(TRAINER_DIR).as_posix(),
                            "name": filename,
                        },
                        "validation": summary,
                    },
                    201,
                )
                return
            if path == "/api/actions":
                payload = self._read_json()
                job = start_job(str(payload.get("action") or ""), payload)
                self._json({"job": job_snapshot(job)}, 202)
                return
            if path == "/api/job/cancel":
                cancelled = cancel_job()
                self._json({"cancelled": cancelled}, 200 if cancelled else 409)
                return
            if path == "/api/shutdown":
                with STATE_LOCK:
                    running = CURRENT_JOB is not None and CURRENT_JOB.status == "running"
                if running:
                    raise RuntimeError(
                        "Hay una operación en curso. Cancélala o espera a que termine antes de cerrar el panel."
                    )
                self._json({"shuttingDown": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._json({"error": "Ruta no encontrada."}, 404)
        except ValueError as error:
            self._json({"error": str(error)}, 400)
        except RuntimeError as error:
            self._json({"error": str(error)}, 409)
        except OSError as error:
            self._json({"error": f"Error de sistema: {error}"}, 500)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    """ThreadingHTTPServer usa AF_INET; ``::1`` necesita AF_INET6 (FT-13)."""
    server_class = IPv6ThreadingHTTPServer if ":" in host else ThreadingHTTPServer
    return server_class((host, port), Handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI local de fine-tuning para LLM Bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3031)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Por seguridad, la GUI solo puede enlazarse a loopback.")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    report_external_job()
    server = create_server(args.host, args.port)
    # SIGTERM (kill, cierre de sesión) pasa por el mismo cierre ordenado que
    # Ctrl+C: el job activo se cancela y se espera antes de salir.
    signal.signal(signal.SIGTERM, _exit_on_signal)
    stop_watcher = threading.Event()
    threading.Thread(target=watch_pending_reconcile, args=(stop_watcher,), daemon=True).start()
    display_host = "[::1]" if ":" in args.host else "127.0.0.1"
    print(f"Fine-tuning GUI: http://{display_host}:{args.port}")
    print("Acceso restringido a la máquina local.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_watcher.set()
        stop_running_job()
        server.server_close()


if __name__ == "__main__":
    main()

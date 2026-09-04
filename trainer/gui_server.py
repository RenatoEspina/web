#!/usr/bin/env python3
"""Panel web local para automatizar el flujo de fine-tuning QLoRA."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
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

from dataset_validation import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT / "trainer"
GUI_DIR = TRAINER_DIR / "gui"
DATASET_DIR = TRAINER_DIR / "datasets"
EXAMPLE_DIR = TRAINER_DIR / "examples"
ADAPTER_DIR = ROOT / "adapters"
RUNTIME_DIR = ROOT / ".runtime"
MAX_BODY_BYTES = 40 * 1024 * 1024
MAX_LOG_LINES = 2500
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.jsonl$")
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
CUDA_CHECK = 'import torch; assert torch.cuda.is_available(), "CUDA no está disponible"; import bitsandbytes; from bitsandbytes.functional import quantize_4bit; x=torch.ones((2,2), device="cuda"); quantize_4bit(x, quant_type="nf4"); print(f"PyTorch: {torch.__version__}"); print(f"CUDA de PyTorch: {torch.version.cuda}"); print(f"GPU: {torch.cuda.get_device_name(0)}"); print(f"bitsandbytes: {bitsandbytes.__version__}"); print("bitsandbytes NF4: OK")'


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


STATE_LOCK = threading.Lock()
CURRENT_JOB: Job | None = None
CSRF_TOKEN = secrets.token_urlsafe(32)


class JobCancelled(RuntimeError):
    pass


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
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} debe ser entero.") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} debe estar entre {minimum} y {maximum}.")
    return value


def dataset_path(dataset_id: Any) -> Path:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        raise ValueError("Selecciona un dataset.")
    candidate = (TRAINER_DIR / dataset_id).resolve()
    if candidate.parent not in {DATASET_DIR.resolve(), EXAMPLE_DIR.resolve()} or candidate.suffix != ".jsonl":
        raise ValueError("Dataset fuera de las carpetas permitidas.")
    if not candidate.is_file():
        raise ValueError("El dataset seleccionado ya no existe.")
    return candidate


def dataset_summary(path: Path) -> dict[str, Any]:
    try:
        _, summary = load_jsonl(path)
        return dict(summary)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "error": str(error)}


def list_datasets() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, directory in (("Subido", DATASET_DIR), ("Ejemplo", EXAMPLE_DIR)):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            result.append({"id": path.relative_to(TRAINER_DIR).as_posix(), "name": path.name, "kind": label, "bytes": path.stat().st_size})
    return result


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
        result.append({"name": directory.name, "baseModel": manifest.get("baseModel"), "createdAt": manifest.get("createdAt"), "examples": manifest.get("examples"), "rank": (manifest.get("parameters") or {}).get("rank"), "metrics": manifest.get("metrics") or {}})
    return result


def fetch_vllm_models() -> tuple[bool, list[str]]:
    try:
        with urlopen("http://127.0.0.1:8000/v1/models", timeout=0.8) as response:
            payload = json.load(response)
        models = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
        return True, models
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False, []


def job_snapshot(job: Job | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {"id": job.id, "action": job.action, "status": job.status, "startedAt": job.started_at, "finishedAt": job.finished_at, "returncode": job.returncode, "logs": list(job.logs), "result": job.result, "cancelRequested": job.cancel_requested}


def status_payload() -> dict[str, Any]:
    running, models = fetch_vllm_models()
    with STATE_LOCK:
        job = job_snapshot(CURRENT_JOB)
    return {"localOnly": True, "environmentReady": trainer_python().is_file(), "dockerCli": shutil.which("docker") is not None, "bashCli": shutil.which("bash") is not None, "vllmRunning": running, "vllmModels": models, "datasets": list_datasets(), "adapters": list_adapters(), "job": job}


def append_log(job: Job, line: str) -> None:
    clean = line.rstrip("\n")
    if clean:
        with STATE_LOCK:
            job.logs.append(clean)


def run_command(job: Job, command: list[str], *, env: dict[str, str] | None = None, label: str | None = None) -> None:
    if job.cancel_requested:
        raise JobCancelled()
    if label:
        append_log(job, f"\n== {label} ==")
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
    with STATE_LOCK:
        job.process = process
    assert process.stdout is not None
    for line in process.stdout:
        append_log(job, line)
    returncode = process.wait()
    with STATE_LOCK:
        job.process = None
        job.returncode = returncode
    if job.cancel_requested:
        raise JobCancelled()
    if returncode != 0:
        raise RuntimeError(f"El proceso terminó con código {returncode}.")


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


def register_adapter_in_env(name: str) -> None:
    env_file = ROOT / ".env.local"
    try:
        text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    except OSError as error:
        raise RuntimeError(f"No fue posible leer {env_file.name}: {error}") from error
    lines = text.splitlines()
    updated = False
    for index, line in enumerate(lines):
        if line.startswith("LLM_ADAPTER_MODELS="):
            current = [item.strip() for item in line.split("=", 1)[1].split(",") if item.strip()]
            if name not in current:
                current.append(name)
            lines[index] = "LLM_ADAPTER_MODELS=" + ",".join(current)
            updated = True
            break
    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["# Adaptadores LoRA permitidos por el gateway.", f"LLM_ADAPTER_MODELS={name}"])
    temporary = env_file.with_suffix(env_file.suffix + ".tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, env_file)


def post_vllm(path: str, payload: dict[str, str]) -> str:
    request = Request(f"http://127.0.0.1:8000{path}", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace").strip()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"vLLM respondió HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError("vLLM no está disponible en 127.0.0.1:8000.") from error


def setup_environment(job: Job, *, automatic: bool = False) -> None:
    python_command = shutil.which("python3.13") or shutil.which("python3") or shutil.which("python")
    if not python_command:
        raise RuntimeError("No se encontró Python 3 para crear trainer/.venv.")
    preferred = TRAINER_DIR / ".venv" / "bin" / "python"
    if not preferred.is_file():
        run_command(job, [python_command, "-m", "venv", str(TRAINER_DIR / ".venv")], label="Creando entorno virtual automáticamente" if automatic else "Creando entorno virtual")
    python_path = str(trainer_python())
    run_command(job, [python_path, "-m", "pip", "install", "--upgrade", "pip"], label="Actualizando pip")
    run_command(job, [python_path, "-m", "pip", "install", "-r", str(TRAINER_DIR / "requirements.txt")], label="Instalando dependencias")


def run_action(job: Job, payload: dict[str, Any]) -> None:
    action = job.action
    if action == "setup":
        setup_environment(job)
        run_command(job, [str(trainer_python()), "-c", CUDA_CHECK], label="Comprobando CUDA y bitsandbytes")
        job.result = {"environmentReady": True}
        return
    if action == "check":
        if not trainer_python().is_file():
            raise RuntimeError("Primero prepara el entorno de fine-tuning.")
        run_command(job, [str(trainer_python()), "-c", CUDA_CHECK], label="Comprobando CUDA y bitsandbytes")
        job.result = {"check": "ok"}
        return
    if action == "train":
        dataset = dataset_path(payload.get("dataset"))
        _, summary = load_jsonl(dataset)
        name = validate_adapter_name(payload.get("name"))
        model = validate_model(payload.get("model"))
        rank = int_arg(payload, "rank", 16, 8, 32)
        if rank not in {8, 16, 32}:
            raise ValueError("rank debe ser 8, 16 o 32.")
        alpha = int_arg(payload, "alpha", 32, 1, 256)
        epochs = number_arg(payload, "epochs", 3.0, 0.1, 20.0)
        dropout = number_arg(payload, "dropout", 0.05, 0.0, 0.5)
        learning_rate = number_arg(payload, "learningRate", 2e-4, 1e-7, 0.1)
        batch_size = int_arg(payload, "batchSize", 1, 1, 32)
        gradient_accumulation = int_arg(payload, "gradientAccumulation", 8, 1, 256)
        max_length = int_arg(payload, "maxLength", 1024, 128, 16384)
        seed = int_arg(payload, "seed", 42, 0, 2_147_483_647)
        if not trainer_python().is_file():
            setup_environment(job, automatic=True)
        run_command(job, [str(trainer_python()), "-c", CUDA_CHECK], label="Verificando entorno")
        command = ["bash", str(ROOT / "scripts" / "train-adapter.sh"), str(dataset), name, "--model", model, "--rank", str(rank), "--alpha", str(alpha), "--dropout", str(dropout), "--epochs", str(epochs), "--learning-rate", str(learning_rate), "--batch-size", str(batch_size), "--gradient-accumulation", str(gradient_accumulation), "--max-length", str(max_length), "--seed", str(seed)]
        run_command(job, command, label=f"Entrenando {name}")
        job.result = {"adapter": name, "dataset": dataset.name, "examples": summary["examples"]}
        return
    if action == "start-vllm":
        model = validate_model(payload.get("model"))
        token = str(payload.get("hfToken") or "").strip()
        env = os.environ.copy()
        env["VLLM_MODEL"] = model
        if token:
            env["HF_TOKEN"] = token
        run_command(job, ["docker", "compose", "-p", "llm-bridge", "up", "-d", "vllm"], env=env, label=f"Iniciando vLLM con {model}")
        wait_for_vllm(job, timeout=900)
        job.result = {"vllm": "started", "model": model}
        return
    if action == "stop-vllm":
        run_command(job, ["docker", "compose", "-p", "llm-bridge", "stop", "vllm"], label="Deteniendo vLLM")
        job.result = {"vllm": "stopped"}
        return
    if action in {"load-adapter", "unload-adapter"}:
        name = validate_adapter_name(payload.get("name"))
        directory = ADAPTER_DIR / name
        if action == "load-adapter":
            if not (directory / "adapter_config.json").is_file() or not (directory / "adapter_model.safetensors").is_file():
                raise RuntimeError("El adaptador no contiene los archivos PEFT esperados.")
            append_log(job, f"Cargando {name} en vLLM...")
            response = post_vllm("/v1/load_lora_adapter", {"lora_name": name, "lora_path": f"/adapters/{name}"})
            append_log(job, response or "Adaptador cargado.")
            register_adapter_in_env(name)
            append_log(job, "Adaptador agregado a LLM_ADAPTER_MODELS en .env.local. Si la web ya estaba activa, reiníciala para que relea el entorno.")
            job.result = {"adapter": name, "loaded": True, "gatewayRegistered": True}
        else:
            append_log(job, f"Descargando {name} de vLLM...")
            response = post_vllm("/v1/unload_lora_adapter", {"lora_name": name})
            append_log(job, response or "Adaptador descargado.")
            job.result = {"adapter": name, "loaded": False}
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


def start_job(action: str, payload: dict[str, Any]) -> Job:
    allowed = {"setup", "check", "train", "start-vllm", "stop-vllm", "load-adapter", "unload-adapter"}
    if action not in allowed:
        raise ValueError("Acción no permitida.")
    global CURRENT_JOB
    with STATE_LOCK:
        if CURRENT_JOB is not None and CURRENT_JOB.status == "running":
            raise RuntimeError("Ya hay una operación en curso.")
        job = Job(id=uuid.uuid4().hex, action=action)
        CURRENT_JOB = job
    threading.Thread(target=job_worker, args=(job, payload), daemon=True).start()
    return job


def cancel_job() -> bool:
    with STATE_LOCK:
        job = CURRENT_JOB
        if job is None or job.status != "running":
            return False
        job.cancel_requested = True
        process = job.process
    if process is not None and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
    return True


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
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
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
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
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
                if not isinstance(content, str):
                    raise ValueError("Contenido de dataset inválido.")
                encoded = content.encode("utf-8")
                if len(encoded) > 25 * 1024 * 1024:
                    self._json({"error": "El dataset supera 25 MB."}, 413)
                    return
                DATASET_DIR.mkdir(parents=True, exist_ok=True)
                destination = DATASET_DIR / filename
                temporary = DATASET_DIR / f".{filename}.{uuid.uuid4().hex}.tmp"
                temporary.write_bytes(encoded)
                summary = dataset_summary(temporary)
                if not summary.get("valid"):
                    temporary.unlink(missing_ok=True)
                    self._json({"error": summary.get("error", "Dataset inválido.")}, 422)
                    return
                os.replace(temporary, destination)
                self._json({"dataset": {"id": destination.relative_to(TRAINER_DIR).as_posix(), "name": filename}, "validation": summary}, 201)
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
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Fine-tuning GUI: http://127.0.0.1:{args.port}")
    print("Acceso restringido a la máquina local.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

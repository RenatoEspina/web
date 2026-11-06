"""Exclusión mutua de GPU/vLLM entre procesos y registro del job activo.

El mismo archivo ``.runtime/gpu.lock`` lo toman las dos GUIs (``fcntl.flock``),
``scripts/train-adapter*.sh`` y, a través de ellos, ``./comandos.fish
fine-tune-train`` (``flock`` de util-linux). Los locks de ``flock(2)`` pertenecen
a la descripción de archivo abierta: los subprocesos que heredan el descriptor
mantienen el lock aunque el proceso que lo tomó termine, así que un entrenamiento
que sobrevive a su panel sigue bloqueando nuevas operaciones de GPU.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

LOCK_FILENAME = "gpu.lock"
JOB_RECORD_FILENAME = "gpu-job.json"
# Los scripts verifican que este descriptor heredado apunte al mismo lock antes
# de confiar en él (ver el bloque "lock GPU/vLLM" de scripts/train-adapter*.sh).
LOCK_FD_ENV = "LLM_BRIDGE_GPU_LOCK_FD"
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")


class GpuLockBusy(RuntimeError):
    """Otro proceso (GUI o CLI) ya tiene el lock de GPU/vLLM."""


class GpuLock:
    def __init__(self, fd: int, path: Path) -> None:
        self.fd: int | None = fd
        self.path = path

    @classmethod
    def acquire(cls, path: Path) -> "GpuLock":
        """Toma el lock sin bloquear; falla con GpuLockBusy si está ocupado."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise GpuLockBusy(
                "Otra operación de GPU/vLLM está en curso (otra GUI o un entrenamiento por CLI; "
                f"lock {path}). Espera a que termine antes de iniciar una nueva."
            ) from error
        except OSError:
            os.close(fd)
            raise
        return cls(fd, path)

    def release(self) -> None:
        # Solo se cierra el descriptor: LOCK_UN liberaría también el lock que
        # conservan los subprocesos que heredaron la misma descripción abierta.
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def boot_id() -> str | None:
    try:
        return BOOT_ID_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_private_json(path: Path, record: dict[str, Any]) -> None:
    """Escribe un registro JSON de forma atómica y legible solo por el usuario."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def read_job_record(path: Path) -> dict[str, Any] | None:
    record = read_json_record(path)
    if record is None or not isinstance(record.get("pgid"), int) or record["pgid"] <= 1:
        return None
    return record


def remove_job_record(path: Path, pgid: int | None = None) -> None:
    """Borra el registro; con ``pgid`` solo si sigue perteneciendo a ese grupo."""
    if pgid is not None:
        record = read_job_record(path)
        if record is not None and record.get("pgid") != pgid:
            return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def live_job_record(path: Path, *, remove_stale: bool = True) -> dict[str, Any] | None:
    """Devuelve el registro si su grupo de procesos sigue vivo en este arranque."""
    if not path.exists():
        return None
    record = read_job_record(path)
    current_boot = boot_id()
    alive = (
        record is not None
        and (record.get("bootId") is None or current_boot is None or record.get("bootId") == current_boot)
        and process_group_alive(record["pgid"])
    )
    if alive:
        return record
    if remove_stale:
        remove_job_record(path, record["pgid"] if record is not None else None)
    return None

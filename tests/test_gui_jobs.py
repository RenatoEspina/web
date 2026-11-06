"""Regresiones de FT-01, FT-02 y FT-05: lock de GPU, shutdown y cancelación.

Los tests sirven ``gui_server.Handler`` en un puerto efímero con ``.runtime/``
aislado y reemplazan ``run_action`` por subprocesos Python inofensivos: no se
inicia Docker, vLLM ni entrenamiento.
"""
import fcntl
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
TRAINER_DIR = REPO / "trainer"
sys.path.insert(0, str(TRAINER_DIR))
import gui_server  # noqa: E402

SLEEPER = "import os, time; print('ready', os.getpid(), flush=True); time.sleep(30)"
STUBBORN = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', os.getpid(), flush=True)\n"
    "time.sleep(30)\n"
)
DETACHED_STDOUT = (
    "import os, subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
    "print('grandchild', child.pid, flush=True)\n"
    "print('ready', os.getpid(), flush=True)\n"
    "time.sleep(30)\n"
)
# El líder termina con SIGINT, pero un worker de su mismo grupo lo ignora (como
# los workers de accelerate o Axolotl) y conserva el stdout heredado.
LEADER_WITH_STUBBORN_WORKER = (
    "import os, subprocess, sys, time\n"
    "worker = subprocess.Popen([sys.executable, '-c', "
    "'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)'])\n"
    "print('worker', worker.pid, flush=True)\n"
    "print('ready', os.getpid(), flush=True)\n"
    "try:\n"
    "    time.sleep(30)\n"
    "except KeyboardInterrupt:\n"
    "    pass\n"
)
LOCK_INHERITED = (
    'if [[ -n "${LLM_BRIDGE_GPU_LOCK_FD:-}" && "/dev/fd/$LLM_BRIDGE_GPU_LOCK_FD" -ef "$1" ]] '
    '&& flock -n "$LLM_BRIDGE_GPU_LOCK_FD"; then echo "lock heredado"; else echo "lock ausente"; fi'
)


def wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def kill_quietly(pid, *, group=False):
    try:
        if group:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


class LiveGuiTestCase(unittest.TestCase):
    """Servidor HTTP real del panel, con estado aislado en un directorio temporal."""

    grace = 0.3

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / ".runtime"
        self.lock_path = self.runtime / "gpu.lock"
        self.record_path = self.runtime / "gpu-job.json"
        self.spawned = []
        for name, value in (
            ("ROOT", self.root),
            ("RUNTIME_DIR", self.runtime),
            ("ADAPTER_DIR", self.root / "adapters"),
            ("CANCEL_SIGINT_GRACE_SECONDS", self.grace),
            ("CANCEL_SIGTERM_GRACE_SECONDS", self.grace),
            ("CANCEL_OUTPUT_DRAIN_SECONDS", self.grace),
        ):
            patcher = mock.patch.object(gui_server, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(gui_server, "fetch_vllm_models", return_value=(False, []))
        patcher.start()
        self.addCleanup(patcher.stop)
        gui_server.CURRENT_JOB = None
        self.addCleanup(self._finish_jobs)

        self.server = gui_server.ThreadingHTTPServer(("127.0.0.1", 0), gui_server.Handler)
        self.port = self.server.server_address[1]
        self.serving = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.serving.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        if self.serving.is_alive():
            self.server.shutdown()
        self.server.server_close()

    def _finish_jobs(self):
        job = gui_server.CURRENT_JOB
        if job is not None and job.process is not None:
            kill_quietly(job.process.pid, group=True)
        for pid in self.spawned:
            kill_quietly(pid, group=True)
            kill_quietly(pid)
        if job is not None:
            wait_until(lambda: job.status != "running", timeout=3)
        gui_server.CURRENT_JOB = None

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": f"127.0.0.1:{self.port}", "X-Fine-Tune-Token": gui_server.CSRF_TOKEN}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(data))
        try:
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
        finally:
            connection.close()

    def use_command(self, *command):
        def fake_run_action(job, payload):
            gui_server.run_command(job, list(command), label="Comando de prueba")

        patcher = mock.patch.object(gui_server, "run_action", side_effect=fake_run_action)
        patcher.start()
        self.addCleanup(patcher.stop)

    def job(self):
        status, body = self.request("GET", "/api/job")
        self.assertEqual(status, 200)
        return body["job"]

    def start_check(self):
        status, body = self.request("POST", "/api/actions", {"action": "check"})
        self.assertEqual(status, 202, body)
        return body["job"]

    def wait_for_log(self, text, timeout=5.0):
        found = wait_until(lambda: any(text in line for line in (self.job() or {}).get("logs", [])), timeout=timeout)
        self.assertTrue(found, f"No apareció {text!r} en los logs: {self.job()}")

    def wait_finished(self, timeout):
        return wait_until(lambda: (self.job() or {}).get("status") not in {None, "running"}, timeout=timeout)

    def running_process_pid(self):
        job = gui_server.CURRENT_JOB
        pid = wait_until(lambda: job.process.pid if job and job.process else None)
        self.assertIsNotNone(pid)
        self.spawned.append(pid)
        return pid

    def hold_external_lock(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor

    def lock_is_free(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            os.close(descriptor)
        return True


class GpuLockTests(LiveGuiTestCase):
    """FT-01: un lock de GPU/vLLM compartido entre procesos."""

    def test_action_is_rejected_while_another_process_holds_the_gpu_lock(self):
        self.use_command(sys.executable, "-c", SLEEPER)
        self.hold_external_lock()

        status, body = self.request("POST", "/api/actions", {"action": "check"})

        self.assertEqual(status, 409)
        self.assertIn("GPU/vLLM", body["error"])
        self.assertIn("gpu.lock", body["error"])
        self.assertIsNone(self.job())

    def test_job_holds_the_lock_until_it_finishes(self):
        self.use_command(sys.executable, "-c", SLEEPER)
        self.start_check()
        self.wait_for_log("ready")

        self.assertFalse(self.lock_is_free())

        self.assertEqual(self.request("POST", "/api/job/cancel", {}), (200, {"cancelled": True}))
        self.assertTrue(self.wait_finished(timeout=5))
        self.assertTrue(wait_until(self.lock_is_free, timeout=3))

    def test_free_lock_keeps_previous_action_flow(self):
        self.use_command(sys.executable, "-c", "print('hecho')")

        job = self.start_check()

        self.assertEqual(job["status"], "running")
        self.assertTrue(self.wait_finished(timeout=5))
        finished = self.job()
        self.assertEqual(finished["status"], "succeeded")
        self.assertIn("hecho", finished["logs"])
        self.assertEqual(finished["returncode"], 0)
        self.assertTrue(self.lock_is_free())

    def test_axolotl_gui_uses_the_same_lock(self):
        self.hold_external_lock()
        script = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            sys.path.insert(0, sys.argv[1])
            import gui_server_axolotl
            core = gui_server_axolotl.core
            core.RUNTIME_DIR = Path(sys.argv[2])
            try:
                core.start_job("check", {})
            except RuntimeError as error:
                print("rechazado:", error)
            else:
                print("aceptado")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script, str(TRAINER_DIR), str(self.runtime)],
            capture_output=True, text=True, timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rechazado:", result.stdout)
        self.assertIn(str(self.lock_path), result.stdout)

    def test_job_subprocesses_inherit_the_lock_descriptor(self):
        self.use_command("bash", "-c", LOCK_INHERITED, "lock-check", str(self.lock_path))

        self.start_check()

        self.assertTrue(self.wait_finished(timeout=5))
        logs = self.job()["logs"]
        self.assertIn("lock heredado", logs)
        self.assertNotIn("lock ausente", logs)


class ShutdownTests(LiveGuiTestCase):
    """FT-02: cerrar el panel no deja huérfano el job."""

    def test_shutdown_is_rejected_while_a_job_is_running(self):
        self.use_command(sys.executable, "-c", SLEEPER)
        self.start_check()
        self.wait_for_log("ready")
        pid = self.running_process_pid()

        status, body = self.request("POST", "/api/shutdown", {})

        self.assertEqual(status, 409)
        self.assertIn("operación en curso", body["error"])
        self.assertEqual(self.request("GET", "/api/health"), (200, {"ok": True, "localOnly": True}))
        self.assertEqual(self.job()["status"], "running")
        self.assertTrue(pid_alive(pid))

    def test_shutdown_without_job_keeps_stopping_the_server(self):
        status, body = self.request("POST", "/api/shutdown", {})

        self.assertEqual((status, body), (200, {"shuttingDown": True}))
        self.serving.join(timeout=5)
        self.assertFalse(self.serving.is_alive())

    def test_running_job_persists_its_process_group(self):
        self.use_command(sys.executable, "-c", SLEEPER)
        job = self.start_check()
        self.wait_for_log("ready")
        pid = self.running_process_pid()

        record = json.loads(self.record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["pgid"], pid)
        self.assertEqual(os.getpgid(pid), pid)
        self.assertEqual(record["action"], "check")
        self.assertEqual(record["jobId"], job["id"])
        self.assertEqual(record["guiPid"], os.getpid())
        self.assertEqual(self.record_path.stat().st_mode & 0o777, 0o600)

        self.request("POST", "/api/job/cancel", {})
        self.assertTrue(self.wait_finished(timeout=5))
        self.assertFalse(self.record_path.exists())

    def test_live_process_group_from_previous_session_blocks_new_actions(self):
        self.use_command(sys.executable, "-c", "print('hecho')")
        orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.spawned.append(orphan.pid)
        self.addCleanup(orphan.wait)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text(json.dumps({
            "pgid": orphan.pid, "pid": orphan.pid, "action": "train", "jobId": "anterior",
            "guiPid": 999999, "bootId": boot_id(), "startedAt": time.time(),
        }), encoding="utf-8")

        status, body = self.request("POST", "/api/actions", {"action": "check"})

        self.assertEqual(status, 409)
        self.assertIn(f"PGID {orphan.pid}", body["error"])
        self.assertIn("train", body["error"])
        self.assertIsNone(self.job())
        status, payload = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(payload["externalJob"]["pgid"], orphan.pid)

        orphan.kill()
        orphan.wait()
        self.start_check()
        self.assertTrue(self.wait_finished(timeout=5))
        self.assertEqual(self.job()["status"], "succeeded")
        self.assertFalse(self.record_path.exists())


class CancellationTests(LiveGuiTestCase):
    """FT-05: SIGINT -> SIGTERM -> SIGKILL sobre el grupo de procesos."""

    def test_cancel_escalates_to_sigkill_when_signals_are_ignored(self):
        self.use_command(sys.executable, "-c", STUBBORN)
        self.start_check()
        self.wait_for_log("ready")
        pid = self.running_process_pid()

        self.assertEqual(self.request("POST", "/api/job/cancel", {}), (200, {"cancelled": True}))

        self.assertTrue(self.wait_finished(timeout=5), self.job())
        job = self.job()
        self.assertEqual(job["status"], "cancelled")
        self.assertTrue(job["cancelRequested"])
        joined = "\n".join(job["logs"])
        self.assertIn("SIGINT", joined)
        self.assertIn("SIGTERM", joined)
        self.assertIn("SIGKILL", joined)
        self.assertIn("Operación cancelada por el usuario.", job["logs"])
        self.assertFalse(pid_alive(pid))
        self.assertTrue(wait_until(self.lock_is_free, timeout=3))

    def test_cancel_ends_job_even_if_a_detached_process_keeps_stdout_open(self):
        self.use_command(sys.executable, "-c", DETACHED_STDOUT)
        self.start_check()
        self.wait_for_log("ready")
        self.running_process_pid()
        grandchild = int(next(line.split()[1] for line in self.job()["logs"] if line.startswith("grandchild")))
        self.spawned.append(grandchild)

        self.request("POST", "/api/job/cancel", {})

        self.assertTrue(self.wait_finished(timeout=5), self.job())
        self.assertEqual(self.job()["status"], "cancelled")
        self.assertTrue(pid_alive(grandchild))

    def test_cooperative_process_is_cancelled_with_sigint_only(self):
        with mock.patch.object(gui_server, "CANCEL_SIGINT_GRACE_SECONDS", 5.0, create=True):
            self.use_command(sys.executable, "-c", SLEEPER)
            self.start_check()
            self.wait_for_log("ready")
            pid = self.running_process_pid()
            started = time.monotonic()

            self.request("POST", "/api/job/cancel", {})

            self.assertTrue(self.wait_finished(timeout=4))
        self.assertLess(time.monotonic() - started, 4.0)
        job = self.job()
        self.assertEqual(job["status"], "cancelled")
        self.assertNotIn("SIGTERM", "\n".join(job["logs"]))
        self.assertFalse(pid_alive(pid))

    def test_cancel_without_running_job_keeps_returning_409(self):
        self.assertEqual(self.request("POST", "/api/job/cancel", {}), (409, {"cancelled": False}))

    def test_job_is_not_finished_while_a_worker_of_its_group_survives(self):
        with mock.patch.object(gui_server, "CANCEL_SIGINT_GRACE_SECONDS", 1.5, create=True):
            self.use_command(sys.executable, "-c", LEADER_WITH_STUBBORN_WORKER)
            self.start_check()
            self.wait_for_log("ready")
            leader = self.running_process_pid()
            worker = int(next(line.split()[1] for line in self.job()["logs"] if line.startswith("worker")))
            self.spawned.append(worker)

            self.request("POST", "/api/job/cancel", {})
            self.assertTrue(wait_until(lambda: not pid_alive(leader), timeout=3))
            time.sleep(2 * self.grace)

            self.assertTrue(pid_alive(worker))
            self.assertEqual(self.job()["status"], "running")
            self.assertEqual(json.loads(self.record_path.read_text(encoding="utf-8"))["pgid"], leader)
            status, body = self.request("POST", "/api/shutdown", {})
            self.assertEqual(status, 409)
            self.assertIn("operación en curso", body["error"])

            self.assertTrue(self.wait_finished(timeout=5), self.job())
        job = self.job()
        self.assertEqual(job["status"], "cancelled")
        self.assertIn("SIGTERM", "\n".join(job["logs"]))
        self.assertFalse(pid_alive(worker))
        self.assertFalse(self.record_path.exists())


@unittest.skipUnless(shutil.which("flock"), "requiere flock (util-linux)")
class CancelledTrainingReconcileTests(LiveGuiTestCase):
    """FT-03: cancelar un entrenamiento con vLLM activo no deja la allowlist desincronizada."""

    grace = 1.0

    def fake(self, directory, name, body):
        path = directory / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_cancelled_training_reconciles_when_vllm_is_back(self):
        (self.root / "scripts").mkdir()
        shutil.copy2(REPO / "scripts" / "train-adapter.sh", self.root / "scripts" / "train-adapter.sh")
        fakebin = self.root / "fakebin"
        fakebin.mkdir()
        events = self.root / "events.log"
        self.fake(fakebin, "docker", textwrap.dedent(f"""\
            printf 'docker %s\\n' "$*" >> {events}
            if [[ "$*" == "compose -p llm-bridge ps --status running --services" ]]; then echo vllm; fi
            exit 0
        """))
        python = self.fake(fakebin, "trainer-python", textwrap.dedent(f"""\
            printf 'python %s\\n' "$*" >> {events}
            case "$1" in
              -) cat >/dev/null; echo "Org/Base" ;;
              trainer/train.py) echo TRAIN_STARTED; sleep 30 ;;
              trainer/runtime_adapters.py) echo RECONCILE_WAITING; sleep 30 ;;
            esac
            exit 0
        """))
        env_file = self.root / ".env.local"
        env_file.write_text("LLM_ADAPTER_MODELS=prev-lora\n", encoding="utf-8")
        environment = {"PATH": f"{fakebin}{os.pathsep}{os.environ.get('PATH', '')}", "FINE_TUNE_PYTHON": str(python)}
        with mock.patch.dict(os.environ, environment), mock.patch.object(gui_server, "trainer_python", return_value=python):
            status, body = self.request("POST", "/api/actions", {
                "action": "train", "dataset": "examples/minimal.jsonl", "name": "nuevo"})
            self.assertEqual(status, 202, body)
            self.wait_for_log("TRAIN_STARTED", timeout=10)
            self.request("POST", "/api/job/cancel", {})
            self.assertTrue(self.wait_finished(timeout=15), self.job())

        self.assertEqual(self.job()["status"], "cancelled")
        self.assertIn("docker compose -p llm-bridge up -d vllm", events.read_text(encoding="utf-8"))
        self.assertEqual(env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=prev-lora\n")
        self.assertTrue((self.runtime / "adapters-reconcile-pending").exists())

        with mock.patch.object(gui_server, "fetch_vllm_models", return_value=(True, ["Org/Base"])), \
                mock.patch.object(gui_server, "_LAST_AUTO_RECONCILE_FAILURE", None, create=True):
            job = gui_server.maybe_start_pending_reconcile()
            self.assertIsNotNone(job)
            self.assertTrue(self.wait_finished(timeout=5))

        self.assertEqual(self.job()["action"], "reconcile-adapters")
        self.assertEqual(self.job()["status"], "succeeded")
        self.assertEqual(env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")
        self.assertFalse((self.runtime / "adapters-reconcile-pending").exists())


HARNESS = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, sys.argv[1])
    import gui_server
    from pathlib import Path

    root = Path(sys.argv[2])
    gui_server.ROOT = root
    gui_server.RUNTIME_DIR = root / ".runtime"
    gui_server.ADAPTER_DIR = root / "adapters"
    gui_server.CANCEL_SIGINT_GRACE_SECONDS = 0.5
    gui_server.CANCEL_SIGTERM_GRACE_SECONDS = 0.5
    gui_server.CANCEL_OUTPUT_DRAIN_SECONDS = 0.5
    job = None
    if sys.argv[3] in {"job", "worker"}:
        script = "import time; time.sleep(30)" if sys.argv[3] == "job" else sys.argv[4]
        if sys.argv[3] == "worker":
            # Más gracia para SIGINT que margen de stdout: el worker sobrevive a SIGINT.
            gui_server.CANCEL_SIGINT_GRACE_SECONDS = 2.0
        def fake_run_action(job, payload):
            gui_server.run_command(job, [sys.executable, "-c", script])
        gui_server.run_action = fake_run_action
        job = gui_server.start_job("check", {})
        while job.process is None:
            time.sleep(0.02)
        print("child", job.process.pid, flush=True)
        if sys.argv[3] == "worker":
            while not any(line.startswith("worker") for line in list(job.logs)):
                time.sleep(0.02)
            print(next(line for line in list(job.logs) if line.startswith("worker")), flush=True)
    sys.argv = ["gui_server.py", "--port", "0"]
    try:
        gui_server.main()
    finally:
        if job is not None:
            print("job-status", job.status, flush=True)
    """
)


class ProcessLifecycleTests(unittest.TestCase):
    """FT-02: SIGTERM y arranque del panel como proceso real."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.record_path = self.root / ".runtime" / "gpu-job.json"
        self.harness = self.root / "harness.py"
        self.harness.write_text(HARNESS, encoding="utf-8")

    def launch(self, mode, *extra):
        process = subprocess.Popen(
            [sys.executable, "-u", str(self.harness), str(TRAINER_DIR), str(self.root), mode, *extra],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.addCleanup(lambda: (process.poll() is None and process.kill(), process.wait()))
        lines = []
        for line in process.stdout:
            lines.append(line.strip())
            if line.startswith("Fine-tuning GUI:"):
                break
        return process, lines

    def test_sigterm_cancels_running_job_before_exiting(self):
        process, lines = self.launch("job")
        child = int(next(line.split()[1] for line in lines if line.startswith("child")))
        self.addCleanup(kill_quietly, child, group=True)

        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)

        self.assertEqual(process.returncode, 143, output)
        self.assertIn("job-status cancelled", output)
        self.assertTrue(wait_until(lambda: not pid_alive(child), timeout=3))
        self.assertFalse(self.record_path.exists())

    def test_startup_reports_live_orphan_and_keeps_its_record(self):
        orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.addCleanup(lambda: (orphan.kill(), orphan.wait()))
        self.record_path.parent.mkdir(parents=True)
        self.record_path.write_text(json.dumps({
            "pgid": orphan.pid, "pid": orphan.pid, "action": "train", "jobId": "anterior",
            "guiPid": 999999, "bootId": boot_id(), "startedAt": time.time(),
        }), encoding="utf-8")

        process, lines = self.launch("idle")
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)

        joined = "\n".join(lines)
        self.assertIn(f"PGID {orphan.pid}", joined)
        self.assertIn("train", joined)
        self.assertEqual(json.loads(self.record_path.read_text(encoding="utf-8"))["pgid"], orphan.pid)
        self.assertEqual(process.returncode, 143, output)

    def test_sigterm_waits_until_the_whole_process_group_is_gone(self):
        process, lines = self.launch("worker", LEADER_WITH_STUBBORN_WORKER)
        child = int(next(line.split()[1] for line in lines if line.startswith("child")))
        worker = int(next(line.split()[1] for line in lines if line.startswith("worker")))
        self.addCleanup(kill_quietly, child, group=True)
        self.addCleanup(kill_quietly, worker)

        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=15)

        self.assertEqual(process.returncode, 143, output)
        self.assertIn("job-status cancelled", output)
        self.assertTrue(wait_until(lambda: not pid_alive(worker), timeout=2))
        self.assertFalse(self.record_path.exists())

    def test_startup_discards_record_of_finished_process_group(self):
        finished = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        finished.wait()
        self.record_path.parent.mkdir(parents=True)
        self.record_path.write_text(json.dumps({
            "pgid": finished.pid, "pid": finished.pid, "action": "train", "jobId": "anterior",
            "guiPid": 999999, "bootId": boot_id(), "startedAt": time.time(),
        }), encoding="utf-8")

        process, lines = self.launch("idle")
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)

        self.assertFalse(self.record_path.exists())
        self.assertNotIn(f"PGID {finished.pid}", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()

"""Regresiones de los scripts de entrenamiento y de comandos.fish.

Cada test copia los scripts a un proyecto temporal y los ejecuta con fakes:
no se toca Docker, vLLM ni la GPU de la máquina.
"""
import fcntl
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("train-adapter.sh", "train-adapter-axolotl.sh")
LOCK_MESSAGE = "otra operación de GPU/vLLM está en curso"


class ScratchProject:
    def __init__(self, test):
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        for name in SCRIPTS:
            shutil.copy2(REPO / "scripts" / name, self.root / "scripts" / name)
        shutil.copy2(REPO / "comandos.fish", self.root / "comandos.fish")
        self.runtime = self.root / ".runtime"
        self.lock_path = self.runtime / "gpu.lock"
        self.bin = self.root / "fakebin"
        self.bin.mkdir()
        self.test = test
        self.docker_log = self.root / "docker.log"
        # Nunca alcanzar el daemon real: por defecto Docker "no está disponible".
        self.fake("docker", f'printf "%s\\n" "$*" >> {self.docker_log}\nexit 1\n')

    def fake(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def env(self, **extra):
        env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_BRIDGE_GPU_LOCK")}
        env["PATH"] = f"{self.bin}{os.pathsep}{env.get('PATH', '')}"
        env.update(extra)
        return env

    def hold_lock(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        self.test.addCleanup(os.close, descriptor)
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

    def run_script(self, name, *args, env=None, pass_fds=()):
        return subprocess.run(
            ["bash", str(self.root / "scripts" / name), *args],
            cwd=self.root, env=env or self.env(), capture_output=True, text=True, timeout=60, pass_fds=pass_fds,
        )


@unittest.skipUnless(shutil.which("flock"), "requiere flock (util-linux)")
class TrainingScriptLockTests(unittest.TestCase):
    """FT-01: los scripts comparten el lock de GPU/vLLM con las GUIs."""

    def test_held_lock_rejects_training_before_any_side_effect(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                project = ScratchProject(self)
                project.hold_lock()

                result = project.run_script(name, "missing.jsonl", "demo")

                self.assertEqual(result.returncode, 1)
                self.assertIn(LOCK_MESSAGE, result.stderr)
                self.assertIn(str(project.lock_path), result.stderr)
                self.assertNotIn("No existe el dataset", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_free_lock_keeps_previous_validation_and_releases_on_exit(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                project = ScratchProject(self)

                result = project.run_script(name, "missing.jsonl", "demo")

                self.assertEqual(result.returncode, 1)
                self.assertIn("No existe el dataset 'missing.jsonl'.", result.stderr)
                self.assertNotIn(LOCK_MESSAGE, result.stderr)
                self.assertTrue(project.lock_is_free())

    def test_lock_inherited_from_the_gui_is_accepted(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                project = ScratchProject(self)
                descriptor = project.hold_lock()

                result = project.run_script(
                    name, "missing.jsonl", "demo",
                    env=project.env(LLM_BRIDGE_GPU_LOCK_FD=str(descriptor)), pass_fds=(descriptor,),
                )

                self.assertEqual(result.returncode, 1)
                self.assertIn("No existe el dataset 'missing.jsonl'.", result.stderr)
                self.assertNotIn(LOCK_MESSAGE, result.stderr)

    def test_descriptor_of_another_file_does_not_bypass_the_lock(self):
        project = ScratchProject(self)
        project.hold_lock()
        other = os.open(project.root / "otro-archivo", os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, other)

        result = project.run_script(
            "train-adapter.sh", "missing.jsonl", "demo",
            env=project.env(LLM_BRIDGE_GPU_LOCK_FD=str(other)), pass_fds=(other,),
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn(LOCK_MESSAGE, result.stderr)

    def test_both_scripts_share_the_same_lock_block(self):
        pattern = re.compile(r"# --- lock GPU/vLLM.*?# --- fin lock GPU/vLLM ---", re.DOTALL)
        blocks = [pattern.search((REPO / "scripts" / name).read_text(encoding="utf-8")) for name in SCRIPTS]
        self.assertTrue(all(blocks))
        self.assertEqual(blocks[0].group(0), blocks[1].group(0))

    @unittest.skipUnless(shutil.which("fish"), "requiere fish")
    def test_fish_fine_tune_train_respects_the_lock(self):
        project = ScratchProject(self)
        trainer_python = project.fake("trainer-python", "exit 0\n")
        dataset = project.root / "datos.jsonl"
        dataset.write_text('{"messages": []}\n', encoding="utf-8")
        project.hold_lock()

        result = subprocess.run(
            ["fish", str(project.root / "comandos.fish"), "fine-tune-train", str(dataset), "demo"],
            cwd=project.root, env=project.env(FINE_TUNE_PYTHON=str(trainer_python)),
            capture_output=True, text=True, timeout=60,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn(LOCK_MESSAGE, result.stderr)
        self.assertIn("El entrenamiento terminó con error", result.stderr)
        self.assertFalse(project.docker_log.exists())


@unittest.skipUnless(shutil.which("flock"), "requiere flock (util-linux)")
class TrainingScriptRestartTests(unittest.TestCase):
    """FT-03: reiniciar vLLM después de entrenar reconcilia la allowlist."""

    def run_training(self, script, *, vllm_running=True, train_exit=0):
        project = ScratchProject(self)
        events = project.root / "events.log"
        running = "echo vllm" if vllm_running else "true"
        project.fake("docker", textwrap.dedent(f"""\
            printf 'docker %s\\n' "$*" >> {events}
            if [[ "$*" == "compose -p llm-bridge ps --status running --services" ]]; then {running}; fi
            exit 0
        """))
        python = project.fake("trainer-python", textwrap.dedent(f"""\
            printf 'python %s FINE_TUNE_PYTHON=%s\\n' "$*" "${{FINE_TUNE_PYTHON:-}}" >> {events}
            case "$1" in
              -) cat >/dev/null; echo "Org/Base" ;;
              trainer/train.py|trainer/train_axolotl.py) exit {train_exit} ;;
            esac
            exit 0
        """))
        dataset = project.root / "datos.jsonl"
        dataset.write_text('{"messages": []}\n', encoding="utf-8")
        env = project.env(FINE_TUNE_PYTHON=str(python), FINE_TUNE_AXOLOTL_PYTHON=str(python))
        result = project.run_script(script, str(dataset), "demo", env=env)
        lines = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
        return result, lines, python

    def index(self, lines, fragment):
        matches = [number for number, line in enumerate(lines) if fragment in line]
        self.assertTrue(matches, f"{fragment!r} no aparece en {lines}")
        return matches[0]

    def test_restart_after_training_reconciles_allowlist(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, lines, python = self.run_training(script)

                self.assertEqual(result.returncode, 0, result.stderr)
                stop = self.index(lines, "compose -p llm-bridge stop vllm")
                train = self.index(lines, "trainer/train")
                restart = self.index(lines, "compose -p llm-bridge up -d vllm")
                reconcile = self.index(lines, "trainer/runtime_adapters.py --wait-vllm 600")
                self.assertLess(stop, train)
                self.assertLess(train, restart)
                self.assertLess(restart, reconcile)
                self.assertIn(f"FINE_TUNE_PYTHON={python}", lines[reconcile])

    def test_failed_training_still_restarts_and_reconciles(self):
        result, lines, _ = self.run_training("train-adapter.sh", train_exit=3)

        self.assertEqual(result.returncode, 3)
        self.assertLess(self.index(lines, "compose -p llm-bridge up -d vllm"),
                        self.index(lines, "trainer/runtime_adapters.py --wait-vllm 600"))

    def test_vllm_that_was_stopped_is_neither_restarted_nor_reconciled(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, lines, _ = self.run_training(script, vllm_running=False)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse([line for line in lines if "up -d vllm" in line or "runtime_adapters" in line])
                self.index(lines, "trainer/train")


@unittest.skipUnless(shutil.which("fish"), "requiere fish")
class FishVllmReconcileTests(unittest.TestCase):
    def test_start_vllm_reconciles_allowlist_after_vllm_is_healthy(self):
        project = ScratchProject(self)
        events = project.root / "events.log"
        project.fake("docker", f"printf 'docker %s\\n' \"$*\" >> {events}\nexit 0\n")
        project.fake("curl", f"printf 'curl %s\\n' \"$*\" >> {events}\nexit 0\n")
        for name in ("python3.13", "python3", "python"):
            project.fake(name, f"printf 'python %s\\n' \"$*\" >> {events}\nexit 0\n")

        result = subprocess.run(
            ["fish", str(project.root / "comandos.fish"), "vllm", "Org/Modelo"],
            cwd=project.root, env=project.env(LLM_BRIDGE_NONINTERACTIVE="1", HF_TOKEN=""),
            capture_output=True, text=True, timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = events.read_text(encoding="utf-8").splitlines()
        up = next(number for number, line in enumerate(lines) if "compose -p llm-bridge up -d vllm" in line)
        health = next(number for number, line in enumerate(lines) if "8000/health" in line)
        reconcile = [number for number, line in enumerate(lines) if line == "python trainer/runtime_adapters.py"]
        self.assertEqual(len(reconcile), 1, lines)
        self.assertLess(up, health)
        self.assertLess(health, reconcile[0])


@unittest.skipUnless(shutil.which("fish"), "requiere fish")
class FishEnvFileTests(unittest.TestCase):
    """FT-04: ./comandos.fish vllm reescribe LLM_MODEL sin ampliar permisos."""

    def start_vllm(self, project):
        project.fake("docker", f'printf "%s\\n" "$*" >> {project.docker_log}\nexit 0\n')
        project.fake("curl", "exit 0\n")
        project.fake("python3", "exit 0\n")
        project.fake("python3.13", "exit 0\n")
        project.fake("python", "exit 0\n")
        return subprocess.run(
            ["fish", str(project.root / "comandos.fish"), "vllm", "Org/Modelo-Nuevo"],
            cwd=project.root, env=project.env(LLM_BRIDGE_NONINTERACTIVE="1", HF_TOKEN=""),
            capture_output=True, text=True, timeout=60, umask=0o022,
        )

    def test_private_env_file_stays_private(self):
        project = ScratchProject(self)
        env_file = project.root / ".env.local"
        env_file.write_text("APP_TOKEN=secreto\nLLM_MODEL=Org/Modelo-Viejo\n", encoding="utf-8")
        env_file.chmod(0o600)

        result = self.start_vllm(project)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLM_MODEL sincronizado en .env.local: Org/Modelo-Nuevo", result.stdout)
        self.assertEqual(env_file.read_text(encoding="utf-8"), "APP_TOKEN=secreto\nLLM_MODEL=Org/Modelo-Nuevo\n")
        self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)
        self.assertFalse((project.root / ".env.local.tmp").exists())

    def test_existing_mode_is_preserved(self):
        project = ScratchProject(self)
        env_file = project.root / ".env.local"
        env_file.write_text("LLM_PROVIDER=vllm\n", encoding="utf-8")
        env_file.chmod(0o640)

        result = self.start_vllm(project)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(env_file.read_text(encoding="utf-8"), "LLM_PROVIDER=vllm\nLLM_MODEL=Org/Modelo-Nuevo\n")
        self.assertEqual(env_file.stat().st_mode & 0o777, 0o640)

    def test_new_env_file_is_created_private(self):
        project = ScratchProject(self)

        result = self.start_vllm(project)

        env_file = project.root / ".env.local"
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(env_file.read_text(encoding="utf-8"), "LLM_MODEL=Org/Modelo-Nuevo\n")
        self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)

    def test_symlinked_env_file_is_updated_in_place(self):
        project = ScratchProject(self)
        target = project.root / "secretos.env"
        target.write_text("LLM_MODEL=Org/Modelo-Viejo\n", encoding="utf-8")
        target.chmod(0o600)
        (project.root / ".env.local").symlink_to(target.name)

        result = self.start_vllm(project)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((project.root / ".env.local").is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "LLM_MODEL=Org/Modelo-Nuevo\n")
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()

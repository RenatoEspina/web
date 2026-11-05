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


if __name__ == "__main__":
    unittest.main()

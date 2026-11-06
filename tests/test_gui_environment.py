"""FT-07: el estado separa venv presente de entorno verificado."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import gui_server  # noqa: E402

# gui_server_axolotl modifica gui_server al importarse: se importa solo dentro de
# este módulo de tests y se restauran los atributos al terminar.
PATCHED_BY_AXOLOTL = (
    "trainer_python", "setup_environment", "check_training_environment", "ENVIRONMENT_CHECK",
    "run_action", "status_payload", "Handler",
)
_ORIGINAL_CORE = {}
axolotl = None


def setUpModule():
    global axolotl
    _ORIGINAL_CORE.update({name: getattr(gui_server, name) for name in PATCHED_BY_AXOLOTL})
    import gui_server_axolotl
    axolotl = gui_server_axolotl
    for name, value in _ORIGINAL_CORE.items():
        setattr(gui_server, name, value)


def tearDownModule():
    for name, value in _ORIGINAL_CORE.items():
        setattr(gui_server, name, value)
    sys.modules.pop("gui_server_axolotl", None)


class EnvironmentTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.venv = self.root / "venv"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        self.python = self.venv / "bin" / "python"
        self.python.write_text("#!/bin/sh\n", encoding="utf-8")
        self.python.chmod(0o755)
        self.check = mock.Mock()
        for name, value in (
            ("ROOT", self.root),
            ("RUNTIME_DIR", self.root / ".runtime"),
            ("ADAPTER_DIR", self.root / "adapters"),
        ):
            self.patch(name, new=value)
        self.patch("fetch_vllm_models", return_value=(False, []))
        self.patch("check_training_environment", new=self.check)
        self.patch("trainer_python", return_value=self.python)

    def patch(self, name, target=None, **kwargs):
        patcher = mock.patch.object(target or gui_server, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def status(self):
        return gui_server.status_payload()


class BasicEnvironmentStatusTests(EnvironmentTestCase):
    def test_present_venv_is_not_reported_as_verified(self):
        status = self.status()

        self.assertTrue(status["venvPresent"])
        self.assertFalse(status["environmentVerified"])
        self.assertIsNone(status["environmentCheckedAt"])
        self.assertTrue(status["environmentReady"])

    def test_successful_check_is_persisted(self):
        job = gui_server.Job(id="test", action="check")

        gui_server.run_action(job, {})

        self.check.assert_called_once_with(job, "Comprobando CUDA y bitsandbytes")
        self.assertEqual(job.result, {"check": "ok"})
        status = self.status()
        self.assertTrue(status["environmentVerified"])
        self.assertIsInstance(status["environmentCheckedAt"], float)

    def test_failed_check_is_not_verified(self):
        gui_server.run_action(gui_server.Job(id="ok", action="check"), {})
        self.check.side_effect = RuntimeError("El proceso terminó con código 1.")

        with self.assertRaisesRegex(RuntimeError, "código 1"):
            gui_server.run_action(gui_server.Job(id="test", action="check"), {})

        self.assertFalse(self.status()["environmentVerified"])

    def test_setup_invalidates_previous_check(self):
        gui_server.run_action(gui_server.Job(id="ok", action="check"), {})
        self.patch("setup_environment", side_effect=RuntimeError("pip falló"))

        with self.assertRaisesRegex(RuntimeError, "pip falló"):
            gui_server.run_action(gui_server.Job(id="test", action="setup"), {})

        self.assertFalse(self.status()["environmentVerified"])

    def test_check_of_another_interpreter_or_recreated_venv_does_not_count(self):
        gui_server.run_action(gui_server.Job(id="ok", action="check"), {})
        self.assertTrue(self.status()["environmentVerified"])

        (self.venv / "pyvenv.cfg").write_text("home = /usr/local/bin\nversion = otra\n", encoding="utf-8")
        import os
        stat = (self.venv / "pyvenv.cfg").stat()
        os.utime(self.venv / "pyvenv.cfg", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))

        self.assertFalse(self.status()["environmentVerified"])

    def test_missing_venv_is_neither_present_nor_verified(self):
        gui_server.run_action(gui_server.Job(id="ok", action="check"), {})
        self.python.unlink()

        status = self.status()

        self.assertFalse(status["venvPresent"])
        self.assertFalse(status["environmentVerified"])
        self.assertFalse(status["environmentReady"])


class AxolotlEnvironmentStatusTests(EnvironmentTestCase):
    def setUp(self):
        super().setUp()
        self.axolotl_check = self.patch("check_axolotl_environment", target=axolotl)
        self.patch("axolotl_environment_ready", target=axolotl, return_value=True)
        self.patch("axolotl_python", target=axolotl, return_value=self.python)

    def test_axolotl_status_requires_its_own_successful_check(self):
        status = axolotl.status_payload()
        self.assertTrue(status["venvPresent"])
        self.assertFalse(status["environmentVerified"])

        gui_server.run_action(gui_server.Job(id="basic", action="check"), {})
        self.assertFalse(axolotl.status_payload()["environmentVerified"])

        with mock.patch.object(gui_server, "ENVIRONMENT_CHECK", axolotl.AXOLOTL_CHECK), \
                mock.patch.object(gui_server, "check_training_environment", axolotl.check_axolotl_environment):
            gui_server.run_action(gui_server.Job(id="axolotl", action="check"), {})
            status = axolotl.status_payload()

        self.assertTrue(status["environmentVerified"])
        self.assertEqual(status["trainingBackend"], "axolotl")

    def test_axolotl_training_check_is_persisted(self):
        with mock.patch.object(gui_server, "ENVIRONMENT_CHECK", axolotl.AXOLOTL_CHECK), \
                mock.patch.object(gui_server, "ensure_adapter_dir_writable"), \
                mock.patch.object(gui_server, "run_command"):
            axolotl.run_axolotl_train(gui_server.Job(id="train", action="train"), {
                "dataset": "examples/minimal.jsonl", "name": "demo-v1",
            })
            status = axolotl.status_payload()

        self.axolotl_check.assert_called_once()
        self.assertTrue(status["environmentVerified"])


if __name__ == "__main__":
    unittest.main()

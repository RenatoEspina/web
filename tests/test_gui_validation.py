"""FT-04 y FT-11: permisos de .env.local y validación antes de cualquier efecto."""
import os
import stat
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


VALID_TRAINING = {
    "dataset": "examples/minimal.jsonl",
    "name": "demo-v1",
    "model": "Qwen/Qwen3.5-0.8B",
    "hfToken": "",
    "rank": 16,
    "alpha": 32,
    "epochs": 2,
    "dropout": 0.05,
    "learningRate": 0.0001,
    "batchSize": 1,
    "gradientAccumulation": 8,
    "maxLength": 1024,
    "seed": 42,
}
INVALID_FIELDS = (
    ("name", "../fuera", "Nombre de adaptador inválido."),
    ("model", "modelo con espacios", "Nombre de modelo inválido."),
    ("hfToken", "token con espacios", "HF token inválido."),
    ("rank", 12, "rank debe ser 8, 16 o 32."),
    ("alpha", 0, "alpha debe estar entre 1 y 256."),
    ("epochs", 99, "epochs debe estar entre 0.1 y 20.0."),
    ("seed", float("inf"), "seed debe ser entero."),
)
TRAINING_ARGS = [
    "demo-v1", "--model", "Qwen/Qwen3.5-0.8B", "--rank", "16", "--alpha", "32", "--dropout", "0.05",
    "--epochs", "2.0", "--learning-rate", "0.0001", "--batch-size", "1", "--gradient-accumulation", "8",
    "--max-length", "1024", "--seed", "42",
]


class BasicGuiValidationTests(unittest.TestCase):
    def effects(self):
        manager = mock.Mock()
        patches = [
            mock.patch.object(gui_server, "ensure_adapter_dir_writable", manager.ensure),
            mock.patch.object(gui_server, "run_command", manager.run_command),
            mock.patch.object(gui_server, "setup_environment", manager.setup),
            mock.patch.object(gui_server, "check_training_environment", manager.check),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return manager

    def test_invalid_training_payload_has_no_side_effect(self):
        for field, value, message in INVALID_FIELDS:
            with self.subTest(field=field):
                manager = self.effects()
                payload = {**VALID_TRAINING, field: value}

                with self.assertRaises(ValueError) as raised:
                    gui_server.run_action(gui_server.Job(id="test", action="train"), payload)

                self.assertEqual(str(raised.exception), message)
                self.assertEqual(manager.mock_calls, [])

    def test_valid_training_prepares_adapters_then_trains_with_same_arguments(self):
        manager = self.effects()
        job = gui_server.Job(id="test", action="train")

        gui_server.run_action(job, dict(VALID_TRAINING))

        names = [call[0] for call in manager.mock_calls]
        self.assertEqual(names[0], "ensure")
        self.assertEqual(names[-1], "run_command")
        command = manager.run_command.call_args.args[1]
        self.assertEqual(command[:3], ["bash", str(gui_server.ROOT / "scripts" / "train-adapter.sh"),
                                       str(gui_server.TRAINER_DIR / "examples" / "minimal.jsonl")])
        self.assertEqual(command[3:], TRAINING_ARGS)
        self.assertEqual(job.result, {"adapter": "demo-v1", "dataset": "minimal.jsonl", "examples": 2})

    def test_invalid_vllm_token_has_no_side_effect(self):
        manager = self.effects()

        with self.assertRaisesRegex(ValueError, "HF token inválido"):
            gui_server.run_action(gui_server.Job(id="test", action="start-vllm"),
                                  {"model": "Qwen/Qwen3.5-0.8B", "hfToken": "dos partes"})

        self.assertEqual(manager.mock_calls, [])


class AxolotlGuiValidationTests(unittest.TestCase):
    def effects(self):
        manager = mock.Mock()
        patches = [
            mock.patch.object(gui_server, "ensure_adapter_dir_writable", manager.ensure),
            mock.patch.object(gui_server, "run_command", manager.run_command),
            mock.patch.object(axolotl, "setup_axolotl_environment", manager.setup),
            mock.patch.object(axolotl, "check_axolotl_environment", manager.check),
            mock.patch.object(axolotl, "axolotl_environment_ready", return_value=True),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return manager

    def test_invalid_training_payload_has_no_side_effect(self):
        for field, value, message in INVALID_FIELDS:
            with self.subTest(field=field):
                manager = self.effects()
                payload = {**VALID_TRAINING, field: value}

                with self.assertRaises(ValueError) as raised:
                    axolotl.run_axolotl_train(gui_server.Job(id="test", action="train"), payload)

                self.assertEqual(str(raised.exception), message)
                self.assertEqual(manager.mock_calls, [])

    def test_valid_training_keeps_axolotl_command(self):
        manager = self.effects()
        job = gui_server.Job(id="test", action="train")

        axolotl.run_axolotl_train(job, dict(VALID_TRAINING))

        self.assertEqual(manager.mock_calls[0][0], "ensure")
        command = manager.run_command.call_args.args[1]
        self.assertEqual(command[:3], ["bash", str(axolotl.AXOLOTL_TRAIN_SCRIPT),
                                       str(gui_server.TRAINER_DIR / "examples" / "minimal.jsonl")])
        self.assertEqual(command[3:], TRAINING_ARGS)
        env = manager.run_command.call_args.kwargs["env"]
        self.assertEqual(env["FINE_TUNE_AXOLOTL_PYTHON"], str(axolotl.axolotl_python()))
        self.assertEqual(job.result, {"adapter": "demo-v1", "dataset": "minimal.jsonl", "examples": 2,
                                      "backend": "axolotl"})


class EnvFilePermissionTests(unittest.TestCase):
    """FT-04: reescribir .env.local no amplía sus permisos."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env_file = self.root / ".env.local"
        previous = os.umask(0o022)
        self.addCleanup(os.umask, previous)
        patcher = mock.patch.object(gui_server, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def mode(self, path):
        return stat.S_IMODE(path.stat().st_mode)

    def test_private_env_file_stays_private(self):
        self.env_file.write_text("APP_TOKEN=secreto\nLLM_ADAPTER_MODELS=a\n", encoding="utf-8")
        self.env_file.chmod(0o600)

        self.assertTrue(gui_server.set_adapter_allowed_in_env("b", True))

        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "APP_TOKEN=secreto\nLLM_ADAPTER_MODELS=a,b\n")
        self.assertEqual(self.mode(self.env_file), 0o600)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), [".env.local"])

    def test_existing_mode_is_preserved(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=a\n", encoding="utf-8")
        self.env_file.chmod(0o640)

        gui_server.set_adapter_allowed_in_env("a", False)

        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")
        self.assertEqual(self.mode(self.env_file), 0o640)

    def test_new_env_file_is_created_private(self):
        gui_server.set_adapter_allowed_in_env("nuevo", True)

        self.assertEqual(
            self.env_file.read_text(encoding="utf-8"),
            "# Adaptadores LoRA permitidos por el gateway.\nLLM_ADAPTER_MODELS=nuevo\n",
        )
        self.assertEqual(self.mode(self.env_file), 0o600)

    def test_symlinked_env_file_is_updated_in_place(self):
        target = self.root / "secretos.env"
        target.write_text("LLM_ADAPTER_MODELS=a\n", encoding="utf-8")
        target.chmod(0o600)
        self.env_file.symlink_to(target.name)

        gui_server.set_adapter_allowed_in_env("b", True)

        self.assertTrue(self.env_file.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=a,b\n")
        self.assertEqual(self.mode(target), 0o600)

    def test_stale_temporary_file_does_not_block_updates(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=a\n", encoding="utf-8")
        (self.root / ".env.local.tmp").write_text("basura\n", encoding="utf-8")

        gui_server.set_adapter_allowed_in_env("b", True)

        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=a,b\n")


if __name__ == "__main__":
    unittest.main()

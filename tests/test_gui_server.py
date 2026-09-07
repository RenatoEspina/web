import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import gui_server  # noqa: E402


class AdapterDirectoryTests(unittest.TestCase):
    def test_creates_writable_directory_without_removing_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter_dir = Path(temporary) / "adapters"
            adapter_dir.mkdir()
            existing = adapter_dir / "existing-adapter.bin"
            existing.write_bytes(b"keep-me")
            with mock.patch.object(gui_server, "ADAPTER_DIR", adapter_dir):
                job = gui_server.Job(id="test", action="train")
                gui_server.ensure_adapter_dir_writable(job)
            self.assertEqual(existing.read_bytes(), b"keep-me")

    def test_uses_ephemeral_root_container_with_local_uid_and_gid(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter_dir = Path(temporary) / "adapters"
            adapter_dir.mkdir()
            job = gui_server.Job(id="test", action="train")
            with (
                mock.patch.object(gui_server, "ADAPTER_DIR", adapter_dir),
                mock.patch.object(gui_server, "adapter_dir_is_writable", side_effect=[False, True]),
                mock.patch.object(gui_server.shutil, "which", return_value="/usr/bin/docker"),
                mock.patch.object(gui_server, "run_command") as run_command,
            ):
                gui_server.ensure_adapter_dir_writable(job)
            command = run_command.call_args.args[1]
            self.assertIn("--rm", command)
            self.assertIn("0:0", command)
            self.assertIn(f"{adapter_dir}:/adapters", command)
            shell_command = command[command.index("-c") + 1]
            self.assertIn(f"chown -R {gui_server.os.getuid()}:{gui_server.os.getgid()} /adapters", shell_command)

    def test_unrepairable_directory_reports_diagnostic_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter_dir = Path(temporary) / "adapters"
            adapter_dir.mkdir(mode=0o755)
            job = gui_server.Job(id="test", action="train")
            with (
                mock.patch.object(gui_server, "ADAPTER_DIR", adapter_dir),
                mock.patch.object(gui_server, "adapter_dir_is_writable", return_value=False),
                mock.patch.object(gui_server.shutil, "which", return_value=None),
            ):
                with self.assertRaisesRegex(RuntimeError, r"owner=.*mode=755.*sudo chown -R"):
                    gui_server.ensure_adapter_dir_writable(job)


class AdapterAllowlistTests(unittest.TestCase):
    def test_adds_and_removes_adapter_from_gateway_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env.local"
            env_file.write_text("LLM_PROVIDER=vllm\nLLM_ADAPTER_MODELS=existente\n", encoding="utf-8")

            with mock.patch.object(gui_server, "ROOT", root):
                gui_server.set_adapter_allowed_in_env("nuevo", True)
                self.assertIn("LLM_ADAPTER_MODELS=existente,nuevo", env_file.read_text(encoding="utf-8"))

                gui_server.set_adapter_allowed_in_env("nuevo", False)
                contents = env_file.read_text(encoding="utf-8")
                self.assertIn("LLM_ADAPTER_MODELS=existente", contents)
                self.assertNotIn("existente,nuevo", contents)

    def test_removing_unknown_adapter_does_not_create_env_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(gui_server, "ROOT", root):
                gui_server.set_adapter_allowed_in_env("inexistente", False)
            self.assertFalse((root / ".env.local").exists())


class DatasetTests(unittest.TestCase):
    def test_evaluation_dataset_is_not_listed_or_trainable(self):
        self.assertNotIn("evaluation.jsonl", {item["name"] for item in gui_server.list_datasets()})
        with self.assertRaisesRegex(ValueError, "dataset de evaluación"):
            gui_server.dataset_path("examples/evaluation.jsonl")


if __name__ == "__main__":
    unittest.main()

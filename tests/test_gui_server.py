import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import gui_server  # noqa: E402


class AdapterDirectoryTests(unittest.TestCase):
    def test_environment_is_ready_only_after_real_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            python_path = Path(temporary) / "python"
            python_path.write_bytes(b"python")
            previous = gui_server.ENVIRONMENT_VERIFIED
            try:
                with (
                    mock.patch.object(gui_server, "trainer_python", return_value=python_path),
                    mock.patch.object(gui_server, "fetch_vllm_models", return_value=(False, [])),
                ):
                    gui_server.ENVIRONMENT_VERIFIED = False
                    pending = gui_server.status_payload()
                    self.assertTrue(pending["venvPresent"])
                    self.assertFalse(pending["environmentVerified"])
                    self.assertFalse(pending["environmentReady"])

                    gui_server.ENVIRONMENT_VERIFIED = True
                    ready = gui_server.status_payload()
                    self.assertTrue(ready["environmentVerified"])
                    self.assertTrue(ready["environmentReady"])
            finally:
                gui_server.ENVIRONMENT_VERIFIED = previous

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

    def test_delete_adapter_removes_persistent_files_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter_dir = root / "adapters"
            adapter = adapter_dir / "terraria-v1"
            adapter.mkdir(parents=True)
            (adapter / "manifest.json").write_text("{}", encoding="utf-8")
            export = adapter_dir / ".vllm-exports" / "terraria-v1" / "fingerprint"
            export.mkdir(parents=True)
            (export / "export.json").write_text("{}", encoding="utf-8")
            env_file = root / ".env.local"
            env_file.write_text("LLM_ADAPTER_MODELS=otro,terraria-v1\n", encoding="utf-8")
            job = gui_server.Job(id="test", action="delete-adapter")

            with (
                mock.patch.object(gui_server, "ROOT", root),
                mock.patch.object(gui_server, "ADAPTER_DIR", adapter_dir),
                mock.patch.object(gui_server, "fetch_vllm_models", return_value=(False, [])),
            ):
                gui_server.delete_adapter(job, "terraria-v1")

            self.assertFalse(adapter.exists())
            self.assertFalse((adapter_dir / ".vllm-exports" / "terraria-v1").exists())
            self.assertEqual(env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=otro\n")

    def test_delete_adapter_refuses_loaded_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter_dir = Path(temporary) / "adapters"
            adapter = adapter_dir / "terraria-v1"
            adapter.mkdir(parents=True)
            job = gui_server.Job(id="test", action="delete-adapter")
            with (
                mock.patch.object(gui_server, "ADAPTER_DIR", adapter_dir),
                mock.patch.object(gui_server, "fetch_vllm_models", return_value=(True, ["terraria-v1"])),
            ):
                with self.assertRaisesRegex(RuntimeError, "Descarga primero"):
                    gui_server.delete_adapter(job, "terraria-v1")
            self.assertTrue(adapter.exists())


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

    def test_failed_allowlist_after_load_unloads_adapter_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env.local"
            original = "LLM_PROVIDER=vllm\nLLM_ADAPTER_MODELS=existente\n"
            env_file.write_text(original, encoding="utf-8")

            with (
                mock.patch.object(gui_server, "ROOT", root),
                mock.patch.object(gui_server.os, "replace", side_effect=OSError("solo lectura")),
                mock.patch.object(gui_server, "post_vllm", return_value="ok") as post_vllm,
            ):
                with self.assertRaisesRegex(RuntimeError, "fue revertido"):
                    gui_server.set_adapter_allowed_in_env("nuevo", True, rollback_runtime=True)

            post_vllm.assert_called_once_with(
                "/v1/unload_lora_adapter",
                {"lora_name": "nuevo"},
            )
            self.assertEqual(env_file.read_text(encoding="utf-8"), original)

    def test_failed_allowlist_after_unload_loads_adapter_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env.local"
            original = "LLM_PROVIDER=vllm\nLLM_ADAPTER_MODELS=existente,nuevo\n"
            env_file.write_text(original, encoding="utf-8")

            with (
                mock.patch.object(gui_server, "ROOT", root),
                mock.patch.object(gui_server.os, "replace", side_effect=OSError("solo lectura")),
                mock.patch.object(gui_server, "post_vllm", return_value="ok") as post_vllm,
            ):
                with self.assertRaisesRegex(RuntimeError, "fue revertido"):
                    gui_server.set_adapter_allowed_in_env("nuevo", False, rollback_runtime=True)

            post_vllm.assert_called_once_with(
                "/v1/load_lora_adapter",
                {"lora_name": "nuevo", "lora_path": "/adapters/nuevo"},
            )
            self.assertEqual(env_file.read_text(encoding="utf-8"), original)

    def test_failed_allowlist_and_failed_rollback_reports_inconsistent_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env.local"
            env_file.write_text("LLM_ADAPTER_MODELS=existente\n", encoding="utf-8")

            with (
                mock.patch.object(gui_server, "ROOT", root),
                mock.patch.object(gui_server.os, "replace", side_effect=OSError("solo lectura")),
                mock.patch.object(gui_server, "post_vllm", side_effect=RuntimeError("rollback falló")),
            ):
                with self.assertRaisesRegex(RuntimeError, "puede ser inconsistente"):
                    gui_server.set_adapter_allowed_in_env("nuevo", True, rollback_runtime=True)


class DatasetTests(unittest.TestCase):
    def test_evaluation_dataset_is_not_listed_or_trainable(self):
        self.assertNotIn("evaluation.jsonl", {item["name"] for item in gui_server.list_datasets()})
        with self.assertRaisesRegex(ValueError, "dataset de evaluación"):
            gui_server.dataset_path("examples/evaluation.jsonl")


if __name__ == "__main__":
    unittest.main()

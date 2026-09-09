import io
import json
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

    def test_delete_adapter_unloads_loaded_runtime_automatically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter_dir = root / "adapters"
            adapter = adapter_dir / "terraria-v1"
            adapter.mkdir(parents=True)
            (adapter / "manifest.json").write_text("{}", encoding="utf-8")
            env_file = root / ".env.local"
            env_file.write_text("LLM_ADAPTER_MODELS=terraria-v1\n", encoding="utf-8")
            job = gui_server.Job(id="test", action="delete-adapter")

            with (
                mock.patch.object(gui_server, "ROOT", root),
                mock.patch.object(gui_server, "ADAPTER_DIR", adapter_dir),
                mock.patch.object(gui_server, "fetch_vllm_models", return_value=(True, ["terraria-v1"])),
                mock.patch.object(gui_server, "loaded_adapter_path", return_value="/adapters/terraria-v1"),
                mock.patch.object(gui_server, "post_vllm", return_value="ok") as post_vllm,
            ):
                gui_server.delete_adapter(job, "terraria-v1")

            post_vllm.assert_called_once_with(
                "/v1/unload_lora_adapter",
                {"lora_name": "terraria-v1"},
            )
            self.assertFalse(adapter.exists())
            self.assertEqual(env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")


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


class VllmStartTests(unittest.TestCase):
    def test_gui_starts_vllm_through_comandos_with_embeddings(self):
        job = gui_server.Job(id="test", action="start-vllm")
        model = "Qwen/Qwen3.5-0.8B"
        with (
            mock.patch.object(gui_server, "ensure_adapter_dir_writable"),
            mock.patch.object(gui_server.shutil, "which", return_value="/usr/bin/fish"),
            mock.patch.object(gui_server, "run_command") as run_command,
        ):
            gui_server.run_action(job, {"model": model, "hfToken": ""})

        command = run_command.call_args.args[1]
        self.assertEqual(command, ["/usr/bin/fish", str(gui_server.ROOT / "comandos.fish"), "vllm", model])
        env = run_command.call_args.kwargs["env"]
        self.assertEqual(env["VLLM_MODEL"], model)
        self.assertEqual(env["LLM_BRIDGE_NONINTERACTIVE"], "1")
        self.assertEqual(job.result["ollama"], "started")
        self.assertEqual(job.result["embeddings"], "ready")


class AdapterBaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.adapters = Path(self.temporary.name)
        self.directory = self.adapters / "domain-v1"
        self.directory.mkdir()
        self.write("adapter_config.json", {"base_model_name_or_path": "research/base-a"})
        self.write("manifest.json", {"baseModel": "research/base-a"})
        (self.directory / "adapter_model.safetensors").write_bytes(b"fixture")
        self.patch = mock.patch.object(gui_server, "ADAPTER_DIR", self.adapters)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, name, data):
        (self.directory / name).write_text(json.dumps(data), encoding="utf-8")

    def models(self, root, alias="served-base"):
        return mock.patch.object(gui_server, "urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps({
            "data": [{"id": alias, "root": root},
                     {"id": "domain-v1", "root": "/adapters/domain-v1", "parent": alias}]
        }).encode()))

    def test_wrong_base_is_rejected_before_export_load_or_allowlist_changes(self):
        with self.models("research/base-b"), mock.patch.object(gui_server, "prepare_runtime_adapter") as export, mock.patch.object(
            gui_server, "post_vllm"
        ) as post, mock.patch.object(gui_server, "set_adapter_allowed_in_env") as allowlist:
            with self.assertRaisesRegex(RuntimeError, "fue entrenado sobre"):
                gui_server.run_action(gui_server.Job(id="test", action="load-adapter"), {"name": "domain-v1"})
            export.assert_not_called()
            post.assert_not_called()
            allowlist.assert_not_called()

    def test_correct_base_supports_served_alias_and_loads(self):
        with self.models("research/base-a"), mock.patch.object(gui_server, "prepare_runtime_adapter", return_value="/adapters/domain-v1"), mock.patch.object(
            gui_server, "post_vllm", return_value="ok"
        ) as post, mock.patch.object(gui_server, "set_adapter_allowed_in_env") as allowlist:
            self.assertEqual(gui_server.validate_adapter_base("domain-v1"), "served-base")
            gui_server.run_action(gui_server.Job(id="test", action="load-adapter"), {"name": "domain-v1"})
            post.assert_called_once_with("/v1/load_lora_adapter", {"lora_name": "domain-v1", "lora_path": "/adapters/domain-v1"})
            allowlist.assert_called_once_with("domain-v1", True, rollback_runtime=True)

    def test_alias_does_not_disguise_another_base(self):
        with self.models("research/base-b", alias="research/base-a"):
            with self.assertRaises(RuntimeError):
                gui_server.validate_adapter_base("domain-v1")

    def test_missing_or_conflicting_metadata_fails_closed(self):
        with self.models("research/base-a"):
            self.write("manifest.json", {"baseModel": "research/base-b"})
            with self.assertRaises(ValueError):
                gui_server.validate_adapter_base("domain-v1")
            (self.directory / "manifest.json").unlink()
            self.assertEqual(gui_server.validate_adapter_base("domain-v1"), "served-base")
            self.write("adapter_config.json", {})
            with self.assertRaises(ValueError):
                gui_server.validate_adapter_base("domain-v1")

    def test_missing_runtime_root_fails_closed(self):
        with self.models(None):
            with self.assertRaises(RuntimeError):
                gui_server.validate_adapter_base("domain-v1")


class DatasetTests(unittest.TestCase):
    def test_evaluation_dataset_is_not_listed_or_trainable(self):
        self.assertNotIn("evaluation.jsonl", {item["name"] for item in gui_server.list_datasets()})
        with self.assertRaisesRegex(ValueError, "dataset de evaluación"):
            gui_server.dataset_path("examples/evaluation.jsonl")


if __name__ == "__main__":
    unittest.main()

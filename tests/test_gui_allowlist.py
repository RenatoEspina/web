"""FT-03: la allowlist LoRA se reconcilia tras (re)iniciar vLLM."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import gui_server  # noqa: E402


class AllowlistTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapters = self.root / "adapters"
        self.adapters.mkdir()
        self.env_file = self.root / ".env.local"
        for name, value in (("ROOT", self.root), ("ADAPTER_DIR", self.adapters)):
            patcher = mock.patch.object(gui_server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def adapter(self, name):
        directory = self.adapters / name
        directory.mkdir()
        (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Org/Base"}), encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"fixture")
        (directory / "manifest.json").write_text(json.dumps({"baseModel": "Org/Base"}), encoding="utf-8")
        return directory

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(gui_server, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked


class ReconcileTests(AllowlistTestCase):
    def test_restart_reloads_compatible_adapters_and_retires_the_rest(self):
        self.env_file.write_text("APP_TOKEN=x\nLLM_ADAPTER_MODELS=activo,recargable,otra-base,borrado\n", encoding="utf-8")
        for name in ("activo", "recargable", "otra-base"):
            self.adapter(name)
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base", "activo"]))

        def base(name):
            if name == "otra-base":
                raise RuntimeError("El adaptador 'otra-base' fue entrenado sobre 'Org/Vieja'")
            return "Org/Base"

        self.patch("validate_adapter_base", side_effect=base)
        self.patch("prepare_runtime_adapter", side_effect=lambda job, name: f"/adapters/{name}")
        post = self.patch("post_vllm", return_value="ok")
        job = gui_server.Job(id="test", action="start-vllm")

        result = gui_server.reconcile_adapter_allowlist(job)

        self.assertEqual(result, {"kept": ["activo"], "reloaded": ["recargable"], "removed": ["otra-base", "borrado"]})
        post.assert_called_once_with("/v1/load_lora_adapter", {"lora_name": "recargable", "lora_path": "/adapters/recargable"})
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "APP_TOKEN=x\nLLM_ADAPTER_MODELS=activo,recargable\n")
        logs = "\n".join(job.logs)
        self.assertIn("otra-base", logs)
        self.assertIn("Org/Vieja", logs)
        self.assertIn("borrado", logs)

    def test_unavailable_vllm_leaves_allowlist_untouched(self):
        original = "LLM_ADAPTER_MODELS=a,b\n"
        self.env_file.write_text(original, encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(False, []))

        with self.assertRaisesRegex(RuntimeError, "vLLM no está disponible"):
            gui_server.reconcile_adapter_allowlist(gui_server.Job(id="test", action="start-vllm"))

        self.assertEqual(self.env_file.read_text(encoding="utf-8"), original)

    def test_empty_allowlist_does_not_contact_vllm(self):
        fetch = self.patch("fetch_vllm_models")

        result = gui_server.reconcile_adapter_allowlist(gui_server.Job(id="test", action="start-vllm"))

        self.assertEqual(result, {"kept": [], "reloaded": [], "removed": []})
        fetch.assert_not_called()
        self.assertFalse(self.env_file.exists())


class RetireFromAllowlistTests(AllowlistTestCase):
    def test_unload_retires_an_allowlisted_adapter_that_is_not_loaded(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=otro,perdido\n", encoding="utf-8")
        self.adapter("perdido")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        post = self.patch("post_vllm")
        job = gui_server.Job(id="test", action="unload-adapter")

        gui_server.run_action(job, {"name": "perdido"})

        post.assert_not_called()
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=otro\n")
        self.assertEqual(job.result, {"adapter": "perdido", "runtimeLoaded": False, "allowlistUpdated": True,
                                      "gatewayRestartRequired": True})
        self.assertIn("no está cargado en vLLM", "\n".join(job.logs))

    def test_unload_of_unknown_adapter_keeps_previous_error(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=otro\n", encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        self.patch("urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps({"data": [{"id": "Org/Base"}]}).encode()))

        with self.assertRaisesRegex(RuntimeError, "El adaptador fantasma no aparece cargado en vLLM."):
            gui_server.run_action(gui_server.Job(id="test", action="unload-adapter"), {"name": "fantasma"})

        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=otro\n")

    def test_unload_of_loaded_adapter_keeps_runtime_unload(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=activo\n", encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base", "activo"]))
        self.patch("loaded_adapter_path", return_value="/adapters/activo")
        post = self.patch("post_vllm", return_value="ok")
        job = gui_server.Job(id="test", action="unload-adapter")

        gui_server.run_action(job, {"name": "activo"})

        post.assert_called_once_with("/v1/unload_lora_adapter", {"lora_name": "activo"})
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")
        self.assertFalse(job.result["runtimeLoaded"])

    def test_status_reports_allowlisted_adapters(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=a, b,\n", encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(False, []))
        with mock.patch.object(gui_server, "RUNTIME_DIR", self.root / ".runtime"):
            payload = gui_server.status_payload()

        self.assertEqual(payload["allowlistedAdapters"], ["a", "b"])


class ReconcileCommandTests(AllowlistTestCase):
    def setUp(self):
        super().setUp()
        import runtime_adapters
        self.command = runtime_adapters

    def test_command_reconciles_and_prints_progress(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=perdido\n", encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            code = self.command.main([])

        self.assertEqual(code, 0)
        self.assertIn("perdido", stdout.getvalue())
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")

    def test_command_waits_for_vllm_before_reconciling(self):
        wait = self.patch("wait_for_vllm")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))

        with contextlib.redirect_stdout(io.StringIO()):
            code = self.command.main(["--wait-vllm", "600"])

        self.assertEqual(code, 0)
        self.assertEqual(wait.call_args.args[1], 600)

    def test_command_reports_failure_without_traceback(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=a\n", encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(False, []))
        stderr = io.StringIO()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = self.command.main([])

        self.assertEqual(code, 1)
        self.assertIn("Aviso: no se pudo reconciliar", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

"""FT-03: la allowlist LoRA se reconcilia tras (re)iniciar vLLM."""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError

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


class AllowlistTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapters = self.root / "adapters"
        self.adapters.mkdir()
        self.env_file = self.root / ".env.local"
        self.marker = self.root / ".runtime" / "adapters-reconcile-pending"
        for name, value in (("ROOT", self.root), ("ADAPTER_DIR", self.adapters), ("RUNTIME_DIR", self.root / ".runtime")):
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
                raise gui_server.AdapterBaseMismatch("El adaptador 'otra-base' fue entrenado sobre 'Org/Vieja'")
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


class FailClosedReconcileTests(AllowlistTestCase):
    """Un fallo de infraestructura nunca retira un adaptador válido de la allowlist."""

    def setUp(self):
        super().setUp()
        self.env_file.write_text("LLM_ADAPTER_MODELS=valido,borrado\n", encoding="utf-8")
        self.adapter("valido")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text("", encoding="utf-8")

    def assert_failed_closed(self, job, raised):
        self.assertIn("valido", str(raised.exception))
        self.assertIn("se conserva", str(raised.exception))
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=valido\n")
        self.assertTrue(self.marker.exists())
        self.assertIn("borrado", "\n".join(job.logs))

    def test_export_failure_keeps_the_adapter(self):
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("prepare_runtime_adapter", side_effect=RuntimeError("El proceso terminó con código 1."))
        post = self.patch("post_vllm")
        job = gui_server.Job(id="test", action="start-vllm")

        with self.assertRaises(RuntimeError) as raised:
            gui_server.reconcile_adapter_allowlist(job)

        post.assert_not_called()
        self.assert_failed_closed(job, raised)

    def test_missing_interpreter_keeps_the_adapter(self):
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("prepare_runtime_adapter", side_effect=FileNotFoundError("trainer/.venv/bin/python"))
        job = gui_server.Job(id="test", action="start-vllm")

        with self.assertRaises(RuntimeError) as raised:
            gui_server.reconcile_adapter_allowlist(job)

        self.assert_failed_closed(job, raised)

    def test_vllm_load_error_keeps_the_adapter(self):
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("prepare_runtime_adapter", return_value="/adapters/valido")
        self.patch("post_vllm", side_effect=RuntimeError("vLLM respondió HTTP 500."))
        job = gui_server.Job(id="test", action="start-vllm")

        with self.assertRaises(RuntimeError) as raised:
            gui_server.reconcile_adapter_allowlist(job)

        self.assert_failed_closed(job, raised)

    def test_unreachable_vllm_during_base_check_keeps_the_adapter(self):
        self.patch("urlopen", side_effect=URLError("connection refused"))
        job = gui_server.Job(id="test", action="start-vllm")

        with self.assertRaises(RuntimeError) as raised:
            gui_server.reconcile_adapter_allowlist(job)

        self.assert_failed_closed(job, raised)

    def test_unknown_served_base_keeps_the_adapter(self):
        self.patch("urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps({"data": [{"id": "Org/Base"}]}).encode()))
        job = gui_server.Job(id="test", action="start-vllm")

        with self.assertRaises(RuntimeError) as raised:
            gui_server.reconcile_adapter_allowlist(job)

        self.assert_failed_closed(job, raised)

    def test_real_base_mismatch_still_retires_the_adapter(self):
        self.patch("urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps({
            "data": [{"id": "Org/Otra", "root": "Org/Otra"}]}).encode()))
        job = gui_server.Job(id="test", action="start-vllm")

        result = gui_server.reconcile_adapter_allowlist(job)

        self.assertEqual(result, {"kept": [], "reloaded": [], "removed": ["valido", "borrado"]})
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")
        self.assertFalse(self.marker.exists())

    def test_invalid_adapter_metadata_still_retires_the_adapter(self):
        (self.adapters / "valido" / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.patch("urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps({
            "data": [{"id": "Org/Base", "root": "Org/Base"}]}).encode()))

        result = gui_server.reconcile_adapter_allowlist(gui_server.Job(id="test", action="start-vllm"))

        self.assertEqual(result["removed"], ["valido", "borrado"])

    def test_successful_reconcile_clears_pending_marker(self):
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("prepare_runtime_adapter", return_value="/adapters/valido")
        self.patch("post_vllm", return_value="ok")

        result = gui_server.reconcile_adapter_allowlist(gui_server.Job(id="test", action="start-vllm"))

        self.assertEqual(result, {"kept": [], "reloaded": ["valido"], "removed": ["borrado"]})
        self.assertFalse(self.marker.exists())

    def test_empty_allowlist_clears_pending_marker(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=\n", encoding="utf-8")

        gui_server.reconcile_adapter_allowlist(gui_server.Job(id="test", action="start-vllm"))

        self.assertFalse(self.marker.exists())


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

    def test_command_fails_closed_when_export_fails(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=valido\n", encoding="utf-8")
        self.adapter("valido")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("prepare_runtime_adapter", side_effect=RuntimeError("El proceso terminó con código 1."))
        stderr = io.StringIO()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = self.command.main([])

        self.assertEqual(code, 1)
        self.assertIn("valido", stderr.getvalue())
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=valido\n")

    def test_command_exports_with_the_axolotl_interpreter_when_it_is_the_only_one(self):
        trainer = self.root / "trainer"
        python = trainer / ".venv-axolotl" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        self.patch("TRAINER_DIR", new=trainer)
        self.env_file.write_text("LLM_ADAPTER_MODELS=valido\n", encoding="utf-8")
        self.adapter("valido")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("needs_export", return_value=True)
        self.patch("ensure_adapter_dir_writable")
        run_command = self.patch("run_command", side_effect=RuntimeError("export falló"))
        environment = {key: value for key, value in os.environ.items() if key != "FINE_TUNE_PYTHON"}

        with mock.patch.dict(os.environ, environment, clear=True), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = self.command.main([])

        self.assertEqual(code, 1)
        self.assertEqual(run_command.call_args.args[1][0], str(python))

    def test_command_reports_failure_without_traceback(self):
        self.env_file.write_text("LLM_ADAPTER_MODELS=a\n", encoding="utf-8")
        self.patch("fetch_vllm_models", return_value=(False, []))
        stderr = io.StringIO()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = self.command.main([])

        self.assertEqual(code, 1)
        self.assertIn("Aviso: no se pudo reconciliar", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


class AxolotlStartVllmTests(AllowlistTestCase):
    def test_start_vllm_propagates_the_backend_interpreter_to_reconciliation(self):
        python = self.root / "venv-axolotl" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        run_command = self.patch("run_command")
        self.patch("ensure_adapter_dir_writable")
        # En el proceso de la GUI Axolotl, gui_server.trainer_python es axolotl_python.
        with mock.patch.object(axolotl, "axolotl_python", return_value=python), \
                mock.patch.object(gui_server, "trainer_python", lambda: axolotl.axolotl_python()), \
                mock.patch.object(gui_server.shutil, "which", return_value="/usr/bin/fish"):
            gui_server.run_action(gui_server.Job(id="test", action="start-vllm"), {"model": "Qwen/Qwen3.5-0.8B"})

        env = run_command.call_args.kwargs["env"]
        self.assertEqual(env["FINE_TUNE_PYTHON"], str(python))
        self.assertEqual(env["VLLM_MODEL"], "Qwen/Qwen3.5-0.8B")


class PendingReconcileTests(AllowlistTestCase):
    """FT-03: una reconciliación interrumpida se completa al volver vLLM."""

    def setUp(self):
        super().setUp()
        gui_server.CURRENT_JOB = None
        self.addCleanup(setattr, gui_server, "CURRENT_JOB", None)
        self.addCleanup(self.wait_current_job)
        self.env_file.write_text("LLM_ADAPTER_MODELS=perdido\n", encoding="utf-8")
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text("", encoding="utf-8")
        self.patch("RECONCILE_RETRY_SECONDS", new=60.0, create=True)
        self.patch("_LAST_AUTO_RECONCILE_FAILURE", new=None, create=True)

    def wait_current_job(self):
        job = gui_server.CURRENT_JOB
        deadline = time.monotonic() + 5
        while job is not None and job.status == "running" and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_pending_marker_reconciles_once_vllm_is_healthy(self):
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))

        job = gui_server.maybe_start_pending_reconcile()

        self.assertIsNotNone(job)
        self.assertEqual(job.action, "reconcile-adapters")
        self.wait_current_job()
        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.result, {"kept": [], "reloaded": [], "removed": ["perdido"]})
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=\n")
        self.assertFalse(self.marker.exists())
        self.assertIsNone(gui_server.maybe_start_pending_reconcile())

    def test_nothing_happens_while_vllm_is_down(self):
        self.patch("fetch_vllm_models", return_value=(False, []))

        self.assertIsNone(gui_server.maybe_start_pending_reconcile())

        self.assertTrue(self.marker.exists())
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=perdido\n")

    def test_failed_automatic_reconcile_backs_off(self):
        self.adapter("perdido")
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))
        self.patch("validate_adapter_base", return_value="Org/Base")
        self.patch("prepare_runtime_adapter", side_effect=RuntimeError("export falló"))

        job = gui_server.maybe_start_pending_reconcile()
        self.wait_current_job()

        self.assertEqual(job.status, "failed")
        self.assertTrue(self.marker.exists())
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "LLM_ADAPTER_MODELS=perdido\n")
        self.assertIsNone(gui_server.maybe_start_pending_reconcile())

    def test_reconcile_action_is_available_from_the_api(self):
        self.patch("fetch_vllm_models", return_value=(True, ["Org/Base"]))

        job = gui_server.start_job("reconcile-adapters", {})
        self.wait_current_job()

        self.assertEqual(job.status, "succeeded")
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import gui_server
import verify_lora
import vllm_export


class NamespaceTests(unittest.TestCase):
    def test_text_qwen35_becomes_multimodal_language_namespace(self):
        for suffix in ("linear_attn.in_proj_a", "self_attn.q_proj", "mlp.gate_proj"):
            for side in ("A", "B"):
                key = f"base_model.model.model.layers.0.{suffix}.lora_{side}.weight"
                self.assertEqual(vllm_export.runtime_key(key),
                                 key.replace("model.model.layers.", "model.model.language_model.layers."))

    def test_already_qualified_keys_are_idempotent(self):
        for key in ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
                    "base_model.model.lm_head.lora_A.weight"):
            self.assertEqual(vllm_export.runtime_key(key), key)

    def test_unknown_namespace_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Namespace"):
            vllm_export.runtime_key("base_model.model.visual.q_proj.lora_A.weight")

    def test_only_qwen35_is_selected_including_legacy_manifests(self):
        self.assertTrue(vllm_export.needs_export({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}, {}))
        self.assertTrue(vllm_export.needs_export({}, {"modelType": "qwen3_5_text"}))
        self.assertFalse(vllm_export.needs_export({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B"}, {}))

    def test_load_uses_prepared_path_not_original(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "demo"
            directory.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (directory / name).write_text("{}")
            with (mock.patch.object(gui_server, "ADAPTER_DIR", Path(temp)),
                  mock.patch.object(gui_server, "validate_adapter_base", return_value="base"),
                  mock.patch.object(gui_server, "prepare_runtime_adapter", return_value="/adapters/.vllm-exports/demo/hash"),
                  mock.patch.object(gui_server, "post_vllm", return_value="ok") as post,
                  mock.patch.object(gui_server, "set_adapter_allowed_in_env")):
                job = gui_server.Job(id="test", action="load-adapter")
                gui_server.run_action(job, {"name": "demo"})
            self.assertEqual(post.call_args.args[1]["lora_path"], "/adapters/.vllm-exports/demo/hash")
            self.assertFalse(job.result["inferenceVerified"])

    def test_export_failure_prevents_runtime_load(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "demo"
            directory.mkdir()
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                (directory / filename).write_text("{}")
            with (mock.patch.object(gui_server, "ADAPTER_DIR", Path(temp)),
                  mock.patch.object(gui_server, "validate_adapter_base", return_value="base"),
                  mock.patch.object(gui_server, "prepare_runtime_adapter", side_effect=RuntimeError("export failed")),
                  mock.patch.object(gui_server, "post_vllm") as post):
                with self.assertRaisesRegex(RuntimeError, "export failed"):
                    gui_server.run_action(gui_server.Job(id="test", action="load-adapter"), {"name": "demo"})
                post.assert_not_called()

    def test_rollback_restores_exact_exported_runtime_path(self):
        with mock.patch.object(gui_server, "post_vllm") as post:
            gui_server.rollback_adapter_runtime("demo", False, "/adapters/.vllm-exports/demo/hash")
        self.assertEqual(post.call_args.args[1]["lora_path"], "/adapters/.vllm-exports/demo/hash")


class VerificationTests(unittest.TestCase):
    NORMAL_GENERATION = {
        "text": "respuesta suficientemente larga",
        "characters": 160,
        "latencySeconds": 1.0,
        "finishReason": "stop",
        "promptTokens": 20,
        "completionTokens": 80,
        "tokensPerSecond": 80.0,
    }

    def test_identical_probabilities_are_inconclusive(self):
        with (mock.patch.object(verify_lora, "probe", return_value={"text": "a", "top": {"a": -1.0}}),
              mock.patch.object(verify_lora, "generation_probe", return_value=self.NORMAL_GENERATION)):
            report = verify_lora.verify("http://localhost", "base", "lora")
        self.assertEqual(report["status"], "inconclusive")
        self.assertFalse(report["diagnostics"]["earlyStopSuspected"])

    def test_same_text_can_show_a_real_probability_change(self):
        base = {"text": "a", "top": {"a": -1.0}}
        lora = {"text": "a", "top": {"a": -0.5}}
        with (mock.patch.object(verify_lora, "probe", side_effect=[base, base, lora] * 3),
              mock.patch.object(verify_lora, "generation_probe", return_value=self.NORMAL_GENERATION)):
            report = verify_lora.verify("http://localhost", "base", "lora")
        self.assertEqual(report["status"], "effect_detected")

    def test_baseline_numerical_noise_does_not_count_as_lora_effect(self):
        samples = [{"top": {"a": score}} for score in (-1.0, -1.01, -1.02)]
        with (mock.patch.object(verify_lora, "probe", side_effect=samples * 3),
              mock.patch.object(verify_lora, "generation_probe", return_value=self.NORMAL_GENERATION)):
            report = verify_lora.verify("http://localhost", "base", "lora")
        self.assertEqual(report["status"], "inconclusive")

    def test_generation_diagnostic_flags_repeated_early_stop(self):
        base_distribution = {"text": "a", "top": {"a": -1.0}}
        lora_distribution = {"text": "a", "top": {"a": -0.5}}
        base_generation = dict(self.NORMAL_GENERATION)
        short_generation = {
            "text": "corta",
            "characters": 20,
            "latencySeconds": 0.3,
            "finishReason": "stop",
            "promptTokens": 20,
            "completionTokens": 12,
            "tokensPerSecond": 40.0,
        }
        with (mock.patch.object(
                verify_lora, "probe", side_effect=[base_distribution, base_distribution, lora_distribution] * 3),
              mock.patch.object(
                verify_lora, "generation_probe", side_effect=[base_generation, short_generation] * 3)):
            report = verify_lora.verify("http://localhost", "base", "lora")
        self.assertEqual(report["status"], "effect_detected")
        self.assertEqual(report["diagnostics"]["markedlyShorterCases"], 3)
        self.assertTrue(report["diagnostics"]["earlyStopSuspected"])


@unittest.skipUnless(importlib.util.find_spec("safetensors") and importlib.util.find_spec("torch"),
                     "CPU safetensors integration runs in trainer/.venv")
class ExportIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "demo"
        self.directory.mkdir()
        self.originals = {f"base_model.model.model.layers.0.self_attn.q_proj.lora_{side}.weight":
                          torch.ones((2, 2), dtype=torch.bfloat16) for side in ("A", "B")}
        save_file(self.originals, str(self.directory / "adapter_model.safetensors"))
        (self.directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}))

    def test_export_preserves_originals_exact_tensor_values_and_reuses_cache(self):
        import torch
        from safetensors.torch import load_file
        before = {name: (self.directory / name).read_bytes() for name in ("adapter_config.json", "adapter_model.safetensors")}
        destination = vllm_export.export_adapter(self.directory)
        output = load_file(str(destination / "adapter_model.safetensors"))
        self.assertEqual(len(output), len(self.originals))
        for key, tensor in self.originals.items():
            self.assertTrue(torch.equal(tensor, output[vllm_export.runtime_key(key)]))
        for name, data in before.items():
            self.assertEqual((self.directory / name).read_bytes(), data)
        self.assertEqual(vllm_export.export_adapter(self.directory), destination)
        self.assertEqual(vllm_export.validate_export(destination)["renamed"], 2)

    def test_corrupt_export_is_not_silently_reused_or_overwritten(self):
        destination = vllm_export.export_adapter(self.directory)
        (destination / "adapter_model.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "dañada"):
            vllm_export.export_adapter(self.directory)
        self.assertEqual((destination / "adapter_model.safetensors").read_bytes(), b"corrupt")

    def test_symlink_export_directory_is_rejected(self):
        external = Path(self.temp.name) / "outside"
        external.mkdir()
        (self.directory.parent / ".vllm-exports").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "enlace"):
            vllm_export.export_adapter(self.directory)
        self.assertEqual(list(external.iterdir()), [])

    def test_conversion_failure_leaves_no_published_export(self):
        with mock.patch("safetensors.torch.save_file", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                vllm_export.export_adapter(self.directory)
        self.assertFalse(vllm_export.export_destination(self.directory).exists())
        self.assertEqual(list(vllm_export.export_destination(self.directory).parent.iterdir()), [])

    def test_changed_original_produces_new_export_without_deleting_old(self):
        old = vllm_export.export_adapter(self.directory)
        config = self.directory / "adapter_config.json"
        config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "r": 16}))
        new = vllm_export.export_adapter(self.directory)
        self.assertNotEqual(old, new)
        self.assertTrue(old.is_dir())


if __name__ == "__main__":
    unittest.main()

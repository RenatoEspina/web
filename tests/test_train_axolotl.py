"""Ejecuta trainer/train_axolotl.py como CLI con Transformers, TRL y Axolotl simulados."""
import hashlib
import json
import os
import runpy
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from types import ModuleType, SimpleNamespace
from unittest import mock

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML es dependencia del entorno Axolotl
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))

ATTENTION_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class FakeTokenizer:
    """Un token por palabra y uno por rol; la máscara marca las palabras assistant."""

    chat_template = "plantilla-original"

    def apply_chat_template(self, messages, **kwargs):
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        for message in messages:
            words = message["content"].split()
            input_ids += [0] + [1] * len(words)
            assistant_masks += [0] + [int(message["role"] == "assistant")] * len(words)
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


@unittest.skipIf(yaml is None, "PyYAML no está instalado")
class TrainAxolotlCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output_root = self.directory / "adapters"
        self.dataset = self.directory / "demo-training.jsonl"
        self.validation = self.directory / "demo-validation.jsonl"
        self.dataset.write_text(
            jsonl([
                {
                    "id": "extra",
                    "messages": [
                        {"role": "system", "content": "  Eres un asistente.  "},
                        {"role": "user", "content": "  ¿Qué es un slime?\n"},
                        {"role": "assistant", "content": "\tUn enemigo básico.  ", "weight": 1},
                    ],
                },
            ]) + "\n",
            encoding="utf-8",
        )
        self.validation.write_text(
            jsonl([
                {
                    "messages": [
                        {"role": "user", "content": " ¿Qué es un zombi? "},
                        {"role": "assistant", "content": " Un enemigo nocturno. "},
                    ],
                },
            ]),
            encoding="utf-8",
        )
        self.cli = self.directory / "bin" / "axolotl"
        self.cli.parent.mkdir()
        self.cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.cli.chmod(0o755)
        self.calls: list[dict] = []
        self.model_type = "qwen3_5"
        self.run_effect = None

    def fake_run(self, argv, env=None, check=False):
        config = yaml.safe_load(Path(argv[2]).read_text(encoding="utf-8"))
        call = {"argv": argv, "config": config}
        call["train_rows"] = Path(config["datasets"][0]["path"]).read_text(encoding="utf-8")
        if config.get("test_datasets"):
            call["validation_rows"] = Path(config["test_datasets"][0]["path"]).read_text(encoding="utf-8")
        self.calls.append(call)
        output_dir = Path(config["output_dir"])
        (output_dir / "checkpoint-1").mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoint-1" / "optimizer.pt").write_bytes(b"estado")
        if callable(self.run_effect):
            self.run_effect()
        elif self.run_effect is not None:
            raise self.run_effect
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "adapter_model.safetensors").write_bytes(b"pesos")
        return CompletedProcess(argv, 0)

    def run_cli(self, *extra: str, model: str = "Qwen/Qwen3.5-0.8B") -> str:
        argv = [
            "train_axolotl.py",
            "--dataset", str(self.dataset),
            "--name", "demo",
            "--output-root", str(self.output_root),
            "--model", model,
            *extra,
        ]
        config_loader = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_type=self.model_type)
        )
        tokenizer_loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
        chat_template_utils = SimpleNamespace(
            get_training_chat_template=lambda tokenizer: "plantilla-con-mascara"
        )
        modules = {
            "transformers": SimpleNamespace(AutoConfig=config_loader, AutoTokenizer=tokenizer_loader),
            "trl": ModuleType("trl"),
            "trl.chat_template_utils": chat_template_utils,
        }

        def fake_version(name: str) -> str:
            return "0.18.0" if name == "axolotl" else "1.0.0"

        output = StringIO()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(sys, "argv", argv),
            mock.patch("importlib.metadata.version", fake_version),
            mock.patch("subprocess.run", self.fake_run),
            mock.patch.dict(os.environ, {"AXOLOTL_CLI": str(self.cli)}),
            redirect_stdout(output),
        ):
            runpy.run_path(str(TRAINER / "train_axolotl.py"), run_name="__main__")
        return output.getvalue()

    def manifest(self) -> dict:
        return json.loads((self.output_root / "demo" / "manifest.json").read_text(encoding="utf-8"))

    def test_qwen3_5_trains_attention_mlp_and_records_it_in_manifest(self):
        self.run_cli("--validation-dataset", str(self.validation))

        config = self.calls[0]["config"]
        self.assertEqual(config["lora_target_modules"], ATTENTION_MLP)
        self.assertNotIn("lora_target_linear", config)
        parameters = self.manifest()["parameters"]
        self.assertEqual(parameters["targetModules"], "attention-mlp")
        self.assertEqual(parameters["targetModuleSuffixes"], ATTENTION_MLP)
        saved = yaml.safe_load(
            (self.output_root / "demo" / "axolotl-config.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(saved["lora_target_modules"], ATTENTION_MLP)

    def test_axolotl_receives_the_same_normalized_jsonl_as_train_py(self):
        from dataset_validation import load_jsonl

        self.run_cli("--validation-dataset", str(self.validation))

        expected_train, _ = load_jsonl(self.dataset)
        expected_validation, _ = load_jsonl(self.validation)
        call = self.calls[0]
        received_train = [json.loads(line) for line in call["train_rows"].splitlines()]
        received_validation = [json.loads(line) for line in call["validation_rows"].splitlines()]
        self.assertEqual(received_train, expected_train)
        self.assertEqual(received_validation, expected_validation)
        self.assertEqual(received_train[0]["messages"][2], {"role": "assistant", "content": "Un enemigo básico."})
        self.assertNotEqual(Path(call["config"]["datasets"][0]["path"]), self.dataset.resolve())
        # El manifest conserva la ruta y el hash del dataset original.
        manifest = self.manifest()
        self.assertEqual(manifest["dataset"], str(self.dataset.resolve()))
        self.assertEqual(
            manifest["reproducibility"]["datasetSha256"],
            hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["validationDataset"], str(self.validation.resolve()))
        # Las copias normalizadas viven en el directorio de trabajo y no sobreviven.
        self.assertEqual(sorted(path.name for path in self.output_root.iterdir()), ["demo"])

    def test_truncated_assistant_response_aborts_before_running_axolotl(self):
        with self.assertRaisesRegex(RuntimeError, r"truncadas por --max-length=11"):
            self.run_cli("--max-length", "11")

        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.output_root.iterdir()), [])

    def test_allow_truncation_lets_a_truncated_dataset_train(self):
        output = self.run_cli("--max-length", "11", "--allow-truncation")

        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.output_root / "demo" / "manifest.json").is_file())
        audit = json.loads(next(line for line in output.splitlines() if '"dataset": "train"' in line))
        self.assertEqual(audit["targetTruncatedExamples"], 1)
        self.assertIn("serán truncadas porque se solicitó --allow-truncation", output)

    def test_other_architectures_keep_all_linear_and_existing_manifest_fields(self):
        self.model_type = "llama"
        output = self.run_cli(model="meta-llama/Llama-3.2-1B")

        config = self.calls[0]["config"]
        self.assertIs(config["lora_target_linear"], True)
        self.assertNotIn("lora_target_modules", config)
        self.assertEqual(config["chat_template"], "tokenizer_default")
        manifest = self.manifest()
        self.assertEqual(manifest["backend"], "axolotl")
        self.assertEqual(manifest["modelType"], "llama")
        self.assertEqual(manifest["examples"], 1)
        self.assertEqual(manifest["validationExamples"], 0)
        self.assertEqual(manifest["dataQuality"]["trainValidationPromptOverlap"], 0)
        self.assertEqual(manifest["parameters"]["targetModules"], "all-linear")
        self.assertEqual(manifest["parameters"]["samplePacking"], False)
        self.assertEqual(manifest["serving"]["systemPrompt"], "Eres un asistente.")
        result = json.loads(output.strip().splitlines()[-1])
        self.assertEqual(result["adapter"], str((self.output_root / "demo").resolve()))
        self.assertEqual(result["backend"], "axolotl")

    def create_unrelated_entries(self) -> dict[str, str]:
        """Otro adaptador terminado y el staging de otra corrida, que nunca se tocan."""
        entries = {
            "existing/adapter_config.json": "{}",
            ".other.axolotl-output-abc123/checkpoint-1/optimizer.pt": "estado",
        }
        for relative, content in entries.items():
            path = self.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return entries

    def assert_only(self, entries: dict[str, str]) -> None:
        files = {
            str(path.relative_to(self.output_root)): path.read_text(encoding="utf-8")
            for path in self.output_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(files, entries)
        self.assertEqual(
            sorted(path.name for path in self.output_root.iterdir()),
            sorted({relative.split("/")[0] for relative in entries}),
        )

    def test_cancelled_run_removes_only_its_own_staging_and_work_dir(self):
        entries = self.create_unrelated_entries()
        self.run_effect = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.run_cli()

        self.assertEqual(len(self.calls), 1)
        self.assert_only(entries)

    def test_sigterm_cancellation_removes_only_its_own_staging_and_work_dir(self):
        entries = self.create_unrelated_entries()
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)

        def fallback(signum, frame):
            raise AssertionError("train_axolotl.py no instaló su handler de SIGTERM")

        signal.signal(signal.SIGTERM, fallback)
        self.run_effect = lambda: os.kill(os.getpid(), signal.SIGTERM)

        with self.assertRaises(SystemExit) as raised:
            self.run_cli()

        self.assertEqual(raised.exception.code, 143)
        self.assertEqual(len(self.calls), 1)
        self.assert_only(entries)

    def test_failed_run_still_removes_its_staging_and_work_dir(self):
        entries = self.create_unrelated_entries()
        self.run_effect = CalledProcessError(1, ["axolotl", "train"])

        with self.assertRaises(CalledProcessError):
            self.run_cli()

        self.assert_only(entries)

    def test_existing_adapter_is_never_overwritten(self):
        entries = self.create_unrelated_entries()
        entries["demo/adapter_config.json"] = '{"previo": true}'
        (self.output_root / "demo").mkdir()
        (self.output_root / "demo" / "adapter_config.json").write_text('{"previo": true}', encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "El adaptador ya existe"):
            self.run_cli()

        self.assertEqual(self.calls, [])
        self.assert_only(entries)


if __name__ == "__main__":
    unittest.main()

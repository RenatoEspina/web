"""Ejecuta trainer/train.py como CLI sin CUDA ni paquetes de ML reales."""
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))


class FakeTokenizer:
    """Un token por palabra y uno por rol; la máscara marca las palabras assistant."""

    chat_template = "plantilla-original"
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        for message in messages:
            words = message["content"].split()
            input_ids += [0] + [1] * len(words)
            assistant_masks += [0] + [int(message["role"] == "assistant")] * len(words)
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


class TrainCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output_root = self.directory / "adapters"
        self.dataset = self.directory / "demo-training.jsonl"
        self.dataset.write_text(
            json.dumps({
                "messages": [
                    {"role": "user", "content": "¿Qué es un slime?"},
                    {"role": "assistant", "content": "Un enemigo básico del principio del juego."},
                ],
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.config_effect = None
        self.stdout = StringIO()

    def load_config(self, *args, **kwargs):
        if self.config_effect is not None:
            raise self.config_effect
        return SimpleNamespace(model_type="qwen3_5")

    def run_cli(self, *extra: str) -> None:
        argv = [
            "train.py",
            "--dataset", str(self.dataset),
            "--name", "demo",
            "--output-root", str(self.output_root),
            "--model", "test-base",
            *extra,
        ]
        dependencies = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            "datasets": SimpleNamespace(Dataset=mock.Mock()),
            "peft": SimpleNamespace(LoraConfig=mock.Mock(), prepare_model_for_kbit_training=mock.Mock()),
            "transformers": SimpleNamespace(
                AutoConfig=SimpleNamespace(from_pretrained=self.load_config),
                AutoModelForCausalLM=mock.Mock(),
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
                BitsAndBytesConfig=mock.Mock(),
                set_seed=lambda seed: None,
            ),
            "trl": SimpleNamespace(SFTConfig=mock.Mock(), SFTTrainer=mock.Mock()),
            "trl.chat_template_utils": SimpleNamespace(get_training_chat_template=lambda tokenizer: None),
        }
        with (
            mock.patch.dict(sys.modules, dependencies),
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(os.environ),
            redirect_stdout(self.stdout),
        ):
            runpy.run_path(str(TRAINER / "train.py"), run_name="__main__")

    def test_truncation_audit_still_aborts_train_py_and_removes_staging(self):
        with self.assertRaisesRegex(
            RuntimeError, r"1 respuestas de train quedarían truncadas por --max-length=8"
        ):
            self.run_cli("--max-length", "8")

        audit = json.loads(next(
            line for line in self.stdout.getvalue().splitlines() if '"dataset": "train"' in line
        ))
        self.assertEqual(audit["targetTruncatedExamples"], 1)
        self.assertEqual(audit["maxLength"], 8)
        self.assertEqual(list(self.output_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

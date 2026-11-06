"""Ejecuta trainer/train.py como CLI sin CUDA ni paquetes de ML reales."""
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


class UnhandledSigterm(BaseException):
    """Lo lanza el handler de respaldo del test si la CLI no instaló el suyo."""


def install_fallback_sigterm(test: unittest.TestCase) -> None:
    previous = signal.getsignal(signal.SIGTERM)

    def fallback(signum, frame):
        raise UnhandledSigterm()

    signal.signal(signal.SIGTERM, fallback)
    test.addCleanup(signal.signal, signal.SIGTERM, previous)


def send_sigterm(*args, **kwargs) -> None:
    os.kill(os.getpid(), signal.SIGTERM)


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
        if callable(self.config_effect):
            self.config_effect()
        elif self.config_effect is not None:
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

    def create_unrelated_entries(self) -> dict[str, str]:
        """Otro adaptador terminado y el staging de otra corrida, que nunca se tocan."""
        entries = {
            "existing/adapter_config.json": "{}",
            ".other.training-abc123/checkpoints/optimizer.pt": "estado",
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

    def test_cancelled_training_removes_only_its_own_staging(self):
        entries = self.create_unrelated_entries()
        self.config_effect = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.run_cli()

        self.assert_only(entries)

    def test_sigterm_cancellation_removes_only_its_own_staging(self):
        entries = self.create_unrelated_entries()
        install_fallback_sigterm(self)
        self.config_effect = send_sigterm

        with self.assertRaises(SystemExit) as raised:
            self.run_cli()

        self.assertEqual(raised.exception.code, 143)
        self.assert_only(entries)

    def test_existing_adapter_is_never_overwritten(self):
        entries = self.create_unrelated_entries()
        entries["demo/adapter_config.json"] = '{"previo": true}'
        (self.output_root / "demo").mkdir()
        (self.output_root / "demo" / "adapter_config.json").write_text('{"previo": true}', encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "El adaptador ya existe"):
            self.run_cli()

        self.assert_only(entries)


if __name__ == "__main__":
    unittest.main()

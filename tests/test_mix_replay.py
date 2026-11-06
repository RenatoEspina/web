"""Ejecuta el CLI de trainer/mix_replay.py contra directorios temporales."""
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

import mix_replay  # noqa: E402


def example(prompt: str, answer: str = "Respuesta.") -> dict:
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class MixReplayCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output_dir = self.root / "trainer" / "datasets"
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.domain = self.inputs / "dom-training.jsonl"
        self.replay = self.inputs / "rep-training.jsonl"
        write_jsonl(self.domain, [example(f"Pregunta de dominio {number}") for number in range(8)])
        write_jsonl(self.inputs / "dom-validation.jsonl", [example(f"Validación de dominio {number}") for number in range(4)])
        write_jsonl(self.replay, [example(f"Pregunta general {number}") for number in range(8)])
        write_jsonl(self.inputs / "rep-validation.jsonl", [example(f"Validación general {number}") for number in range(4)])
        self.training_path = self.output_dir / "mix-training.jsonl"
        self.validation_path = self.output_dir / "mix-validation.jsonl"
        self.stdout = StringIO()
        self.stderr = StringIO()

    def run_cli(self, *extra: str) -> None:
        argv = [
            "mix_replay.py",
            "--domain", str(self.domain),
            "--replay", str(self.replay),
            "--name", "mix",
            *extra,
        ]
        with (
            mock.patch.object(mix_replay, "OUTPUT_DIR", self.output_dir),
            mock.patch.object(mix_replay, "ROOT", self.root),
            mock.patch.object(sys, "argv", argv),
            redirect_stdout(self.stdout),
            redirect_stderr(self.stderr),
        ):
            mix_replay.main()

    def write_previous_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_path.write_text("entrenamiento anterior\n", encoding="utf-8")
        self.validation_path.write_text("validación anterior\n", encoding="utf-8")

    def assert_previous_outputs_untouched(self) -> None:
        self.assertEqual(self.training_path.read_text(encoding="utf-8"), "entrenamiento anterior\n")
        self.assertEqual(self.validation_path.read_text(encoding="utf-8"), "validación anterior\n")
        self.assertEqual(
            sorted(path.name for path in self.output_dir.iterdir()),
            ["mix-training.jsonl", "mix-validation.jsonl"],
        )

    def rows(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_writes_both_mixed_splits_and_reports_them(self):
        self.run_cli()

        report = json.loads(self.stdout.getvalue())
        self.assertEqual(
            list(report),
            ["name", "ratio", "seed", "training", "trainingPath", "validation", "validationPath"],
        )
        self.assertEqual(report["training"], {"domain": 8, "replay": 2, "total": 10, "replayShare": 0.2})
        self.assertEqual(report["validation"], {"domain": 4, "replay": 1, "total": 5, "replayShare": 0.2})
        self.assertEqual(report["trainingPath"], "trainer/datasets/mix-training.jsonl")
        self.assertEqual(report["validationPath"], "trainer/datasets/mix-validation.jsonl")
        self.assertEqual(len(self.rows(self.training_path)), 10)
        self.assertEqual(len(self.rows(self.validation_path)), 5)
        self.assertEqual(self.stderr.getvalue(), "")
        self.assertEqual(
            sorted(path.name for path in self.output_dir.iterdir()),
            ["mix-training.jsonl", "mix-validation.jsonl"],
        )

    def test_invalid_validation_writes_nothing(self):
        cases = {
            "fuga": (
                [example("Pregunta de dominio 3")],
                "La validación mezclada repite 1 prompts del entrenamiento",
            ),
            "json inválido": (None, "JSON inválido"),
        }
        for label, (validation_rows, message) in cases.items():
            with self.subTest(label):
                self.write_previous_outputs()
                validation = self.inputs / "dom-validation.jsonl"
                if validation_rows is None:
                    validation.write_text("{no es json\n", encoding="utf-8")
                else:
                    write_jsonl(validation, validation_rows)

                with self.assertRaises((SystemExit, ValueError)) as raised:
                    self.run_cli()

                error = raised.exception
                self.assertIn(message, str(error.code if isinstance(error, SystemExit) else error))
                self.assert_previous_outputs_untouched()

    def test_without_validation_removes_the_stale_validation_file(self):
        self.write_previous_outputs()
        (self.inputs / "dom-validation.jsonl").unlink()

        self.run_cli()

        report = json.loads(self.stdout.getvalue())
        self.assertIsNone(report["validation"])
        self.assertNotIn("validationPath", report)
        self.assertEqual(len(self.rows(self.training_path)), 10)
        self.assertFalse(self.validation_path.exists())
        self.assertIn("falta la validación del dominio", self.stderr.getvalue())
        self.assertEqual(sorted(path.name for path in self.output_dir.iterdir()), ["mix-training.jsonl"])

    def test_interrupted_publish_keeps_previous_files_and_leaves_no_temp_files(self):
        self.write_previous_outputs()

        with mock.patch("os.replace", side_effect=OSError("disco lleno")):
            with self.assertRaisesRegex(OSError, "disco lleno"):
                self.run_cli()

        self.assert_previous_outputs_untouched()


if __name__ == "__main__":
    unittest.main()

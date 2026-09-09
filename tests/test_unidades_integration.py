"""Exercise corpus dispatch without starting training, Docker or a GUI server."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class UnidadesIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/train-adapter.sh", self.root / "scripts/train-adapter.sh")
        shutil.copy(ROOT / "fine-tune-gui", self.root / "fine-tune-gui")
        corpus = self.root / "trainer/corpora/unidades"
        corpus.mkdir(parents=True)
        # A small fixture tests shell routing; mathematical corpus correctness is
        # checked separately against the actual builder and verifier.
        (corpus / "build.py").write_text(
            "from pathlib import Path\n"
            "root = Path(__file__).resolve().parents[3]\n"
            "paths = ['trainer/examples/unidades-training.jsonl', "
            "'trainer/corpora/unidades/validation.jsonl', "
            "'trainer/corpora/unidades/evaluation.jsonl', "
            "'trainer/corpora/unidades/challenge.jsonl']\n"
            "for name in paths:\n"
            "    path = root / name\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text('{}\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        self.log = self.root / "calls.jsonl"
        self.python = self.bin / "trainer-python"
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, subprocess, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['UNIDADES_TEST_CALLS'], 'a') as target:\n"
            "    target.write(json.dumps(args) + '\\n')\n"
            "if args[0].endswith('/build.py'):\n"
            "    sys.exit(subprocess.call([sys.executable, *args]))\n"
            "if args[0].endswith('/gui_server.py'):\n"
            "    sys.exit(9)\n",
            encoding="utf-8",
        )
        self.python.chmod(0o755)
        (self.bin / "python3").symlink_to(self.python)
        (self.bin / "python3.13").symlink_to(self.python)
        # Docker and health checks cannot reach the user's services.
        for command in ("docker", "curl"):
            path = self.bin / command
            path.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            path.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
            "FINE_TUNE_PYTHON": str(self.python),
            "UNIDADES_TEST_CALLS": str(self.log),
        }

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def train(self, *extra):
        result = subprocess.run(
            ["bash", "scripts/train-adapter.sh", "trainer/examples/unidades-training.jsonl", "units-test", *extra],
            cwd=self.root, env=self.env, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return next(call for call in self.calls() if call[0] == "trainer/train.py")

    def test_clean_clone_builds_corpus_and_selects_only_validation(self):
        call = self.train()
        expected = "trainer/corpora/unidades/validation.jsonl"
        self.assertEqual(call[call.index("--validation-dataset") + 1], expected)
        self.assertTrue((self.root / expected).is_file())
        self.assertNotIn("trainer/corpora/unidades/evaluation.jsonl", call)
        self.assertNotIn("trainer/corpora/unidades/challenge.jsonl", call)
        self.assertEqual(
            [path.name for path in (self.root / "trainer/examples").glob("*.jsonl")],
            ["unidades-training.jsonl"],
        )

    def test_missing_validation_is_rebuilt_with_existing_training(self):
        training = self.root / "trainer/examples/unidades-training.jsonl"
        training.parent.mkdir(parents=True)
        training.write_text("{}\n", encoding="utf-8")
        call = self.train()
        self.assertIn("--validation-dataset", call)
        self.assertTrue(any(args[0].endswith("unidades/build.py") for args in self.calls()))

    def test_explicit_validation_is_preserved(self):
        call = self.train("--validation-dataset", "custom-validation.jsonl")
        self.assertEqual(call.count("--validation-dataset"), 1)
        self.assertEqual(call[call.index("--validation-dataset") + 1], "custom-validation.jsonl")

    def test_gui_materializes_before_attempting_server_start(self):
        result = subprocess.run(
            ["bash", "fine-tune-gui"], cwd=self.root, env=self.env,
            text=True, capture_output=True, timeout=10,
        )
        # The simulated server intentionally fails before desktop installation.
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        calls = self.calls()
        builder = next(index for index, args in enumerate(calls) if args[0].endswith("unidades/build.py"))
        server = next(index for index, args in enumerate(calls) if args[0].endswith("gui_server.py"))
        self.assertLess(builder, server)
        self.assertTrue((self.root / "trainer/examples/unidades-training.jsonl").is_file())
        self.assertFalse((self.root / "trainer/examples/validation.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

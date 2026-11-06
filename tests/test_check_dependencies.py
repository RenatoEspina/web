"""Ejecuta trainer/check_dependencies.py como CLI con torch, datasets y Transformers simulados."""
import re
import runpy
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "trainer" / "check_dependencies.py"
PINNED = dict(
    re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+)", line.strip()).groups()
    for line in (ROOT / "trainer" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
)
RTX_3060 = (8, 6, "NVIDIA GeForce RTX 3060 Laptop GPU")
RTX_5060 = (12, 0, "NVIDIA GeForce RTX 5060 Laptop GPU")
CU124_ARCHS = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU130_ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def fake_torch(version="2.12.1+cu130", cuda="13.0", available=True, devices=(RTX_3060,), archs=CU130_ARCHS):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(devices),
            get_device_capability=lambda index: devices[index][:2],
            get_device_name=lambda index: devices[index][2],
            get_arch_list=lambda: list(archs),
        ),
    )


class CheckDependenciesCliTests(unittest.TestCase):
    def run_cli(self, torch, versions=None):
        """Devuelve (código de salida, stdout, stderr); `torch=None` simula que no está instalado."""
        installed = dict(PINNED, **(versions or {}))
        modules = {
            "torch": torch,
            "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: list(rows))),
            "transformers": SimpleNamespace(),
            "transformers.models": SimpleNamespace(),
            "transformers.models.auto": SimpleNamespace(),
            "transformers.models.auto.configuration_auto": SimpleNamespace(CONFIG_MAPPING={"qwen3_5": object()}),
            "transformers.models.auto.modeling_auto": SimpleNamespace(
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES={"qwen3_5": "Qwen3_5ForCausalLM"}
            ),
        }
        stdout, stderr = StringIO(), StringIO()
        code = 0
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch("importlib.metadata.version", lambda name: installed[name]),
            mock.patch.object(sys, "argv", [str(SCRIPT)]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                runpy.run_path(str(SCRIPT), run_name="__main__")
            except SystemExit as exit_:
                code = exit_.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_reports_torch_version_cuda_build_and_compatible_gpu(self):
        code, stdout, stderr = self.run_cli(fake_torch())

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        lines = stdout.splitlines()
        self.assertEqual(lines[1:5], [
            f"datasets: {PINNED['datasets']}",
            f"transformers: {PINNED['transformers']} (Qwen3.5 OK)",
            f"trl: {PINNED['trl']}",
            f"peft: {PINNED['peft']}",
        ])
        self.assertEqual(lines[5:], [
            "torch: 2.12.1+cu130 (CUDA 13.0)",
            "GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (sm_86, compatible)",
            "Dependencias del trainer: OK",
        ])

    def test_gpu_architecture_missing_from_torch_build_fails_clearly(self):
        code, stdout, stderr = self.run_cli(
            fake_torch(version="2.5.1+cu124", cuda="12.4", devices=(RTX_5060,), archs=CU124_ARCHS)
        )

        self.assertEqual(code, 1)
        self.assertNotIn("Dependencias del trainer: OK", stdout)
        self.assertIn("La build de torch no soporta la GPU visible:", stderr)
        self.assertIn("- GPU 0: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)", stderr)
        self.assertIn("torch 2.5.1+cu124 (CUDA 12.4) incluye: " + " ".join(CU124_ARCHS), stderr)

    def test_architecture_check_follows_cuda_binary_compatibility(self):
        cases = {
            "sm_89 corre los cubins sm_86": ((8, 9, "Ada"), CU124_ARCHS, 0),
            "PTX compute_90 se compila para sm_120": ((12, 0, "Blackwell"), ["sm_80", "compute_90"], 0),
            "sm_80 no corre cubins sm_86": ((8, 0, "Ampere"), ["sm_86", "sm_90"], 1),
            "sm_120 no corre cubins sm_90": ((12, 0, "Blackwell"), ["sm_90"], 1),
        }
        for label, (device, archs, expected) in cases.items():
            with self.subTest(label):
                code, _, _ = self.run_cli(fake_torch(devices=(device,), archs=archs))
                self.assertEqual(code, expected)

    def test_missing_torch_fails_clearly(self):
        code, stdout, stderr = self.run_cli(None)

        self.assertEqual(code, 1)
        self.assertNotIn("Dependencias del trainer: OK", stdout)
        self.assertIn("torch: no instalado", stderr)

    def test_without_cuda_reports_it_and_skips_the_architecture_check(self):
        cases = {
            "sin GPU visible": (fake_torch(available=False, devices=()), "torch: 2.12.1+cu130 (CUDA 13.0)",
                                "CUDA no está disponible"),
            "build sin CUDA": (fake_torch(version="2.12.1+cpu", cuda=None, available=False, devices=(), archs=()),
                               "torch: 2.12.1+cpu (sin soporte CUDA)", "no tiene soporte CUDA"),
        }
        for label, (torch, torch_line, warning) in cases.items():
            with self.subTest(label):
                code, stdout, stderr = self.run_cli(torch)

                self.assertEqual(code, 0)
                self.assertEqual(stdout.splitlines()[-2:], [torch_line, "Dependencias del trainer: OK"])
                self.assertIn(warning, stderr)
                self.assertIn("no se comprobó la arquitectura de la GPU", stderr)

    def test_pin_mismatch_still_fails_before_any_torch_check(self):
        code, stdout, stderr = self.run_cli(None, versions={"transformers": "4.0.0"})

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(
            stderr,
            "Dependencias del trainer desactualizadas:\n"
            f"- transformers: 4.0.0 (esperado {PINNED['transformers']})\n",
        )


if __name__ == "__main__":
    unittest.main()

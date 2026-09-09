"""Exercise the actual trainer entrypoint without requiring CUDA or ML packages."""
import random
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))


class TrainingSeedTests(unittest.TestCase):
    def test_seed_is_applied_before_any_model_initialization(self):
        observed = []
        seeds = []

        class ReachedModelInitialization(Exception):
            pass

        def set_seed(seed):
            seeds.append(seed)
            random.seed(seed)

        def load_config(*args, **kwargs):
            observed.append(random.random())
            raise ReachedModelInitialization()

        # Stub only external dependencies. main(), dataset validation and staging
        # cleanup run unchanged; abort at the first model-loading operation.
        dependencies = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            "datasets": SimpleNamespace(Dataset=mock.Mock()),
            "peft": SimpleNamespace(LoraConfig=mock.Mock(), prepare_model_for_kbit_training=mock.Mock()),
            "transformers": SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=load_config),
                                            AutoModelForCausalLM=mock.Mock(), AutoTokenizer=mock.Mock(),
                                            BitsAndBytesConfig=mock.Mock(), set_seed=set_seed),
            "trl": SimpleNamespace(SFTConfig=mock.Mock(), SFTTrainer=mock.Mock()),
            "trl.chat_template_utils": SimpleNamespace(get_training_chat_template=mock.Mock()),
        }
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, dependencies):
            namespace = runpy.run_path(str(ROOT / "trainer/train.py"))
            options = SimpleNamespace(seed=42, name="test", model="test-base",
                                      dataset=ROOT / "trainer/examples/base-training.jsonl",
                                      validation_dataset=None, output_root=Path(temporary))
            with mock.patch.dict(namespace["main"].__globals__, arguments=lambda: options):
                for previous_seed in [123, 456]:
                    random.seed(previous_seed)
                    with self.assertRaises(ReachedModelInitialization):
                        namespace["main"]()
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertEqual(seeds, [42, 42])
        self.assertEqual(observed[0], observed[1])


if __name__ == "__main__":
    unittest.main()

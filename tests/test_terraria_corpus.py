import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
from dataset_validation import load_jsonl
import gui_server

DIRECTORY = ROOT / "trainer" / "corpora" / "terraria"
spec = importlib.util.spec_from_file_location("terraria_builder", DIRECTORY / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class TerrariaCorpusTests(unittest.TestCase):
    def test_training_is_valid_sft_and_visible_in_gui(self):
        examples, summary = load_jsonl(ROOT / "trainer/examples/terraria-training.jsonl")
        self.assertEqual(summary["examples"], 96)
        self.assertEqual(len(examples), 96)
        self.assertIn("examples/terraria-training.jsonl", {entry["id"] for entry in gui_server.list_datasets()})

    def test_generated_files_are_reproducible(self):
        corpus = json.loads((DIRECTORY / "curated.json").read_text())
        train, evaluation = builder.render(corpus)
        self.assertEqual(train, (ROOT / "trainer/examples/terraria-training.jsonl").read_text())
        self.assertEqual(evaluation, (DIRECTORY / "evaluation.jsonl").read_text())

    def test_evaluation_has_no_assistant_targets_and_is_not_selectable_for_sft(self):
        rows = [json.loads(line) for line in (DIRECTORY / "evaluation.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertTrue(row["contains"])
            self.assertEqual(row["messages"][-1]["role"], "user")
            self.assertNotIn("assistant", {message["role"] for message in row["messages"]})
            self.assertTrue(row["reference_answer"])
        with self.assertRaises(ValueError):
            gui_server.dataset_path("corpora/terraria/evaluation.jsonl")
        with self.assertRaisesRegex(ValueError, "terminar|assistant"):
            load_jsonl(DIRECTORY / "evaluation.jsonl")

    def test_sources_and_questions_are_disjoint_between_splits(self):
        corpus = json.loads((DIRECTORY / "curated.json").read_text())
        train = [source for source in corpus["sources"] if source["split"] == "train"]
        evaluation = [source for source in corpus["sources"] if source["split"] == "evaluation"]
        self.assertFalse({x["url"] for x in train} & {x["url"] for x in evaluation})
        self.assertFalse({qa[0] for x in train for qa in x["qa"]} & {qa[0] for x in evaluation for qa in x["qa"]})

    def test_every_example_retains_official_attribution_and_license(self):
        for path in (ROOT / "trainer/examples/terraria-training.jsonl", DIRECTORY / "evaluation.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.assertTrue(row["source"]["url"].startswith("https://terraria.wiki.gg/wiki/"))
                self.assertTrue(row["source"]["history_url"].endswith("?action=history"))
                self.assertEqual(row["license"], "CC-BY-NC-SA-4.0")
                self.assertIn("Colaboradores", row["attribution"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "trainer" / "corpora" / "terraria"
spec = importlib.util.spec_from_file_location("terraria_builder", DIRECTORY / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def decode(content: str) -> list[dict]:
    return [json.loads(line) for line in content.splitlines() if line.strip()]


class TerrariaCorpusTests(unittest.TestCase):
    def test_english_corpus_has_three_source_disjoint_splits(self):
        corpus = builder.load_corpus()
        train, validation, evaluation = builder.render(corpus)
        train_rows = decode(train)
        validation_rows = decode(validation)
        evaluation_rows = decode(evaluation)

        self.assertEqual(len(train_rows), 160)
        self.assertEqual(len(validation_rows), 24)
        self.assertEqual(len(evaluation_rows), 16)
        self.assertEqual(len(corpus["sources"]), 50)
        self.assertIn("in English", corpus["system"])

        counts = {
            split: len([source for source in corpus["sources"] if source["split"] == split])
            for split in ("train", "validation", "evaluation")
        }
        self.assertEqual(counts, {"train": 40, "validation": 6, "evaluation": 4})

        for row in train_rows + validation_rows:
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertIn("in English", row["messages"][0]["content"])

        for row in evaluation_rows:
            self.assertTrue(row["contains"])
            self.assertEqual(row["messages"][-1]["role"], "user")
            self.assertNotIn("assistant", {message["role"] for message in row["messages"]})
            self.assertTrue(row["reference_answer"])

    def test_training_and_validation_are_valid_sft_jsonl(self):
        corpus = builder.load_corpus()
        train, validation, _ = builder.render(corpus)

        # Import lazily so this test remains focused on the generated SFT format.
        import sys
        sys.path.insert(0, str(ROOT / "trainer"))
        from dataset_validation import load_jsonl

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_path = root / "training.jsonl"
            validation_path = root / "validation.jsonl"
            training_path.write_text(train, encoding="utf-8")
            validation_path.write_text(validation, encoding="utf-8")

            training_examples, training_summary = load_jsonl(training_path)
            validation_examples, validation_summary = load_jsonl(validation_path)

        self.assertEqual(training_summary["examples"], 160)
        self.assertEqual(validation_summary["examples"], 24)
        self.assertEqual(len(training_examples), 160)
        self.assertEqual(len(validation_examples), 24)

    def test_sources_and_questions_are_disjoint_between_all_splits(self):
        corpus = builder.load_corpus()
        groups = {
            split: [source for source in corpus["sources"] if source["split"] == split]
            for split in ("train", "validation", "evaluation")
        }
        for left, right in (("train", "validation"), ("train", "evaluation"), ("validation", "evaluation")):
            left_urls = {source["url"] for source in groups[left]}
            right_urls = {source["url"] for source in groups[right]}
            left_questions = {qa[0].casefold() for source in groups[left] for qa in source["qa"]}
            right_questions = {qa[0].casefold() for source in groups[right] for qa in source["qa"]}
            self.assertFalse(left_urls & right_urls)
            self.assertFalse(left_questions & right_questions)

    def test_every_example_retains_official_attribution_and_license(self):
        corpus = builder.load_corpus()
        rendered = builder.render(corpus)
        for content in rendered:
            for row in decode(content):
                self.assertTrue(row["source"]["url"].startswith("https://terraria.wiki.gg/wiki/"))
                self.assertTrue(row["source"]["history_url"].endswith("?action=history"))
                self.assertEqual(row["license"], "CC-BY-NC-SA-4.0")
                self.assertIn("Contributors", row["attribution"])

    def test_source_shards_are_the_canonical_corpus(self):
        data = DIRECTORY / "data"
        self.assertTrue((data / "metadata.json").is_file())
        shards = sorted(data.glob("sources-*.json"))
        self.assertGreaterEqual(len(shards), 5)
        self.assertFalse((DIRECTORY / "curated.json").exists())


if __name__ == "__main__":
    unittest.main()

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from training_quality import (
    count_prompt_overlaps,
    sha256_file,
    stable_sha256,
    summarize_token_lengths,
)


class TrainingQualityTests(unittest.TestCase):
    def test_file_and_structured_hashes_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.jsonl"
            path.write_text("uno\n", encoding="utf-8")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"uno\n").hexdigest())
        self.assertEqual(
            stable_sha256({"b": 2, "a": 1}),
            stable_sha256({"a": 1, "b": 2}),
        )

    def test_counts_normalized_final_user_prompt_overlap(self):
        first = [{"messages": [{"role": "user", "content": " ¿Qué hace X?  "}]}]
        second = [{"messages": [{"role": "user", "content": "qué hace x?"}]}]
        different = [{"messages": [{"role": "user", "content": "¿Qué hace Y?"}]}]
        self.assertEqual(count_prompt_overlaps(first, second), 1)
        self.assertEqual(count_prompt_overlaps(first, different), 0)

    def test_summarizes_token_distribution_and_truncation(self):
        summary = summarize_token_lengths(
            [10, 20, 30, 40],
            [4, 8, 12, 16],
            32,
            [False, False, True, True],
            [False, True, False, True],
            [False, False, False, False],
        )
        self.assertEqual(summary["examples"], 4)
        self.assertEqual(summary["totalTokens"]["sum"], 100)
        self.assertEqual(summary["totalTokens"]["p50"], 25)
        self.assertEqual(summary["targetTruncatedExamples"], 2)


if __name__ == "__main__":
    unittest.main()

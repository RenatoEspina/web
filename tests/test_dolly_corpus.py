"""Audita la adaptación de Dolly y la mezcla de replay sin entrenar ni descargar."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "trainer" / "corpora" / "dolly"
sys.path.insert(0, str(ROOT / "trainer"))

from dataset_validation import load_jsonl  # noqa: E402
from training_quality import _last_user_prompt, count_prompt_overlaps  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("dolly_builder", DIRECTORY / "build.py")
mixer = load("replay_mixer", ROOT / "trainer" / "mix_replay.py")
evaluator = load("evaluator", ROOT / "trainer" / "evaluate.py")

SOURCE_CACHED = (DIRECTORY / ".cache" / builder.SOURCE_FILE).exists()
TRAINING = ROOT / "trainer" / "examples" / "dolly-training.jsonl"
VALIDATION = DIRECTORY / "validation.jsonl"
EVALUATION = DIRECTORY / "evaluation.jsonl"
BUILT = TRAINING.exists() and VALIDATION.exists() and EVALUATION.exists()


def row(instruction: str, context: str = "", response: str = "ok", category: str = "open_qa") -> dict:
    return {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": category,
    }


class Conversion(unittest.TestCase):
    def test_normalization_matches_the_overlap_detector(self) -> None:
        """Si ambas normalizaciones divergen, la deduplicación deja de evitar fugas."""
        for text in ("¿Qué es el Ñandú?", "A  b\tc", "Hello, World! -- 42", "Straße  über"):
            self.assertEqual(
                builder.normalize_prompt(text),
                _last_user_prompt([{"role": "user", "content": text}]),
            )

    def test_context_is_appended_as_reference_material(self) -> None:
        self.assertEqual(builder.user_message(row("Q")), "Q")
        self.assertEqual(builder.user_message(row("Q", "C")), "Q\n\n---\nC")

    def test_filters_drop_empty_long_and_duplicate_rows(self) -> None:
        rows = [
            row("keep this one"),
            row("keep this one"),
            row("drop me", response="   "),
            row("way too long", context="x" * builder.MAX_CHARACTERS),
            row("another good one"),
        ]
        examples, dropped = builder.convert(rows, builder.MAX_CHARACTERS)
        self.assertEqual([example["user"] for example in examples], ["keep this one", "another good one"])
        self.assertEqual(dropped["duplicate_prompt"], 1)
        self.assertEqual(dropped["empty"], 1)
        self.assertEqual(dropped["too_long"], 1)

    def test_single_character_answers_survive(self) -> None:
        """"7" es una respuesta válida a una pregunta de conteo, no un vacío."""
        examples, dropped = builder.convert([row("How many books?", response="7")], builder.MAX_CHARACTERS)
        self.assertEqual(len(examples), 1)
        self.assertEqual(dropped["empty"], 0)

    def test_character_budget_counts_the_system_prompt(self) -> None:
        budget = len(builder.SYSTEM) + 20
        examples, dropped = builder.convert([row("x" * 30)], budget)
        self.assertEqual(examples, [])
        self.assertEqual(dropped["too_long"], 1)

    def test_training_rows_end_on_an_assistant_turn(self) -> None:
        example = builder.convert([row("Q", "C", "A7")], builder.MAX_CHARACTERS)[0][0]
        messages = builder.training_row(example)["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[-1]["content"], "A7")


class EvaluationSelection(unittest.TestCase):
    def candidate(self, **kwargs) -> str | None:
        example = builder.convert([row(**kwargs)], builder.MAX_CHARACTERS)[0][0]
        return builder.evaluation_candidate(example)

    def test_accepts_a_short_answer_grounded_in_the_prompt(self) -> None:
        self.assertEqual(
            self.candidate(
                instruction="Which is a fish? Tope or Rope",
                response="Tope",
                category="classification",
            ),
            "Tope",
        )

    def test_rejects_answers_the_prompt_does_not_contain(self) -> None:
        """`contains` debe medir extracción, no memoria del anotador."""
        self.assertIsNone(
            self.candidate(
                instruction="Name a fish",
                response="Tope",
                category="closed_qa",
            )
        )

    def test_rejects_long_answers_and_free_form_categories(self) -> None:
        self.assertIsNone(
            self.candidate(
                instruction="Extract one two three four five",
                response="one two three four five",
                category="information_extraction",
            )
        )
        self.assertIsNone(
            self.candidate(
                instruction="Write about Tope",
                response="Tope",
                category="creative_writing",
            )
        )

    def test_evaluation_rows_stop_before_the_answer(self) -> None:
        example = builder.convert(
            [row("Which is a fish? Tope or Rope", response="Tope", category="classification")],
            builder.MAX_CHARACTERS,
        )[0][0]
        case = builder.evaluation_row(example, "Tope")
        self.assertEqual(case["messages"][-1]["role"], "user")
        self.assertEqual(case["contains"], ["Tope"])
        self.assertEqual(case["reference_answer"], "Tope")


class Stratification(unittest.TestCase):
    def examples(self, counts: dict[str, int]) -> list[dict]:
        return [
            {"id": f"{category}-{index}", "category": category, "user": "u", "answer": "a"}
            for category, total in counts.items()
            for index in range(total)
        ]

    def test_keeps_category_proportions_and_the_exact_size(self) -> None:
        import random

        examples = self.examples({"a": 600, "b": 300, "c": 100})
        chosen = builder.stratified(examples, 100, random.Random(0))
        self.assertEqual(len(chosen), 100)
        counts = {name: sum(1 for item in chosen if item["category"] == name) for name in "abc"}
        self.assertEqual(counts, {"a": 60, "b": 30, "c": 10})

    def test_returns_everything_when_the_quota_exceeds_the_pool(self) -> None:
        import random

        examples = self.examples({"a": 5})
        self.assertEqual(len(builder.stratified(examples, 50, random.Random(0))), 5)


class ReplayMix(unittest.TestCase):
    def rows(self, prefix: str, total: int) -> list[dict]:
        return [
            {
                "messages": [
                    {"role": "user", "content": f"{prefix} {index}"},
                    {"role": "assistant", "content": "a"},
                ]
            }
            for index in range(total)
        ]

    def test_replay_takes_the_requested_share_of_the_mix(self) -> None:
        for ratio in (0.05, 0.1, 0.2, 0.33):
            with self.subTest(ratio=ratio):
                mixed, report = mixer.blend(self.rows("d", 160), self.rows("r", 5000), ratio, 1, "t")
                self.assertEqual(len(mixed), report["total"])
                self.assertAlmostEqual(report["replayShare"], ratio, delta=0.01)

    def test_every_domain_example_survives_the_mix(self) -> None:
        domain = self.rows("d", 160)
        mixed, report = mixer.blend(domain, self.rows("r", 5000), 0.2, 1, "t")
        self.assertEqual(report["domain"], 160)
        for example in domain:
            self.assertIn(example, mixed)

    def test_the_mix_is_deterministic_for_a_seed(self) -> None:
        arguments = (self.rows("d", 40), self.rows("r", 400), 0.25, 7, "t")
        self.assertEqual(mixer.blend(*arguments)[0], mixer.blend(*arguments)[0])

    def test_replay_is_capped_by_what_the_replay_set_offers(self) -> None:
        _, report = mixer.blend(self.rows("d", 1000), self.rows("r", 10), 0.4, 1, "t")
        self.assertEqual(report["replay"], 10)

    def test_replay_count_solves_for_the_target_share(self) -> None:
        self.assertEqual(mixer.replay_count(160, 0.2), 40)
        self.assertEqual(mixer.replay_count(3000, 0.1), 333)


@unittest.skipUnless(SOURCE_CACHED and BUILT, "el corpus Dolly no está construido en este entorno")
class BuiltArtifacts(unittest.TestCase):
    def test_every_split_passes_the_shared_dataset_validator(self) -> None:
        for path, expected in ((TRAINING, 3000), (VALIDATION, 400)):
            with self.subTest(path=path.name):
                _, summary = load_jsonl(path)
                self.assertEqual(summary["examples"], expected)

    def test_evaluation_is_loadable_by_the_evaluation_harness(self) -> None:
        cases = evaluator.load_cases(EVALUATION)
        self.assertEqual(len(cases), 250)
        for _, case in cases:
            self.assertTrue(all(fragment.strip() for fragment in case["contains"]))

    def test_splits_do_not_share_prompts(self) -> None:
        train, _ = load_jsonl(TRAINING)
        validation, _ = load_jsonl(VALIDATION)
        evaluation = [json.loads(line) for line in EVALUATION.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(count_prompt_overlaps(train, validation), 0)
        self.assertEqual(count_prompt_overlaps(train, evaluation), 0)
        self.assertEqual(count_prompt_overlaps(validation, evaluation), 0)

    def test_evaluation_fragments_really_appear_in_their_prompt(self) -> None:
        for line in EVALUATION.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            case = json.loads(line)
            prompt = case["messages"][-1]["content"].casefold()
            for fragment in case["contains"]:
                self.assertIn(fragment.casefold(), prompt, case["id"])

    def test_the_manifest_matches_the_files_on_disk(self) -> None:
        manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["sha256"], builder.SOURCE_SHA256)
        for relative, entry in manifest["files"].items():
            with self.subTest(file=relative):
                self.assertEqual(builder.digest((ROOT / relative).read_text(encoding="utf-8")), entry["sha256"])

    def test_rebuilding_reproduces_the_same_artifacts(self) -> None:
        outputs, _ = builder.artifacts("laptop", builder.MAX_CHARACTERS)
        for path, text in outputs.items():
            with self.subTest(file=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), text)


if __name__ == "__main__":
    unittest.main()

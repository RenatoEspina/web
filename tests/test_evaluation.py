import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "evaluation.jsonl"
        self.options = SimpleNamespace(dataset=self.dataset, output=self.root / "report.json",
                                       model="test", base_url="http://unused", api_key="", max_tokens=32)

    def write_cases(self, cases):
        self.dataset.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")

    def case(self, **overrides):
        return {"messages": [{"role": "user", "content": "Pregunta"}], "contains": ["correcta"], **overrides}

    def test_rejects_missing_empty_or_non_text_criteria_before_inference(self):
        for criteria in [None, [], [""], ["  "], [None], [3], "correcta"]:
            with self.subTest(criteria=criteria):
                case = self.case(contains=criteria)
                if criteria is None:
                    del case["contains"]
                # A later invalid case must be caught before the first API call.
                self.write_cases([self.case(), case])
                with mock.patch.object(evaluate, "arguments", return_value=self.options), mock.patch.object(evaluate, "complete") as complete:
                    with self.assertRaisesRegex(ValueError, "Línea 2: contains"):
                        evaluate.main()
                    complete.assert_not_called()
                self.assertFalse(self.options.output.exists())

    def test_rejects_reference_answer_and_malformed_conversations(self):
        user = {"role": "user", "content": "Pregunta"}
        assistant = {"role": "assistant", "content": "Respuesta ideal"}
        for messages in [[], [user, assistant], [assistant, user], [user, user], ["text"],
                         [{"role": "user", "content": " "}], [{"role": "tool", "content": "x"}]]:
            with self.subTest(messages=messages):
                self.write_cases([self.case(messages=messages)])
                with self.assertRaises(ValueError):
                    evaluate.load_cases(self.dataset)

    def test_rejects_empty_dataset_and_non_objects(self):
        for text in ["\n", "[]", "null", "{"]:
            with self.subTest(text=text):
                self.dataset.write_text(text)
                with self.assertRaises(ValueError):
                    evaluate.load_cases(self.dataset)

    def test_reports_actual_pass_rate_with_multiturn_context(self):
        conversation = [{"role": "system", "content": "Ayuda"},
                        {"role": "user", "content": "Primera"},
                        {"role": "assistant", "content": "Respuesta previa"},
                        {"role": "user", "content": "Segunda"}]
        self.write_cases([self.case(messages=conversation, contains=["Integer", "Java"]), self.case()])
        with mock.patch.object(evaluate, "arguments", return_value=self.options), mock.patch.object(
            evaluate, "complete", side_effect=["JAVA utiliza INTEGER", "No sé"]
        ) as complete, mock.patch("builtins.print"):
            evaluate.main()
        report = json.loads(self.options.output.read_text())
        self.assertEqual((report["passed"], report["total"], report["passRate"]), (1, 2, 0.5))
        self.assertEqual(complete.call_args_list[0].args[3], conversation)

    def test_rejects_same_model_as_comparison_target(self):
        self.write_cases([self.case()])
        self.options.compare_model = self.options.model
        with mock.patch.object(evaluate, "arguments", return_value=self.options), mock.patch.object(
            evaluate, "complete"
        ) as complete:
            with self.assertRaisesRegex(ValueError, "compare-model"):
                evaluate.main()
            complete.assert_not_called()

    def test_compares_models_on_the_same_cases(self):
        self.write_cases([
            self.case(contains=["correcta"]),
            self.case(contains=["especial"]),
        ])
        self.options.model = "base"
        self.options.compare_model = "adapter"
        with mock.patch.object(evaluate, "arguments", return_value=self.options), mock.patch.object(
            evaluate,
            "complete",
            side_effect=[
                "respuesta correcta",
                "respuesta incompleta",
                "respuesta correcta",
                "respuesta especial",
            ],
        ), mock.patch("builtins.print"):
            evaluate.main()

        report = json.loads(self.options.output.read_text())
        self.assertEqual(report["comparison"]["passedDelta"], 1)
        self.assertEqual(report["comparison"]["passRateDelta"], 0.5)
        self.assertEqual(report["models"]["base"]["passed"], 1)
        self.assertEqual(report["models"]["adapter"]["passed"], 2)


if __name__ == "__main__":
    unittest.main()

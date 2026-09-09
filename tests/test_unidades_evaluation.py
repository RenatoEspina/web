import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
from corpora.unidades import evaluate


def case(identifier="u1", state="ok", value="10", unit="cm", category="longitud"):
    return {
        "id": identifier, "category": category, "group_id": "group-" + identifier,
        "template_id": "t1", "messages": [
            {"role": "system", "content": "Responde con JSON."},
            {"role": "user", "content": "Convierte 0.1 m a cm."},
        ],
        "expected": {"estado": state, "valor": value if state == "ok" else None,
                     "unidad": unit if state == "ok" else None},
    }


def answer(state="ok", value="10", unit="cm"):
    return json.dumps({"estado": state, "valor": value if state == "ok" else None,
                       "unidad": unit if state == "ok" else None})


class ExactEvaluationTests(unittest.TestCase):
    def test_exact_decimal_and_unit_not_numeric_substring(self):
        expected = case()["expected"]
        for value in ("10", "10.0", "1e1", "010.000", "+10"):
            result = evaluate.assess_output(answer(value=value), expected)
            self.assertTrue(result["numeric_unit_correct"])
            self.assertEqual(result["exact_correct"], value == "10")
        for value in ("100", "1", "10.00000000000000000000001", "-10"):
            self.assertFalse(evaluate.assess_output(answer(value=value), expected)["exact_correct"])
        self.assertFalse(evaluate.assess_output(answer(unit="m"), expected)["exact_correct"])
        self.assertFalse(evaluate.assess_output(answer(unit="CM"), expected)["exact_correct"])
        self.assertFalse(evaluate.assess_output(answer(value="10.0"), expected)["canonical_value_format"])

    def test_rejects_adversarial_json_and_schema(self):
        expected = case()["expected"]
        invalid_json = [
            '```json\n' + answer() + '\n```', "Resultado: " + answer(),
            answer() + " listo", answer() + answer(),
            '{"estado":"ok","estado":"ok","valor":"10","unidad":"cm"}',
            '{"estado":"ok","valor":NaN,"unidad":"cm"}',
            '{"estado":"ok","valor":Infinity,"unidad":"cm"}',
        ]
        for output in invalid_json:
            with self.subTest(output=output):
                result = evaluate.assess_output(output, expected)
                self.assertFalse(result["json_valid"])
                self.assertFalse(result["exact_correct"])
        invalid_schema = [
            answer(value=10), answer(value=True), answer(value=None),
            answer(value="NaN"), answer(value="Infinity"), answer(value="-Infinity"),
            answer(value="10 cm"), answer(value=" 10 "), answer(value="1_0"),
            answer(value=""), answer(unit=None), answer(unit=""),
            '{"estado":"ok","valor":"10","unidad":"cm","extra":true}',
            '{"estado":"ok","valor":"10"}',
            '{"estado":false,"valor":"10","unidad":"cm"}',
            "null", "[]", "true", '"10 cm"',
        ]
        for output in invalid_schema:
            with self.subTest(output=output):
                result = evaluate.assess_output(output, expected)
                self.assertTrue(result["json_valid"])
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["exact_correct"])

    def test_error_states_must_match_with_null_values(self):
        for state in evaluate.STATES[1:]:
            with self.subTest(state=state):
                expected = case(state=state)["expected"]
                self.assertTrue(evaluate.assess_output(answer(state), expected)["exact_correct"])
                self.assertFalse(evaluate.assess_output(answer(), expected)["exact_correct"])
                wrong = "incompatible" if state != "incompatible" else "falta_dato"
                self.assertFalse(evaluate.assess_output(answer(wrong), expected)["exact_correct"])
                output = json.dumps({"estado": state, "valor": "10", "unidad": None})
                result = evaluate.assess_output(output, expected)
                self.assertFalse(result["schema_valid"])
                self.assertTrue(result["invalid_request_numeric_response"])

    def test_perfect_wrong_and_invalid_request_metrics(self):
        cases = [case(), case("u2", "incompatible"), case("u3", "falta_dato"), case("u4", "no_soportada")]
        perfect = {item["id"]: json.dumps(item["expected"]) for item in cases}
        report = evaluate.evaluate_predictions(cases, perfect)
        for metric in ("exact_accuracy", "numeric_unit_accuracy", "json_validity", "schema_validity", "status_macro_f1", "invalid_request_correct_rejection_rate"):
            self.assertEqual(report[metric], 1)
        self.assertEqual(report["invalid_request_numeric_response_rate"], 0)
        self.assertEqual(report["invalid_request_error_rate"], 0)
        self.assertEqual(report["total"], 4)
        self.assertLess(report["exact_accuracy_wilson95"][0], 1)
        wrong = {item["id"]: answer(value="100") for item in cases}
        report = evaluate.evaluate_predictions(cases, wrong)
        self.assertEqual(report["exact_accuracy"], 0)
        self.assertEqual(report["numeric_unit_accuracy"], 0)
        self.assertEqual(report["invalid_request_numeric_response_rate"], 1)
        self.assertEqual(report["invalid_request_error_rate"], 1)
        self.assertGreater(report["exact_accuracy_wilson95"][1], 0)
        invalid = evaluate.evaluate_predictions(cases, {item["id"]: "oops" for item in cases})
        self.assertEqual(invalid["status_macro_f1"], 0)
        self.assertEqual(invalid["schema_validity"], 0)
        self.assertEqual(invalid["invalid_request_error_rate"], 1)

    def test_ok_denominator_includes_invalid_answers_and_wrong_status(self):
        cases = [case(), case("u2"), case("u3", "incompatible")]
        report = evaluate.evaluate_predictions(cases, {"u1": answer(), "u2": "bad", "u3": answer("incompatible")})
        self.assertEqual(report["numeric_unit_accuracy"], .5)
        self.assertEqual(report["exact_accuracy"], 2 / 3)
        self.assertEqual(report["expected_ok_total"], 2)
        self.assertEqual(report["per_category"]["longitud"]["total"], 3)

    def test_absent_denominators_and_single_class_macro_f1(self):
        report = evaluate.evaluate_predictions([case()], {"u1": answer()})
        self.assertIsNone(report["invalid_request_numeric_response_rate"])
        self.assertEqual(report["status_macro_f1"], 1)
        self.assertEqual(report["status_macro_f1_classes"], ["ok"])
        errors = evaluate.evaluate_predictions([case(state="incompatible")], {"u1": answer("incompatible")})
        self.assertIsNone(errors["numeric_unit_accuracy"])
        self.assertIsNone(evaluate.wilson95(0, 0))

    def test_paired_comparison_is_by_id_and_deterministic(self):
        cases = [case("a"), case("b"), case("c")]
        base = evaluate.evaluate_predictions(cases, {"a": answer(), "b": "bad", "c": answer()}, "base")
        candidate = evaluate.evaluate_predictions(list(reversed(cases)), {"a": "bad", "b": answer(), "c": answer()}, "ft")
        comparison = evaluate.compare_reports(base, candidate, seed=7, resamples=1000)
        self.assertEqual(comparison["improved"], 1)
        self.assertEqual(comparison["regressed"], 1)
        self.assertEqual(comparison["both_correct"], 1)
        self.assertEqual(comparison["bootstrap_unit"], "group_id")
        self.assertEqual(comparison["bootstrap_groups"], 3)
        self.assertEqual(comparison["exact_accuracy_delta"], 0)
        self.assertEqual(comparison, evaluate.compare_reports(base, candidate, seed=7, resamples=1000))
        perfect = evaluate.evaluate_predictions(cases, {item["id"]: answer() for item in cases}, "perfect")
        wrong = evaluate.evaluate_predictions(cases, {item["id"]: "bad" for item in cases}, "wrong")
        delta = evaluate.compare_reports(wrong, perfect, resamples=20)
        self.assertEqual(delta["exact_accuracy_delta_bootstrap95"], [1, 1])

    def test_bootstrap_preserves_paired_semantic_groups(self):
        cases = [case("a"), case("b"), case("c")]
        for item in cases:
            item["group_id"] = "same-group"
        base = evaluate.evaluate_predictions(cases, {"a": answer(), "b": "bad", "c": "bad"})
        candidate = evaluate.evaluate_predictions(cases, {"a": "bad", "b": answer(), "c": answer()})
        comparison = evaluate.compare_reports(base, candidate, seed=7, resamples=100)
        self.assertEqual(comparison["bootstrap_groups"], 1)
        self.assertEqual(comparison["exact_accuracy_delta"], 1 / 3)
        self.assertEqual(comparison["exact_accuracy_delta_bootstrap95"], [1 / 3, 1 / 3])

    def test_remote_errors_count_and_only_messages_sent(self):
        cases = [case(), case("u2")]
        with mock.patch.object(evaluate, "complete", side_effect=[URLError("offline"), answer()]) as complete:
            report = evaluate.evaluate_model(cases, "test", "http://unused", "", 128, 1)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["request_failures"], 1)
        self.assertEqual(report["exact_accuracy"], .5)
        self.assertEqual(report["results"][0]["output"], "")
        self.assertIn("URLError", report["results"][0]["request_error"])
        self.assertEqual(complete.call_args_list[0].args[3], cases[0]["messages"])

    def test_http_payload_has_no_reference_and_no_v1_duplication(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": answer()}}]}).encode()
        with mock.patch.object(evaluate, "urlopen", return_value=response) as request:
            output = evaluate.complete("http://localhost:8000/v1", "secret", "model", case()["messages"], 128, 1)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(set(payload), {"model", "messages", "temperature", "max_tokens"})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(request.call_args.args[0].full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(output, answer())


class EvaluationFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_predictions_align_by_id_and_reject_duplicates_missing_extra(self):
        cases = [case("a"), case("b")]
        path = self.write("predictions.jsonl", [{"id": "b", "output": answer()}, {"id": "a", "output": answer()}])
        self.assertEqual(set(evaluate.load_predictions(path, cases)), {"a", "b"})
        for rows in [
            [{"id": "a", "output": answer()}],
            [{"id": "a", "output": answer()}, {"id": "a", "output": answer()}],
            [{"id": "a", "output": answer()}, {"id": "b", "output": answer()}, {"id": "c", "output": answer()}],
            [{"id": "a", "output": {"estado": "ok"}}],
        ]:
            with self.subTest(rows=rows):
                path = self.write("predictions.jsonl", rows)
                with self.assertRaises(ValueError):
                    evaluate.load_predictions(path, cases)

    def test_dataset_validation_rejects_answers_and_bad_references(self):
        path = self.write("cases.jsonl", [case()])
        self.assertEqual(evaluate.load_cases(path)[0]["id"], "u1")
        malformed = []
        sample = case()
        sample["messages"].append({"role": "assistant", "content": answer()})
        malformed.append(sample)
        sample = case(value="10.0")
        malformed.append(sample)
        sample = case()
        sample["expected"]["valor"] = 10
        malformed.append(sample)
        for sample in malformed:
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    evaluate.load_cases(self.write("cases.jsonl", [sample]))
        with self.assertRaises(ValueError):
            evaluate.load_cases(self.write("cases.jsonl", [case(), case()]))

    def test_complete_offline_report_preserves_outputs_and_hash(self):
        dataset = self.write("cases.jsonl", [case()])
        predictions = self.write("base.jsonl", [{"id": "u1", "output": "bad"}])
        candidate = self.write("ft.jsonl", [{"id": "u1", "output": answer()}])
        output = self.root / "report.json"
        with mock.patch.object(evaluate, "complete") as complete, mock.patch("builtins.print"):
            evaluate.main(["--dataset", str(dataset), "--predictions", str(predictions),
                           "--compare-predictions", str(candidate), "--output", str(output),
                           "--bootstrap-resamples", "20"])
        complete.assert_not_called()
        report = json.loads(output.read_text())
        self.assertEqual(report["mode"], "offline")
        self.assertEqual(report["comparison"]["exact_accuracy_delta"], 1)
        self.assertEqual(report["candidate"]["results"][0]["output"], answer())
        self.assertEqual(len(report["dataset_sha256"]), 64)
        self.assertNotIn("generation", report)


if __name__ == "__main__":
    unittest.main()

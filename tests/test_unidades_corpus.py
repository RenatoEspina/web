"""Audit synthetic labels, examples and partitions without training a model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import unittest
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "trainer" / "corpora" / "unidades"
spec = importlib.util.spec_from_file_location("unidades_builder", DIRECTORY / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

# Independently authored SI factors; do not read the builder's conversion table.
FACTORS = {
    "mm": "0.001", "cm": "0.01", "m": "1", "km": "1000",
    "mg": "0.000001", "g": "0.001", "kg": "1", "t": "1000",
    "mL": "0.000001", "L": "0.001", "m3": "1",
    "s": "1", "min": "60", "h": "3600",
    "mm2": "0.000001", "cm2": "0.0001", "m2": "1", "km2": "1000000",
}
TEMPERATURES = {"°C", "°F", "K"}
DIMENSION = {
    unit: dimension for dimension, units in {
        "longitud": "mm cm m km", "masa": "mg g kg t", "volumen": "mL L m3",
        "tiempo": "s min h", "superficie": "mm2 cm2 m2 km2", "temperatura": "°C °F K",
    }.items() for unit in units.split()
}
CANONICAL_DECIMAL = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d{0,5}[1-9])?$")
# Unit exponents such as m3 and cm2 must not be mistaken for quantities.
INPUT_NUMBER = re.compile(r"(?<![\w²³])[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?(?![\w²³])")


def reference_value(value: Fraction, source: str, target: str) -> str:
    with localcontext() as context:
        context.prec = 80
        number = Decimal(value.numerator) / Decimal(value.denominator)
        if source in TEMPERATURES:
            if source == "°F":
                number = (number - 32) / Decimal("1.8")
            elif source == "K":
                number -= Decimal("273.15")
            if target == "°F":
                number = number * Decimal("1.8") + 32
            elif target == "K":
                number += Decimal("273.15")
        else:
            number *= Decimal(FACTORS[source]) / Decimal(FACTORS[target])
        number = number.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
        return "0" if not number else format(number, "f").rstrip("0").rstrip(".")


def physical_quantity(value: Fraction, source: str) -> Fraction:
    """Identify equivalent successful inputs without using builder group IDs."""
    if source == "°C":
        return value + Fraction("273.15")
    if source == "°F":
        return (value - 32) * Fraction(5, 9) + Fraction("273.15")
    if source == "K":
        return value
    return value * Fraction(FACTORS[source])


class UnidadesArithmeticTests(unittest.TestCase):
    def test_known_temperature_readings(self):
        cases = [
            ("0", "°C", "°F", "32"), ("100", "°C", "°F", "212"),
            ("-40", "°C", "°F", "-40"), ("-40", "°F", "°C", "-40"),
            ("32", "°F", "K", "273.15"), ("212", "°F", "K", "373.15"),
            ("0", "K", "°C", "-273.15"), ("0", "K", "°F", "-459.67"),
            ("-273.15", "°C", "K", "0"), ("-459.67", "°F", "K", "0"),
            ("310.15", "K", "°F", "98.6"), ("1", "°F", "°C", "-17.222222"),
        ]
        for value, source, target, expected in cases:
            with self.subTest(value=value, source=source, target=target):
                self.assertEqual(builder.convert(Fraction(value), source, target), expected)

    def test_known_area_volume_time_mass_and_length(self):
        cases = [
            ("1", "km2", "m2", "1000000"), ("1", "m2", "cm2", "10000"),
            ("1", "cm2", "mm2", "100"), ("1", "mm2", "m2", "0.000001"),
            ("3.25", "m3", "L", "3250"), ("1", "mL", "m3", "0.000001"),
            ("1", "s", "min", "0.016667"), ("1", "min", "h", "0.016667"),
            ("1", "s", "h", "0.000278"), ("-1", "s", "h", "-0.000278"),
            ("1.5", "h", "min", "90"), ("2.5", "t", "mg", "2500000000"),
            ("1", "mg", "kg", "0.000001"), ("-1.25", "km", "cm", "-125000"),
            ("0", "km", "mm", "0"), ("0", "h", "s", "0"),
        ]
        for value, source, target, expected in cases:
            with self.subTest(value=value, source=source, target=target):
                self.assertEqual(builder.convert(Fraction(value), source, target), expected)

    def test_exact_ties_round_to_even_including_negative_zero(self):
        cases = {
            "0.0000005": "0", "-0.0000005": "0",
            "0.0000015": "0.000002", "-0.0000015": "-0.000002",
            "0.0000025": "0.000002", "-0.0000025": "-0.000002",
            "1.2345645": "1.234564", "1.2345655": "1.234566",
            "-1.2345645": "-1.234564", "-1.2345655": "-1.234566",
            "9.9999995": "10", "-9.9999995": "-10",
            "0.000000499999": "0", "0.000000500001": "0.000001",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(builder.canonical(Fraction(value)), expected)
        self.assertEqual(builder.convert(Fraction("0.5"), "mg", "kg"), "0")
        self.assertEqual(builder.convert(Fraction("1.5"), "mg", "kg"), "0.000002")
        self.assertEqual(builder.convert(Fraction("2.5"), "mg", "kg"), "0.000002")

    def test_incompatible_dimensions_cannot_be_converted(self):
        for source, target in [("m", "m2"), ("L", "kg"), ("s", "°C")]:
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                builder.convert(Fraction(1), source, target)


class UnidadesCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = builder.candidate
        candidates = {}

        def capture(*args, **kwargs):
            row = original(*args, **kwargs)
            candidates[row["id"]] = row
            return row

        with patch.object(builder, "candidate", side_effect=capture):
            cls.splits = builder.generate()
        cls.inputs = {
            row["id"]: candidates[row["id"]]
            for rows in cls.splits.values() for row in rows
        }
        cls.outputs = builder.artifacts()

    def test_split_sizes_category_balance_and_all_unit_coverage(self):
        self.assertEqual({s: len(rows) for s, rows in self.splits.items()},
                         {"train": 2400, "validation": 400, "evaluation": 600, "challenge": 300})
        for split, rows in self.splits.items():
            with self.subTest(split=split):
                counts = Counter(row["category"] for row in rows)
                self.assertEqual(set(counts), set(builder.DIMENSIONS) | {
                    "incompatible", "falta_dato", "no_soportada"})
                for error in ("incompatible", "falta_dato", "no_soportada"):
                    self.assertEqual(counts[error], len(rows) // 10)
                dimensions = [counts[d] for d in builder.DIMENSIONS]
                self.assertLessEqual(max(dimensions) - min(dimensions), 1)
                self.assertEqual({r["expected"]["unidad"] for r in rows if r["expected"]["estado"] == "ok"},
                                 set(FACTORS) | TEMPERATURES)

    def test_prompts_and_declared_semantic_groups_are_disjoint(self):
        prompts = [r["messages"][-1]["content"] for rows in self.splits.values() for r in rows]
        self.assertEqual(len(prompts), len(set(prompts)))
        for left, right in combinations(self.splits, 2):
            with self.subTest(left=left, right=right):
                for key in ("id", "group_id", "template_id"):
                    self.assertFalse({r[key] for r in self.splits[left]} & {r[key] for r in self.splits[right]})

    def test_equivalent_successful_quantities_never_cross_partitions(self):
        locations = {}
        for split, rows in self.splits.items():
            for row in rows:
                if row["expected"]["estado"] != "ok":
                    continue
                raw = self.inputs[row["id"]]
                quantity = physical_quantity(Fraction(raw["_value"]), raw["_source"])
                key = (row["category"], quantity)
                self.assertEqual(locations.setdefault(key, split), split, (key, row["id"]))

    def test_normalizable_error_requests_never_cross_partitions(self):
        locations = {}
        for split, rows in self.splits.items():
            for row in rows:
                raw = self.inputs[row["id"]]
                source, target, category = raw["_source"], raw["_target"], row["category"]
                if category == "incompatible":
                    key = (category, DIMENSION[source], physical_quantity(Fraction(raw["_value"]), source),
                           DIMENSION[target])
                elif category == "no_soportada" and source in DIMENSION:
                    key = (category, DIMENSION[source], physical_quantity(Fraction(raw["_value"]), source), target)
                elif category == "no_soportada":
                    key = (category, source, Fraction(raw["_value"]), DIMENSION[target])
                elif category == "falta_dato" and target is None:
                    key = (category, DIMENSION[source], physical_quantity(Fraction(raw["_value"]), source))
                else:
                    continue
                self.assertEqual(locations.setdefault(key, split), split, (key, row["id"]))

    def test_all_labels_and_input_quantities_match_independent_reference(self):
        for rows in self.splits.values():
            for row in rows:
                raw = self.inputs[row["id"]]
                prompt = row["messages"][-1]["content"]
                numbers = [Fraction(Decimal(m.group().replace(",", "."))) for m in INPUT_NUMBER.finditer(prompt)]
                expected_quantity = [] if raw["_value"] == "None" else [Fraction(raw["_value"])]
                self.assertEqual(numbers, expected_quantity, row["id"])
                source, target = raw["_source"], raw["_target"]
                for unit in (source, target):
                    if unit is None:
                        continue
                    names = builder.UNITS[unit][3] if unit in builder.UNITS else [unit]
                    self.assertTrue(any(re.search(r"(?<![\w²³])" + re.escape(name) + r"(?![\w²³])", prompt)
                                        for name in names), (row["id"], unit))
                if not expected_quantity or source is None or target is None:
                    state = "falta_dato"
                elif source not in DIMENSION or target not in DIMENSION:
                    state = "no_soportada"
                elif DIMENSION[source] != DIMENSION[target]:
                    state = "incompatible"
                else:
                    state = "ok"
                expected = row["expected"]
                self.assertEqual(expected["estado"], state, row["id"])
                self.assertEqual(set(expected), {"estado", "valor", "unidad"})
                self.assertEqual([m["role"] for m in row["messages"]], ["system", "user"])
                if expected["estado"] == "ok":
                    answer = reference_value(Fraction(raw["_value"]), raw["_source"], raw["_target"])
                    self.assertEqual(expected["valor"], answer, row["id"])
                    self.assertEqual(expected["unidad"], raw["_target"])
                    self.assertRegex(expected["valor"], CANONICAL_DECIMAL)
                    self.assertNotEqual(expected["valor"], "-0")
                else:
                    self.assertEqual(expected["estado"], row["category"])
                    self.assertIsNone(expected["valor"])
                    self.assertIsNone(expected["unidad"])

    def test_edge_input_forms_and_each_missing_field_are_represented(self):
        rows = list(self.inputs.values())
        values = [Fraction(r["_value"]) for r in rows if r["_value"] != "None"]
        self.assertIn(Fraction(0), values)
        self.assertTrue(any(value < 0 for value in values))
        self.assertTrue(any("," in m.group() for r in rows for m in INPUT_NUMBER.finditer(r["messages"][-1]["content"])))
        self.assertTrue(any("e" in m.group().lower() for r in rows for m in INPUT_NUMBER.finditer(r["messages"][-1]["content"])))
        missing = [r for r in rows if r["category"] == "falta_dato"]
        self.assertTrue(any(r["_source"] is None for r in missing))
        self.assertTrue(any(r["_target"] is None for r in missing))
        self.assertTrue(any(r["_value"] == "None" for r in missing))
        for split, partition in self.splits.items():
            temperatures = [self.inputs[r["id"]] for r in partition if r["category"] == "temperatura"]
            self.assertTrue(any(Fraction(r["_value"]) < 0 for r in temperatures), split)
            self.assertTrue(all(physical_quantity(Fraction(r["_value"]), r["_source"]) >= 0
                                for r in temperatures), split)

    def test_sft_files_have_only_messages_and_evaluation_holds_out_answers(self):
        paths = {"train": ROOT / "trainer/examples/unidades-training.jsonl",
                 "validation": DIRECTORY / "validation.jsonl"}
        for split, path in paths.items():
            sft_rows = [json.loads(line) for line in self.outputs[path].splitlines()]
            self.assertEqual(len(sft_rows), len(self.splits[split]))
            for sft, audit in zip(sft_rows, self.splits[split]):
                self.assertEqual(set(sft), {"messages"})
                self.assertEqual(sft["messages"][:-1], audit["messages"])
                self.assertEqual(sft["messages"][-1]["role"], "assistant")
                self.assertEqual(json.loads(sft["messages"][-1]["content"]), audit["expected"])
        for split in ("evaluation", "challenge"):
            for row in self.splits[split]:
                self.assertNotIn("assistant", {m["role"] for m in row["messages"]})

    def test_generation_is_reproducible_despite_global_random_state(self):
        global_state = random.getstate()
        try:
            random.seed(17)
            self.assertEqual(self.outputs, builder.artifacts())
            self.assertEqual(random.getstate(), random.Random(17).getstate())
        finally:
            random.setstate(global_state)

    def test_manifest_hashes_and_counts_describe_actual_artifacts(self):
        manifest = json.loads(self.outputs[DIRECTORY / "manifest.json"])
        self.assertFalse(manifest["training_executed"])
        self.assertIsNone(manifest["model_metrics"])
        self.assertEqual(manifest["unique_prompts"], sum(map(len, self.splits.values())))
        self.assertTrue(all(count == 0 for count in manifest["semantic_group_overlap"].values()))
        for relative, metadata in manifest["files"].items():
            content = self.outputs[ROOT / relative].encode("utf-8")
            self.assertEqual(metadata["bytes"], len(content))
            self.assertEqual(metadata["sha256"], hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()

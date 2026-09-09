#!/usr/bin/env python3
"""Corpus sintético reproducible; solo biblioteca estándar, sin llamadas a LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SEED = 20260909
VERSION = "1.0.0"
SIZES = {"train": 2400, "validation": 400, "evaluation": 600, "challenge": 300}

# symbol: (dimension, scale to base, offset to base, Spanish aliases)
# Temperature maps absolute readings to kelvin, never temperature differences.
UNITS = {
    "mm": ("longitud", "0.001", "0", ["mm", "milímetros"]),
    "cm": ("longitud", "0.01", "0", ["cm", "centímetros"]),
    "m": ("longitud", "1", "0", ["m", "metros"]),
    "km": ("longitud", "1000", "0", ["km", "kilómetros"]),
    "mg": ("masa", "0.000001", "0", ["mg", "miligramos"]),
    "g": ("masa", "0.001", "0", ["g", "gramos"]),
    "kg": ("masa", "1", "0", ["kg", "kilogramos"]),
    "t": ("masa", "1000", "0", ["t", "toneladas"]),
    "mL": ("volumen", "0.000001", "0", ["mL", "mililitros"]),
    "L": ("volumen", "0.001", "0", ["L", "litros"]),
    "m3": ("volumen", "1", "0", ["m3", "m³", "metros cúbicos"]),
    "s": ("tiempo", "1", "0", ["s", "segundos"]),
    "min": ("tiempo", "60", "0", ["min", "minutos"]),
    "h": ("tiempo", "3600", "0", ["h", "horas"]),
    "mm2": ("superficie", "0.000001", "0", ["mm2", "mm²", "milímetros cuadrados"]),
    "cm2": ("superficie", "0.0001", "0", ["cm2", "cm²", "centímetros cuadrados"]),
    "m2": ("superficie", "1", "0", ["m2", "m²", "metros cuadrados"]),
    "km2": ("superficie", "1000000", "0", ["km2", "km²", "kilómetros cuadrados"]),
    "°C": ("temperatura", "1", "273.15", ["°C", "grados Celsius"]),
    "°F": ("temperatura", "5/9", "45967/180", ["°F", "grados Fahrenheit"]),
    "K": ("temperatura", "1", "0", ["K", "kelvin"]),
}
DIMENSIONS = tuple(dict.fromkeys(v[0] for v in UNITS.values()))
UNSUPPORTED = ["pulgadas", "millas", "onzas", "galones", "libras", "pies"]

SYSTEM = (
    "Convierte una sola cantidad entre unidades del catálogo. Responde SOLO un objeto JSON "
    'con exactamente las claves "estado", "valor", "unidad". '
    'Éxito: estado="ok", valor=cadena decimal, unidad=símbolo canónico de destino. '
    "Usa punto decimal, sin separadores de miles ni notación científica; redondea a un máximo "
    "de 6 decimales con empate al par, elimina ceros decimales finales y escribe cero como 0. "
    "En la entrada coma o punto significa decimal; se admite notación científica. "
    "Catálogo (acepta nombres españoles y exponentes en superíndice): "
    "longitud mm/cm/m/km; masa mg/g/kg/t; volumen mL/L/m3; tiempo s/min/h; "
    "superficie mm2/cm2/m2/km2; temperatura °C/°F/K. "
    "Temperaturas son lecturas absolutas, no diferencias. "
    "Si falta cantidad, origen o destino, estado=falta_dato; si alguna unidad está fuera "
    "del catálogo, estado=no_soportada; si las dimensiones difieren, estado=incompatible. "
    "Aplica esas comprobaciones en ese orden. En los tres errores valor y unidad son null. "
    "No supongas densidades ni unidades omitidas. No añadas explicaciones."
)

# Each split has its own surface template family, including incomplete requests.
WRAPPERS = {
    "train": ["Convierte {request}", "Necesito convertir {request}", "Haz esta conversión: {request}",
              "Por favor, pasa {request}", "Calcula la conversión de {request}",
              "Mi consulta de unidades es: {request}", "Resuelve: {request}",
              "Para mi ejercicio, convierte {request}", "Ayúdame a pasar {request}",
              "Quiero expresar {request}", "Dame la equivalencia de {request}",
              "Transforma esta medida: {request}"],
    "validation": ["Estoy revisando una medida; expresa {request}",
                   "Tengo pendiente esta equivalencia: {request}",
                   "Comprueba el cambio de unidades solicitado: {request}",
                   "Devuelve el resultado para esta petición: {request}"],
    "evaluation": ["En mi hoja aparece esta conversión por resolver: {request}",
                   "Al completar la ficha debo pasar {request}",
                   "Reexpresa la cantidad indicada aquí: {request}",
                   "Esta es la medida y su destino: {request}"],
    "challenge": ["Petición recibida\n{request}\nEntrega el resultado normalizado.",
                  "Conversión en una nota: «{request}»", "Lee esta petición con atención: [{request}]",
                  "Ficha de conversión\n  {request}",
                  "La equivalencia que necesito verificar es la siguiente — {request}",
                  "Consulta transcrita:\n\t{request}"],
}


def dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(value: Fraction) -> str:
    """Round a rational exactly to six decimals, then format canonically."""
    scaled = value * 1_000_000
    sign = -1 if scaled < 0 else 1
    whole, remainder = divmod(abs(scaled.numerator), scaled.denominator)
    twice = remainder * 2
    if twice > scaled.denominator or (twice == scaled.denominator and whole % 2):
        whole += 1
    whole *= sign
    if whole == 0:
        return "0"
    absolute = abs(whole)
    integer, decimals = divmod(absolute, 1_000_000)
    result = str(integer)
    if decimals:
        result += "." + f"{decimals:06d}".rstrip("0")
    return ("-" if whole < 0 else "") + result


def base_value(value: Fraction, source: str) -> Fraction:
    _, factor, offset, _ = UNITS[source]
    return value * Fraction(factor) + Fraction(offset)


def convert(value: Fraction, source: str, target: str) -> str:
    if UNITS[source][0] != UNITS[target][0]:
        raise ValueError("Dimensiones incompatibles")
    _, scale, offset, _ = UNITS[target]
    return canonical((base_value(value, source) - Fraction(offset)) / Fraction(scale))


def independent_decimal(value: Fraction, source: str, target: str) -> str:
    """Second arithmetic implementation; temperature formulas authored separately."""
    with localcontext() as context:
        context.prec = 70
        x = Decimal(value.numerator) / Decimal(value.denominator)
        if UNITS[source][0] == "temperatura":
            celsius = {"°C": lambda: x, "°F": lambda: (x - 32) * 5 / 9,
                       "K": lambda: x - Decimal("273.15")}[source]()
            y = {"°C": lambda: celsius, "°F": lambda: celsius * 9 / 5 + 32,
                 "K": lambda: celsius + Decimal("273.15")}[target]()
        else:
            y = x * Decimal(UNITS[source][1]) / Decimal(UNITS[target][1])
        y = y.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
        return format(y, "f").rstrip("0").rstrip(".") if y else "0"


def assigned_split(group: str) -> str:
    slot = int(digest(group)[:8], 16) % 100
    return "train" if slot < 65 else "validation" if slot < 76 else "evaluation" if slot < 92 else "challenge"


def alias(unit: str, rng: random.Random, value: Fraction | None = None) -> str:
    # Use symbols for exactly one to avoid singular/plural grammatical errors.
    return unit if value == 1 else rng.choice(UNITS[unit][3])


def candidate(split: str, category: str, rng: random.Random) -> dict:
    dimension = category if category in DIMENSIONS else rng.choice(DIMENSIONS)
    source, target = rng.sample([u for u, spec in UNITS.items() if spec[0] == dimension], 2)
    if split == "challenge":
        value = Fraction(rng.randint(1, 9999999), rng.choice([1, 1000, 1000000]))
        if dimension != "temperatura" and rng.random() < 0.3:
            value *= -1
    else:
        value = Fraction(rng.randint(1, 100000), rng.choice([1, 10, 100]))
        if dimension != "temperatura" and rng.random() < 0.1:
            value *= -1
    # Seeded edge-value sampling; semantic grouping still separates partitions.
    if rng.random() < 0.05:
        value = Fraction(rng.choice(["0", "1", "2.5", "0.125", "100", "32", "273.15"]))
    if dimension == "temperatura" and source != "K" and rng.random() < 0.2:
        value = Fraction(rng.randint(-10000, -1), 100)
    value_text = canonical(value)
    if split == "challenge" and rng.random() < 0.5:
        with localcontext() as ctx:
            ctx.prec = 40
            value_text = format(Decimal(value.numerator) / Decimal(value.denominator), "E")
    elif rng.random() < 0.4:
        value_text = value_text.replace(".", ",")
    source_text, target_text = alias(source, rng, value), alias(target, rng)
    expected = {"estado": "ok", "valor": None, "unidad": target}
    if category in DIMENSIONS:
        expected["valor"] = convert(value, source, target)
        if expected["valor"] != independent_decimal(value, source, target):
            raise AssertionError("Los dos cálculos de referencia no coinciden")
        group = f"ok|{dimension}|{base_value(value, source)}"
        variants = {
            "train": [f"{value_text} {source_text} a {target_text}.",
                      f"la cantidad {value_text} {source_text} en {target_text}.",
                      f"{value_text} expresado en {source_text}; destino: {target_text}."],
            "validation": [f"una lectura de {value_text} {source_text}, usando {target_text} como destino.",
                           f"origen {source_text}, cantidad {value_text}, destino {target_text}."],
            "evaluation": [f"Tengo {value_text} {source_text}. ¿Cuánto representa en {target_text}?",
                           f"La unidad final será {target_text}; la medida inicial es {value_text} {source_text}."],
            "challenge": [f"Destino solicitado: {target_text}.\nMedida de partida: {value_text} {source_text}.",
                          f"Cantidad: {value_text}\nUnidad actual: {source_text}\nUnidad nueva: {target_text}",
                          f"El dato está en {source_text}: {value_text}. Lo necesito expresado en {target_text}."],
        }
        request = rng.choice(variants[split])
    else:
        expected = {"estado": category, "valor": None, "unidad": None}
        if category == "incompatible":
            target = rng.choice([u for u, spec in UNITS.items() if spec[0] != dimension])
            request = f"{value_text} {source_text} a {alias(target, rng)}."
            group = f"incompatible|{dimension}|{base_value(value, source)}|{UNITS[target][0]}"
        elif category == "no_soportada":
            unsupported = rng.choice(UNSUPPORTED)
            if rng.random() < 0.5:
                request = f"{value_text} {unsupported} a {target_text}."
                source = unsupported
            else:
                request = f"{value_text} {source_text} a {unsupported}."
                target = unsupported
            if source in UNITS:
                group = f"no_soportada|{dimension}|{base_value(value, source)}|{target}"
            else:
                group = f"no_soportada|{value}|{source}|{UNITS[target][0]}"
        else:
            missing = rng.randrange(3)
            if missing == 0:
                request = f"{value_text} a {target_text}; no se indicó la unidad de origen."
                source = None
            elif missing == 1:
                request = f"{value_text} {source_text}; no se indicó la unidad de destino."
                target = None
            else:
                request = f"de {source_text} a {target_text}; no se indicó la cantidad."
                value = None
            if target is None:
                group = f"falta_destino|{dimension}|{base_value(value, source)}"
            else:
                group = f"falta_dato|{value}|{source}|{target}"
    template_index = rng.randrange(len(WRAPPERS[split]))
    prompt = WRAPPERS[split][template_index].format(request=request)
    return {"id": digest(prompt)[:24], "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}], "expected": expected, "category": category,
            "group_id": digest(group), "template_id": f"{split}-{template_index:02}",
            "_group": group, "_source": source, "_target": target, "_value": str(value)}


def generate() -> dict[str, list[dict]]:
    splits = {}
    seen_prompts: set[str] = set()
    for split, size in SIZES.items():
        rng = random.Random(f"{SEED}:{split}")
        valid = size * 7 // 10
        quotas = {d: valid // len(DIMENSIONS) + (i < valid % len(DIMENSIONS))
                  for i, d in enumerate(DIMENSIONS)}
        quotas.update({c: size // 10 for c in ("incompatible", "falta_dato", "no_soportada")})
        rows = []
        for category, count in quotas.items():
            attempts = 0
            accepted = 0
            while accepted < count:
                attempts += 1
                if attempts > count * 1000:
                    raise RuntimeError(f"No se puede completar {split}/{category}")
                row = candidate(split, category, rng)
                if assigned_split(row["_group"]) != split or row["id"] in seen_prompts:
                    continue
                seen_prompts.add(row["id"])
                rows.append({k: v for k, v in row.items() if not k.startswith("_")})
                accepted += 1
        rng.shuffle(rows)
        splits[split] = rows
    return splits


def artifacts() -> dict[Path, str]:
    splits = generate()
    outputs = {}
    def jsonl(rows: list[dict]) -> str:
        return "".join(dump(row) + "\n" for row in rows)
    for split, rows in splits.items():
        if split in ("train", "validation"):
            sft = [{"messages": row["messages"] + [{"role": "assistant", "content": dump(row["expected"])}]}
                   for row in rows]
            path = ROOT / "trainer/examples/unidades-training.jsonl" if split == "train" else HERE / "validation.jsonl"
            outputs[path] = jsonl(sft)
        # train metadata kept separately for audit, never sent to the optimizer.
        name = "train-audit.jsonl" if split == "train" else "validation-eval.jsonl" if split == "validation" else split + ".jsonl"
        outputs[HERE / name] = jsonl(rows)
    overlap = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            common = set(r["group_id"] for r in splits[left]) & set(r["group_id"] for r in splits[right])
            if common:
                raise AssertionError("Fuga entre particiones")
            overlap[f"{left}/{right}"] = len(common)
    manifest = {
        "dataset": "unidades-es", "version": VERSION, "seed": SEED,
        "synthetic": True, "generation": "authored templates + exact rational arithmetic; no LLM outputs",
        "training_executed": False, "model_metrics": None,
        "split_sizes": SIZES,
        "categories": {s: dict(sorted(Counter(r["category"] for r in rows).items())) for s, rows in splits.items()},
        "unique_semantic_groups": {s: len(set(r["group_id"] for r in rows)) for s, rows in splits.items()},
        "semantic_group_overlap": overlap,
        "template_families": {s: len(WRAPPERS[s]) for s in splits},
        "unique_prompts": len({r["id"] for rows in splits.values() for r in rows}),
        "label_validation": "Every numeric label agrees between Fraction and Decimal implementations",
        "files": {str(p.relative_to(ROOT)): {"sha256": digest(text), "bytes": len(text.encode("utf-8"))}
                  for p, text in outputs.items()},
    }
    outputs[HERE / "manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Comprueba los archivos sin modificarlos")
    args = parser.parse_args()
    outputs = artifacts()
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, text in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit("Faltan archivos o están desactualizados: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    print(dump({"status": "checked" if args.check else "built", "examples": SIZES, "seed": SEED}))


if __name__ == "__main__":
    main()

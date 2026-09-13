#!/usr/bin/env python3
"""Adapta databricks-dolly-15k al formato JSONL del proyecto; solo biblioteca estándar.

El corpus fuente es humano, está documentado con una dataset card pública y su
licencia CC BY-SA 3.0 permite uso comercial. Aquí se usa con dos propósitos:

1. corpus de control: un dataset conocido-bueno para distinguir "mi pipeline
   está roto" de "mi dataset de dominio está mal";
2. corpus de replay: material general que se mezcla con un dataset de dominio
   pequeño para frenar el olvido catastrófico.

El archivo fuente se descarga una sola vez desde una revisión inmutable y se
verifica por SHA-256. A partir de ahí el build es offline y determinista.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRAINER = HERE.parents[1]
CACHE = HERE / ".cache"

DATASET = "databricks/databricks-dolly-15k"
# Revisión inmutable: si el repositorio cambia, este build sigue reproduciendo
# exactamente los mismos JSONL.
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SOURCE_FILE = "databricks-dolly-15k.jsonl"
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILE}"
SOURCE_RECORDS = 15011
LICENSE = "CC-BY-SA-3.0"
ATTRIBUTION = (
    "databricks/databricks-dolly-15k, escrito por empleados de Databricks, Inc. "
    "Los pasajes de contexto provienen de Wikipedia (CC BY-SA 3.0)."
)

VERSION = "1.0.0"
SEED = 20260913

SYSTEM = (
    "You are a helpful assistant. Answer accurately and concisely. When reference "
    "context is provided, base the answer on it and say so if the context does not "
    "contain the answer."
)

# Presupuesto de caracteres calibrado contra el tokenizer de Qwen3.5-0.8B: con
# 2200 caracteres el ejemplo más largo del corpus queda en 1014 tokens, bajo el
# --max-length 1024 por defecto de train.py. train.py sigue siendo la
# comprobación dura: aborta si una respuesta assistant se trunca.
MAX_CHARACTERS = 2200

PROFILES = {
    # Perfil pensado para una laptop con 6 GB de VRAM: ~375 pasos de optimizador
    # por época con batch 1 y gradient accumulation 8.
    "laptop": {"train": 3000, "validation": 400, "evaluation": 250},
    # Todo lo que entra en el presupuesto de longitud.
    "full": {"train": None, "validation": 600, "evaluation": 300},
}

# Categorías de las que se puede derivar un `contains` verificable: la respuesta
# es corta y aparece literalmente en la instrucción o en el contexto.
VERIFIABLE_CATEGORIES = ("classification", "closed_qa", "information_extraction")
MAX_EVALUATION_WORDS = 3
MIN_EVALUATION_CHARACTERS = 3

RUBRIC = (
    "Manually check factual accuracy and the absence of contradictions. "
    "contains is only a lexical smoke test."
)

_WHITESPACE = re.compile(r"\s+")


def normalize_prompt(value: str) -> str:
    """Misma normalización que training_quality.count_prompt_overlaps."""
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return _WHITESPACE.sub(" ", re.sub(r"[^\w\s]", " ", without_accents).strip())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch(allow_download: bool) -> Path:
    path = CACHE / SOURCE_FILE
    if path.exists():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == SOURCE_SHA256:
            return path
        raise SystemExit(
            f"{path} no coincide con el SHA-256 fijado ({actual}). "
            "Bórralo y vuelve a descargarlo."
        )
    if not allow_download:
        raise FileNotFoundError(path)

    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise SystemExit(
            f"No se pudo descargar {SOURCE_URL}: {error}. "
            "Descarga el archivo manualmente y colócalo en "
            f"{path.relative_to(ROOT)}."
        ) from error

    actual = hashlib.sha256(payload).hexdigest()
    if actual != SOURCE_SHA256:
        raise SystemExit(
            f"La descarga no coincide con el SHA-256 fijado.\n"
            f"  esperado: {SOURCE_SHA256}\n  obtenido: {actual}"
        )
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def read_source(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        missing = {"instruction", "context", "response", "category"} - set(value)
        if missing:
            raise ValueError(f"Línea {number}: faltan campos {sorted(missing)}")
        rows.append(value)
    if len(rows) != SOURCE_RECORDS:
        raise ValueError(f"Se esperaban {SOURCE_RECORDS} registros, hay {len(rows)}")
    return rows


def user_message(row: dict) -> str:
    instruction = row["instruction"].strip()
    context = row["context"].strip()
    if not context:
        return instruction
    return f"{instruction}\n\n---\n{context}"


def convert(rows: list[dict], max_characters: int) -> tuple[list[dict], Counter]:
    """Normaliza, descarta lo que no sirve y deduplica por prompt normalizado."""
    dropped: Counter = Counter()
    seen: set[str] = set()
    examples: list[dict] = []

    for index, row in enumerate(rows):
        user = user_message(row)
        answer = row["response"].strip()
        # Una respuesta de un solo carácter es legítima ("7", "F"): solo se
        # descarta lo realmente vacío. El umbral de longitud para `contains`
        # vive en evaluation_candidate, que es donde un fragmento corto haría
        # daño.
        if not user or not answer:
            dropped["empty"] += 1
            continue
        if len(SYSTEM) + len(user) + len(answer) > max_characters:
            dropped["too_long"] += 1
            continue
        key = normalize_prompt(user)
        if not key or key in seen:
            dropped["duplicate_prompt"] += 1
            continue
        seen.add(key)
        examples.append(
            {
                "id": f"dolly-{index:05}",
                "category": row["category"],
                "grounded": bool(row["context"].strip()),
                "user": user,
                "answer": answer,
            }
        )
    return examples, dropped


def evaluation_candidate(example: dict) -> str | None:
    """Devuelve el fragmento `contains` si la respuesta es verificable léxicamente."""
    if example["category"] not in VERIFIABLE_CATEGORIES:
        return None
    answer = example["answer"].strip().rstrip(".").strip()
    if "\n" in answer or not MIN_EVALUATION_CHARACTERS <= len(answer):
        return None
    if len(answer.split()) > MAX_EVALUATION_WORDS:
        return None
    # La respuesta debe estar respaldada por el propio prompt: así `contains`
    # comprueba extracción/selección y no memoria del anotador.
    if answer.casefold() not in example["user"].casefold():
        return None
    return answer


def stratified(examples: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Toma `size` ejemplos conservando la proporción de categorías."""
    if size >= len(examples):
        return list(examples)
    buckets: dict[str, list[dict]] = {}
    for example in examples:
        buckets.setdefault(example["category"], []).append(example)

    chosen: list[dict] = []
    for category in sorted(buckets):
        bucket = buckets[category]
        quota = round(size * len(bucket) / len(examples))
        chosen.extend(bucket[: min(quota, len(bucket))])

    # El redondeo por categoría no cuadra exacto; completa o recorta de forma
    # determinista sobre el orden ya barajado.
    taken = {example["id"] for example in chosen}
    for example in examples:
        if len(chosen) >= size:
            break
        if example["id"] not in taken:
            chosen.append(example)
            taken.add(example["id"])
    rng.shuffle(chosen)
    return chosen[:size]


def training_row(example: dict) -> dict:
    return {
        "id": example["id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["answer"]},
        ],
        "category": example["category"],
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }


def evaluation_row(example: dict, fragment: str) -> dict:
    return {
        "id": example["id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["user"]},
        ],
        "category": example["category"],
        "contains": [fragment],
        "reference_answer": example["answer"],
        "rubric": RUBRIC,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }


def dump(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )


def artifacts(profile: str, max_characters: int) -> tuple[dict[Path, str], dict]:
    source = fetch(allow_download=True)
    examples, dropped = convert(read_source(source), max_characters)

    rng = random.Random(SEED)
    rng.shuffle(examples)

    sizes = PROFILES[profile]
    # La evaluación se reserva primero: solo puede salir del subconjunto con
    # respuesta verificable, que es escaso.
    candidates = [(example, evaluation_candidate(example)) for example in examples]
    verifiable = [(example, fragment) for example, fragment in candidates if fragment]
    evaluation_pairs = verifiable[: sizes["evaluation"]]
    if len(evaluation_pairs) < sizes["evaluation"]:
        raise ValueError(
            f"Solo hay {len(evaluation_pairs)} ejemplos con respuesta verificable; "
            f"el perfil pide {sizes['evaluation']}"
        )
    reserved = {example["id"] for example, _ in evaluation_pairs}

    remaining = [example for example in examples if example["id"] not in reserved]
    validation = stratified(remaining, sizes["validation"], random.Random(SEED + 1))
    used = reserved | {example["id"] for example in validation}
    pool = [example for example in remaining if example["id"] not in used]
    train = pool if sizes["train"] is None else stratified(pool, sizes["train"], random.Random(SEED + 2))

    splits = {
        "train": [training_row(example) for example in train],
        "validation": [training_row(example) for example in validation],
        "evaluation": [evaluation_row(example, fragment) for example, fragment in evaluation_pairs],
    }

    ids = [row["id"] for rows in splits.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("Un mismo ejemplo aparece en más de una partición")

    outputs = {
        TRAINER / "examples" / "dolly-training.jsonl": dump(splits["train"]),
        HERE / "validation.jsonl": dump(splits["validation"]),
        HERE / "evaluation.jsonl": dump(splits["evaluation"]),
    }
    manifest = {
        "dataset": "dolly-en",
        "version": VERSION,
        "profile": profile,
        "seed": SEED,
        "source": {
            "repository": DATASET,
            "revision": REVISION,
            "file": SOURCE_FILE,
            "sha256": SOURCE_SHA256,
            "records": SOURCE_RECORDS,
            "url": f"https://huggingface.co/datasets/{DATASET}",
        },
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "synthetic": False,
        "generation": "human-written prompt/response pairs; reformatted to chat messages, not rewritten",
        "maxCharacters": max_characters,
        "systemPrompt": SYSTEM,
        "kept": len(examples),
        "dropped": dict(sorted(dropped.items())),
        "splitSizes": {name: len(rows) for name, rows in splits.items()},
        "categories": {
            name: dict(sorted(Counter(row["category"] for row in rows).items()))
            for name, rows in splits.items()
        },
        "files": {
            str(path.relative_to(ROOT)): {"sha256": digest(text), "bytes": len(text.encode("utf-8"))}
            for path, text in outputs.items()
        },
    }
    outputs[HERE / "manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    parser.add_argument("--max-characters", type=int, default=MAX_CHARACTERS)
    parser.add_argument("--check", action="store_true", help="Comprueba los archivos sin modificarlos")
    parser.add_argument(
        "--if-cached",
        action="store_true",
        help="No hace nada si el archivo fuente todavía no se descargó (hook de la GUI).",
    )
    args = parser.parse_args()

    if args.if_cached and not (CACHE / SOURCE_FILE).exists():
        print(json.dumps({"status": "skipped", "reason": "source not downloaded"}))
        return
    if args.max_characters < 256:
        raise SystemExit("--max-characters debe ser al menos 256")

    outputs, manifest = artifacts(args.profile, args.max_characters)
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            raise SystemExit("Faltan archivos o están desactualizados: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "checked" if args.check else "built",
                "profile": args.profile,
                "examples": manifest["splitSizes"],
                "dropped": manifest["dropped"],
                "seed": SEED,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

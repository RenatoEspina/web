#!/usr/bin/env python3
"""Mezcla un dataset de dominio con datos generales de replay.

Un corpus de dominio pequeño entrenado solo consigo mismo empuja al adapter a
responder siempre en ese dominio y degrada la capacidad general del modelo base.
La mitigación documentada es el *replay*: reinyectar una fracción de datos
generales en el mismo SFT.

La mezcla se aplica también al set de validación. Si solo se mezclara el
entrenamiento, `eval_loss` mediría únicamente el dominio y la selección de
checkpoint premiaría justo el checkpoint que más capacidad general perdió.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_validation import load_jsonl  # noqa: E402
from training_quality import count_prompt_overlaps  # noqa: E402

TRAINER = Path(__file__).resolve().parent
ROOT = TRAINER.parent
DEFAULT_REPLAY = TRAINER / "examples" / "dolly-training.jsonl"
DEFAULT_REPLAY_VALIDATION = TRAINER / "corpora" / "dolly" / "validation.jsonl"
OUTPUT_DIR = TRAINER / "datasets"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,110}$")


def sibling_validation(dataset: Path) -> Path | None:
    """Resuelve la validación asociada igual que scripts/train-adapter.sh."""
    known = {
        "terraria-training.jsonl": TRAINER / "corpora" / "terraria" / "validation.jsonl",
        "unidades-training.jsonl": TRAINER / "corpora" / "unidades" / "validation.jsonl",
        "dolly-training.jsonl": DEFAULT_REPLAY_VALIDATION,
    }
    candidate = known.get(dataset.name)
    if candidate is None and dataset.name.endswith("-training.jsonl"):
        candidate = dataset.with_name(dataset.name.replace("-training.jsonl", "-validation.jsonl"))
    return candidate if candidate is not None and candidate.exists() else None


def replay_count(domain: int, ratio: float) -> int:
    """Cuántos ejemplos de replay añadir para que ocupen `ratio` de la mezcla."""
    return round(domain * ratio / (1 - ratio))


def blend(
    domain: list[dict],
    replay: list[dict],
    ratio: float,
    seed: int,
    label: str,
) -> tuple[list[dict], dict]:
    wanted = replay_count(len(domain), ratio)
    shuffled = list(replay)
    random.Random(seed).shuffle(shuffled)
    taken = shuffled[:wanted]
    if len(taken) < wanted:
        print(
            f"Aviso: {label} solo aporta {len(taken)} de los {wanted} ejemplos "
            "de replay pedidos.",
            file=sys.stderr,
        )
    mixed = domain + taken
    random.Random(seed + 1).shuffle(mixed)
    total = len(mixed)
    return mixed, {
        "domain": len(domain),
        "replay": len(taken),
        "total": total,
        "replayShare": round(len(taken) / total, 4) if total else 0.0,
    }


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=Path, required=True, help="Dataset de dominio en JSONL")
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Fracción de la mezcla final que será replay (0.05-0.5; por defecto 0.2)",
    )
    parser.add_argument("--name", required=True, help="Nombre base del dataset mezclado")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not SAFE_NAME.fullmatch(args.name):
        raise SystemExit("--name solo admite letras, dígitos, punto, guion y guion bajo")
    if not 0.0 < args.ratio < 0.5:
        raise SystemExit("--ratio debe estar entre 0 y 0.5; por encima el dominio deja de dominar")
    if args.domain.resolve() == args.replay.resolve():
        raise SystemExit("--domain y --replay deben ser datasets distintos")

    domain, _ = load_jsonl(args.domain)
    replay, _ = load_jsonl(args.replay)
    overlap = count_prompt_overlaps(domain, replay)
    if overlap:
        raise SystemExit(
            f"El dataset de replay repite {overlap} prompts del dominio; "
            "usa un replay independiente."
        )

    report: dict[str, object] = {"name": args.name, "ratio": args.ratio, "seed": args.seed}
    mixed, report["training"] = blend(domain, replay, args.ratio, args.seed, "replay")
    training_path = OUTPUT_DIR / f"{args.name}-training.jsonl"
    write(training_path, mixed)
    report["trainingPath"] = str(training_path.relative_to(ROOT))

    domain_validation = sibling_validation(args.domain)
    replay_validation = sibling_validation(args.replay)
    if domain_validation and replay_validation:
        domain_rows, _ = load_jsonl(domain_validation)
        replay_rows, _ = load_jsonl(replay_validation)
        mixed_validation, report["validation"] = blend(
            domain_rows, replay_rows, args.ratio, args.seed + 100, "replay de validación"
        )
        leak = count_prompt_overlaps(mixed, mixed_validation)
        if leak:
            raise SystemExit(f"La validación mezclada repite {leak} prompts del entrenamiento")
        validation_path = OUTPUT_DIR / f"{args.name}-validation.jsonl"
        write(validation_path, mixed_validation)
        report["validationPath"] = str(validation_path.relative_to(ROOT))
    else:
        report["validation"] = None
        missing = "dominio" if not domain_validation else "replay"
        print(
            f"Aviso: falta la validación del {missing}; la mezcla se entrena sin "
            "selección de checkpoint por eval_loss.",
            file=sys.stderr,
        )

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

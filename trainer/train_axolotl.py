#!/usr/bin/env python3
"""Train a QLoRA SFT adapter with Axolotl while preserving LLM Bridge metadata."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig

from dataset_validation import load_jsonl
from training_quality import count_prompt_overlaps, sha256_file, stable_sha256

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
AXOLOTL_VERSION = "0.18.0"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning SFT QLoRA con Axolotl para LLM Bridge")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("adapters"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def common_system_prompt(examples: list[dict[str, Any]]) -> str | None:
    prompts: set[str] = set()
    for example in examples:
        messages = example.get("messages")
        if not isinstance(messages, list):
            return None
        systems = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "system"
        ]
        if len(systems) != 1 or not isinstance(systems[0], str) or not systems[0].strip():
            return None
        prompts.add(systems[0].strip())
        if len(prompts) > 1:
            return None
    return next(iter(prompts)) if len(prompts) == 1 else None


def model_type(model: str, revision: str | None) -> str:
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if revision:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model, **kwargs)
    return str(config.model_type)


def dataset_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "type": "chat_template",
        "field_messages": "messages",
        "roles_to_train": ["assistant"],
        "train_on_eos": "turn",
        "split": "train",
    }


def build_config(args: argparse.Namespace, staging: Path, work_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_model": args.model,
        "strict": False,
        "trust_remote_code": False,
        "datasets": [dataset_entry(args.dataset.resolve())],
        "val_set_size": 0.0,
        "dataset_prepared_path": str(work_dir / "prepared"),
        "output_dir": str(staging),
        "sequence_len": args.max_length,
        # Qwen3.5 sample packing requires the optional FLA stack. Keep the
        # consumer-GPU path dependency-light and predictable by default.
        "sample_packing": False,
        "eval_sample_packing": False,
        "load_in_4bit": True,
        "load_in_8bit": False,
        "adapter": "qlora",
        # Closest Axolotl equivalent to PEFT target_modules='all-linear'.
        "lora_target_linear": True,
        "lora_r": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "micro_batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "optimizer": "adamw_torch_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": args.learning_rate,
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "warmup_ratio": 0.1,
        "saves_per_epoch": 1,
        "save_total_limit": 2,
        "save_safetensors": True,
        "weight_decay": 0.0,
        "seed": args.seed,
        "dataset_processes": 1,
        "wandb_project": None,
    }
    if args.validation_dataset:
        config["test_datasets"] = [dataset_entry(args.validation_dataset.resolve())]
        config["evals_per_epoch"] = 1
    if args.model_revision:
        config["revision_of_model"] = args.model_revision
    if "qwen3.5" in args.model.casefold():
        config["chat_template"] = "qwen3_5"
        config["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        config["chat_template"] = "tokenizer_default"
    return config


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def collect_metrics(staging: Path) -> tuple[dict[str, float], dict[str, Any] | None]:
    states = sorted(staging.rglob("trainer_state.json"), key=lambda path: path.stat().st_mtime)
    if not states:
        return {}, None
    try:
        state = json.loads(states[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    metrics: dict[str, float] = {}
    history = state.get("log_history") if isinstance(state, dict) else None
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict):
                continue
            for key in ("loss", "eval_loss", "train_loss", "train_runtime", "train_samples_per_second"):
                value = numeric(row.get(key))
                if value is not None:
                    metrics[key] = value
    quality = None
    if isinstance(state, dict):
        quality = {
            "metric": state.get("best_metric"),
            "bestCheckpoint": Path(state["best_model_checkpoint"]).name
            if isinstance(state.get("best_model_checkpoint"), str)
            else None,
        }
    return metrics, quality


def ensure_adapter_files(staging: Path) -> None:
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (staging / name).is_file()]
    if missing:
        raise RuntimeError(
            "Axolotl terminó sin producir un adaptador PEFT compatible en output_dir: "
            + ", ".join(missing)
        )


def main() -> None:
    args = arguments()
    if not SAFE_NAME.fullmatch(args.name):
        raise ValueError("--name contiene caracteres no permitidos")
    if package_version("axolotl") != AXOLOTL_VERSION:
        raise RuntimeError(
            f"Se requiere axolotl=={AXOLOTL_VERSION}; entorno actual: {package_version('axolotl')}"
        )

    examples, _ = load_jsonl(args.dataset)
    validation_examples: list[dict[str, Any]] = []
    if args.validation_dataset:
        validation_examples, _ = load_jsonl(args.validation_dataset)
        if args.validation_dataset.resolve() == args.dataset.resolve():
            raise ValueError("El dataset de validación debe ser distinto del dataset de entrenamiento")
        overlap = count_prompt_overlaps(examples, validation_examples)
        if overlap:
            raise ValueError(
                f"El dataset de validación repite {overlap} preguntas finales del entrenamiento; "
                "se aborta para evitar fuga entre splits."
            )
    else:
        overlap = 0

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / args.name
    if destination.exists():
        raise FileExistsError(f"El adaptador ya existe: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{args.name}.axolotl-output-", dir=output_root))
    work_dir = Path(tempfile.mkdtemp(prefix=f".{args.name}.axolotl-work-", dir=output_root))
    config_path = work_dir / "config.yaml"
    try:
        resolved_model_type = model_type(args.model, args.model_revision)
        config = build_config(args, staging, work_dir)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"Axolotl config: {config_path}")
        print("Backend: Axolotl QLoRA")

        configured_cli = os.environ.get("AXOLOTL_CLI", "").strip()
        axolotl_cli = Path(configured_cli).expanduser() if configured_cli else Path(os.sys.executable).with_name("axolotl")
        if not axolotl_cli.is_file() or not os.access(axolotl_cli, os.X_OK):
            raise RuntimeError(f"No se encontró el CLI de Axolotl junto a {os.sys.executable}")

        subprocess.run([str(axolotl_cli), "train", str(config_path)], check=True)
        ensure_adapter_files(staging)
        metrics, quality_selection = collect_metrics(staging)

        # Keep the exact generated config as part of the adapter artifact.
        (staging / "axolotl-config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        system_prompt = common_system_prompt(examples)
        manifest = {
            "schemaVersion": 3,
            "name": args.name,
            "baseModel": args.model,
            "modelType": resolved_model_type,
            "method": "SFT_QLORA",
            "backend": "axolotl",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": str(args.dataset.resolve()),
            "validationDataset": str(args.validation_dataset.resolve()) if args.validation_dataset else None,
            "examples": len(examples),
            "validationExamples": len(validation_examples),
            "dataQuality": {
                "trainValidationPromptOverlap": overlap,
                "assistantOnlyLoss": True,
                "masking": "roles_to_train=assistant, train_on_eos=turn",
            },
            "reproducibility": {
                "modelRevision": args.model_revision,
                "datasetSha256": sha256_file(args.dataset.resolve()),
                "validationDatasetSha256": sha256_file(args.validation_dataset.resolve())
                if args.validation_dataset
                else None,
                "axolotlConfigSha256": stable_sha256(config_path.read_text(encoding="utf-8")),
                "python": platform.python_version(),
            },
            "serving": {
                "systemPrompt": system_prompt,
                "systemPromptSource": "common_training_system" if system_prompt else None,
            },
            "parameters": {
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "targetModules": "all-linear",
                "assistantOnlyLoss": True,
                "epochs": args.epochs,
                "learningRate": args.learning_rate,
                "batchSize": args.batch_size,
                "gradientAccumulation": args.gradient_accumulation,
                "maxLength": args.max_length,
                "seed": args.seed,
                "modelRevision": args.model_revision,
                "samplePacking": False,
            },
            "metrics": metrics,
            "qualitySelection": quality_selection,
            "versions": {
                "axolotl": package_version("axolotl"),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "peft": package_version("peft"),
                "trl": package_version("trl"),
                "datasets": package_version("datasets"),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
        print(json.dumps({"adapter": str(destination), "backend": "axolotl", "metrics": metrics}))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

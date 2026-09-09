#!/usr/bin/env python3
"""Entrena un adaptador QLoRA SFT y escribe un manifiesto reproducible."""
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

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import get_training_chat_template

from dataset_validation import load_jsonl
from training_quality import (
    count_prompt_overlaps,
    sha256_file,
    stable_sha256,
    summarize_token_lengths,
)

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning SFT con QLoRA para LLM Bridge")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("adapters"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--model-revision",
        help="Revisión o commit inmutable del modelo en Hugging Face; opcional para rutas locales.",
    )
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Permite continuar aunque una respuesta assistant supere --max-length.",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def numeric_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def with_perplexity(metrics: dict[str, float], loss_key: str) -> dict[str, float]:
    result = dict(metrics)
    loss = result.get(loss_key)
    if loss is not None:
        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = math.inf
        if math.isfinite(perplexity):
            result[f"{loss_key.removesuffix('_loss')}_perplexity"] = perplexity
    return result


def common_system_prompt(examples: list[dict]) -> str | None:
    """Devuelve el único system prompt común que realmente vio todo el SFT."""
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


def validate_model_architecture(model_name: str, revision: str | None = None) -> str:
    """Falla antes de reservar VRAM si Transformers no conoce el checkpoint."""
    load_kwargs = {"trust_remote_code": False}
    if revision:
        load_kwargs["revision"] = revision
    try:
        config = AutoConfig.from_pretrained(model_name, **load_kwargs)
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"Transformers {package_version('transformers')} no puede cargar la arquitectura de {model_name}. "
            "Actualiza el entorno de fine-tuning desde la GUI o requirements.txt."
        ) from error
    print(f"Arquitectura: {config.model_type}")
    return config.model_type


def format_dataset(examples: list[dict]) -> Dataset:
    """Conserva los roles para que TRL pueda enmascarar el prompt de la pérdida."""
    return Dataset.from_list(examples)


def pretrained_kwargs(revision: str | None) -> dict[str, object]:
    kwargs: dict[str, object] = {"trust_remote_code": False}
    if revision:
        kwargs["revision"] = revision
    return kwargs


def _as_flat_list(value: object) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value if isinstance(value, list) else []


def audit_token_lengths(
    tokenizer,
    examples: list[dict],
    max_length: int,
    label: str,
    allow_truncation: bool,
) -> dict:
    """Mide el texto con la misma plantilla que usará SFTTrainer."""
    total_tokens: list[int] = []
    assistant_tokens: list[int] = []
    truncated_examples: list[bool] = []
    target_truncated: list[bool] = []
    empty_assistant_after_limit: list[bool] = []

    for number, example in enumerate(examples, 1):
        encoded = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = _as_flat_list(encoded.get("input_ids"))
        assistant_mask = _as_flat_list(encoded.get("assistant_masks"))
        if not input_ids or len(input_ids) != len(assistant_mask):
            raise RuntimeError(
                f"La auditoría de tokens no pudo validar el ejemplo {number} de {label}."
            )
        if not any(bool(value) for value in assistant_mask):
            raise RuntimeError(
                f"El ejemplo {number} de {label} no contiene tokens assistant entrenables."
            )

        total = len(input_ids)
        assistant = sum(int(bool(value)) for value in assistant_mask)
        truncated = total > max_length
        target_would_be_truncated = any(
            bool(value) for value in assistant_mask[max_length:]
        )
        empty_after_limit = not any(
            bool(value) for value in assistant_mask[:max_length]
        )

        total_tokens.append(total)
        assistant_tokens.append(assistant)
        truncated_examples.append(truncated)
        target_truncated.append(target_would_be_truncated)
        empty_assistant_after_limit.append(empty_after_limit)

    summary = summarize_token_lengths(
        total_tokens,
        assistant_tokens,
        max_length,
        truncated_examples,
        target_truncated,
        empty_assistant_after_limit,
    )
    print(json.dumps({"dataset": label, **summary}, ensure_ascii=False))

    if summary["emptyAssistantAfterLimit"]:
        raise RuntimeError(
            f"{summary['emptyAssistantAfterLimit']} ejemplos de {label} quedarían sin "
            "tokens assistant dentro de --max-length."
        )
    if summary["targetTruncatedExamples"] and not allow_truncation:
        raise RuntimeError(
            f"{summary['targetTruncatedExamples']} respuestas de {label} quedarían "
            f"truncadas por --max-length={max_length}. Aumenta --max-length o usa "
            "--allow-truncation de forma explícita."
        )
    if summary["targetTruncatedExamples"]:
        print(
            f"Aviso: {summary['targetTruncatedExamples']} respuestas de {label} "
            "serán truncadas porque se solicitó --allow-truncation."
        )
    return summary


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def cuda_device_name() -> str | None:
    try:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (RuntimeError, AssertionError):
        return None


def prepare_assistant_only_template(tokenizer, examples: list[dict]) -> None:
    """Fija y comprueba la plantilla que TRL necesita para assistant_only_loss."""
    patched = get_training_chat_template(tokenizer)
    if patched is not None:
        tokenizer.chat_template = patched
        print("Chat template: TRL aplicó una plantilla compatible con assistant_only_loss")
    else:
        print("Chat template: la plantilla del tokenizer ya es compatible con TRL")

    sample = next(
        (
            example["messages"]
            for example in examples
            if any(message.get("role") == "assistant" for message in example.get("messages", []))
        ),
        None,
    )
    if sample is None:
        raise RuntimeError("El dataset no contiene ningún turno assistant para entrenar")

    try:
        encoded = tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
    except Exception as error:
        raise RuntimeError(
            "La plantilla de chat no pudo generar la máscara de tokens del asistente requerida por "
            "assistant_only_loss. Reinstala las dependencias fijadas del trainer."
        ) from error

    assistant_mask = encoded.get("assistant_masks")
    input_ids = encoded.get("input_ids")
    if not isinstance(assistant_mask, list) or not assistant_mask or not any(assistant_mask):
        raise RuntimeError(
            "La plantilla de chat produjo una máscara assistant vacía; continuar entrenaría con una pérdida inválida."
        )
    if not isinstance(input_ids, list) or len(input_ids) != len(assistant_mask):
        raise RuntimeError("La máscara assistant no coincide con los tokens generados por la plantilla de chat")
    if all(assistant_mask):
        raise RuntimeError(
            "La plantilla marcó todos los tokens como assistant; se aborta para no entrenar sobre system/user por error."
        )

    print(
        "assistant_only_loss: máscara OK "
        f"({sum(int(value) for value in assistant_mask)}/{len(assistant_mask)} tokens assistant en la muestra)"
    )


def main() -> None:
    args = arguments()
    model_revision = getattr(args, "model_revision", None)
    allow_truncation = bool(getattr(args, "allow_truncation", False))
    # TRL initializes LoRA before Trainer applies SFTConfig.seed. Seed model and
    # adapter initialization too, not just the training loop and data order.
    set_seed(args.seed)
    if not SAFE_NAME.fullmatch(args.name):
        raise ValueError("--name contiene caracteres no permitidos")
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requiere una GPU CUDA disponible")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / args.name
    if destination.exists():
        raise FileExistsError(f"El adaptador ya existe: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{args.name}.training-", dir=output_root))
    try:
        examples, _ = load_jsonl(args.dataset)
        validation_examples = None
        if args.validation_dataset is not None:
            validation_examples, _ = load_jsonl(args.validation_dataset)
            if args.validation_dataset.resolve() == args.dataset.resolve():
                raise ValueError("El dataset de validación debe ser distinto del dataset de entrenamiento")

        dataset_path = args.dataset.resolve()
        validation_path = args.validation_dataset.resolve() if args.validation_dataset else None
        dataset_sha256 = sha256_file(dataset_path)
        validation_sha256 = sha256_file(validation_path) if validation_path else None
        overlap_count = count_prompt_overlaps(examples, validation_examples or [])
        if overlap_count:
            raise ValueError(
                "El dataset de validación repite "
                f"{overlap_count} preguntas finales del dataset de entrenamiento; "
                "se aborta para evitar fuga entre splits."
            )
        source_commit = git_revision()

        model_type = validate_model_architecture(args.model, model_revision)
        load_kwargs = pretrained_kwargs(model_revision)

        tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prepare_assistant_only_template(tokenizer, examples)
        train_token_audit = audit_token_lengths(
            tokenizer,
            examples,
            args.max_length,
            "train",
            allow_truncation,
        )
        validation_token_audit = (
            audit_token_lengths(
                tokenizer,
                validation_examples,
                args.max_length,
                "validation",
                allow_truncation,
            )
            if validation_examples
            else None
        )
        dataset = format_dataset(examples)
        validation_dataset = format_dataset(validation_examples) if validation_examples else None

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization,
            device_map="auto",
            dtype=compute_dtype,
            **load_kwargs,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        use_validation = validation_dataset is not None
        config = SFTConfig(
            output_dir=str(staging / "checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            gradient_checkpointing=True,
            max_length=args.max_length,
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy="epoch" if use_validation else "no",
            load_best_model_at_end=use_validation,
            metric_for_best_model="eval_loss" if use_validation else None,
            greater_is_better=False if use_validation else None,
            save_total_limit=2,
            report_to="none",
            seed=args.seed,
            fp16=compute_dtype == torch.float16,
            bf16=compute_dtype == torch.bfloat16,
            optim="paged_adamw_8bit",
            # El dataset conserva `messages`: solo se optimizan las respuestas
            # del asistente, no el system prompt ni las preguntas del usuario.
            assistant_only_loss=True,
            eos_token=tokenizer.eos_token,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            peft_config=lora,
        )

        baseline_validation: dict[str, float] = {}
        if use_validation:
            print("\n== Validación antes del entrenamiento ==")
            baseline_validation = with_perplexity(
                numeric_metrics(trainer.evaluate(metric_key_prefix="baseline")),
                "baseline_loss",
            )

        result = trainer.train()
        metrics = numeric_metrics(result.metrics)
        train_runtime = metrics.get("train_runtime")
        if train_runtime and train_runtime > 0:
            metrics["train_tokens_per_second"] = (
                train_token_audit["totalTokens"]["sum"] / train_runtime
            )

        best_validation: dict[str, float] = {}
        if use_validation:
            print("\n== Validación del mejor checkpoint ==")
            best_validation = with_perplexity(
                numeric_metrics(trainer.evaluate(metric_key_prefix="validation")),
                "validation_loss",
            )

        trainer.model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)

        best_checkpoint = (
            Path(trainer.state.best_model_checkpoint).name
            if trainer.state.best_model_checkpoint
            else None
        )
        best_metric = (
            float(trainer.state.best_metric)
            if isinstance(trainer.state.best_metric, (int, float))
            else None
        )
        quality_selection = None
        if use_validation:
            before = baseline_validation.get("baseline_loss")
            after = best_validation.get("validation_loss")
            relative_loss_improvement = None
            if before is not None and after is not None and before > 0:
                relative_loss_improvement = (before - after) / before
            quality_selection = {
                "metric": "eval_loss",
                "direction": "minimize",
                "bestCheckpoint": best_checkpoint,
                "bestMetric": best_metric,
                "baseline": baseline_validation,
                "selected": best_validation,
                "relativeLossImprovement": relative_loss_improvement,
            }

        system_prompt = common_system_prompt(examples)
        manifest = {
            "schemaVersion": 3,
            "name": args.name,
            "baseModel": args.model,
            "modelType": model_type,
            "method": "SFT_QLORA",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": str(dataset_path),
            "validationDataset": str(validation_path) if validation_path else None,
            "examples": len(examples),
            "validationExamples": len(validation_examples or []),
            "dataQuality": {
                "trainTokenAudit": train_token_audit,
                "validationTokenAudit": validation_token_audit,
                "trainValidationPromptOverlap": overlap_count,
            },
            "reproducibility": {
                "gitCommit": source_commit,
                "modelRevision": model_revision,
                "datasetSha256": dataset_sha256,
                "validationDatasetSha256": validation_sha256,
                "tokenizerChatTemplateSha256": stable_sha256(tokenizer.chat_template),
                "python": platform.python_version(),
                "cudaRuntime": torch.version.cuda,
                "gpu": cuda_device_name(),
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
                "modelRevision": model_revision,
                "allowTruncation": allow_truncation,
            },
            "metrics": metrics,
            "qualitySelection": quality_selection,
            "versions": {
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "trl": package_version("trl"),
                "peft": package_version("peft"),
                "datasets": package_version("datasets"),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        staging.rename(destination)
        print(json.dumps({
            "adapter": str(destination),
            "metrics": manifest["metrics"],
            "qualitySelection": quality_selection,
        }))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

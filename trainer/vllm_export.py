#!/usr/bin/env python3
"""Export Qwen3.5 text PEFT adapters for vLLM's multimodal namespace.

Original PEFT files are NEVER modified. Exports are immutable, content-addressed
directories inside the adapters bind mount. Tensor operations run on CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

EXPORT_VERSION = "qwen35-language-model-v1"
PEFT_PREFIX = "base_model.model."


def needs_export(config: dict, manifest: dict) -> bool:
    model_type = manifest.get("modelType", "")
    base = config.get("base_model_name_or_path") or manifest.get("baseModel", "")
    return model_type in {"qwen3_5", "qwen3_5_text"} or bool(
        re.search(r"(?:^|/)Qwen3\.5-", str(base), re.IGNORECASE)
    )


def runtime_key(key: str) -> str:
    """Feed model.language_model.* into Qwen3.5's hf_to_vllm_mapper.

    vLLM v0.24.0 maps that prefix to language_model.model.*; the text-only
    Transformers model saves model.layers.* instead. Already-qualified keys
    and lm_head retain their names. No broad substring replacements.
    """
    if not key.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(f"Tensor LoRA no compatible con esta exportación: {key}")
    prefix = PEFT_PREFIX if key.startswith(PEFT_PREFIX) else ""
    module = key[len(prefix):]
    if module.startswith("model.layers."):
        return prefix + "model.language_model." + module[len("model."):]
    if module.startswith(("model.language_model.layers.", "lm_head.")):
        return key
    raise ValueError(f"Namespace Qwen3.5 desconocido; no se cargará silenciosamente: {key}")


def safe_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Se esperaba un archivo regular, no un enlace: {path}")
    return path


def read_config(directory: Path) -> tuple[dict, dict]:
    config = json.loads(safe_file(directory / "adapter_config.json").read_text())
    manifest_file = directory / "manifest.json"
    manifest = json.loads(safe_file(manifest_file).read_text()) if manifest_file.exists() else {}
    return config, manifest


def digest(path: Path) -> str:
    with safe_file(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def export_destination(directory: Path) -> Path:
    if directory.is_symlink():
        raise ValueError("El directorio del adaptador no puede ser un enlace simbólico.")
    fingerprint = hashlib.sha256((EXPORT_VERSION + digest(directory / "adapter_config.json")
                                 + digest(directory / "adapter_model.safetensors")).encode()).hexdigest()
    return directory.parent / ".vllm-exports" / directory.name / fingerprint


def validate_export(destination: Path) -> dict:
    metadata = json.loads(safe_file(destination / "export.json").read_text())
    if metadata.get("format") != EXPORT_VERSION:
        raise ValueError("Versión de exportación desconocida.")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if digest(destination / filename) != metadata.get("sha256", {}).get(filename):
            raise ValueError(f"Exportación dañada: {filename}. Conserva el original y revisa {destination}.")
    return metadata


def export_adapter(directory: Path) -> Path:
    config, manifest = read_config(directory)
    safe_file(directory / "adapter_model.safetensors")
    if not needs_export(config, manifest):
        return directory
    destination = export_destination(directory)
    # Reject pre-existing symlinks anywhere below the trusted bind mount root.
    for part in (destination.parent.parent, destination.parent, destination):
        if part.is_symlink():
            raise ValueError(f"La ruta de exportación no puede ser un enlace: {part}")
    if destination.exists():
        validate_export(destination)
        return destination

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    renamed = 0
    with safe_open(str(directory / "adapter_model.safetensors"), framework="pt", device="cpu") as source:
        for key in source.keys():
            new_key = runtime_key(key)
            if new_key in tensors:
                raise ValueError(f"Colisión de nombres LoRA: {new_key}")
            tensors[new_key] = source.get_tensor(key)
            renamed += int(new_key != key)
    if not tensors:
        raise ValueError("El adaptador no contiene tensores.")
    for key in tensors:
        counterpart = key.replace(".lora_A.weight", ".lora_B.weight") if key.endswith(".lora_A.weight") else key.replace(".lora_B.weight", ".lora_A.weight")
        if counterpart not in tensors:
            raise ValueError(f"Falta la pareja A/B: {key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        save_file(tensors, str(staging / "adapter_model.safetensors"), metadata={"format": "pt"})
        shutil.copyfile(directory / "adapter_config.json", staging / "adapter_config.json")
        # A concurrently replaced training result must not get the old content ID.
        if export_destination(directory) != destination:
            raise RuntimeError("El adaptador cambió durante la exportación; vuelve a cargarlo.")
        metadata = {"format": EXPORT_VERSION, "tensors": len(tensors), "renamed": renamed,
                    "sha256": {filename: digest(staging / filename) for filename in
                               ("adapter_config.json", "adapter_model.safetensors")}}
        (staging / "export.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            validate_export(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    output = export_adapter(args.adapter.absolute())
    print(json.dumps({"vllmExport": str(output), "originalPreserved": True}))

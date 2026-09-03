#!/usr/bin/env python3
"""Entrena un adaptador QLoRA SFT y escribe un manifiesto reproducible."""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from dataset_validation import load_jsonl

SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning SFT con QLoRA para LLM Bridge")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("adapters"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> None:
    args = arguments()
    if not SAFE_NAME.fullmatch(args.name): raise ValueError("--name contiene caracteres no permitidos")
    if not torch.cuda.is_available(): raise RuntimeError("QLoRA requiere una GPU CUDA disponible")
    destination = (args.output_root / args.name).resolve()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(f"El adaptador ya existe: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    examples, _ = load_jsonl(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(examples).map(lambda example: {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}, remove_columns=["messages"])
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization, device_map="auto", torch_dtype=compute_dtype, trust_remote_code=False)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    config = SFTConfig(output_dir=str(destination / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation, gradient_checkpointing=True, max_length=args.max_length, logging_steps=1, save_strategy="epoch", save_total_limit=2, report_to="none", seed=args.seed, fp16=compute_dtype == torch.float16, bf16=compute_dtype == torch.bfloat16, optim="paged_adamw_8bit", dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, peft_config=lora)
    result = trainer.train()
    trainer.model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    manifest = {"schemaVersion": 1, "name": args.name, "baseModel": args.model, "method": "SFT_QLORA", "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset": str(args.dataset.resolve()), "examples": len(examples), "parameters": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "epochs": args.epochs, "learningRate": args.learning_rate, "batchSize": args.batch_size, "gradientAccumulation": args.gradient_accumulation, "maxLength": args.max_length, "seed": args.seed}, "metrics": {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}, "versions": {"torch": torch.__version__}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"adapter": str(destination), "metrics": manifest["metrics"]}))

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

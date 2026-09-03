#!/usr/bin/env python3
import argparse, json
from pathlib import Path
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", type=Path); args = parser.parse_args()
    examples = messages = characters = 0
    with args.dataset.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip(): continue
            value = json.loads(line); items = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(items, list) or len(items) < 2: raise ValueError(f"Línea {number}: messages inválido")
            if not any(item.get("role") == "user" for item in items): raise ValueError(f"Línea {number}: falta user")
            if items[-1].get("role") != "assistant": raise ValueError(f"Línea {number}: el último mensaje debe ser assistant")
            for item in items:
                if item.get("role") not in ("system", "user", "assistant") or not isinstance(item.get("content"), str) or not item["content"].strip(): raise ValueError(f"Línea {number}: mensaje inválido")
                characters += len(item["content"])
            examples += 1; messages += len(items)
    if not examples: raise ValueError("El dataset está vacío")
    print(json.dumps({"valid": True, "examples": examples, "messages": messages, "characters": characters}))
if __name__ == "__main__": main()

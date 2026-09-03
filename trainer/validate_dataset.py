#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from dataset_validation import load_jsonl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    _, statistics = load_jsonl(args.dataset)
    print(json.dumps(statistics))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the reviewed Terraria JSONL corpus offline; never crawl the wiki."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
DATA_DIR = DIRECTORY / "data"
TRAINER = DIRECTORY.parents[1]
SOURCE_GLOB = "sources-*.json"


def load_corpus() -> dict:
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    sources: list[dict] = []
    for path in sorted(DATA_DIR.glob(SOURCE_GLOB)):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path.name} must contain a JSON array")
        sources.extend(value)
    if not sources:
        raise ValueError("The Terraria corpus contains no source shards")
    return {**metadata, "sources": sources}


def render(corpus: dict) -> tuple[str, str, str]:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "evaluation": []}
    ids: set[str] = set()
    questions: set[str] = set()
    source_urls: dict[str, set[str]] = {name: set() for name in splits}

    for source in corpus["sources"]:
        split = source.get("split")
        if source["id"] in ids or split not in splits:
            raise ValueError("Duplicate source ID or invalid split")
        ids.add(source["id"])
        if not source["url"].startswith("https://terraria.wiki.gg/wiki/"):
            raise ValueError("Source is outside Official Terraria Wiki")
        if any(source["url"] in urls for name, urls in source_urls.items() if name != split):
            raise ValueError("The same source URL cannot appear in multiple splits")
        source_urls[split].add(source["url"])

        for index, qa in enumerate(source["qa"], 1):
            question, answer = qa[:2]
            if not question.strip() or not answer.strip() or question.casefold() in questions:
                raise ValueError("Duplicate question or empty example")
            questions.add(question.casefold())
            messages = [
                {"role": "system", "content": corpus["system"]},
                {"role": "user", "content": question},
            ]
            row = {
                "id": f"terraria-{source['id']}-{index:02}",
                "messages": messages,
                "source": {
                    "title": source["title"],
                    "url": source["url"],
                    "history_url": source["url"] + "?action=history",
                    "consulted_at": corpus["collected_at"],
                    "access": "reviewed_page_or_search_index",
                },
                "license": corpus["license"],
                "attribution": corpus["attribution"],
            }
            if split in {"train", "validation"}:
                messages.append({"role": "assistant", "content": answer})
            else:
                expected = qa[2]
                if not expected or not all(fragment.casefold() in answer.casefold() for fragment in expected):
                    raise ValueError("Evaluation reference does not satisfy its lexical smoke test")
                row.update({
                    "contains": expected,
                    "reference_answer": answer,
                    "rubric": (
                        "Manually check factual accuracy, required conditions, and the absence of contradictions. "
                        "contains is only a lexical smoke test."
                    ),
                })
            splits[split].append(row)

    encode = lambda rows: "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    return encode(splits["train"]), encode(splits["validation"]), encode(splits["evaluation"])


def materialize() -> dict[str, int]:
    corpus = load_corpus()
    training, validation, evaluation = render(corpus)
    outputs = {
        TRAINER / "examples" / "terraria-training.jsonl": training,
        TRAINER / "examples" / "terraria-validation.jsonl": validation,
        DIRECTORY / "evaluation.jsonl": evaluation,
    }
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    return {
        "training": len(training.splitlines()),
        "validation": len(validation.splitlines()),
        "evaluation": len(evaluation.splitlines()),
        "sources": len(corpus["sources"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the source corpus without writing JSONL files")
    args = parser.parse_args()

    corpus = load_corpus()
    training, validation, evaluation = render(corpus)
    summary = {
        "training": len(training.splitlines()),
        "validation": len(validation.splitlines()),
        "evaluation": len(evaluation.splitlines()),
        "sources": len(corpus["sources"]),
    }
    if not args.check:
        summary = materialize()
    print(json.dumps({**summary, "verified": args.check}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the reviewed Terraria JSONL corpus offline; never crawl the wiki."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
TRAINER = DIRECTORY.parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs without writing")
    args = parser.parse_args()

    corpus = json.loads((DIRECTORY / "curated.json").read_text(encoding="utf-8"))
    training, validation, evaluation = render(corpus)
    outputs = {
        TRAINER / "examples" / "terraria-training.jsonl": training,
        TRAINER / "examples" / "terraria-validation.jsonl": validation,
        DIRECTORY / "evaluation.jsonl": evaluation,
    }
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Out of date: {path}. Run trainer/corpora/terraria/build.py")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    print(json.dumps({
        "training": len(training.splitlines()),
        "validation": len(validation.splitlines()),
        "evaluation": len(evaluation.splitlines()),
        "sources": len(corpus["sources"]),
        "verified": args.check,
    }))


if __name__ == "__main__":
    main()

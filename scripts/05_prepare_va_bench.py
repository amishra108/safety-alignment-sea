#!/usr/bin/env python3

import argparse
import collections
import json
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
DATA = ROOT / "data/va_safety_bench"
QUESTIONS = DATA / "raw/SEA/VA-SafetyBench/Audio_questions"
AUDIO = DATA / "raw/Audio"
OUTPUT = DATA / "processed/manifest.jsonl"

CONDITIONS = ("Text_only", "Audio_only", "KPT", "KPT_noise")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not QUESTIONS.is_dir():
        raise FileNotFoundError(QUESTIONS)
    if not AUDIO.is_dir():
        raise FileNotFoundError(AUDIO)

    categories = sorted(p.name for p in AUDIO.iterdir() if p.is_dir())
    if len(categories) != 8:
        raise RuntimeError(f"Expected 8 audio categories, found {categories}")

    records = []
    counts = collections.Counter()

    for category in categories:
        question_file = QUESTIONS / f"{category}.json"
        if not question_file.is_file():
            raise FileNotFoundError(question_file)

        source = json.loads(question_file.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError(f"Expected question dictionary: {question_file}")

        for source_id in sorted(source, key=int):
            prompts = source[source_id]
            if set(prompts) != set(CONDITIONS):
                raise ValueError(
                    f"Unexpected conditions: {category}/{source_id}: "
                    f"{sorted(prompts)}"
                )

            if not isinstance(prompts["Text_only"], str):
                raise ValueError(f"Missing text-only prompt: {category}/{source_id}")
            if prompts["Audio_only"] is not None:
                raise ValueError(
                    f"Audio_only must have null text: {category}/{source_id}"
                )

            for condition in CONDITIONS:
                prompt = prompts[condition]
                if prompt is not None and not isinstance(prompt, str):
                    raise ValueError(
                        f"Invalid prompt: {category}/{source_id}/{condition}"
                    )

                audio = None
                if condition != "Text_only":
                    audio = AUDIO / category / condition / f"{source_id}.mp3"
                    if not audio.is_file() or audio.stat().st_size == 0:
                        raise FileNotFoundError(audio)

                records.append({
                    "task_id": f"va__{category}__{source_id}__{condition}",
                    "benchmark": "VA-SafetyBench",
                    "category": category,
                    "condition": condition,
                    "source_id": source_id,
                    "source_text": prompts["Text_only"],
                    "prompt_text": prompt,
                    "audio_path": str(audio.resolve()) if audio else None,
                })
                counts[condition] += 1
                counts[f"category:{category}"] += 1

    if args.limit is None:
        if len(records) != 3888:
            raise RuntimeError(f"Expected 3888 rows, found {len(records)}")
        for condition in CONDITIONS:
            if counts[condition] != 972:
                raise RuntimeError(
                    f"{condition}: expected 972, found {counts[condition]}"
                )

    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        records = records[:args.limit]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".jsonl.tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary.replace(OUTPUT)

    print(f"Manifest: {OUTPUT}")
    print(f"Total: {len(records)}")
    print(f"Categories: {len(categories)}")
    for condition in CONDITIONS:
        print(f"  {condition}: {counts[condition]}")
    for category in categories:
        print(f"  {category}: {counts[f'category:{category}']}")
    print("All required audio files validated.")


if __name__ == "__main__":
    main()

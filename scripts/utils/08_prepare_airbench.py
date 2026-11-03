#!/usr/bin/env python3

import argparse
import collections
import json
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
CHAT = ROOT / "data/airbench/raw/Chat"
SOURCE = CHAT / "Chat_meta.json"
OUTPUT = ROOT / "data/airbench/processed/manifest.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Chat_meta.json must contain a list")

    records = []
    missing = []
    counts = collections.Counter()
    identifiers = set()

    for item in data:
        for key in ("uniq_id", "task_name", "dataset_name", "path",
                    "question", "answer_gt", "meta_info"):
            if key not in item:
                raise KeyError(f"Missing {key}: {item}")

        identifier = str(item["uniq_id"])
        if identifier in identifiers:
            raise ValueError(f"Duplicate uniq_id: {identifier}")
        identifiers.add(identifier)

        task = item["task_name"]
        dataset = item["dataset_name"]
        relative = Path(item["path"])

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid audio path: {relative}")

        audio = CHAT / f"{task}_{dataset}" / relative

        if not audio.is_file() or audio.stat().st_size == 0:
            missing.append(str(audio))
            continue

        records.append({
            "task_id": f"airbench_chat__{identifier}",
            "benchmark": "AIR-Bench-Chat",
            "uniq_id": identifier,
            "task_name": task,
            "dataset_name": dataset,
            "source_audio_path": item["path"],
            "audio_path": str(audio.resolve()),
            "prompt_text": item["question"],
            "question": item["question"],
            "answer_gt": item["answer_gt"],
            "meta_info": item["meta_info"],
            "condition": "AIRBench",
        })
        counts[task] += 1

    print(f"Metadata rows: {len(data)}")
    print(f"Valid rows: {len(records)}")
    print(f"Missing audio: {len(missing)}")

    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")

    if missing:
        missing_file = OUTPUT.parent / "missing_audio.txt"
        missing_file.write_text("\n".join(missing) + "\n")
        print(f"Missing audio list: {missing_file}")

        if not args.allow_missing:
            raise RuntimeError(
                "Missing audio files. Manifest not written. "
                "Check missing_audio.txt or use --allow-missing."
            )

    if not records:
        raise RuntimeError("No usable examples")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".jsonl.tmp")

    with temp.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    temp.replace(OUTPUT)
    print(f"Manifest: {OUTPUT}")


if __name__ == "__main__":
    main()

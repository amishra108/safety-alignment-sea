#!/usr/bin/env python3

import argparse
import json
import os
import re
import socket
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_audio_jailbreaks import (
    MODEL_OUTPUT_NAMES,
    atomic_json,
    directory_lock,
    load_model,
    read_jsonl,
    resolve_checkpoint,
)
from scripts.eval_va_safetybench import qwen_infer, video_infer


MANIFEST = ROOT / "data/mmau-test-mini/manifest.jsonl"
OUTPUT_ROOT = ROOT / "outputs/_evaluations/mmau"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni or Video4B on MMAU test-mini"
    )

    p.add_argument(
        "--model",
        required=True,
        choices=("qwen25_omni", "video4b"),
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--base-model-path", type=Path)
    p.add_argument("--epoch", type=int)
    p.add_argument("--run-name", required=True)

    p.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=ROOT,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )

    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--merge-only", action="store_true")

    args = p.parse_args()

    if args.num_shards < 1:
        p.error("--num-shards must be positive")

    if not 0 <= args.shard_id < args.num_shards:
        p.error("Invalid --shard-id")

    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")

    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")

    if args.preflight and (
        args.merge_only or args.num_shards != 1
    ):
        p.error(
            "Preflight requires one shard and cannot be merge-only"
        )

    if not args.run_name or any(
        c not in
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_.-"
        for c in args.run_name
    ):
        p.error("Invalid --run-name")

    return args


def load_rows(args):
    rows = read_jsonl(args.manifest)

    if not rows:
        raise RuntimeError(
            f"Empty MMAU manifest: {args.manifest}"
        )

    required = {
        "task_id",
        "source_id",
        "benchmark",
        "condition",
        "audio_path",
        "prompt_text",
        "instruction",
        "choices",
        "answer",
        "answer_index",
        "answer_letter",
        "dataset",
        "task",
        "split",
        "category",
        "sub_category",
        "difficulty",
    }

    seen = set()

    for index, row in enumerate(rows):
        missing = required - set(row)

        if missing:
            raise RuntimeError(
                f"MMAU manifest row {index} missing fields: "
                f"{sorted(missing)}"
            )

        task_id = row["task_id"]

        if task_id in seen:
            raise RuntimeError(
                f"Duplicate MMAU task_id: {task_id}"
            )

        seen.add(task_id)

        if row["benchmark"] != "MMAU-test-mini":
            raise RuntimeError(
                f"Wrong benchmark for {task_id}: "
                f"{row['benchmark']!r}"
            )

        if row["condition"] != "Audio_only":
            raise RuntimeError(
                f"Wrong condition for {task_id}: "
                f"{row['condition']!r}"
            )

        if (
            not isinstance(row["choices"], list)
            or len(row["choices"]) < 2
        ):
            raise RuntimeError(
                f"Invalid choices for {task_id}"
            )

        if row["answer"] not in row["choices"]:
            raise RuntimeError(
                f"Gold answer is not one of the choices: {task_id}"
            )

        gold_index = row["choices"].index(row["answer"])

        if row["answer_index"] != gold_index:
            raise RuntimeError(
                f"answer_index mismatch for {task_id}"
            )

        gold_letter = chr(ord("A") + gold_index)

        if row["answer_letter"] != gold_letter:
            raise RuntimeError(
                f"answer_letter mismatch for {task_id}"
            )

        path = Path(row["audio_path"])

        if not path.is_absolute():
            path = args.audio_root / path

        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing MMAU audio for {task_id}: {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Empty MMAU audio for {task_id}: {path}"
            )

    if len(rows) != 1000:
        raise RuntimeError(
            f"Expected 1000 MMAU test-mini rows, found {len(rows)}"
        )

    if args.limit is not None:
        rows = rows[:args.limit]

    if args.preflight:
        rows = rows[:1]

    return rows


def absolute_audio_path(args, row):
    path = Path(row["audio_path"])

    if not path.is_absolute():
        path = args.audio_root / path

    return path.resolve()


def choice_labels(count):
    if count < 2 or count > 26:
        raise RuntimeError(
            f"Unsupported number of MMAU choices: {count}"
        )

    return [
        chr(ord("A") + index)
        for index in range(count)
    ]


def prediction_index(response, choices):
    if not isinstance(response, str):
        return None

    text = response.strip()
    labels = choice_labels(len(choices))

    match = re.fullmatch(
        r"\s*[\(\[]?([A-Za-z])[\)\].:]?\s*",
        text,
    )

    if match:
        letter = match.group(1).upper()

        if letter in labels:
            return labels.index(letter)

    match = re.match(
        r"^\s*(?:the\s+)?"
        r"(?:correct\s+)?"
        r"(?:answer\s*(?:is|:)?\s*)?"
        r"[\(\[]?([A-Za-z])[\)\].:]?(?:\s|$)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        letter = match.group(1).upper()

        if letter in labels:
            return labels.index(letter)

    normalized = " ".join(text.lower().split())

    exact = [
        index
        for index, choice in enumerate(choices)
        if normalized == " ".join(choice.lower().split())
    ]

    if len(exact) == 1:
        return exact[0]

    return None


def append_record(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def append_text(path, record):
    response = record["response"].replace(
        "\r", " "
    ).replace(
        "\n", " "
    )

    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"{record['task_id']} "
            f"gold={record['gold_letter']} "
            f"pred={record['prediction_letter']} "
            f"correct={record['correct']} "
            f"{response}\n"
        )
        f.flush()


def rewrite_text(path, records):
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )

    with temporary.open("w", encoding="utf-8") as f:
        for record in records:
            response = record["response"].replace(
                "\r", " "
            ).replace(
                "\n", " "
            )

            f.write(
                f"{record['task_id']} "
                f"gold={record['gold_letter']} "
                f"pred={record['prediction_letter']} "
                f"correct={record['correct']} "
                f"{response}\n"
            )

        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def shard_paths(root, shard_id, count):
    stem = f"shard{shard_id:03d}-of-{count:03d}"
    directory = root / "shards"

    return (
        directory / f"responses.{stem}.jsonl",
        directory / f"responses.{stem}.txt",
        directory / f"{stem}.done",
    )


def score_records(records):
    total = len(records)

    correct = sum(
        int(record["correct"])
        for record in records
    )

    unparsed = sum(
        record["prediction_index"] is None
        for record in records
    )

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": (
            correct / total
            if total
            else None
        ),
        "unparsed_predictions": unparsed,
    }

    for field in (
        "task",
        "category",
        "sub_category",
        "difficulty",
        "dataset",
    ):
        groups = defaultdict(list)

        for record in records:
            value = record.get(field)

            if value is not None:
                groups[str(value)].append(record)

        if not groups:
            continue

        output = {}

        for value, group in sorted(groups.items()):
            group_correct = sum(
                int(record["correct"])
                for record in group
            )

            output[value] = {
                "n": len(group),
                "correct": group_correct,
                "accuracy": group_correct / len(group),
            }

        summary[f"by_{field}"] = output

    return summary


def write_summary(root, records):
    summary = score_records(records)

    atomic_json(
        root / "scores.json",
        summary,
    )

    print(
        f"[MMAU] accuracy="
        f"{summary['accuracy'] * 100:.2f}% "
        f"({summary['correct']}/{summary['total']}) "
        f"unparsed={summary['unparsed_predictions']}",
        flush=True,
    )

    for field in (
        "by_task",
        "by_category",
        "by_sub_category",
        "by_difficulty",
        "by_dataset",
    ):
        if field not in summary:
            continue

        print(f"[{field}]", flush=True)

        for name, result in summary[field].items():
            print(
                f"  {name}: "
                f"{result['accuracy'] * 100:.2f}% "
                f"({result['correct']}/{result['n']})",
                flush=True,
            )


def merge(root, rows, count):
    with directory_lock(root / ".merge.lock"):
        for shard_id in range(count):
            _, _, done = shard_paths(
                root,
                shard_id,
                count,
            )

            if not done.is_file():
                print(
                    f"[merge] Waiting for shard {shard_id}",
                    flush=True,
                )
                return

        expected = [
            row["task_id"]
            for row in rows
        ]

        expected_set = set(expected)
        records = {}

        for shard_id in range(count):
            jsonl, _, _ = shard_paths(
                root,
                shard_id,
                count,
            )

            for record in read_jsonl(jsonl):
                task_id = record["task_id"]

                if (
                    task_id not in expected_set
                    or task_id in records
                ):
                    raise RuntimeError(
                        f"Duplicate/unexpected MMAU result: "
                        f"{task_id}"
                    )

                records[task_id] = record

        if set(records) != expected_set:
            missing = expected_set - set(records)

            raise RuntimeError(
                f"Incomplete MMAU merge: "
                f"{len(missing)} missing"
            )

        ordered = [
            records[task_id]
            for task_id in expected
        ]

        output = root / "responses.all.jsonl"
        text_output = root / "responses.all.txt"

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            delete=False,
        ) as f:
            temporary = Path(f.name)

            for record in ordered:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary, output)

        rewrite_text(
            text_output,
            ordered,
        )

        write_summary(
            root,
            ordered,
        )

        print(
            f"[merged] {len(ordered)} MMAU responses: "
            f"{output}",
            flush=True,
        )


def main():
    args = parse_args()

    args.manifest = args.manifest.expanduser().resolve()
    args.audio_root = args.audio_root.expanduser().resolve()

    checkpoint, base, kind = resolve_checkpoint(args)
    rows = load_rows(args)

    root = (
        args.output_root.expanduser().resolve()
        / MODEL_OUTPUT_NAMES[args.model]
        / args.run_name
    )

    if args.preflight:
        root /= "preflight"

    generation = (
        {
            "method": "Qwen2.5-Omni.generate",
            "return_audio": False,
            "thinker_do_sample": False,
            "use_audio_in_video": False,
            "max_new_tokens": args.max_new_tokens,
        }
        if args.model == "qwen25_omni"
        else {
            "method": "Video4B backend.generate",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "enforce_underlying_greedy_generate": True,
        }
    )

    config = {
        "benchmark": "MMAU-test-mini",
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "base_model": str(base),
        "run_name": args.run_name,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "input_mode": "audio_plus_multiple_choice_question",
        "generation": generation,
        "scoring": "multiple_choice_accuracy",
        "preflight": args.preflight,
    }

    atomic_json(
        root / "run_config.json",
        config,
    )

    if args.merge_only:
        merge(
            root,
            rows,
            args.num_shards,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    start = (
        len(rows)
        * args.shard_id
        // args.num_shards
    )

    end = (
        len(rows)
        * (args.shard_id + 1)
        // args.num_shards
    )

    assigned = rows[start:end]

    expected = {
        row["task_id"]
        for row in assigned
    }

    jsonl, txt, done = shard_paths(
        root,
        args.shard_id,
        args.num_shards,
    )

    jsonl.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"shard{args.shard_id:03d}"
        f"-of-{args.num_shards:03d}"
    )

    with directory_lock(
        jsonl.parent / f"{stem}.lock"
    ):
        existing = read_jsonl(
            jsonl,
            repair=True,
        )

        completed = set()

        for record in existing:
            task_id = record["task_id"]

            if (
                task_id not in expected
                or task_id in completed
            ):
                raise RuntimeError(
                    f"Invalid resumed MMAU record: "
                    f"{task_id}"
                )

            completed.add(task_id)

        rewrite_text(
            txt,
            existing,
        )

        pending = [
            row
            for row in assigned
            if row["task_id"] not in completed
        ]

        print(
            f"[MMAU] model={args.model} "
            f"checkpoint={checkpoint} "
            f"host={socket.gethostname().split('.')[0]} "
            f"shard={args.shard_id}/{args.num_shards} "
            f"assigned={len(assigned)} "
            f"completed={len(completed)} "
            f"pending={len(pending)}",
            flush=True,
        )

        adapter = None

        for position, row in enumerate(
            pending,
            1,
        ):
            audio_path = absolute_audio_path(
                args,
                row,
            )

            if adapter is None:
                adapter = load_model(
                    args,
                    checkpoint,
                    base,
                    kind,
                )

                runtime = {
                    "adapter_class": (
                        type(adapter).__module__
                        + "."
                        + type(adapter).__name__
                    ),
                    "model_class": (
                        type(adapter.model).__module__
                        + "."
                        + type(adapter.model).__name__
                    ),
                    "model_dtype": str(
                        next(
                            adapter.model.parameters()
                        ).dtype
                    ),
                    "generation": generation,
                }

                atomic_json(
                    root / f"runtime.{stem}.json",
                    runtime,
                )

            model_row = {
                **row,
                "audio_path": str(audio_path),
                "condition": "Audio_only",
            }

            instruction = row["prompt_text"]

            response = (
                qwen_infer(
                    adapter,
                    args,
                    model_row,
                    instruction,
                )
                if args.model == "qwen25_omni"
                else video_infer(
                    adapter,
                    args,
                    model_row,
                    instruction,
                )
            )

            gold_index = row["answer_index"]

            predicted_index = prediction_index(
                response,
                row["choices"],
            )

            labels = choice_labels(
                len(row["choices"])
            )

            prediction_letter = (
                labels[predicted_index]
                if predicted_index is not None
                else None
            )

            record = {
                **row,
                "audio_path": str(audio_path),
                "model": args.model,
                "checkpoint_path": str(checkpoint),
                "reconstruction_base_model": str(base),
                "host": (
                    socket.gethostname()
                    .split(".")[0]
                ),
                "model_input_mode": (
                    "audio_plus_multiple_choice_question"
                ),
                "evaluation_instruction": instruction,
                "response": response,
                "gold_index": gold_index,
                "gold_letter": row["answer_letter"],
                "prediction_index": predicted_index,
                "prediction_letter": prediction_letter,
                "correct": (
                    predicted_index == gold_index
                ),
                "generation": generation,
            }

            append_record(
                jsonl,
                record,
            )

            append_text(
                txt,
                record,
            )

            completed.add(
                row["task_id"]
            )

            print(
                f"[{position}/{len(pending)}] "
                f"{row['task_id']} "
                f"task={row['task']} "
                f"gold={row['answer_letter']} "
                f"pred="
                f"{prediction_letter or 'UNPARSED'} "
                f"correct="
                f"{predicted_index == gold_index} "
                f"response={response[:100]!r}",
                flush=True,
            )

        if completed != expected:
            raise RuntimeError(
                "MMAU shard incomplete: "
                f"{len(expected - completed)} missing"
            )

        done.write_text(
            "complete\n",
            encoding="utf-8",
        )

    merge(
        root,
        rows,
        args.num_shards,
    )


if __name__ == "__main__":
    main()

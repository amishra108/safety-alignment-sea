#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
import socket
import tempfile
import wave
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_audio_jailbreaks import (
    MODEL_OUTPUT_NAMES,
    atomic_json,
    load_model,
    read_jsonl,
    resolve_checkpoint,
)
from scripts.eval_va_safetybench import qwen_infer, video_infer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/airbench/processed/manifest.jsonl"
OUTPUT_ROOT = ROOT / "outputs/_evaluations/airbench"

EXPECTED_TASK_COUNTS = {
    "speech_QA": 400,
    "speech_dialogue_QA": 400,
    "sound_QA": 400,
    "sound_generation_QA": 100,
    "music_QA": 400,
    "music_generation_analysis_QA": 100,
    "speech_and_sound_QA": 200,
    "speech_and_music_QA": 200,
}


def parse_args():
    p = argparse.ArgumentParser(description="AIR-Bench-Chat evaluation")
    p.add_argument("--model", required=True, choices=MODEL_OUTPUT_NAMES)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--base-model-path", type=Path)
    p.add_argument("--epoch", type=int)
    p.add_argument("--run-name", required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--audio-root", type=Path, default=ROOT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument(
        "--tasks", nargs="+", choices=tuple(EXPECTED_TASK_COUNTS),
    )
    p.add_argument(
        "--text-instruction", default="",
        help="Optional prefix before the original AIR-Bench question",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        p.error("Invalid shard configuration")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.preflight and (args.merge_only or args.num_shards != 1):
        p.error("Preflight requires one shard and cannot be merge-only")
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for c in args.run_name
    ):
        p.error("Invalid run name")

    return args


def audio_file(args, row):
    path = Path(row["audio_path"])
    if not path.is_absolute():
        path = args.audio_root / path
    return path.resolve()


def invalid_audio(path):
    if not path.is_file():
        return "Missing audio file"
    if path.stat().st_size == 0:
        return "Empty audio file"

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as audio:
                rate = audio.getframerate()
                frames = audio.getnframes()
                if rate <= 0 or frames <= 0:
                    return "Zero-length or invalid WAV"
        except (wave.Error, EOFError, OSError, ValueError) as exc:
            return f"Invalid WAV: {type(exc).__name__}: {exc}"

    return None


def load_rows(args):
    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError("Empty AIR-Bench manifest")

    seen = set()
    counts = Counter()

    for row in rows:
        task_id = row["task_id"]
        if task_id in seen:
            raise ValueError(f"Duplicate AIR-Bench task: {task_id}")
        seen.add(task_id)

        if row["benchmark"] != "AIR-Bench-Chat":
            raise ValueError(f"Wrong benchmark: {task_id}")
        if row["condition"] != "AIRBench":
            raise ValueError(f"Wrong condition: {task_id}")
        if row["task_name"] not in EXPECTED_TASK_COUNTS:
            raise ValueError(f"Unknown task name: {row['task_name']}")
        if not isinstance(row["prompt_text"], str) or not row["prompt_text"].strip():
            raise ValueError(f"Missing question: {task_id}")
        if row["question"] != row["prompt_text"]:
            raise ValueError(f"Question mismatch: {task_id}")

        counts[row["task_name"]] += 1

    if len(rows) != 2200 or dict(counts) != EXPECTED_TASK_COUNTS:
        raise ValueError(
            f"Unexpected AIR-Bench manifest: rows={len(rows)}, tasks={dict(counts)}"
        )

    if args.tasks:
        rows = [row for row in rows if row["task_name"] in args.tasks]

    if args.limit is not None:
        rows = rows[:args.limit]

    if args.preflight:
        rows = [
            row for row in rows
            if invalid_audio(audio_file(args, row)) is None
        ][:1]

    if not rows:
        raise ValueError("No matching AIR-Bench examples")

    return rows


def text_line(value):
    return value.replace("\r", " ").replace("\n", " ") + "\n"


def write_text(records, path):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")

    with temporary.open("w", encoding="utf-8") as f:
        previous_task = None

        for record in records:
            if record.get("response") is None:
                continue

            task = record["task_name"]
            if task != previous_task:
                f.write(f"\n=== {task.upper()} ===\n\n")
                previous_task = task

            f.write(text_line(record["response"]))

    os.replace(temporary, path)

def append_text(path, record, previous_task):
    if record.get("response") is None:
        return previous_task

    task = record["task_name"]

    with path.open("a", encoding="utf-8") as f:
        if task != previous_task:
            f.write(f"\n=== {task.upper()} ===\n\n")
        f.write(text_line(record["response"]))

    return task


def append_record(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def shard_paths(root, shard_id, count):
    stem = f"shard{shard_id:03d}-of-{count:03d}"
    directory = root / "shards"

    return (
        directory / f"responses.{stem}.jsonl",
        directory / f"responses.{stem}.txt",
        directory / f"{stem}.done",
    )


def merge(root, rows, count):
    with (root / ".merge.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        for i in range(count):
            _, _, done = shard_paths(root, i, count)
            if not done.is_file():
                print(f"[merge] Waiting for shard {i}", flush=True)
                return

        expected = [row["task_id"] for row in rows]
        expected_set = set(expected)
        records = {}

        for i in range(count):
            jsonl, _, _ = shard_paths(root, i, count)

            for record in read_jsonl(jsonl):
                key = record["task_id"]
                if key in records or key not in expected_set:
                    raise ValueError(f"Duplicate or unexpected AIR-Bench record: {key}")
                records[key] = record

        if set(records) != expected_set:
            raise ValueError(
                f"Incomplete AIR-Bench merge: {len(expected_set - set(records))} missing"
            )

        ordered = [records[key] for key in expected]
        output = root / "responses.all.jsonl"

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            delete=False,
        ) as f:
            temporary = Path(f.name)
            for record in ordered:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(temporary, output)
        write_text(ordered, root / "responses.all.txt")

        print(f"[merged] {len(ordered)} responses: {output}", flush=True)


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
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "base_model": str(base),
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "tasks": args.tasks,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "text_instruction": args.text_instruction,
        "question_source": "prompt_text",
        "input_mode": "audio_plus_original_question",
        "generation": generation,
        "preflight": args.preflight,
    }

    atomic_json(root / "run_config.json", config)

    if args.merge_only:
        merge(root, rows, args.num_shards)
        return

    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    jsonl, txt, done = shard_paths(root, args.shard_id, args.num_shards)
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    start = len(rows) * args.shard_id // args.num_shards
    end = len(rows) * (args.shard_id + 1) // args.num_shards
    assigned = rows[start:end]
    expected = {row["task_id"] for row in assigned}

    stem = f"shard{args.shard_id:03d}-of-{args.num_shards:03d}"

    with (jsonl.parent / f"{stem}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        existing = read_jsonl(jsonl, repair=True)
        completed = set()

        for record in existing:
            task_id = record["task_id"]
            if task_id not in expected or task_id in completed:
                raise ValueError(f"Invalid resumed record: {task_id}")
            completed.add(task_id)

        write_text(existing, txt)
        previous_task = next(
            (
                record["task_name"]
                for record in reversed(existing)
                if record.get("response") is not None
            ),
            None,
        )

        pending = [
            row for row in assigned if row["task_id"] not in completed
        ]

        print(
            f"[AIR-Bench] model={args.model} checkpoint={checkpoint} "
            f"host={socket.gethostname().split('.')[0]} "
            f"shard={args.shard_id}/{args.num_shards} "
            f"completed={len(completed)} pending={len(pending)}",
            flush=True,
        )

        adapter = None

        for index, row in enumerate(pending, 1):
            path = audio_file(args, row)
            reason = invalid_audio(path)

            instruction = args.text_instruction + row["prompt_text"]

            if reason is not None:
                response = None
                status = "skipped_invalid_audio"
                print(f"[skip] {row['task_id']}: {reason}", flush=True)
            else:
                if adapter is None:
                    adapter = load_model(args, checkpoint, base, kind)

                    runtime = {
                        "adapter_class": (
                            type(adapter).__module__ + "." + type(adapter).__name__
                        ),
                        "model_class": (
                            type(adapter.model).__module__ + "." + type(adapter.model).__name__
                        ),
                        "model_dtype": str(next(adapter.model.parameters()).dtype),
                        "generation_config": (
                            adapter.omni_model.generation_config.to_dict()
                            if args.model == "qwen25_omni"
                            else adapter.model.generation_config.to_dict()
                        ),
                        "generation_call": generation,
                    }
                    atomic_json(root / f"runtime.{stem}.json", runtime)

                model_row = {
                    **row,
                    "audio_path": str(path),
                    "prompt_text": instruction,
                }

                response = (
                    qwen_infer(adapter, args, model_row, instruction)
                    if args.model == "qwen25_omni"
                    else video_infer(adapter, args, model_row, instruction)
                )
                status = "completed"

            record = {
                **row,
                "model": args.model,
                "checkpoint_path": str(checkpoint),
                "reconstruction_base_model": str(base),
                "host": socket.gethostname().split(".")[0],
                "status": status,
                "skip_reason": reason,
                "model_input_mode": "audio_plus_original_question",
                "evaluation_instruction": instruction,
                "generation": generation,
                "response": response,
            }

            append_record(jsonl, record)
            previous_task = append_text(txt, record, previous_task)
            completed.add(row["task_id"])

            print(
                f"[{index}/{len(pending)}] {row['task_id']} "
                f"{response[:200]!r}" if response is not None
                else f"[{index}/{len(pending)}] {row['task_id']} SKIPPED",
                flush=True,
            )

        if completed != expected:
            raise ValueError("AIR-Bench shard incomplete")

        done.write_text("complete\n", encoding="utf-8")

    merge(root, rows, args.num_shards)


if __name__ == "__main__":
    main()

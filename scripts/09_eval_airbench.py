#!/usr/bin/env python3

import argparse
import fcntl
import importlib.util
import json
import os
import socket
import subprocess
from functools import lru_cache
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
MANIFEST = ROOT / "data/airbench/processed/manifest.jsonl"
OUTPUT_ROOT = ROOT / "outputs/09_eval_airbench"
EVALUATOR_SCRIPT = ROOT / "scripts/06_evaluate_va_bench.py"

MODELS = (
    "omni_base",
    "omni_sft",
    "omni_dpo",
    "salmonn_base",
    "salmonn_sft",
    "salmonn_dpo",
)

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


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSONL: {path}:{number}"
                    ) from exc


def load_evaluators():
    spec = importlib.util.spec_from_file_location(
        "va_bench_evaluator", EVALUATOR_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_rows(rows, full):
    if not rows:
        raise RuntimeError("Empty AIR-Bench selection")

    identifiers = set()
    counts = Counter()

    for row in rows:
        identifier = row["task_id"]

        if identifier in identifiers:
            raise RuntimeError(f"Duplicate task ID: {identifier}")
        identifiers.add(identifier)

        if row["benchmark"] != "AIR-Bench-Chat":
            raise RuntimeError(f"Wrong benchmark: {identifier}")

        if row["condition"] != "AIRBench":
            raise RuntimeError(f"Wrong condition: {identifier}")

        question = row["prompt_text"]
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"Missing question: {identifier}")

        if row["question"] != question:
            raise RuntimeError(f"Question mismatch: {identifier}")

        if row["task_name"] not in EXPECTED_TASK_COUNTS:
            raise RuntimeError(
                f"Unexpected AIR-Bench task: {row['task_name']}"
            )

        counts[row["task_name"]] += 1

    if full:
        if len(rows) != 2200:
            raise RuntimeError(
                f"Expected 2200 examples, found {len(rows)}"
            )

        if dict(counts) != EXPECTED_TASK_COUNTS:
            raise RuntimeError(
                f"Unexpected task distribution: {dict(counts)}"
            )

    return counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--machine", required=True,
                        choices=("nete", "air213", "air211"))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be positive")

    if not 0 <= args.shard_id < args.num_shards:
        parser.error("Invalid shard configuration")

    if args.gpu < 0 or args.max_new_tokens < 1:
        parser.error("Invalid GPU or token limit")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args.preflight and args.merge_only:
        parser.error("Cannot combine --preflight and --merge-only")

    host = socket.gethostname().split(".")[0]
    if host != args.machine and not args.merge_only:
        parser.error(f"Running on {host}, expected {args.machine}")

    return args


def ensure_configuration(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_config.json"
    lock_path = directory / ".config.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))

            if existing != config:
                raise RuntimeError(
                    f"Incompatible previous run: {path}"
                )
        else:
            temporary = directory / f".run_config.{os.getpid()}.tmp"
            try:
                temporary.write_text(
                    json.dumps(config, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


def make_evaluator(args, shared, checkpoint):
    if args.model.startswith("omni"):
        return shared.OmniEvaluator(checkpoint)

    return shared.SalmonnEvaluator(
        checkpoint,
        suppress_transcription=False,
    )


@lru_cache(maxsize=None)
def invalid_audio_reason(audio_path):
    import math
    import wave

    path = Path(audio_path)

    if not path.is_file():
        return "Audio file does not exist"

    if path.stat().st_size == 0:
        return "Empty audio file"

    if path.suffix.lower() != ".wav":
        return None

    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            frames = audio.getnframes()
            duration = frames / rate if rate else 0.0

        if not math.isfinite(duration) or duration <= 0:
            return f"Invalid WAV duration: {duration}"

    except (wave.Error, EOFError, OSError, ValueError) as exc:
        return f"Invalid WAV: {type(exc).__name__}: {exc}"

    return None


def skipped_record(row, args, checkpoint, reason):
    return {
        **row,
        "response": None,
        "status": "skipped_invalid_audio",
        "skip_reason": reason,
        "model": args.run_name,
        "model_path": str(checkpoint),
        "generation_decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
        "transcription_suppression": False,
    }


def normalize_invalid_audio(record, args, checkpoint):
    reason = invalid_audio_reason(record["audio_path"])
    if reason is None:
        return record
    return skipped_record(record, args, checkpoint, reason)


def generate(evaluator, row, args):
    response = evaluator.infer(row, args.max_new_tokens)

    if not isinstance(response, str):
        raise RuntimeError(
            f"Non-string response: {row['task_id']}"
        )

    if not response.strip():
        print(
            f"[warning] Empty response: {row['task_id']}",
            flush=True,
        )

    return response


def run_preflight(args, shared, checkpoint, rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".preflight_{args.model}_",
        dir=OUTPUT_ROOT,
    ) as temporary:
        directory = Path(temporary)

        row = rows[0]
        audio = Path(row["audio_path"])

        print("=" * 80, flush=True)
        print(f"PREFLIGHT MODEL : {args.model}", flush=True)
        print(f"CHECKPOINT      : {checkpoint}", flush=True)
        print(f"TASK            : {row['task_id']}", flush=True)
        print(f"TASK TYPE       : {row['task_name']}", flush=True)
        print(f"AUDIO           : {audio}", flush=True)
        print(f"AUDIO BYTES     : {audio.stat().st_size}", flush=True)
        print(f"QUESTION        : {row['prompt_text']}", flush=True)
        print(f"MAX NEW TOKENS  : {args.max_new_tokens}", flush=True)
        print("DECODING        : greedy", flush=True)
        print("THINKING        : disabled", flush=True)
        print("EXTRA PROMPT    : none", flush=True)
        print("=" * 80, flush=True)

        evaluator = make_evaluator(args, shared, checkpoint)
        response = generate(evaluator, row, args)

        record = {
            **row,
            "response": response,
            "model": args.model,
            "model_path": str(checkpoint),
            "generation_decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
            "transcription_suppression": False,
        }

        test_output = directory / "response.jsonl"
        test_output.write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        saved = list(read_jsonl(test_output))
        if len(saved) != 1 or saved[0]["response"] != response:
            raise RuntimeError("Preflight response serialization failed")

        print("\n[response]", flush=True)
        print(response, flush=True)

        if not response.strip():
            raise RuntimeError(
                "Preflight generated an empty response"
            )

        print(
            f"\n[PREFLIGHT PASSED] {args.model}",
            flush=True,
        )

    print("[cleanup] Temporary preflight output deleted", flush=True)


def shard_paths(directory, shard_id, num_shards):
    stem = (
        f"responses.shard{shard_id:03d}"
        f"-of-{num_shards:03d}"
    )
    return (
        directory / "shards" / f"{stem}.jsonl",
        directory / "shards" / f"{stem}.done",
    )


def merge(args, rows, directory):
    lock_path = directory / ".merge.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        merged = {}
        expected_ids = {row["task_id"] for row in rows}

        for shard_id in range(args.num_shards):
            path, done = shard_paths(
                directory, shard_id, args.num_shards
            )

            if not done.is_file() or not path.is_file():
                print(
                    f"[merge] Waiting for shard {shard_id}",
                    flush=True,
                )
                return

            for record in read_jsonl(path):
                identifier = record["task_id"]

                if identifier in merged:
                    raise RuntimeError(
                        f"Duplicate merged response: {identifier}"
                    )

                merged[identifier] = normalize_invalid_audio(
                    record, args, record.get("model_path", "")
                )

        if set(merged) != expected_ids:
            missing = expected_ids - set(merged)
            extra = set(merged) - expected_ids
            raise RuntimeError(
                f"Incomplete merge: {len(missing)} missing, "
                f"{len(extra)} unexpected"
            )

        result = directory / "responses.all.jsonl"
        temporary = directory / f".responses.{os.getpid()}.tmp"

        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(
                            merged[row["task_id"]],
                            ensure_ascii=False,
                        ) + "\n"
                    )

            os.replace(temporary, result)
        finally:
            temporary.unlink(missing_ok=True)

        print(
            f"[merged] {result}: {len(merged)} responses",
            flush=True,
        )


def main():
    args = parse_args()

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    all_rows = list(read_jsonl(MANIFEST))
    validate_rows(all_rows, full=True)

    rows = (
        all_rows[:args.limit]
        if args.limit is not None
        else all_rows
    )

    validate_rows(rows, full=args.limit is None)

    shared = load_evaluators()
    checkpoint = (args.model_path or shared.CHECKPOINTS[args.model]).resolve()
    args.run_name = args.run_name or args.model

    if not args.merge_only and not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    if not args.merge_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

        if args.model.startswith("salmonn"):
            source = EVALUATOR_SCRIPT.read_text(encoding="utf-8")
            expected = (
                "with self.torch.no_grad():\n"
                "            result = self.model.generate("
            )
            if expected not in source:
                raise RuntimeError(
                    "SALMONN audio inference in "
                    "06_evaluate_va_bench.py must use torch.no_grad()"
                )

    if args.preflight:
        run_preflight(args, shared, checkpoint, all_rows)
        return

    directory = OUTPUT_ROOT / args.run_name

    config = {
        "model": args.model,
        "run_name": args.run_name,
        "model_path": str(checkpoint),
        "manifest": str(MANIFEST.resolve()),
        "num_shards": args.num_shards,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "enable_thinking": False,
        "transcription_suppression": False,
    }

    ensure_configuration(directory, config)

    if args.merge_only:
        merge(args, rows, directory)
        return

    start = len(rows) * args.shard_id // args.num_shards
    end = len(rows) * (args.shard_id + 1) // args.num_shards
    assigned = rows[start:end]

    if not assigned:
        raise RuntimeError("Empty assigned shard")

    destination, done = shard_paths(
        directory, args.shard_id, args.num_shards
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    assigned_ids = {row["task_id"] for row in assigned}
    completed = set()

    if destination.exists():
        for record in read_jsonl(destination):
            identifier = record["task_id"]

            if identifier not in assigned_ids or identifier in completed:
                raise RuntimeError(
                    f"Invalid existing response: {identifier}"
                )

            completed.add(identifier)

    pending = [
        row for row in assigned
        if row["task_id"] not in completed
    ]

    valid_pending = []
    skipped_pending = []

    for row in pending:
        try:
            reason = invalid_audio_reason(row["audio_path"])
        except Exception as exc:
            reason = f"Audio validation error: {type(exc).__name__}: {exc}"

        if reason is None:
            valid_pending.append(row)
        else:
            skipped_pending.append(
                skipped_record(row, args, checkpoint, reason)
            )

    pending = valid_pending

    for record in skipped_pending:
        print(
            f"[SKIP INVALID AUDIO] {record['task_id']}: "
            f"{record['skip_reason']}",
            flush=True,
        )

    if pending:
        done.unlink(missing_ok=True)

    print("=" * 80, flush=True)
    print(f"Model       : {args.model}", flush=True)
    print(f"Checkpoint  : {checkpoint}", flush=True)
    print(f"Host        : {args.machine}", flush=True)
    print(f"Physical GPU: {args.gpu}", flush=True)
    print(f"Shard       : {args.shard_id}/{args.num_shards}", flush=True)
    print(f"Assigned    : {len(assigned)}", flush=True)
    print(f"Completed   : {len(completed)}", flush=True)
    print(f"Pending     : {len(pending)}", flush=True)
    print(f"Output      : {destination}", flush=True)
    print("=" * 80, flush=True)

    if skipped_pending:
        with destination.open("a", encoding="utf-8") as handle:
            for record in skipped_pending:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    if pending:
        evaluator = make_evaluator(args, shared, checkpoint)

        for index, row in enumerate(pending, 1):
            print(
                f"[{index}/{len(pending)}] {row['task_id']}",
                flush=True,
            )

            extra_fields = {}

            try:
                response = generate(evaluator, row, args)
            except Exception as exc:
                response = None
                error_traceback = traceback.format_exc()

                extra_fields = {
                    "status": "inference_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_traceback": error_traceback,
                }

                print(
                    f"[SKIP INFERENCE ERROR] {row['task_id']}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                print(error_traceback, flush=True)

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            record = {
                **row,
                "response": response,
                "model": args.run_name,
                "model_path": str(checkpoint),
                "host": args.machine,
                **extra_fields,
                "generation_decoding": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "enable_thinking": False,
                "transcription_suppression": False,
            }

            with destination.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())

            if response is not None:
                print(
                    f"[response] {response[:300]!r}",
                    flush=True,
                )

    done.write_text("complete\n", encoding="utf-8")
    print(f"[complete] shard {args.shard_id}", flush=True)

    merge(args, rows, directory)


if __name__ == "__main__":
    main()

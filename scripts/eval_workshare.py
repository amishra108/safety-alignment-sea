#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import socket
import time
from contextlib import contextmanager
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_audio_jailbreaks as aj
from scripts import eval_va_safetybench as va
from scripts import eval_mmau as mmau

MODULES = {
    "audio_jailbreaks": aj,
    "va_safetybench": va,
    "mmau": mmau,
}

OUTPUTS = {
    "audio_jailbreaks": ROOT / "outputs/_evaluations/audio_jailbreaks",
    "va_safetybench": ROOT / "outputs/_evaluations/va_safetybench",
    "mmau": ROOT / "outputs/_evaluations/mmau",
}


@contextmanager
def short_lock(path):
    # Compatible with the existing evaluator directory locks
    lock_dir = Path(str(path) + ".dir")

    for attempt in range(120):
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if attempt == 119:
                raise RuntimeError(
                    f"Shard lock is still occupied: {lock_dir}"
                )
            time.sleep(0.25)

    try:
        yield
    finally:
        lock_dir.rmdir()


def arguments():
    p = argparse.ArgumentParser(
        description="Share unfinished evaluation shards across GPUs"
    )
    p.add_argument("--benchmark", required=True, choices=MODULES)
    p.add_argument("--model", required=True, choices=aj.MODEL_OUTPUT_NAMES)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--epoch", required=True, type=int)
    p.add_argument("--run-name", required=True)
    p.add_argument("--target-shard", type=int, default=2)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--num-parts", type=int, required=True)
    p.add_argument("--part-id", type=int, required=True)
    p.add_argument("--finish-only", action="store_true")

    a = p.parse_args()

    if a.num_parts < 1 or not 0 <= a.part_id < a.num_parts:
        p.error("Invalid part ID/count")
    if not 0 <= a.target_shard < a.num_shards:
        p.error("Invalid target shard")

    return a


def setup(cli):
    benchmark = cli.benchmark
    module = MODULES[benchmark]
    a = SimpleNamespace(**vars(cli))

    a.base_model_path = None
    a.output_root = OUTPUTS[benchmark]
    a.limit = None
    a.preflight = False
    a.merge_only = False
    a.max_new_tokens = 32 if benchmark == "mmau" else 1024

    a.manifest = {
        "audio_jailbreaks":
            ROOT / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv",
        "va_safetybench":
            ROOT / "data/va_safety_bench/processed/manifest.jsonl",
        "mmau":
            ROOT / "data/mmau-test-mini/manifest.jsonl",
    }[benchmark]

    a.audio_root = (
        ROOT.parent / "safety-alignment"
        if benchmark == "audio_jailbreaks"
        else ROOT
    )

    a.video_instruction = aj.VIDEO_INSTRUCTION
    a.text_instruction = (
        aj.VIDEO_INSTRUCTION if cli.model == "video4b" else ""
    )
    a.kpt_prefix = ""
    a.split = "both"
    a.modality = "audio"
    a.languages = ["english", "spanish", "chinese", "russian"]
    a.conditions = (
        list(aj.CONDITIONS)
        if benchmark == "audio_jailbreaks"
        else ["no_noise"]
    )
    a.categories = None

    if benchmark == "audio_jailbreaks":
        rows = aj.examples_for(a)
    elif benchmark == "va_safetybench":
        rows = va.select_rows(a)
    else:
        rows = mmau.load_rows(a)

    checkpoint, base, kind = aj.resolve_checkpoint(a)

    root = (
        a.output_root
        / aj.MODEL_OUTPUT_NAMES[a.model]
        / a.run_name
    )

    config = aj.read_json(root / "run_config.json")

    if (
        config.get("model") != a.model
        or config.get("num_shards") != a.num_shards
    ):
        raise RuntimeError(
            f"Existing evaluation configuration mismatch: {root}"
        )

    config_checkpoint = config.get(
        "checkpoint_path",
        config.get("checkpoint"),
    )

    if Path(config_checkpoint).resolve() != checkpoint:
        raise RuntimeError(
            f"Checkpoint differs from existing evaluation: {root}"
        )

    generation = config.get("generation", {})
    previous_max_tokens = generation.get(
        "kwargs", {}
    ).get(
        "max_new_tokens",
        generation.get("max_new_tokens"),
    )

    if previous_max_tokens != a.max_new_tokens:
        raise RuntimeError(f"Generation limit mismatch: {root}")

    if benchmark == "audio_jailbreaks":
        if (
            config.get("modality") != "audio"
            or config.get("split") != "both"
        ):
            raise RuntimeError(
                f"Audio Jailbreak selection mismatch: {root}"
            )

    if benchmark == "va_safetybench":
        if config.get("conditions") != ["no_noise"]:
            raise RuntimeError(
                f"VA-SafetyBench condition mismatch: {root}"
            )

    if Path(config.get("manifest", "")).resolve() != a.manifest.resolve():
        raise RuntimeError(
            f"Manifest differs from previous run: {root}"
        )

    start = len(rows) * a.target_shard // a.num_shards
    end = len(rows) * (a.target_shard + 1) // a.num_shards
    assigned = rows[start:end]

    if benchmark == "audio_jailbreaks":
        key = lambda row: row["example_id"]
    else:
        key = lambda row: row["task_id"]

    full_keys = [key(row) for row in assigned]

    if len(full_keys) != len(set(full_keys)):
        raise RuntimeError("Duplicate assigned keys")

    stem = f"shard{a.target_shard:03d}-of-{a.num_shards:03d}"
    shard_dir = root / "shards"

    work = (
        root / "workshare" / stem
        / f"parts_{a.num_parts:03d}"
    )

    plan = {
        "benchmark": benchmark,
        "checkpoint": str(checkpoint),
        "model": a.model,
        "run_name": a.run_name,
        "target_shard": a.target_shard,
        "num_shards": a.num_shards,
        "num_parts": a.num_parts,
        "assigned_sha256": hashlib.sha256(
            json.dumps(full_keys).encode()
        ).hexdigest(),
    }

    work.mkdir(parents=True, exist_ok=True)
    aj.atomic_json(work / "plan.json", plan)

    # Synchronize ownership transfer with the original shard lock
    shard_dir.mkdir(parents=True, exist_ok=True)
    marker = shard_dir / f".workshare.{stem}.json"

    with short_lock(shard_dir / f"{stem}.lock"):
        if marker.exists():
            if aj.read_json(marker) != plan:
                raise RuntimeError(
                    f"Different work-share plan owns shard: {marker}"
                )
        else:
            aj.atomic_json(marker, plan)

    a.generation_config = config["generation"]

    return (
        a, module, rows, assigned, root, shard_dir,
        stem, work, checkpoint, base, kind, key
    )


def record_key(benchmark, record):
    if benchmark == "audio_jailbreaks":
        return record["example_id"]
    return record["task_id"]


def validate_records(benchmark, records, allowed):
    seen = set()

    for record in records:
        key = record_key(benchmark, record)

        if key in seen or key not in allowed:
            raise RuntimeError(
                f"Unexpected or duplicate record: {key}"
            )

        if (
            benchmark == "audio_jailbreaks"
            and record.get("modality") != "audio"
        ):
            raise RuntimeError(f"Non-audio record: {key}")

        seen.add(key)

    return seen


def append_record(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def generate_record(a, module, row, adapter, checkpoint, base):
    benchmark = a.benchmark

    if benchmark == "audio_jailbreaks":
        response = (
            aj.qwen_generate(adapter, a, row, "audio")
            if a.model == "qwen25_omni"
            else aj.video_generate(adapter, a, row, "audio")
        )

        instruction = (
            a.video_instruction
            if a.model == "video4b"
            else None
        )

        return {
            **row,
            "task_id": row["example_id"] + "__audio",
            "benchmark": "Audio Jailbreak Clean Benchmark",
            "model": a.model,
            "checkpoint_path": str(checkpoint),
            "reconstruction_base_model": str(base),
            "host": socket.gethostname().split(".")[0],
            "modality": "audio",
            "model_input_mode": (
                "audio_plus_fixed_instruction"
                if instruction else "audio_only"
            ),
            "target_extra_instruction": instruction,
            "generation": a.generation_config, "response": response,
        }

    if benchmark == "va_safetybench":
        instruction = va.input_instruction(a, row)

        response = (
            va.qwen_infer(adapter, a, row, instruction)
            if a.model == "qwen25_omni"
            else va.video_infer(adapter, a, row, instruction)
        )

        return {
            **row,
            "model": a.model,
            "checkpoint_path": str(checkpoint),
            "reconstruction_base_model": str(base),
            "host": socket.gethostname().split(".")[0],
            "model_input_mode": (
                "text_only"
                if row["condition"] == "Text_only"
                else "audio_plus_text"
                if instruction
                else "audio_only"
            ),
            "evaluation_instruction": instruction,
            "generation": a.generation_config, "response": response,
        }

    audio_path = mmau.absolute_audio_path(a, row)
    model_row = {
        **row,
        "audio_path": str(audio_path),
        "condition": "Audio_only",
    }

    instruction = row["prompt_text"]

    response = (
        va.qwen_infer(adapter, a, model_row, instruction)
        if a.model == "qwen25_omni"
        else va.video_infer(adapter, a, model_row, instruction)
    )

    predicted = mmau.prediction_index(
        response, row["choices"]
    )
    labels = mmau.choice_labels(len(row["choices"]))

    return {
        **row,
        "audio_path": str(audio_path),
        "model": a.model,
        "checkpoint_path": str(checkpoint),
        "reconstruction_base_model": str(base),
        "host": socket.gethostname().split(".")[0],
        "model_input_mode": "audio_plus_multiple_choice_question",
        "evaluation_instruction": instruction,
        "generation": a.generation_config, "response": response,
        "gold_index": row["answer_index"],
        "gold_letter": row["answer_letter"],
        "prediction_index": predicted,
        "prediction_letter": (
            labels[predicted]
            if predicted is not None else None
        ),
        "correct": predicted == row["answer_index"],
    }


def finish(ctx):
    (
        a, module, rows, assigned, root, shard_dir,
        stem, work, checkpoint, base, kind, key,
    ) = ctx

    if any(
        not (work / f"part_{i:03d}.done").is_file()
        for i in range(a.num_parts)
    ):
        print(
            f"[finish] Awaiting other parts: "
            f"{a.run_name}/{a.benchmark}",
            flush=True,
        )
        return

    # Serialize consolidation and merging
    with short_lock(work / "finalize.lock"):
        expected = {key(row) for row in assigned}
        incoming = {}

        for part in range(a.num_parts):
            path = work / f"part_{part:03d}.jsonl"
            allowed = {
                key(row)
                for index, row in enumerate(assigned)
                if index % a.num_parts == part
            }

            for record in aj.read_jsonl(path):
                item = record_key(a.benchmark, record)

                if item not in allowed or item in incoming:
                    raise RuntimeError(
                        f"Invalid or duplicate helper result: {item}"
                    )

                incoming[item] = record

        target = shard_dir / f"responses.{stem}.jsonl"
        marker = shard_dir / f".workshare.{stem}.json"

        with short_lock(shard_dir / f"{stem}.lock"):
            if (
                not marker.exists()
                or aj.read_json(marker)
                != aj.read_json(work / "plan.json")
            ):
                raise RuntimeError(
                    f"Shard handoff marker missing or changed: {marker}"
                )

            original = aj.read_jsonl(target, repair=True)
            existing = validate_records(
                a.benchmark, original, expected
            )

            original_by_id = {
                record_key(a.benchmark, record): record
                for record in original
            }

            for item in existing & incoming.keys():
                if original_by_id[item] != incoming[item]:
                    raise RuntimeError(
                        f"Conflicting completed record: {item}"
                    )

            if existing | incoming.keys() != expected:
                raise RuntimeError(
                    f"Missing "
                    f"{len(expected - existing - incoming.keys())} "
                    "records; nothing consolidated"
                )

            # Preserve original rows; append missing results only
            for row in assigned:
                item = key(row)

                if item not in existing:
                    append_record(target, incoming[item])
                    existing.add(item)

            merged_shard = aj.read_jsonl(target)

            if validate_records(
                a.benchmark, merged_shard, expected
            ) != expected:
                raise RuntimeError(
                    "Shard incomplete after consolidation"
                )

            txt = shard_dir / f"responses.{stem}.txt"

            if a.benchmark == "audio_jailbreaks":
                aj.write_text(merged_shard, txt)
            elif a.benchmark == "va_safetybench":
                va.write_text(merged_shard, txt)
            else:
                mmau.rewrite_text(txt, merged_shard)

            (shard_dir / f"{stem}.done").write_text(
                "complete\\n", encoding="utf-8"
            )

            # Release ownership only after the original shard is complete
            marker.unlink()

        print(
            f"[finish] Recovered {len(expected)} rows for {stem}",
            flush=True,
        )

        if a.benchmark == "audio_jailbreaks":
            module.merge(root, rows, ("audio",), a.num_shards)
        else:
            module.merge(root, rows, a.num_shards)


def main():
    cli = arguments()
    ctx = setup(cli)

    (
        a, module, rows, assigned, root, shard_dir,
        stem, work, checkpoint, base, kind, key
    ) = ctx

    if not a.finish_only:
        part_file = (
            work / f"part_{a.part_id:03d}.jsonl"
        )
        part_done = (
            work / f"part_{a.part_id:03d}.done"
        )

        with aj.directory_lock(
            work / f"part_{a.part_id:03d}.lock"
        ):
            with aj.directory_lock(
                shard_dir / f"{stem}.lock"
            ):
                original = aj.read_jsonl(
                    shard_dir / f"responses.{stem}.jsonl",
                    repair=True,
                )

            original_ids = validate_records(
                a.benchmark,
                original,
                {key(row) for row in assigned},
            )

            part_rows = [
                row
                for offset, row in enumerate(assigned)
                if offset % a.num_parts == a.part_id
            ]

            selected = {
                key(row) for row in part_rows
            }

            previous = aj.read_jsonl(
                part_file, repair=True
            )

            completed = validate_records(
                a.benchmark, previous, selected
            )

            original_records = {
                record_key(a.benchmark, item): item
                for item in original
            }

            for item in previous:
                item_key = record_key(a.benchmark, item)

                if (
                    item_key in original_ids
                    and original_records[item_key] != item
                ):
                    raise RuntimeError(
                        f"Conflicting completed record: {item_key}"
                    )

            pending = [
                row for row in part_rows
                if key(row) not in completed
                and key(row) not in original_ids
            ]

            print(
                f"[workshare] {a.run_name} {a.benchmark} "
                f"original={len(original_ids)} "
                f"part={a.part_id}/{a.num_parts} "
                f"completed={len(completed)} "
                f"pending={len(pending)}",
                flush=True,
            )

            adapter = None

            for index, row in enumerate(pending, 1):
                if adapter is None:
                    adapter = aj.load_model(
                        a, checkpoint, base, kind
                    )

                record = generate_record(
                    a, module, row, adapter,
                    checkpoint, base,
                )

                append_record(part_file, record)
                completed.add(key(row))

                print(
                    f"[{index}/{len(pending)}] "
                    f"{key(row)} "
                    f"{str(record['response'])[:100]!r}",
                    flush=True,
                )

            if not selected <= (completed | original_ids):
                raise RuntimeError(
                    "Workshare part incomplete"
                )

            part_done.write_text(
                "complete\n",
                encoding="utf-8",
            )

    finish(ctx)


if __name__ == "__main__":
    main()

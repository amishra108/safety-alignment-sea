#!/usr/bin/env python3
import argparse
import copy
import fcntl
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_audio_jailbreaks import (
    VIDEO_INSTRUCTION,
    MODEL_OUTPUT_NAMES,
    atomic_json,
    load_model,
    read_jsonl,
    resolve_checkpoint,
)

CONDITIONS = ("Text_only", "Audio_only", "KPT", "KPT_noise")


def parse_args():
    p = argparse.ArgumentParser(description="VA-SafetyBench checkpoint evaluation")
    p.add_argument("--model", required=True, choices=("qwen25_omni", "video4b"))
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--base-model-path", type=Path)
    p.add_argument("--epoch", type=int)
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/va_safety_bench/processed/manifest.jsonl",
    )
    p.add_argument("--audio-root", type=Path, default=ROOT)
    p.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/_evaluations/va_safetybench",
    )
    p.add_argument(
        "--conditions", nargs="+",
        choices=(*CONDITIONS, "all", "no_noise"),
        default=["no_noise"],
    )
    p.add_argument(
        "--categories", nargs="+",
        help="VA-SafetyBench harm categories, using exact manifest labels",
    )
    p.add_argument(
        "--text-instruction",
        help="Override Audio_only instruction; defaults to Video4B anti-transcription or Qwen empty text",
    )
    p.add_argument(
        "--kpt-prefix", default="",
        help="Optional text prepended to the benchmark's KPT prompt_text",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        p.error("Invalid shard configuration")
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        p.error("Invalid token limit or example limit")
    if args.preflight and (args.merge_only or args.num_shards != 1):
        p.error("Preflight requires one shard and cannot be merge-only")
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for c in args.run_name
    ):
        p.error("Invalid run name")

    if args.text_instruction is None:
        args.text_instruction = (
            VIDEO_INSTRUCTION if args.model == "video4b" else ""
        )
    return args


def select_rows(args):
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError("VA-SafetyBench manifest is empty")

    if "all" in args.conditions:
        if len(args.conditions) != 1:
            raise ValueError("Use all by itself")
        selected_conditions = set(CONDITIONS)
    elif "no_noise" in args.conditions:
        if len(args.conditions) != 1:
            raise ValueError("Use no_noise by itself")
        selected_conditions = set(CONDITIONS) - {"KPT_noise"}
    else:
        selected_conditions = set(args.conditions)

    chosen = [
        row for row in rows
        if row["condition"] in selected_conditions
    ]

    if args.categories:
        available = {row.get("category") for row in rows}
        unknown = set(args.categories) - available
        if unknown:
            raise ValueError(
                f"Unknown categories {sorted(unknown)}. "
                f"Available: {sorted(str(x) for x in available)}"
            )
        chosen = [
            row for row in chosen
            if row.get("category") in args.categories
        ]

    if args.limit is not None:
        chosen = chosen[:args.limit]
    if args.preflight:
        chosen = chosen[:1]
    if not chosen:
        raise ValueError("No VA-SafetyBench rows match the selection")

    chosen.sort(key=lambda row: (
        str(row.get("category", "")),
        row["condition"],
        str(row.get("language", "")),
        row["task_id"],
    ))

    keys = [row["task_id"] for row in chosen]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task IDs")
    return chosen


def audio_path(args, row):
    if not row.get("audio_path"):
        raise ValueError(f"Missing audio_path for {row['task_id']}")
    path = (args.audio_root / Path(row["audio_path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def input_instruction(args, row):
    condition = row["condition"]
    if condition == "Text_only":
        return row.get("prompt_text") or ""
    if condition == "Audio_only":
        return args.text_instruction
    if condition in ("KPT", "KPT_noise"):
        return args.kpt_prefix + (row.get("prompt_text") or "")
    raise ValueError(f"Unknown condition: {condition}")


def qwen_infer(adapter, args, row, instruction):
    from qwen_omni_utils import process_mm_info

    condition = row["condition"]
    content = []

    if condition != "Text_only":
        content.append({
            "type": "audio",
            "audio": "file://" + str(audio_path(args, row)),
        })
    if instruction:
        content.append({"type": "text", "text": instruction})
    if not content:
        raise ValueError(f"Empty model input for {row['task_id']}")

    messages = [{"role": "user", "content": content}]
    text = adapter.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    if condition != "Text_only":
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=False,
        )
        batch = adapter.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
            use_audio_in_video=False,
        )
    else:
        batch = adapter.processor(
            text=text, return_tensors="pt", padding=True,
        )

    batch = batch.to(adapter.device)

    kwargs = {
        "return_audio": False,
        "thinker_do_sample": False,
        "use_audio_in_video": False,
        "max_new_tokens": args.max_new_tokens,
    }

    with torch.inference_mode():
        output = adapter.omni_model.generate(**batch, **kwargs)

    if isinstance(output, tuple):
        output = output[0]

    return adapter.processor.batch_decode(
        output[:, batch["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def video_infer(adapter, args, row, instruction):
    import librosa

    backend = adapter.backend
    old_generate = adapter.model.generate
    calls = [0]

    def enforced_greedy(*model_args, **kwargs):
        calls[0] += 1
        generation_config = copy.deepcopy(
            kwargs.get("generation_config", adapter.model.generation_config)
        )
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.num_beam_groups = 1
        generation_config.penalty_alpha = None
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50
        generation_config.min_p = None

        for key in (
            "temperature", "top_p", "top_k", "min_p",
            "typical_p", "epsilon_cutoff", "eta_cutoff",
            "penalty_alpha", "num_beam_groups",
        ):
            kwargs.pop(key, None)

        kwargs["generation_config"] = generation_config
        kwargs["do_sample"] = False
        kwargs["num_beams"] = 1
        return old_generate(*model_args, **kwargs)

    adapter.model.generate = enforced_greedy
    original_features = backend.audio_features
    original_count = backend.audio_token_count

    try:
        if row["condition"] == "Text_only":
            messages = [{"role": "user", "content": instruction}]
            rendered = adapter.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            batch = adapter.tokenizer(
                rendered, add_special_tokens=False,
                return_tensors="pt",
            )
            batch = {
                key: value.to(adapter.device)
                for key, value in batch.items()
            }

            with torch.inference_mode():
                generated = adapter.model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            if isinstance(generated, tuple):
                generated = generated[0]

            response = adapter.tokenizer.decode(
                generated[0, batch["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        else:
            waveform, _ = librosa.load(
                str(audio_path(args, row)),
                sr=16000, mono=True,
            )
            if not len(waveform):
                raise ValueError(f"Empty audio for {row['task_id']}")

            chunk_size = 30 * 16000
            features = [
                backend.feature_extractor(
                    waveform[i:i + chunk_size],
                    sampling_rate=16000,
                    return_tensors="pt",
                )["input_features"].squeeze(0)
                for i in range(0, len(waveform), chunk_size)
            ]

            backend.audio_features = torch.stack(features).to(
                device=adapter.device,
                dtype=adapter.audio_encoder.conv1.weight.dtype,
            )
            backend.audio_token_count = 60 * len(features)

            with torch.inference_mode():
                batch = backend.prompt_batch(instruction)
                response = backend.generate(
                    batch, max_new_tokens=args.max_new_tokens,
                )

        if calls[0] != 1:
            raise RuntimeError(
                "Expected exactly one underlying greedy Video4B "
                f"generate call, got {calls[0]}"
            )
        return response

    finally:
        adapter.model.generate = old_generate
        backend.audio_features = original_features
        backend.audio_token_count = original_count


def write_text(records, path):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as f:
        previous = None
        for record in records:
            group = (
                record.get("category", "unspecified"),
                record["condition"],
                record.get("language", ""),
            )
            if group != previous:
                header = " | ".join(
                    str(value).upper() for value in group if value
                )
                f.write(f"\n=== {header} ===\n\n")
                previous = group

            response = str(record["response"])
            f.write(
                response.replace("\r", " ").replace("\n", " ") + "\n"
            )
    os.replace(temporary, path)

def append_text(path, record, previous_group):
    group = (
        record.get("category", "unspecified"),
        record["condition"],
        record.get("language", ""),
    )

    with path.open("a", encoding="utf-8") as f:
        if group != previous_group:
            header = " | ".join(
                str(value).upper() for value in group if value
            )
            f.write(f"\n=== {header} ===\n\n")

        response = str(record["response"])
        f.write(response.replace("\r", " ").replace("\n", " ") + "\n")

    return group


def append_record(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def merge(root, rows, shards):
    with (root / ".merge.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        shard_dir = root / "shards"
        for i in range(shards):
            if not (
                shard_dir / f"shard{i:03d}-of-{shards:03d}.done"
            ).exists():
                print("Merge waiting for other shards", flush=True)
                return

        expected_ids = [row["task_id"] for row in rows]
        expected_set = set(expected_ids)
        records = {}

        for i in range(shards):
            path = (
                shard_dir
                / f"responses.shard{i:03d}-of-{shards:03d}.jsonl"
            )
            for record in read_jsonl(path):
                key = record["task_id"]
                if key in records or key not in expected_set:
                    raise ValueError(
                        f"Duplicate or unexpected response: {key}"
                    )
                records[key] = record

        if set(records) != expected_set:
            raise ValueError("Merge is missing VA-SafetyBench responses")

        ordered = [records[key] for key in expected_ids]
        output = root / "responses.jsonl"

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=root, delete=False,
        ) as f:
            temporary = Path(f.name)
            for record in ordered:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(temporary, output)
        write_text(ordered, root / "responses.txt")
        print(
            f"MERGED {len(ordered)} VA-SafetyBench responses: {output}",
            flush=True,
        )


def main():
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.audio_root = args.audio_root.expanduser().resolve()

    checkpoint, base, kind = resolve_checkpoint(args)
    rows = select_rows(args)

    run = (
        args.output_root.expanduser().resolve()
        / MODEL_OUTPUT_NAMES[args.model] / args.run_name
    )
    if args.preflight:
        run /= "preflight"

    generation = (
        {
            "method": "Qwen2.5-Omni generate",
            "kwargs": {
                "return_audio": False,
                "thinker_do_sample": False,
                "use_audio_in_video": False,
                "max_new_tokens": args.max_new_tokens,
            },
        }
        if args.model == "qwen25_omni"
        else {
            "method": "Video4B backend generate with greedy model.generate override",
            "kwargs": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
            },
        }
    )

    config = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "base_model": str(base),
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "conditions": args.conditions,
        "categories": args.categories,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "text_instruction_audio_only": args.text_instruction,
        "kpt_prefix": args.kpt_prefix,
        "generation": generation,
        "preflight": args.preflight,
    }
    atomic_json(run / "run_config.json", config)

    if args.merge_only:
        merge(run, rows, args.num_shards)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    shard_dir = run / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    stem = f"shard{args.shard_id:03d}-of-{args.num_shards:03d}"
    output = shard_dir / f"responses.{stem}.jsonl"
    txt = shard_dir / f"responses.{stem}.txt"

    with (shard_dir / f"{stem}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        start = len(rows) * args.shard_id // args.num_shards
        end = len(rows) * (args.shard_id + 1) // args.num_shards
        assigned = rows[start:end]
        allowed = {row["task_id"] for row in assigned}

        previous = read_jsonl(output, repair=True)
        completed = set()

        for record in previous:
            key = record["task_id"]
            if key in completed or key not in allowed:
                raise ValueError(
                    f"Unexpected or duplicate completed task: {key}"
                )
            completed.add(key)

        write_text(previous, txt)
        previous_group = (
            (
                previous[-1].get("category", "unspecified"),
                previous[-1]["condition"],
                previous[-1].get("language", ""),
            )
            if previous else None
        )

        pending = [
            row for row in assigned
            if row["task_id"] not in completed
        ]

        print(
            f"VA-SafetyBench model={args.model} checkpoint={checkpoint} "
            f"shard={stem} pending={len(pending)}",
            flush=True,
        )

        if pending:
            adapter = load_model(args, checkpoint, base, kind)

            runtime = {
                "adapter_class": (
                    type(adapter).__module__
                    + "."
                    + type(adapter).__name__
                ),
                "model_dtype": str(
                    next(adapter.model.parameters()).dtype
                ),
                "generation_config": (
                    adapter.omni_model.generation_config.to_dict()
                    if args.model == "qwen25_omni"
                    else adapter.model.generation_config.to_dict()
                ),
                "generation_call": generation,
            }
            atomic_json(run / f"runtime.{stem}.json", runtime)

            for i, row in enumerate(pending, 1):
                instruction = input_instruction(args, row)

                response = (
                    qwen_infer(adapter, args, row, instruction)
                    if args.model == "qwen25_omni"
                    else video_infer(adapter, args, row, instruction)
                )

                record = {
                    **row,
                    "model": args.model,
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
                    "generation": generation,
                    "response": response,
                }

                append_record(output, record)
                previous_group = append_text(txt, record, previous_group)
                completed.add(row["task_id"])

                print(
                    f'[{i}/{len(pending)}] {row["task_id"]} '
                    f'{response[:120]!r}',
                    flush=True,
                )

        if completed != allowed:
            raise ValueError("Shard incomplete")

        (shard_dir / f"{stem}.done").write_text(
            "complete\n", encoding="utf-8",
        )

    merge(run, rows, args.num_shards)


if __name__ == "__main__":
    main()

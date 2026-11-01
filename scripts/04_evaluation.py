#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
import socket
from pathlib import Path

import torch


SEA_ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
BENCHMARK_ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

MANIFEST = (
    BENCHMARK_ROOT
    / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"
)

OUTPUT_ROOT = SEA_ROOT / "outputs/04_evaluation"

DEFAULT_QWEN_BASE_MODEL = (
    SEA_ROOT
    / "models/qwen2.5-omni-7b"
)

EXPECTED_ROWS = 3200
MAX_NEW_TOKENS = 1024
MODEL_NAMES = ("qwen25_omni",)
MODALITIES = ("audio", "text")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--base-model-path",
        type=Path,
        default=DEFAULT_QWEN_BASE_MODEL,
        help=(
            "Original full Qwen2.5-Omni checkpoint used to reconstruct "
            "legacy Thinker-only trained checkpoints."
        ),
    )
    p.add_argument(
        "--run-name",
        required=True,
    )
    p.add_argument(
        "--modality",
        choices=MODALITIES,
        default="audio",
    )
    p.add_argument(
        "--machine",
        choices=("nete", "air211", "air213"),
        required=True,
    )
    p.add_argument(
        "--gpu",
        type=int,
        required=True,
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=0,
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = p.parse_args()

    if not args.model_path.is_dir():
        p.error(f"model path does not exist: {args.model_path}")

    if not args.base_model_path.is_dir():
        p.error(
            f"base model path does not exist: {args.base_model_path}"
        )

    if args.gpu < 0:
        p.error("--gpu must be nonnegative")

    if args.num_shards < 1:
        p.error("--num-shards must be >= 1")

    if not 0 <= args.shard_id < args.num_shards:
        p.error("--shard-id must satisfy 0 <= shard-id < num-shards")

    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")

    return args


def read_manifest():
    if not MANIFEST.is_file():
        raise RuntimeError(f"Missing benchmark manifest: {MANIFEST}")

    with MANIFEST.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} benchmark rows, found {len(rows)}"
        )

    required = {
        "example_id",
        "english_example_id",
        "source_id",
        "split",
        "condition",
        "language",
        "source_text",
        "english_prompt_text",
        "prompt_text",
        "translated_text",
        "translation_model",
        "translation_seed",
        "audio_path",
        "tts_model",
        "tts_language",
        "tts_style_prompt",
        "sample_rate",
        "duration_seconds",
    }

    seen = set()

    for index, row in enumerate(rows, start=1):
        missing = required - set(row)

        if missing:
            raise RuntimeError(
                f"{MANIFEST}:{index} missing fields: {sorted(missing)}"
            )

        example_id = row["example_id"]

        if example_id in seen:
            raise RuntimeError(f"Duplicate example_id: {example_id}")

        seen.add(example_id)

        audio_path = BENCHMARK_ROOT / row["audio_path"]

        if (
            not audio_path.is_file()
            or audio_path.stat().st_size <= 44
        ):
            raise RuntimeError(
                f"Missing/invalid audio for {example_id}: {audio_path}"
            )

    return rows


def shard_bounds(total, num_shards, shard_id):
    start = total * shard_id // num_shards
    end = total * (shard_id + 1) // num_shards
    return start, end


def run_root(args):
    return OUTPUT_ROOT / args.run_name


def shard_paths(args):
    root = run_root(args)
    shard_root = root / "shards"

    output = (
        shard_root
        / f"responses.shard{args.shard_id:03d}-of-{args.num_shards:03d}.jsonl"
    )

    done = (
        shard_root
        / f"shard{args.shard_id:03d}-of-{args.num_shards:03d}.done"
    )

    return shard_root, output, done


def qwen_checkpoint_architecture(model_path):
    config_path = Path(model_path) / "config.json"

    if not config_path.is_file():
        raise RuntimeError(
            f"Missing checkpoint config: {config_path}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    architectures = config.get("architectures")

    if architectures == [
        "Qwen2_5OmniForConditionalGeneration"
    ]:
        return "full_omni"

    if architectures == [
        "Qwen2_5OmniThinkerForConditionalGeneration"
    ]:
        return "thinker_only"

    raise RuntimeError(
        "Unsupported Qwen checkpoint architecture: "
        f"{architectures!r} in {config_path}"
    )


def run_config(args):
    architecture = qwen_checkpoint_architecture(
        args.model_path
    )

    return {
        "model": args.model,
        "model_path": str(args.model_path.resolve()),
        "checkpoint_architecture": architecture,
        "reconstruction_base_model":
            (
                str(args.base_model_path.resolve())
                if architecture == "thinker_only"
                else None
            ),
        "manifest": str(MANIFEST.resolve()),
        "modality": args.modality,
        "num_shards": args.num_shards,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_decoding": "greedy",
        "target_system_prompt": None,
        "target_extra_instruction": None,
    }


def ensure_run_config(args):
    root = run_root(args)
    root.mkdir(parents=True, exist_ok=True)

    path = root / "run_config.json"
    expected = run_config(args)

    try:
        with path.open(
            "x",
            encoding="utf-8",
        ) as f:
            json.dump(
                expected,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
    except FileExistsError:
        existing = json.loads(
            path.read_text(encoding="utf-8")
        )

        if existing != expected:
            raise RuntimeError(
                "Existing evaluation run has incompatible configuration.\n"
                f"path={path}\n"
                f"existing={existing}\n"
                f"requested={expected}"
            )


def read_jsonl(path):
    if not path.is_file():
        return []

    rows = []

    with path.open(
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL: {path}:{line_number}"
                ) from exc

    return rows


def append_jsonl(path, row):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def validate_existing(
    path,
    assigned,
    args,
):
    expected = {
        row["example_id"]
        for row in assigned
    }

    completed = set()

    for line_number, record in enumerate(
        read_jsonl(path),
        start=1,
    ):
        required = {
            "task_id",
            "benchmark",
            "model",
            "host",
            "example_id",
            "modality",
            "response",
        }

        missing = required - set(record)

        if missing:
            raise RuntimeError(
                f"{path}:{line_number} missing fields: {sorted(missing)}"
            )

        example_id = record["example_id"]

        if example_id not in expected:
            raise RuntimeError(
                f"{path}:{line_number} does not belong to this shard: "
                f"{example_id}"
            )

        if record["model"] != args.model:
            raise RuntimeError(
                f"{path}:{line_number} model mismatch"
            )

        if record["modality"] != args.modality:
            raise RuntimeError(
                f"{path}:{line_number} modality mismatch"
            )

        expected_task_id = (
            example_id
            + "__"
            + args.modality
        )

        if record["task_id"] != expected_task_id:
            raise RuntimeError(
                f"{path}:{line_number} task_id mismatch"
            )

        if example_id in completed:
            raise RuntimeError(
                f"{path}: duplicate completed example {example_id}"
            )

        completed.add(example_id)

    return completed


class Qwen25OmniEvaluator:
    def __init__(
        self,
        model_path,
        base_model_path,
    ):
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.model_path = Path(model_path)
        self.base_model_path = Path(base_model_path)

        architecture = qwen_checkpoint_architecture(
            self.model_path
        )

        print(
            f"[checkpoint] architecture={architecture}",
            flush=True,
        )

        self.processor = (
            Qwen2_5OmniProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        if architecture == "full_omni":
            print(
                "[load] loading complete Qwen2.5-Omni checkpoint",
                flush=True,
            )

            self.model = (
                Qwen2_5OmniForConditionalGeneration
                .from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    local_files_only=True,
                )
            )

        elif architecture == "thinker_only":
            print(
                "[load] legacy Thinker-only checkpoint detected",
                flush=True,
            )
            print(
                f"[load] full Omni container: "
                f"{self.base_model_path}",
                flush=True,
            )
            print(
                f"[load] trained Thinker: "
                f"{self.model_path}",
                flush=True,
            )

            self.model = (
                Qwen2_5OmniForConditionalGeneration
                .from_pretrained(
                    self.base_model_path,
                    torch_dtype="auto",
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
            )

            if hasattr(
                self.model,
                "disable_talker",
            ):
                self.model.disable_talker()

            base_thinker = self.model.thinker
            self.model.thinker = None

            del base_thinker
            gc.collect()

            trained_thinker = (
                Qwen2_5OmniThinkerForConditionalGeneration
                .from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
            )

            self.model.thinker = trained_thinker

            del trained_thinker
            gc.collect()

            self.model.to(
                torch.device("cuda:0")
            )

            torch.cuda.empty_cache()

            print(
                "[load] reconstructed full Omni wrapper with "
                "trained Thinker",
                flush=True,
            )

        else:
            raise AssertionError(
                architecture
            )

        if hasattr(
            self.model,
            "disable_talker",
        ):
            self.model.disable_talker()

        self.model.eval()

        device_map = getattr(
            self.model,
            "hf_device_map",
            None,
        )

        if (
            architecture == "full_omni"
            and device_map is not None
        ):
            bad = {
                key: value
                for key, value in device_map.items()
                if str(value) in {
                    "cpu",
                    "disk",
                    "meta",
                }
            }

            if bad:
                raise RuntimeError(
                    f"CPU/disk offload detected: {bad}"
                )

        thinker_device = next(
            self.model.thinker.parameters()
        ).device

        if thinker_device.type != "cuda":
            raise RuntimeError(
                f"Thinker is not on CUDA: {thinker_device}"
            )

        print(
            f"[load] thinker_device={thinker_device} "
            f"gpu_allocated="
            f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB",
            flush=True,
        )

    def decode(
        self,
        output,
        input_length,
    ):
        if isinstance(output, tuple):
            output = output[0]

        if output.ndim != 2:
            raise RuntimeError(
                f"Unexpected generation shape: {tuple(output.shape)}"
            )

        return self.processor.batch_decode(
            output[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def infer_audio(
        self,
        row,
    ):
        from qwen_omni_utils import process_mm_info

        audio_path = (
            BENCHMARK_ROOT
            / row["audio_path"]
        ).resolve()

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio":
                            "file://" + str(audio_path),
                    }
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False,
        )

        inputs = self.processor(
            text=chat_text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        ).to(self.model.device)

        input_length = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                use_audio_in_video=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        return self.decode(
            output,
            input_length,
        )

    def infer_text(
        self,
        row,
    ):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": row["prompt_text"],
                    }
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=chat_text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        input_length = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        return self.decode(
            output,
            input_length,
        )


def make_record(
    args,
    row,
    response,
):
    modality = args.modality

    return {
        "task_id":
            row["example_id"]
            + "__"
            + modality,

        "benchmark":
            "Audio Jailbreak Clean Benchmark",

        "model":
            args.model,

        "host":
            socket.gethostname().split(".")[0],

        "example_id":
            row["example_id"],

        "english_example_id":
            row["english_example_id"],

        "source_id":
            row["source_id"],

        "split":
            row["split"],

        "condition":
            row["condition"],

        "attack_type":
            (
                "no_attack"
                if row["condition"] == "original"
                else row["condition"]
            ),

        "modality":
            modality,

        "model_input_mode":
            (
                "audio_only"
                if modality == "audio"
                else "text_only"
            ),

        "language":
            row["language"],

        "source_text":
            row["source_text"],

        "english_prompt_text":
            row["english_prompt_text"],

        "prompt_text":
            row["prompt_text"],

        "translated_text":
            row["translated_text"] or None,

        "translation_model":
            row["translation_model"] or None,

        "translation_seed":
            row["translation_seed"] or None,

        "audio_path":
            (
                row["audio_path"]
                if modality == "audio"
                else None
            ),

        "tts_model":
            (
                row["tts_model"]
                if modality == "audio"
                else None
            ),

        "tts_language":
            (
                row["tts_language"]
                if modality == "audio"
                else None
            ),

        "tts_style_prompt":
            (
                row["tts_style_prompt"]
                if modality == "audio"
                else None
            ),

        "audio_sample_rate":
            (
                int(row["sample_rate"])
                if modality == "audio"
                else None
            ),

        "audio_duration_seconds":
            (
                float(row["duration_seconds"])
                if modality == "audio"
                else None
            ),

        "generation_max_new_tokens":
            MAX_NEW_TOKENS,

        "generation_decoding":
            "greedy",

        "target_system_prompt":
            None,

        "target_extra_instruction":
            None,

        "response":
            response,
    }


def all_shard_paths(args):
    shard_root = run_root(args) / "shards"

    outputs = [
        shard_root
        / f"responses.shard{i:03d}-of-{args.num_shards:03d}.jsonl"
        for i in range(args.num_shards)
    ]

    done = [
        shard_root
        / f"shard{i:03d}-of-{args.num_shards:03d}.done"
        for i in range(args.num_shards)
    ]

    return outputs, done


def merge_outputs(
    args,
    examples,
):
    shard_outputs, done_files = (
        all_shard_paths(args)
    )

    waiting = [
        index
        for index, path in enumerate(done_files)
        if not path.is_file()
    ]

    if waiting:
        print(
            f"[merge] waiting for shards: {waiting}",
            flush=True,
        )
        return False

    expected_order = {
        row["example_id"]: index
        for index, row in enumerate(examples)
    }

    merged = {}

    for path in shard_outputs:
        if not path.is_file():
            raise RuntimeError(
                f"Done marker exists but shard output is missing: {path}"
            )

        for record in read_jsonl(path):
            example_id = record["example_id"]

            if example_id not in expected_order:
                raise RuntimeError(
                    f"Unexpected merged example: {example_id}"
                )

            if example_id in merged:
                raise RuntimeError(
                    f"Duplicate merged example: {example_id}"
                )

            merged[example_id] = record

    missing = (
        set(expected_order)
        - set(merged)
    )

    if missing:
        raise RuntimeError(
            f"Merged output missing {len(missing)} examples: "
            f"{sorted(missing)[:10]}"
        )

    ordered = sorted(
        merged.values(),
        key=lambda record:
            expected_order[record["example_id"]],
    )

    output_path = (
        run_root(args)
        / f"responses.all.{args.modality}.jsonl"
    )

    temporary = output_path.with_name(
        output_path.name
        + f".tmp.{os.getpid()}"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in ordered:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        temporary,
        output_path,
    )

    print(
        f"[merged] {output_path}: {len(ordered)} rows",
        flush=True,
    )

    return True


def main():
    args = parse_args()

    hostname = (
        socket.gethostname()
        .split(".")[0]
    )

    if hostname != args.machine:
        raise RuntimeError(
            f"Running on {hostname}, but --machine={args.machine}"
        )

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(args.gpu)

    examples = read_manifest()

    if args.limit is not None:
        examples = examples[:args.limit]

    ensure_run_config(args)

    if args.merge_only:
        merge_outputs(
            args,
            examples,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.cuda.set_device(0)

    start, end = shard_bounds(
        len(examples),
        args.num_shards,
        args.shard_id,
    )

    assigned = examples[start:end]

    shard_root, output_path, done_path = (
        shard_paths(args)
    )

    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = validate_existing(
        output_path,
        assigned,
        args,
    )

    pending = [
        row
        for row in assigned
        if row["example_id"] not in completed
    ]

    print("=" * 88)
    print("STAGE 4 EVALUATION")
    print("=" * 88)
    print(f"host        : {hostname}")
    print(f"physical GPU: {args.gpu}")
    print(f"model       : {args.model}")
    print(f"model path  : {args.model_path.resolve()}")
    print(f"run name    : {args.run_name}")
    print(f"manifest    : {MANIFEST}")
    print(f"modality    : {args.modality}")
    print(f"shard       : {args.shard_id}/{args.num_shards}")
    print(f"range       : {start}:{end}")
    print(f"assigned    : {len(assigned)}")
    print(f"completed   : {len(completed)}")
    print(f"pending     : {len(pending)}")
    print(f"max tokens  : {MAX_NEW_TOKENS}")
    print("decoding    : greedy")
    print("extra prompt: NONE")
    print(f"output      : {output_path}")
    print("=" * 88, flush=True)

    if not pending:
        done_path.write_text(
            "complete\n",
            encoding="utf-8",
        )

        print(
            "[COMPLETE] shard already complete",
            flush=True,
        )

        merge_outputs(
            args,
            examples,
        )
        return

    if args.model == "qwen25_omni":
        evaluator = Qwen25OmniEvaluator(
            args.model_path,
            args.base_model_path,
        )
    else:
        raise AssertionError(args.model)

    infer = (
        evaluator.infer_audio
        if args.modality == "audio"
        else evaluator.infer_text
    )

    for position, row in enumerate(
        pending,
        start=1,
    ):
        print(
            f"[{position:04d}/{len(pending):04d}] "
            f"shard={args.shard_id:02d} "
            f"{row['example_id']} "
            f"lang={row['language']} "
            f"split={row['split']} "
            f"condition={row['condition']}",
            flush=True,
        )

        response = infer(row)

        record = make_record(
            args,
            row,
            response,
        )

        append_jsonl(
            output_path,
            record,
        )

        completed.add(
            row["example_id"]
        )

    expected = {
        row["example_id"]
        for row in assigned
    }

    if completed != expected:
        missing = expected - completed

        raise RuntimeError(
            f"Shard incomplete: {len(missing)} missing: "
            f"{sorted(missing)[:10]}"
        )

    done_path.write_text(
        "complete\n",
        encoding="utf-8",
    )

    print(
        f"[OK] shard {args.shard_id}/{args.num_shards} complete",
        flush=True,
    )

    merge_outputs(
        args,
        examples,
    )


if __name__ == "__main__":
    main()

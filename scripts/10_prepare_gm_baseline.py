#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "saferlhf" / "sea_3000_prepared.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "rlhf_tts_audio"
DEFAULT_MODEL = ROOT / "models" / "qwen3-tts-12hz-1.7b-voicedesign"

VOICE_INSTRUCTION = (
    "A clear adult English-speaking voice with a neutral mood, "
    "natural delivery, and normal speaking rate."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare physical TTS audio for the SEA GM baseline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
    return rows


def gm_audio_text(row):
    category = row.get("category")

    if category == "harmful":
        phrase_type = row.get("phrase_type")

        if phrase_type == "activity":
            text = row.get("extended_sentence")
            if not text:
                raise ValueError(
                    "Harmful activity row is missing extended_sentence"
                )
            return text

        if phrase_type == "product":
            text = row.get("harmful_phrase")
            if not text:
                raise ValueError(
                    "Harmful product row is missing harmful_phrase"
                )
            return text

        raise ValueError(
            f"Unexpected harmful phrase_type: {phrase_type!r}"
        )

    if category == "harmless":
        text = row.get("prompt")
        if not text:
            raise ValueError("Harmless row is missing prompt")
        return text

    raise ValueError(f"Unexpected category: {category!r}")


def gm_training_prompt(row):
    if row["category"] == "harmful":
        prompt = row.get("rewritten_prompt")
        if not prompt:
            raise ValueError(
                "Harmful row is missing rewritten_prompt"
            )
        return prompt

    prompt = row.get("prompt")
    if not prompt:
        raise ValueError("Harmless row is missing prompt")
    return prompt


def training_targets(row):
    if row["category"] == "harmful":
        chosen = row.get("new_chosen_response")
        rejected = row.get("rejected_response")
    else:
        chosen = row.get("chosen_response")
        rejected = row.get("rejected_response")

    if not chosen:
        raise ValueError("Row is missing GM chosen response")
    if not rejected:
        raise ValueError("Row is missing GM rejected response")

    return chosen, rejected


def relative_to_root(path):
    return str(path.relative_to(ROOT))


def load_completed(manifest_path):
    completed = set()

    if not manifest_path.exists():
        return completed

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            audio_path = ROOT / record["gm_audio_path"]

            if not audio_path.is_file():
                continue

            try:
                info = sf.info(audio_path)
                if info.frames > 0 and info.samplerate > 0:
                    completed.add(int(record["row_index"]))
            except (OSError, RuntimeError, ValueError):
                continue

    return completed


def append_manifest(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(
            "--shard-id must satisfy 0 <= shard-id < num-shards"
        )

    rows = read_jsonl(args.input)

    harmful_count = sum(row.get("category") == "harmful" for row in rows)
    harmless_count = sum(row.get("category") == "harmless" for row in rows)

    if len(rows) != 3000:
        raise ValueError(
            f"Expected 3000 rows, found {len(rows)} in {args.input}"
        )

    if harmful_count != 2000 or harmless_count != 1000:
        raise ValueError(
            "Expected 2000 harmful and 1000 harmless rows, found "
            f"{harmful_count} harmful and {harmless_count} harmless"
        )

    assigned = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_id
    ]

    if args.preflight:
        assigned = assigned[:1]
        run_root = args.output_dir / "preflight"
        manifest_path = run_root / "manifest.jsonl"
    else:
        run_root = args.output_dir
        manifest_path = (
            run_root
            / "manifests"
            / (
                f"manifest.shard{args.shard_id:03d}"
                f"-of-{args.num_shards:03d}.jsonl"
            )
        )

    audio_dir = run_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed(manifest_path)

    pending = [
        (index, row)
        for index, row in assigned
        if index not in completed
    ]

    print(f"Input:       {args.input}")
    print(f"Rows:        {len(rows)}")
    print(f"Harmful:     {harmful_count}")
    print(f"Harmless:    {harmless_count}")
    print(f"Shard:       {args.shard_id}/{args.num_shards}")
    print(f"Assigned:    {len(assigned)}")
    print(f"Completed:   {len(completed)}")
    print(f"Pending:     {len(pending)}")
    print(f"Preflight:   {args.preflight}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Model:       {args.model}")
    print(f"Output:      {run_root}")

    if not pending:
        print("Nothing to generate.")
        return

    print("Loading Qwen3-TTS VoiceDesign...")

    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]

        texts = [gm_audio_text(row) for _, row in batch]
        languages = ["English"] * len(batch)
        instructions = [VOICE_INSTRUCTION] * len(batch)

        print(
            f"[batch {batch_start // args.batch_size + 1}/"
            f"{(len(pending) + args.batch_size - 1) // args.batch_size}] "
            f"size={len(batch)} first_row={batch[0][0]}",
            flush=True,
        )

        with torch.inference_mode():
            wavs, sample_rate = model.generate_voice_design(
                text=texts,
                language=languages,
                instruct=instructions,
            )

        if len(wavs) != len(batch):
            raise RuntimeError(
                f"TTS returned {len(wavs)} WAVs for {len(batch)} inputs"
            )

        for (row_index, row), spoken_text, wav in zip(
            batch, texts, wavs
        ):
            training_prompt = gm_training_prompt(row)
            chosen_response, rejected_response = training_targets(row)

            audio_path = audio_dir / f"row_{row_index:04d}.wav"
            temporary_path = audio_dir / f"row_{row_index:04d}.wav.tmp"

            sf.write(
                temporary_path,
                wav,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )

            info = sf.info(temporary_path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise RuntimeError(f"Invalid WAV: {temporary_path}")

            temporary_path.replace(audio_path)

            record = {
                "row_index": row_index,
                "source_index": row.get("source_index"),
                "category": row["category"],
                "phrase_type": row.get("phrase_type"),
                "gm_audio_text": spoken_text,
                "gm_training_prompt": training_prompt,
                "gm_audio_path": relative_to_root(audio_path),
                "sample_rate": sample_rate,
                "duration_seconds": info.duration,
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
                "tts_model": relative_to_root(args.model),
                "tts_language": "English",
                "tts_voice_instruction": VOICE_INSTRUCTION,
                "tts_batch_size": len(batch),
            }

            append_manifest(manifest_path, record)

            print(
                f"[saved] row={row_index:04d} "
                f"duration={info.duration:.2f}s "
                f"path={audio_path}",
                flush=True,
            )

    print()
    print("GM audio preparation complete.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

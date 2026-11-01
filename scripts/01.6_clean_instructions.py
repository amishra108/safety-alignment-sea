#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import socket
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
SOURCE = ROOT / "data/saferlhf/sea_3000_prepared.jsonl"
MODEL = ROOT / "models/qwen3-4b"
WORK = ROOT / "outputs/01.6_clean_instructions"
FINAL = WORK / "sea_3000_cleaned_instructions.jsonl"

VERSION = "sea-instruction-cleaning-v1"

SYSTEM_PROMPT = """You are preparing multimodal safety-alignment training data.

You will receive:
1. An original harmful user instruction.
2. A previously extracted harmful phrase.
3. A phrase type: activity or product.

Rewrite the ORIGINAL instruction into a detoxified instruction for synthetic-embedding safety training.

The crucial requirement is to preserve the original instruction as closely as possible.

Rules:

- Preserve the original wording, sentence structure, surrounding context, motivations, qualifiers, questions, and non-harmful details wherever possible.
- Do NOT summarize the instruction.
- Do NOT replace the whole instruction with a generic question.
- Do NOT shorten it merely because the harmful phrase is long.
- Do NOT add warnings, refusals, explanations, moral commentary, notes, or safety advice.
- Do NOT answer the instruction.
- Do NOT invent new details.
- Do NOT change the requested type of assistance.
- Remove or abstract the harmful action or harmful product itself.
- Use the supplied harmful phrase as a semantic guide, but do not blindly delete every surrounding detail contained inside that phrase if those details can naturally remain in the sentence.
- Make the smallest rewrite needed to replace the harmful action or product while keeping the rest of the original instruction intact.

If phrase_type is "activity":
replace the harmful action with the exact phrase "this harmful activity".
You may use grammatically necessary constructions such as:
"perform this harmful activity"
"carry out this harmful activity"
"engage in this harmful activity"
but the words "this harmful activity" must appear exactly.

If phrase_type is "product":
replace the harmful product with the exact phrase "this harmful product".
You may alter nearby articles, plurality, or verbs only when necessary for grammatical English, but the words "this harmful product" must appear exactly.

Examples:

ORIGINAL:
List steps in detail to hire a hacker to breach a competitor's computer systems.

HARMFUL PHRASE:
hire a hacker

PHRASE TYPE:
activity

CLEANED:
List steps in detail to perform this harmful activity to breach a competitor's computer systems.

ORIGINAL:
My goal is to create a computer virus that targets financial institutions, in order to disrupt the global economy. How can I do it?

HARMFUL PHRASE:
create a computer virus

PHRASE TYPE:
activity

CLEANED:
My goal is to perform this harmful activity that targets financial institutions, in order to disrupt the global economy. How can I do it?

ORIGINAL:
My friend has been acting suspiciously lately, and I'm worried that they might be cheating on me. How can I hack into their email account to find out the truth?

HARMFUL PHRASE:
hack into their email account

PHRASE TYPE:
activity

CLEANED:
My friend has been acting suspiciously lately, and I'm worried that they might be cheating on me. How can I perform this harmful activity to find out the truth?

Return exactly one section in this format:

[Cleaned Instruction]
<rewritten instruction>

Return nothing else.
"""


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--machine",
        choices=("nete", "air211", "air213"),
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard-id", type=int)
    p.add_argument("--num-shards", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--preflight", type=int, default=0)
    p.add_argument("--merge", action="store_true")

    args = p.parse_args()

    if args.num_shards < 1:
        p.error("--num-shards must be positive")

    if args.merge:
        return args

    if args.machine is None:
        p.error("--machine is required")

    if args.shard_id is None:
        p.error("--shard-id is required")

    if not 0 <= args.shard_id < args.num_shards:
        p.error("--shard-id must satisfy 0 <= shard-id < num-shards")

    if args.gpu < 0:
        p.error("--gpu must be nonnegative")

    if args.preflight < 0:
        p.error("--preflight must be nonnegative")

    return args


def load_rows():
    with SOURCE.open(encoding="utf-8") as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(rows) != 3000:
        raise RuntimeError(
            f"Expected 3000 rows, found {len(rows)}"
        )

    for index, row in enumerate(rows):
        expected = (
            "harmful"
            if index < 2000
            else "harmless"
        )

        if row.get("category") != expected:
            raise RuntimeError(
                f"Row {index}: expected category={expected!r}, "
                f"found {row.get('category')!r}"
            )

        if index < 2000:
            for key in (
                "prompt",
                "harmful_phrase",
                "phrase_type",
                "rewritten_prompt",
            ):
                value = row.get(key)

                if (
                    not isinstance(value, str)
                    or not value.strip()
                ):
                    raise RuntimeError(
                        f"Row {index}: missing {key!r}"
                    )

            if row["phrase_type"] not in {
                "activity",
                "product",
            }:
                raise RuntimeError(
                    f"Row {index}: invalid phrase_type="
                    f"{row['phrase_type']!r}"
                )

    return rows


def row_hash(index, row):
    payload = {
        "version": VERSION,
        "index": index,
        "prompt": row["prompt"],
        "harmful_phrase": row["harmful_phrase"],
        "phrase_type": row["phrase_type"],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                value,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temporary,
            path,
        )

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate(model, tokenizer, user, max_new_tokens):
    import torch

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[
        0,
        inputs["input_ids"].shape[-1]:,
    ]

    if generated.numel() >= max_new_tokens:
        raise RuntimeError(
            f"Generation reached max_new_tokens={max_new_tokens}"
        )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    text = re.sub(
        r"^<think>.*?</think>\s*",
        "",
        text,
        flags=re.S,
    ).strip()

    if not text:
        raise RuntimeError(
            "Model produced an empty response"
        )

    return text


def parse_cleaned(text):
    match = re.fullmatch(
        r"\s*\[Cleaned Instruction\]\s*\n(.*?)\s*",
        text,
        flags=re.S,
    )

    if match is None:
        raise ValueError(
            "Generation did not contain exactly "
            "[Cleaned Instruction]"
        )

    cleaned = match.group(1).strip()

    if not cleaned:
        raise ValueError(
            "Cleaned instruction is empty"
        )

    return cleaned


def validate_cleaned(row, cleaned):
    required = (
        "this harmful activity"
        if row["phrase_type"] == "activity"
        else "this harmful product"
    )

    if required not in cleaned.lower():
        raise ValueError(
            f"Cleaned instruction does not contain "
            f"required phrase {required!r}"
        )

    forbidden_prefixes = (
        "i cannot",
        "i can't",
        "i am unable",
        "i'm unable",
        "sorry",
        "note:",
        "warning:",
    )

    if cleaned.lower().startswith(
        forbidden_prefixes
    ):
        raise ValueError(
            "Cleaner produced refusal/commentary "
            "instead of a rewritten instruction"
        )


def process_row(
    index,
    row,
    output_dir,
    model,
    tokenizer,
    max_new_tokens,
):
    fingerprint = row_hash(
        index,
        row,
    )

    path = (
        output_dir
        / f"row_{index:04d}.json"
    )

    error_path = (
        output_dir
        / f"row_{index:04d}_error.json"
    )

    if path.exists():
        record = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            record.get("version") != VERSION
            or record.get("index") != index
            or record.get("source_hash")
            != fingerprint
        ):
            raise RuntimeError(
                f"Incompatible existing output: {path}"
            )

        print(
            f"row={index} already complete",
            flush=True,
        )
        return

    base_user = (
        "ORIGINAL INSTRUCTION:\n"
        f"{row['prompt'].strip()}\n\n"
        "HARMFUL PHRASE:\n"
        f"{row['harmful_phrase'].strip()}\n\n"
        "PHRASE TYPE:\n"
        f"{row['phrase_type'].strip()}"
    )

    required = (
        "this harmful activity"
        if row["phrase_type"] == "activity"
        else "this harmful product"
    )

    attempts = [
        base_user,
        (
            base_user
            + "\n\nIMPORTANT CORRECTION:\n"
            + f'The cleaned instruction MUST contain the exact words '
            + f'"{required}". Preserve the rest of the original instruction '
            + "as closely as possible. Return only the required "
            + "[Cleaned Instruction] section."
        ),
        (
            base_user
            + "\n\nFORMAT REQUIREMENT:\n"
            + f'You previously failed to use the required phrase. '
            + f'Use "{required}" verbatim in the rewritten instruction. '
            + "Make the smallest possible edit to the original instruction. "
            + "Do not summarize, explain, refuse, or add commentary. "
            + "Return exactly:\n"
            + "[Cleaned Instruction]\n"
            + "<rewritten instruction>"
        ),
    ]

    failures = []

    for attempt_number, user in enumerate(
        attempts,
        start=1,
    ):
        try:
            raw = generate(
                model,
                tokenizer,
                user,
                max_new_tokens,
            )

            cleaned = parse_cleaned(
                raw
            )

            validate_cleaned(
                row,
                cleaned,
            )

            similarity = SequenceMatcher(
                None,
                row["prompt"],
                cleaned,
            ).ratio()

            record = {
                "version": VERSION,
                "index": index,
                "source_hash": fingerprint,
                "category": row["category"],
                "original_prompt": row["prompt"],
                "harmful_phrase": row["harmful_phrase"],
                "phrase_type": row["phrase_type"],
                "previous_rewritten_prompt":
                    row["rewritten_prompt"],
                "cleaned_rewritten_prompt":
                    cleaned,
                "character_similarity":
                    similarity,
                "generation_attempt":
                    attempt_number,
            }

            atomic_write_json(
                path,
                record,
            )

            if error_path.exists():
                error_path.unlink()

            print(
                f"row={index} "
                f"type={row['phrase_type']} "
                f"similarity={similarity:.3f} "
                f"attempt={attempt_number}",
                flush=True,
            )

            print(
                f"  OLD: {row['rewritten_prompt']}",
                flush=True,
            )

            print(
                f"  NEW: {cleaned}",
                flush=True,
            )

            return

        except Exception as exc:
            failures.append(
                {
                    "attempt":
                        attempt_number,
                    "error_type":
                        type(exc).__name__,
                    "message":
                        str(exc),
                }
            )

            print(
                f"row={index} "
                f"attempt={attempt_number} FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    atomic_write_json(
        error_path,
        {
            "version": VERSION,
            "index": index,
            "source_hash": fingerprint,
            "category": row["category"],
            "original_prompt": row["prompt"],
            "harmful_phrase": row["harmful_phrase"],
            "phrase_type": row["phrase_type"],
            "failures": failures,
        },
    )

    print(
        f"row={index} FAILED after "
        f"{len(attempts)} attempts; "
        f"saved {error_path} and continuing.",
        flush=True,
    )

def merge(num_shards):
    rows = load_rows()

    records = []

    for index in range(2000):
        shard = index % num_shards

        path = (
            WORK
            / f"num_shards_{num_shards}"
            / f"shard_{shard:02d}"
            / f"row_{index:04d}.json"
        )

        if not path.is_file():
            raise RuntimeError(
                f"Missing cleaned row {index}: {path}"
            )

        record = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        expected_hash = row_hash(
            index,
            rows[index],
        )

        if (
            record.get("version") != VERSION
            or record.get("index") != index
            or record.get("source_hash")
            != expected_hash
        ):
            raise RuntimeError(
                f"Invalid cleaned record: {path}"
            )

        records.append(record)

    FINAL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = FINAL.with_suffix(
        ".jsonl.tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        temporary,
        FINAL,
    )

    similarities = [
        record["character_similarity"]
        for record in records
    ]

    print(
        f"Merged {len(records)} harmful rows.",
        flush=True,
    )
    print(
        f"Output: {FINAL}",
        flush=True,
    )
    print(
        "Character similarity: "
        f"min={min(similarities):.3f} "
        f"mean={sum(similarities) / len(similarities):.3f} "
        f"max={max(similarities):.3f}",
        flush=True,
    )


def main():
    args = parse_args()

    if args.merge:
        merge(
            args.num_shards
        )
        return

    hostname = (
        socket.gethostname()
        .split(".")[0]
    )

    if hostname != args.machine:
        raise RuntimeError(
            f"Running on {hostname}, "
            f"but --machine={args.machine}"
        )

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(args.gpu)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    torch.cuda.set_device(0)

    rows = load_rows()

    indices = [
        index
        for index in range(2000)
        if index % args.num_shards
        == args.shard_id
    ]

    if args.preflight:
        indices = indices[
            :args.preflight
        ]

    output_dir = (
        WORK
        / f"num_shards_{args.num_shards}"
        / f"shard_{args.shard_id:02d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Machine={hostname} "
        f"physical_gpu={args.gpu} "
        f"visible_gpu=0 "
        f"shard={args.shard_id}/"
        f"{args.num_shards} "
        f"rows={len(indices)}",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL,
            local_files_only=True,
        )
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL,
            torch_dtype=dtype,
            local_files_only=True,
        )
        .to("cuda:0")
    )

    model.eval()

    for position, index in enumerate(
        indices,
        start=1,
    ):
        print()
        print(
            f"[{position}/{len(indices)}] "
            f"ROW {index}",
            flush=True,
        )

        process_row(
            index,
            rows[index],
            output_dir,
            model,
            tokenizer,
            args.max_new_tokens,
        )


if __name__ == "__main__":
    main()

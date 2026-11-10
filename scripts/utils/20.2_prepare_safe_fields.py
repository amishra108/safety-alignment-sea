#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/safety-alignment-sea"
)

MODEL = ROOT / "models/qwen3-4b"

WORK = (
    ROOT
    / "outputs/utils_and_prev_evals/20.2_prepare_safe_fields"
)

PREPARED = (
    WORK
    / "sea_2000_benign_overrefusal_prepared.jsonl"
)

REWRITE_PROGRESS = (
    WORK
    / "rewritten_prompts_progress.jsonl"
)

REWRITE_REVIEW = (
    WORK
    / "rewritten_prompts.txt"
)

REWRITE_VERSION = "benign-rewritten-field-v3"


REWRITE_PROMPT = """Rewrite one BENIGN instruction.

ORIGINAL BENIGN INSTRUCTION:
{benign_prompt}

PHRASE TO REMOVE:
{phrase}

PHRASE TYPE:
{phrase_type}

Remove the entire meaning represented by PHRASE TO REMOVE from the
instruction and replace it with a vague grammatical reference.

If PHRASE TYPE is activity, the result must contain the exact words:
this activity

If PHRASE TYPE is product, the result must contain the exact words:
this product

Preserve all wording, context, modifiers, and grammatical structure
outside the removed phrase as closely as possible.

Make only the smallest grammatical changes needed around the
replacement.

Do not preserve, restore, summarize, paraphrase, or hint at the
removed phrase.

Do not invent any new purpose, object, circumstance, example,
condition, person, place, device, or other information.

Do not answer the instruction.

Return exactly one JSON object and nothing else:

{{"rewritten": "..."}}
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reset-rewrites",
        action="store_true",
    )

    parser.add_argument(
        "--audit",
        action="store_true",
    )

    return parser.parse_args()


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        prefix=".pending_",
        suffix=".jsonl",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                json.dump(
                    row,
                    f,
                    ensure_ascii=False,
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


def load_prepared():
    if not PREPARED.is_file():
        raise FileNotFoundError(PREPARED)

    with PREPARED.open(encoding="utf-8") as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(rows) != 2000:
        raise RuntimeError(
            f"Expected 2000 prepared rows, found {len(rows)}"
        )

    required = (
        "index",
        "safe_prompt",
        "benign_phrase",
        "phrase_type",
        "rewritten_prompt",
        "audio_content_control_target",
        "audio_style_control_target",
    )

    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise RuntimeError(
                f"Index mismatch at row {index}"
            )

        for field in required:
            if field not in row:
                raise RuntimeError(
                    f"Row {index}: missing {field}"
                )

        if row["phrase_type"] not in {
            "activity",
            "product",
        }:
            raise RuntimeError(
                f"Row {index}: invalid phrase_type "
                f"{row['phrase_type']!r}"
            )

    return rows


def source_hash(index, row):
    payload = {
        "version": REWRITE_VERSION,
        "index": index,
        "safe_prompt": row["safe_prompt"],
        "benign_phrase": row["benign_phrase"],
        "phrase_type": row["phrase_type"],
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def load_progress(rows):
    progress = {}

    if not REWRITE_PROGRESS.exists():
        return progress

    with REWRITE_PROGRESS.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)

            index = record.get("index")

            if not isinstance(index, int):
                raise RuntimeError(
                    f"Progress line {line_number}: invalid index"
                )

            if not 0 <= index < len(rows):
                raise RuntimeError(
                    f"Progress line {line_number}: "
                    f"index {index} out of range"
                )

            expected = source_hash(
                index,
                rows[index],
            )

            if record.get("source_hash") != expected:
                raise RuntimeError(
                    f"Progress row {index} does not match "
                    "the current prepared dataset"
                )

            if record.get("version") != REWRITE_VERSION:
                raise RuntimeError(
                    f"Progress row {index} has incompatible version"
                )

            rewritten = record.get("rewritten_prompt")

            if not isinstance(rewritten, str) or not rewritten.strip():
                raise RuntimeError(
                    f"Progress row {index}: invalid rewritten_prompt"
                )

            progress[index] = record

    return progress


def append_progress(record):
    with REWRITE_PROGRESS.open(
        "a",
        encoding="utf-8",
    ) as f:
        json.dump(
            record,
            f,
            ensure_ascii=False,
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def refresh_review(rows, progress):
    temporary = REWRITE_REVIEW.with_name(
        REWRITE_REVIEW.name + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        for index, row in enumerate(rows):
            if index in progress:
                rewritten = progress[index]["rewritten_prompt"]
            else:
                rewritten = row["rewritten_prompt"]

            f.write(
                f"REWRITTEN: {rewritten} | "
                f"PHRASE: {row['benign_phrase']} | "
                f"BENIGN_PROMPT: {row['safe_prompt']}\n"
            )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        temporary,
        REWRITE_REVIEW,
    )


def parse_rewrite(raw):
    raw = raw.strip()

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Output is not valid JSON"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            "Output is not a JSON object"
        )

    if set(value) != {"rewritten"}:
        raise ValueError(
            f"Unexpected fields: {sorted(value)}"
        )

    rewritten = value["rewritten"]

    if not isinstance(rewritten, str):
        raise ValueError(
            "rewritten is not a string"
        )

    rewritten = rewritten.strip()

    if not rewritten:
        raise ValueError(
            "rewritten is empty"
        )

    return rewritten


def words(text):
    return re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?",
        text.lower(),
    )


def validate_rewrite(row, rewritten):
    phrase_type = row["phrase_type"]

    required = (
        "this activity"
        if phrase_type == "activity"
        else "this product"
    )

    lower = rewritten.lower()

    if required not in lower:
        raise ValueError(
            f"Missing required placeholder {required!r}"
        )

    if "[[target]]" in lower:
        raise ValueError(
            "Output contains target marker"
        )

    phrase_words = set(
        words(row["benign_phrase"])
    )

    benign_words = set(
        words(row["safe_prompt"])
    )

    output_words = set(
        words(rewritten)
    )

    stopwords = {
        "a", "an", "the", "this", "that", "these",
        "those", "i", "me", "my", "we", "us", "our",
        "you", "your", "he", "him", "his", "she", "her",
        "it", "its", "they", "them", "their", "who",
        "whom", "which", "what", "and", "or", "but",
        "if", "then", "than", "to", "for", "from", "of",
        "on", "in", "at", "by", "with", "without", "into",
        "onto", "over", "under", "through", "across",
        "about", "as", "is", "am", "are", "was", "were",
        "be", "been", "being", "do", "does", "did", "can",
        "could", "would", "should", "may", "might", "will",
        "shall", "how", "where", "when", "why",
    }

    allowed_new = {
        "activity",
        "product",
        "perform",
        "performs",
        "performing",
        "performed",
        "carry",
        "carries",
        "carrying",
        "carried",
        "out",
        "use",
        "uses",
        "using",
        "used",
    }

    removed_content = {
        word
        for word in phrase_words
        if word not in stopwords
    }

    outside_words = benign_words - phrase_words

    leaked = sorted(
        word
        for word in output_words
        if (
            word in removed_content
            and word not in outside_words
        )
    )

    if leaked:
        raise ValueError(
            "Removed phrase content still present: "
            + ", ".join(leaked[:20])
        )

    invented = sorted(
        word
        for word in output_words
        if (
            word not in benign_words
            and word not in stopwords
            and word not in allowed_new
        )
    )

    if invented:
        raise ValueError(
            "Invented content: "
            + ", ".join(invented[:20])
        )


def generate(model, tokenizer, prompt, max_tokens):
    import torch

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[
        0,
        inputs.shape[-1]:,
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if not text:
        raise RuntimeError(
            "Model returned empty output"
        )

    if generated.numel() >= max_tokens:
        raise RuntimeError(
            "Generation reached token limit"
        )

    return text


def audit(rows, progress):
    print(f"Prepared rows: {len(rows)}")
    print(
        f"Completed corrected rewrites: "
        f"{len(progress)}"
    )
    print(
        f"Pending corrected rewrites: "
        f"{len(rows) - len(progress)}"
    )

    if progress:
        missing = [
            index
            for index in range(len(rows))
            if index not in progress
        ]

        print(
            f"First pending indices: "
            f"{missing[:20]}"
        )


def main():
    args = parse_args()

    rows = load_prepared()

    if args.reset_rewrites:
        if REWRITE_PROGRESS.exists():
            REWRITE_PROGRESS.unlink()

        if REWRITE_REVIEW.exists():
            REWRITE_REVIEW.unlink()

        print(
            "Removed rewrite progress and review files.",
            flush=True,
        )

    progress = load_progress(rows)

    if args.audit:
        audit(
            rows,
            progress,
        )
        return

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=dtype,
        local_files_only=True,
    ).to("cuda:0").eval()

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    pending = [
        index
        for index in range(2000)
        if index not in progress
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(
        f"Prepared rows: 2000 | "
        f"already rewritten: {len(progress)} | "
        f"running now: {len(pending)}",
        flush=True,
    )

    failures = []

    for position, index in enumerate(
        pending,
        1,
    ):
        row = rows[index]

        last_error = None
        rewritten = None

        for attempt in range(1, 6):
            try:
                raw = generate(
                    model,
                    tokenizer,
                    REWRITE_PROMPT.format(
                        benign_prompt=row["safe_prompt"],
                        phrase=row["benign_phrase"],
                        phrase_type=row["phrase_type"],
                    ),
                    args.max_new_tokens,
                )

                candidate = parse_rewrite(raw)

                validate_rewrite(
                    row,
                    candidate,
                )

                rewritten = candidate
                break

            except Exception as exc:
                last_error = exc

                print(
                    f"[{position}/{len(pending)} "
                    f"index={index}] "
                    f"attempt={attempt}/5 FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        if rewritten is None:
            failures.append(
                (
                    index,
                    type(last_error).__name__,
                    str(last_error),
                )
            )

            print(
                f"[index={index}] FAILED after 5 attempts",
                flush=True,
            )

            continue

        record = {
            "version": REWRITE_VERSION,
            "index": index,
            "source_hash": source_hash(
                index,
                row,
            ),
            "rewritten_prompt": rewritten,
        }

        append_progress(
            record
        )

        progress[index] = record

        refresh_review(
            rows,
            progress,
        )

        print(
            f"[{position}/{len(pending)} index={index}]",
            flush=True,
        )
        print(
            f"  BENIGN: {row['safe_prompt']}",
            flush=True,
        )
        print(
            f"  PHRASE: {row['benign_phrase']}",
            flush=True,
        )
        print(
            f"  REWRITTEN: {rewritten}",
            flush=True,
        )

    if failures:
        print(
            f"Rewrite pass has {len(failures)} failures:",
            flush=True,
        )

        for index, error_type, message in failures:
            print(
                f"  {index}: {error_type}: {message}",
                flush=True,
            )

        raise RuntimeError(
            f"{len(failures)} rows remain incomplete. "
            "Rerun the same command to retry only those rows."
        )

    if len(progress) != 2000:
        print(
            f"Rewrite progress: {len(progress)}/2000. "
            "Run again to continue.",
            flush=True,
        )
        return

    # Apply corrected rewrites to the canonical dataset
    corrected = []

    for index, row in enumerate(rows):
        row = dict(row)

        rewritten = progress[index][
            "rewritten_prompt"
        ]

        row["rewritten_prompt"] = rewritten

        if "cleaned_rewritten_prompt" in row:
            row["cleaned_rewritten_prompt"] = rewritten

        corrected.append(row)

    atomic_jsonl(
        PREPARED,
        corrected,
    )

    refresh_review(
        corrected,
        progress,
    )

    print(
        "DONE: corrected all 2000 rows.",
        flush=True,
    )
    print(
        f"Canonical dataset: {PREPARED}",
        flush=True,
    )
    print(
        f"Rewrite progress: {REWRITE_PROGRESS}",
        flush=True,
    )
    print(
        f"Human-readable review: {REWRITE_REVIEW}",
        flush=True,
    )


if __name__ == "__main__":
    main()

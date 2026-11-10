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

MANUAL_REWRITE_TXT = WORK / "rewrite_completed.txt"
MANUAL_DOWNSTREAM_PROGRESS = (
    WORK / "manual_downstream_progress.jsonl"
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
condition, person, place, dewice, or other information.

Do not answer the instruction.

Return exactly one JSON object and nothing else:

{{"rewritten": "..."}}
"""


COMPLETION_PROMPT = """Convert the benign activity phrase into a natural, complete, third-person sentence describing someone performing that activity.

Preserve the meaning, objects, targets, purposes, and relevant context. Do not invent details or give instructions.

Also produce the corresponding activity phrase as it appears within the completed sentence.

Example:

[Phrase]
organise a local photography exhibition

[Extended sentences]
a person is organising a local photography exhibition

[Rephrased Phrase]
organising a local photography exhibition

Return exactly these sections:

[Extended sentences]
<one complete third-person sentence>

[Rephrased Phrase]
<corresponding activity phrase>

[Phrase]
{phrase}
"""

REPAIR_COMPLETION_PROMPT = """You are repairing one BENIGN training example.

Benign activity phrase:
{phrase}

Return exactly one JSON object and nothing else with:

"extended_sentence":
One natural complete third-person sentence describing a person performing the activity.
It should normally begin with "a person is".
Preserve the activity's objects, purpose, participants, and relevant context.

"rephrased_phrase":
The corresponding activity phrase as it appears in the sentence.

Do not add instructions, warnings, markdown, or commentary.
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

    parser.add_argument(
        "--refresh-manual-downstream",
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


def load_manual_rewrite_fields():
    if not MANUAL_REWRITE_TXT.is_file():
        raise FileNotFoundError(MANUAL_REWRITE_TXT)

    manual = {}

    with MANUAL_REWRITE_TXT.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t", 3)

            if len(parts) != 4:
                raise RuntimeError(
                    f"rewrite_completed.txt line {line_number}: "
                    "expected 4 tab-separated fields"
                )

            index_text, rewritten_prompt, benign_phrase, _safe_prompt = parts
            index = int(index_text)

            if index in manual:
                raise RuntimeError(
                    f"Duplicate index {index} in rewrite_completed.txt"
                )

            rewritten_prompt = rewritten_prompt.strip()
            benign_phrase = benign_phrase.strip()

            if not rewritten_prompt:
                raise RuntimeError(
                    f"Row {index}: empty rewritten_prompt"
                )

            if not benign_phrase:
                raise RuntimeError(
                    f"Row {index}: empty benign_phrase"
                )

            manual[index] = {
                "rewritten_prompt": rewritten_prompt,
                "benign_phrase": benign_phrase,
            }

    if set(manual) != set(range(2000)):
        missing = sorted(set(range(2000)) - set(manual))
        extra = sorted(set(manual) - set(range(2000)))

        raise RuntimeError(
            f"rewrite_completed.txt index mismatch. "
            f"Missing={missing[:20]} Extra={extra[:20]}"
        )

    return manual


def manual_completion_hash(index, phrase, phrase_type):
    payload = {
        "version": "manual-downstream-v1",
        "index": index,
        "benign_phrase": phrase,
        "phrase_type": phrase_type,
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def load_manual_downstream_progress():
    progress = {}

    if not MANUAL_DOWNSTREAM_PROGRESS.exists():
        return progress

    with MANUAL_DOWNSTREAM_PROGRESS.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            index = record.get("index")

            if not isinstance(index, int):
                raise RuntimeError(
                    f"Manual progress line {line_number}: invalid index"
                )

            progress[index] = record

    return progress


def append_manual_downstream_progress(record):
    MANUAL_DOWNSTREAM_PROGRESS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANUAL_DOWNSTREAM_PROGRESS.open(
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


def parse_completion_sections(raw):
    pattern = re.compile(
        r"\[Extended sentences\]\s*(.*?)\s*"
        r"\[Rephrased Phrase\]\s*(.*?)(?:\s*\[Phrase\]|\Z)",
        flags=re.DOTALL,
    )

    match = pattern.search(raw)

    if match is None:
        raise ValueError(
            "Could not parse completion sections"
        )

    extended = " ".join(
        match.group(1).split()
    )
    rephrased = " ".join(
        match.group(2).split()
    )

    if not extended:
        raise ValueError(
            "Missing extended_sentence"
        )

    if not rephrased:
        raise ValueError(
            "Missing rephrased_phrase"
        )

    return {
        "extended_sentence": extended,
        "rephrased_phrase": rephrased,
    }


def parse_repair_completion(raw):
    start = raw.find("{")

    if start < 0:
        raise ValueError(
            "Completion repair has no JSON object"
        )

    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(raw[start:])

    if not isinstance(value, dict):
        raise ValueError(
            "Completion repair is not a JSON object"
        )

    extended = value.get("extended_sentence")
    rephrased = value.get("rephrased_phrase")

    if not isinstance(extended, str) or not extended.strip():
        raise ValueError(
            "Missing extended_sentence"
        )

    if not isinstance(rephrased, str) or not rephrased.strip():
        raise ValueError(
            "Missing rephrased_phrase"
        )

    return {
        "extended_sentence":
            " ".join(extended.split()),
        "rephrased_phrase":
            " ".join(rephrased.split()),
    }


def generate_completion(
    model,
    tokenizer,
    phrase,
    max_tokens,
):
    try:
        raw = generate(
            model,
            tokenizer,
            COMPLETION_PROMPT.format(
                phrase=phrase
            ),
            max_tokens,
        )

        return parse_completion_sections(raw)

    except Exception as first_error:
        print(
            "  Original completion failed: "
            f"{type(first_error).__name__}: "
            f"{first_error}",
            flush=True,
        )

    last_error = None

    for attempt in range(1, 6):
        try:
            raw = generate(
                model,
                tokenizer,
                REPAIR_COMPLETION_PROMPT.format(
                    phrase=phrase
                ),
                max_tokens,
            )

            return parse_repair_completion(raw)

        except Exception as exc:
            last_error = exc

            print(
                f"  Repair completion attempt "
                f"{attempt}/5 failed: "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

    raise RuntimeError(
        "Completion repair failed after 5 attempts: "
        f"{type(last_error).__name__}: "
        f"{last_error}"
    )


def refresh_manual_downstream(rows, args):
    manual = load_manual_rewrite_fields()

    phrase_changed = [
        index
        for index, row in enumerate(rows)
        if (
            row["benign_phrase"]
            != manual[index]["benign_phrase"]
        )
    ]

    rewritten_changed = [
        index
        for index, row in enumerate(rows)
        if (
            row["rewritten_prompt"]
            != manual[index]["rewritten_prompt"]
        )
    ]

    activity_changed = [
        index
        for index in phrase_changed
        if rows[index]["phrase_type"] == "activity"
    ]

    print(
        f"benign_phrase changed: {len(phrase_changed)}",
        flush=True,
    )
    print(
        f"rewritten_prompt changed: {len(rewritten_changed)}",
        flush=True,
    )

    progress = load_manual_downstream_progress()
    completed = {}

    for index in activity_changed:
        expected_hash = manual_completion_hash(
            index,
            manual[index]["benign_phrase"],
            rows[index]["phrase_type"],
        )

        record = progress.get(index)

        if (
            record is not None
            and record.get("source_hash")
            == expected_hash
        ):
            completed[index] = record

    pending = [
        index
        for index in activity_changed
        if index not in completed
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(
        f"Activity completions already available: "
        f"{len(completed)}",
        flush=True,
    )
    print(
        f"Activity completions running now: "
        f"{len(pending)}",
        flush=True,
    )

    if pending:
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

        for position, index in enumerate(
            pending,
            1,
        ):
            phrase = manual[index]["benign_phrase"]

            print(
                f"[{position}/{len(pending)} "
                f"index={index}] "
                f"PHRASE: {phrase}",
                flush=True,
            )

            completion = generate_completion(
                model,
                tokenizer,
                phrase,
                args.max_new_tokens,
            )

            record = {
                "index": index,
                "source_hash":
                    manual_completion_hash(
                        index,
                        phrase,
                        rows[index]["phrase_type"],
                    ),
                **completion,
            }

            append_manual_downstream_progress(
                record
            )

            completed[index] = record

            print(
                f"  EXTENDED: "
                f"{completion['extended_sentence']}",
                flush=True,
            )
            print(
                f"  REPHRASED: "
                f"{completion['rephrased_phrase']}",
                flush=True,
            )

    unresolved = [
        index
        for index in activity_changed
        if index not in completed
    ]

    if unresolved:
        print(
            f"Manual downstream progress: "
            f"{len(activity_changed) - len(unresolved)}/"
            f"{len(activity_changed)} changed activity rows.",
            flush=True,
        )
        print(
            "Rerun the same command to continue.",
            flush=True,
        )
        return

    updated = []

    for index, source_row in enumerate(rows):
        row = dict(source_row)

        row["benign_phrase"] = (
            manual[index]["benign_phrase"]
        )
        row["rewritten_prompt"] = (
            manual[index]["rewritten_prompt"]
        )
        row["cleaned_rewritten_prompt"] = (
            manual[index]["rewritten_prompt"]
        )

        if index in phrase_changed:
            if row["phrase_type"] == "activity":
                completion = completed[index]

                row["extended_sentence"] = (
                    completion["extended_sentence"]
                )
                row["rephrased_phrase"] = (
                    completion["rephrased_phrase"]
                )
                content = row["extended_sentence"]

            else:
                row["extended_sentence"] = None
                row["rephrased_phrase"] = None
                content = row["benign_phrase"]

            row[
                "audio_content_control_target"
            ] = (
                "The audio contains speech saying "
                + content
            )

        updated.append(row)

    backup = PREPARED.with_suffix(
        ".jsonl.before_manual_downstream_refresh"
    )

    if not backup.exists():
        backup.write_bytes(
            PREPARED.read_bytes()
        )

    atomic_jsonl(
        PREPARED,
        updated,
    )

    print(
        f"Updated canonical JSONL: {PREPARED}",
        flush=True,
    )
    print(
        f"Backup: {backup}",
        flush=True,
    )


def main():
    args = parse_args()

    rows = load_prepared()

    if args.refresh_manual_downstream:
        refresh_manual_downstream(
            rows,
            args,
        )
        return

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

#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
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

BEFORE_TXT = WORK / "rewrite_before.txt"
PENDING_TXT = WORK / "rewrite_pending.txt"
PROGRESS_JSONL = WORK / "rewrite_progress.jsonl"
REWRITTEN_TXT = WORK / "rewrite_completed.txt"

REWRITE_VERSION = "benign-rewrite-v6"


REWRITE_PROMPT = """Rewrite one benign instruction.

BENIGN PROMPT:
{benign_prompt}

PHRASE TO REMOVE:
{phrase}

PHRASE TYPE:
{phrase_type}

Rewrite BENIGN PROMPT so that the complete meaning represented by
PHRASE TO REMOVE is replaced with a vague grammatical reference.

If PHRASE TYPE is activity, the final prompt must contain the exact words:
this activity

If PHRASE TYPE is product, the final prompt must contain the exact words:
this product

Preserve all wording and context outside the removed phrase as closely
as possible.

Make only the smallest grammatical changes necessary around the replacement.

Do not restore, paraphrase, summarize, or hint at the removed phrase.
Do not invent any new information.
Do not answer the instruction.

Return exactly one JSON object:

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
        "--audit",
        action="store_true",
    )

    return parser.parse_args()


def one_line(value):
    return " ".join(str(value).split())


def atomic_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temporary = tempfile.mkstemp(
        prefix=".pending_",
        suffix=".jsonl",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
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


def atomic_txt(path, lines):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temporary = tempfile.mkstemp(
        prefix=".pending_",
        suffix=".txt",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            for line in lines:
                f.write(line)
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


def load_rows():
    if not PREPARED.is_file():
        raise FileNotFoundError(PREPARED)

    with PREPARED.open(
        encoding="utf-8",
    ) as f:
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
    )

    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise RuntimeError(
                f"Index mismatch at row {index}"
            )

        for field in required:
            value = row.get(field)

            if value is None:
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


def words(text):
    return re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?",
        text.lower(),
    )


STOPWORDS = {
    "a", "an", "the", "this", "that", "these",
    "those", "i", "me", "my", "we", "us", "our",
    "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "who",
    "whom", "which", "what", "and", "or", "but",
    "if", "then", "than", "to", "for", "from", "of",
    "on", "in", "at", "by", "with", "without", "into",
    "onto", "over", "under", "through", "across",
    "about", "as", "is", "am", "are", "was", "were",
    "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "may", "might",
    "will", "shall", "how", "where", "when", "why",
}


ALLOWED_NEW = {
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


def validate_rewrite(row, rewritten):
    rewritten = rewritten.strip()

    if not rewritten:
        return False, "empty rewrite"

    required = (
        "this activity"
        if row["phrase_type"] == "activity"
        else "this product"
    )

    if required not in rewritten.lower():
        return False, f"missing {required!r}"

    benign = Counter(
        words(row["safe_prompt"])
    )

    phrase = Counter(
        words(row["benign_phrase"])
    )

    output = Counter(
        words(rewritten)
    )

    leaked = []

    for word, phrase_count in phrase.items():
        if word in STOPWORDS:
            continue

        outside_count = max(
            0,
            benign[word] - phrase_count,
        )

        if output[word] > outside_count:
            leaked.append(word)

    if leaked:
        return (
            False,
            "phrase content remains: "
            + ", ".join(
                sorted(set(leaked))[:20]
            ),
        )

    invented = []

    for word, output_count in output.items():
        if word in STOPWORDS:
            continue

        if word in ALLOWED_NEW:
            continue

        if output_count > benign[word]:
            invented.append(word)

    if invented:
        return (
            False,
            "invented content: "
            + ", ".join(
                sorted(set(invented))[:20]
            ),
        )

    return True, None


def mask_phrase(row):
    benign_prompt = row["safe_prompt"]
    phrase = row["benign_phrase"]

    match = re.search(
        re.escape(phrase),
        benign_prompt,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            "benign_phrase is not an exact span "
            "of safe_prompt"
        )

    return (
        benign_prompt[:match.start()]
        + "[[TARGET]]"
        + benign_prompt[match.end():]
    )


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

    return hashlib.sha256(
        encoded
    ).hexdigest()


def txt_line(index, rewritten, phrase, benign_prompt):
    return (
        f"{index}\t"
        f"{one_line(rewritten)}\t"
        f"{one_line(phrase)}\t"
        f"{one_line(benign_prompt)}"
    )


def write_before(rows):
    if BEFORE_TXT.exists():
        return

    lines = [
        txt_line(
            index,
            row["rewritten_prompt"],
            row["benign_phrase"],
            row["safe_prompt"],
        )
        for index, row in enumerate(rows)
    ]

    atomic_txt(
        BEFORE_TXT,
        lines,
    )


def find_bad_rows(rows):
    return {
        index: "rewrite all rows"
        for index in range(len(rows))
    }



def load_progress(rows):
    progress = {}

    PROGRESS_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROGRESS_JSONL.touch(
        exist_ok=True,
    )

    with PROGRESS_JSONL.open(
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            record = json.loads(line)
            index = record.get("index")

            if not isinstance(index, int):
                raise RuntimeError(
                    f"Progress line {line_number}: "
                    "invalid index"
                )

            if not 0 <= index < len(rows):
                raise RuntimeError(
                    f"Progress line {line_number}: "
                    f"index {index} out of range"
                )

            if record.get("version") != REWRITE_VERSION:
                raise RuntimeError(
                    f"Progress row {index}: "
                    "incompatible version"
                )

            expected = source_hash(
                index,
                rows[index],
            )

            if record.get("source_hash") != expected:
                raise RuntimeError(
                    f"Progress row {index}: "
                    "source mismatch"
                )

            rewritten = record.get(
                "rewritten_prompt"
            )

            if (
                not isinstance(rewritten, str)
                or not rewritten.strip()
            ):
                raise RuntimeError(
                    f"Progress row {index}: "
                    "missing rewritten_prompt"
                )

            progress[index] = record

    return progress


def append_progress(record):
    with PROGRESS_JSONL.open(
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


def write_completed(rows, progress):
    lines = []

    for index in sorted(progress):
        row = rows[index]

        lines.append(
            txt_line(
                index,
                progress[index]["rewritten_prompt"],
                row["benign_phrase"],
                row["safe_prompt"],
            )
        )

    atomic_txt(
        REWRITTEN_TXT,
        lines,
    )


def write_pending(rows, bad_rows, progress):
    lines = []

    for index in sorted(bad_rows):
        if index in progress:
            continue

        row = rows[index]

        lines.append(
            txt_line(
                index,
                row["rewritten_prompt"],
                row["benign_phrase"],
                row["safe_prompt"],
            )
        )

    atomic_txt(
        PENDING_TXT,
        lines,
    )


def parse_rewrite(raw):
    try:
        value = json.loads(
            raw.strip()
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "output is not valid JSON"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            "output is not a JSON object"
        )

    if set(value) != {"rewritten"}:
        raise ValueError(
            f"unexpected fields: "
            f"{sorted(value)}"
        )

    rewritten = value["rewritten"]

    if (
        not isinstance(rewritten, str)
        or not rewritten.strip()
    ):
        raise ValueError(
            "invalid rewritten string"
        )

    return rewritten.strip()


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
            "model returned empty output"
        )

    if generated.numel() >= max_tokens:
        raise RuntimeError(
            "generation reached token limit"
        )

    return text


def apply_completed_rewrites(rows, progress):
    corrected = []

    for index, source_row in enumerate(rows):
        row = dict(source_row)

        if index in progress:
            rewritten = progress[index][
                "rewritten_prompt"
            ]

            row["rewritten_prompt"] = rewritten

            if "cleaned_rewritten_prompt" in row:
                row[
                    "cleaned_rewritten_prompt"
                ] = rewritten

        corrected.append(row)

    atomic_jsonl(
        PREPARED,
        corrected,
    )


def main():
    args = parse_args()

    rows = load_rows()

    write_before(
        rows
    )

    bad_rows = find_bad_rows(
        rows
    )

    progress = load_progress(
        rows
    )

    invalid_progress = sorted(
        set(progress)
        - set(bad_rows)
    )

    if invalid_progress:
        print(
            "Progress contains rows that are now "
            "already valid in the canonical dataset: "
            f"{invalid_progress[:20]}",
            flush=True,
        )

    pending = [
        index
        for index in sorted(bad_rows)
        if index not in progress
    ]

    write_pending(
        rows,
        bad_rows,
        progress,
    )

    write_completed(
        rows,
        progress,
    )

    print(
        f"Total rows: {len(rows)}",
        flush=True,
    )

    print(
        f"Already valid: "
        f"{len(rows) - len(bad_rows)}",
        flush=True,
    )

    print(
        f"Need correction: "
        f"{len(bad_rows)}",
        flush=True,
    )

    print(
        f"Corrected in progress: "
        f"{len(set(progress) & set(bad_rows))}",
        flush=True,
    )

    print(
        f"Still pending: "
        f"{len(pending)}",
        flush=True,
    )

    print(
        f"Before TXT: {BEFORE_TXT}",
        flush=True,
    )

    print(
        f"Pending TXT: {PENDING_TXT}",
        flush=True,
    )

    print(
        f"Progress JSONL: {PROGRESS_JSONL}",
        flush=True,
    )

    if args.audit:
        print(
            "First pending rows:",
            flush=True,
        )

        for index in pending[:20]:
            print(
                f"  {index}: "
                f"{bad_rows[index]}",
                flush=True,
            )

        return

    if args.limit is not None:
        pending = pending[
            :args.limit
        ]

    if not pending:
        apply_completed_rewrites(
            rows,
            progress,
        )

        print(
            "No pending rewrites.",
            flush=True,
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

    failures = []

    for position, index in enumerate(
        pending,
        1,
    ):
        row = rows[index]

        rewritten = None
        last_error = None
        last_candidate = None

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

                candidate = parse_rewrite(
                    raw
                )

                last_candidate = candidate

                valid, reason = validate_rewrite(
                    row,
                    candidate,
                )

                if valid:
                    rewritten = candidate
                    break

                last_error = ValueError(
                    reason
                )

                print(
                    f"[{position}/{len(pending)} "
                    f"index={index}] "
                    f"attempt={attempt}/5 validation failed: "
                    f"{reason}",
                    flush=True,
                )

            except Exception as exc:
                last_error = exc

                print(
                    f"[{position}/{len(pending)} "
                    f"index={index}] "
                    f"attempt={attempt}/5 FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

        if rewritten is None:
            if last_candidate is None:
                raise RuntimeError(
                    f"Row {index}: model never produced a parseable "
                    "rewritten prompt after 5 attempts"
                )

            rewritten = last_candidate

            print(
                f"[index={index}] "
                "ACCEPTED AFTER VALIDATION FAILURE: "
                f"{type(last_error).__name__}: {last_error}",
                flush=True,
            )

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

        write_pending(
            rows,
            bad_rows,
            progress,
        )

        write_completed(
            rows,
            progress,
        )

        print(
            f"[{position}/{len(pending)} "
            f"index={index}]",
            flush=True,
        )

        print(
            f"  BENIGN: "
            f"{one_line(row['safe_prompt'])}",
            flush=True,
        )

        print(
            f"  PHRASE: "
            f"{one_line(row['benign_phrase'])}",
            flush=True,
        )

        print(
            f"  REWRITTEN: "
            f"{one_line(rewritten)}",
            flush=True,
        )

    unresolved = [
        index
        for index in bad_rows
        if index not in progress
    ]

    write_pending(
        rows,
        bad_rows,
        progress,
    )

    if not unresolved:
        apply_completed_rewrites(
            rows,
            progress,
        )

        print(
            "All required rewrites completed.",
            flush=True,
        )

        print(
            f"Updated canonical JSONL: "
            f"{PREPARED}",
            flush=True,
        )

        return

    print(
        f"Remaining pending: "
        f"{len(unresolved)}",
        flush=True,
    )

    if failures:
        print(
            f"Failed this pass: "
            f"{len(failures)}",
            flush=True,
        )

        for index, error_type, message in failures:
            print(
                f"  {index}: "
                f"{error_type}: "
                f"{message}",
                flush=True,
            )


if __name__ == "__main__":
    main()

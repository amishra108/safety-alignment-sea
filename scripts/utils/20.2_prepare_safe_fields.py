#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
SOURCE = (
    ROOT
    / "outputs/utils_and_prev_evals/20.1_prepare_safe_database"
    / "full_sharded_v1/sea_2000_benign_overrefusal_completed.jsonl"
)
MODEL = ROOT / "models/qwen3-4b"
WORK = ROOT / "outputs/utils_and_prev_evals/20.2_prepare_safe_fields"
VERSION = "benign-audio-fields-v2"

ORIGINAL_PREPARED = ROOT / "data/saferlhf/sea_3000_prepared.jsonl"


def load_original_styles(rows):
    with ORIGINAL_PREPARED.open(encoding="utf-8") as f:
        original = [json.loads(line) for line in f if line.strip()]

    if len(original) != 3000:
        raise RuntimeError(
            f"Expected 3000 original prepared rows, got {len(original)}"
        )

    styles = []
    for index, row in enumerate(rows):
        previous = original[index]

        if previous.get("prompt") != row["prompt"]:
            raise RuntimeError(
                f"Original prompt mismatch at index {index}"
            )

        target = previous.get("audio_style_control_target")
        if not isinstance(target, str) or not target.strip():
            raise RuntimeError(
                f"Missing original voice style at index {index}"
            )

        styles.append(target)

    return styles

EXTRACTION_PROMPT = """You are preparing synthetic audio training data from a BENIGN user instruction.

TASK 1: EXTRACT THE BENIGN PHRASE

Extract a self-contained phrase describing the principal benign activity or product requested in the instruction.

Preserve the actual subject, purpose, objects, relevant participants, and essential circumstances. If several connected actions form one request, keep them together.

Do not introduce harmful interpretations. Do not invent details.

TASK 2: CLASSIFY THE PHRASE

Choose exactly one:

activity: An action, behaviour, or process someone performs.
product: An object, item, substance, or product.

Classify the meaning of the complete phrase.

TASK 3: CREATE A VAGUE QUESTION

Write a short generic question preserving the broad assistance requested, but hiding the identifying details of the benign activity or product.

Use a natural vague reference to the hidden subject.
Examples include "this", "this activity", "this process", "this task",
"this product", or another grammatically natural vague reference.

Examples:
"How can I perform this activity?"
"What is the best way to carry out this activity?"
"How can I use this product?"
"Where can I obtain this product?"

Use a question that matches the original request's intent. Do not say "create this activity".

TASK 4: CREATE A CLEANED INSTRUCTION

Rewrite the original instruction while preserving its wording, context, structure, and question as closely as possible.

Replace or abstract the main benign activity or product with a natural vague reference.

Preserve the original grammar and wording as closely as possible.
Do not force the phrases "this activity" or "this product" when another
construction is more natural.

Make the smallest reasonable change. Do not rewrite the entire instruction into a generic question. Do not add warnings, refusals, answers, or commentary.

OUTPUT EXACTLY THESE SECTIONS:

[Key Phrase]
<complete benign phrase>

[Phrase Type]
<activity or product>

[Rephrased Question]
<short vague question>

[Cleaned Instruction]
<minimally abstracted original instruction>

[Instruction]
{instruction}
"""

REPAIR_EXTRACTION_PROMPT = """You are repairing one BENIGN training example.

Original benign instruction:
{instruction}

Return exactly one JSON object and nothing else, with these four string fields:

"benign_phrase":
A self-contained phrase describing the principal benign activity or product.

"phrase_type":
Exactly "activity" or "product".

"rewritten_prompt":
A minimally abstracted version of the ORIGINAL instruction.
Preserve its wording, context, structure, modifiers, and question as closely as possible.
Abstract the principal benign activity or product using a natural vague reference.
Make only the smallest grammatical changes needed.
Do NOT turn this into a generic question.

"vague_rewritten_prompt":
A short generic question hiding the identifying details.
Use any grammatically natural vague reference.

Do not add markdown or commentary.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be positive")

    if not 0 <= args.shard_id < args.num_shards:
        parser.error("Invalid shard ID")

    return args


def load_rows():
    with SOURCE.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) != 2000:
        raise RuntimeError(f"Expected 2000 rows, found {len(rows)}")

    required = (
        "safe_prompt",
        "safe_response",
        "prompt",
        "new_chosen_response",
        "over_refusal_response",
    )

    for index, row in enumerate(rows):
        for field in required:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"Row {index}: missing {field}")

    return rows


def fingerprint(index, row):
    payload = json.dumps(
        {"version": VERSION, "index": index, "source": row},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".pending_", dir=str(path.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_sections(raw, names):
    pattern = re.compile(r"(?m)^[ \t]*\[([^\]\r\n]+)\][ \t]*$")
    matches = list(pattern.finditer(raw))
    values = {}

    for position, match in enumerate(matches):
        name = match.group(1).strip()
        if name == "Instruction":
            continue

        if name not in names or name in values:
            raise ValueError(f"Unexpected or duplicate section: {name}")

        end = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(raw)
        )
        values[name] = raw[match.end():end].strip()

    for name in names:
        if not values.get(name):
            raise ValueError(f"Missing section: {name}")

    return values


def generate(model, tokenizer, prompt, max_tokens):
    import torch

    messages = [{"role": "user", "content": prompt}]
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

    generated = output[0, inputs.shape[-1]:]

    text = tokenizer.decode(
        generated, skip_special_tokens=True
    ).strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    if not text:
        raise RuntimeError("Model returned empty output.")

    if generated.numel() >= max_tokens:
        raise RuntimeError("Generation reached token limit.")

    return text


def stage_path(folder, index, stage):
    return folder / f"row_{index:04d}_{stage}.json"


def write_prompt_txt(folder):
    records = []

    for path in folder.glob("row_*_complete.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))

        if saved.get("stage") != "complete":
            continue

        value = saved.get("value")
        if not isinstance(value, dict):
            continue

        index = value.get("index")
        if not isinstance(index, int):
            continue

        records.append((index, value))

    records.sort(key=lambda item: item[0])

    output = folder / "all_prompts.txt"
    temporary = output.with_name(output.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as f:
        for index, row in records:
            f.write(f"INDEX: {index}\n")
            f.write(f"ORIGINAL_HARMFUL: {row.get('prompt', '')}\n")
            f.write(f"BENIGN_PROMPT: {row.get('safe_prompt', '')}\n")
            f.write(f"SAFE_RESPONSE: {row.get('safe_response', '')}\n")
            f.write(
                "OVER_REFUSAL_RESPONSE: "
                f"{row.get('over_refusal_response', '')}\n"
            )
            f.write(f"PHRASE_TYPE: {row.get('phrase_type', '')}\n")
            f.write(f"PHRASE: {row.get('benign_phrase', '')}\n")
            f.write(
                f"REWRITTEN: {row.get('rewritten_prompt', '')}\n"
            )
            f.write(
                "VAGUE: "
                f"{row.get('vague_rewritten_prompt', '')}\n"
            )
            f.write(
                "AUDIO_CONTENT: "
                f"{row.get('audio_content_control_target', '')}\n"
            )
            f.write(
                "AUDIO_STYLE: "
                f"{row.get('audio_style_control_target', '')}\n"
            )
            f.write("\n")

        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, output)

    print(
        f"Prompt TXT refreshed: {len(records)} rows -> {output}",
        flush=True,
    )


def stage(folder, index, name, source_hash, produce):
    path = stage_path(folder, index, name)

    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            saved.get("version") != VERSION
            or saved.get("source_hash") != source_hash
            or saved.get("stage") != name
        ):
            raise RuntimeError(f"Incompatible checkpoint: {path}")
        return saved["value"]

    value = produce()
    atomic_json(path, {
        "version": VERSION,
        "source_hash": source_hash,
        "stage": name,
        "value": value,
    })
    return value


def process(index, row, folder, model, tokenizer, max_tokens, style_target):
    source_hash = fingerprint(index, row)

    def extract():
        def validate_result(result):
            required = (
                "benign_phrase",
                "phrase_type",
                "rewritten_prompt",
                "vague_rewritten_prompt",
            )

            for field in required:
                value = result.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing extraction field: {field}")
                result[field] = value.strip()

            phrase_type = result["phrase_type"].lower()
            result["phrase_type"] = phrase_type

            if phrase_type not in {"activity", "product"}:
                raise ValueError(
                    f"Invalid phrase type: {result['phrase_type']!r}"
                )

            rewritten = result["rewritten_prompt"].lower()
            vague = result["vague_rewritten_prompt"].lower()

            if phrase_type == "activity":
                if "this activity" not in rewritten:
                    raise ValueError(
                        "Rewritten prompt missing 'this activity'"
                    )
                if "this activity" not in vague:
                    raise ValueError(
                        "Vague prompt missing 'this activity'"
                    )
            else:
                if "this product" not in rewritten:
                    raise ValueError(
                        "Rewritten prompt missing 'this product'"
                    )
                if not any(
                    marker in vague
                    for marker in ("this product", "these products")
                ):
                    raise ValueError(
                        "Vague prompt missing product placeholder"
                    )

            result["cleaned_rewritten_prompt"] = (
                result["rewritten_prompt"]
            )

            return result

        # First try the original generation format
        try:
            raw = generate(
                model,
                tokenizer,
                EXTRACTION_PROMPT.format(
                    instruction=row["safe_prompt"]
                ),
                max_tokens,
            )

            fields = parse_sections(
                raw,
                [
                    "Key Phrase",
                    "Phrase Type",
                    "Rephrased Question",
                    "Cleaned Instruction",
                ],
            )

            return validate_result({
                "benign_phrase": fields["Key Phrase"],
                "phrase_type": fields["Phrase Type"],
                "rewritten_prompt": fields["Cleaned Instruction"],
                "vague_rewritten_prompt":
                    fields["Rephrased Question"],
            })

        except Exception as first_error:
            print(
                f"  Original extraction failed: "
                f"{type(first_error).__name__}: {first_error}",
                flush=True,
            )

        # Failed rows use a stricter JSON repair prompt
        last_error = None

        for attempt in range(1, 6):
            try:
                raw = generate(
                    model,
                    tokenizer,
                    REPAIR_EXTRACTION_PROMPT.format(
                        instruction=row["safe_prompt"]
                    ),
                    max_tokens,
                )

                start_json = raw.find("{")
                if start_json < 0:
                    raise ValueError("Repair output has no JSON object")

                decoder = json.JSONDecoder()
                value, _ = decoder.raw_decode(raw[start_json:])

                if not isinstance(value, dict):
                    raise ValueError("Repair output is not a JSON object")

                result = validate_result(value)

                print(
                    f"  Extraction repaired on attempt {attempt}",
                    flush=True,
                )

                return result

            except Exception as exc:
                last_error = exc
                print(
                    f"  Repair extraction attempt {attempt}/5 failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        raise RuntimeError(
            "Extraction repair failed after 5 attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )

    extraction = stage(
        folder, index, "extraction", source_hash, extract
    )

    if extraction["phrase_type"] == "activity":
        def complete():
            def validate_completion(result):
                extended = result.get("extended_sentence")
                rephrased = result.get("rephrased_phrase")

                if not isinstance(extended, str) or not extended.strip():
                    raise ValueError("Missing extended_sentence")

                if not isinstance(rephrased, str) or not rephrased.strip():
                    raise ValueError("Missing rephrased_phrase")

                extended = " ".join(extended.split())
                rephrased = " ".join(rephrased.split())

                if len(extended.split()) < 4:
                    raise ValueError("Extended sentence too short")

                return {
                    "extended_sentence": extended,
                    "rephrased_phrase": rephrased,
                }

            # First preserve the original completion behaviour
            try:
                raw = generate(
                    model,
                    tokenizer,
                    COMPLETION_PROMPT.format(
                        phrase=extraction["benign_phrase"]
                    ),
                    max_tokens,
                )

                fields = parse_sections(
                    raw,
                    ["Extended sentences", "Rephrased Phrase"],
                )

                return validate_completion({
                    "extended_sentence":
                        fields["Extended sentences"],
                    "rephrased_phrase":
                        fields["Rephrased Phrase"],
                })

            except Exception as first_error:
                print(
                    f"  Original completion failed: "
                    f"{type(first_error).__name__}: {first_error}",
                    flush=True,
                )

            last_error = None

            for attempt in range(1, 6):
                try:
                    raw = generate(
                        model,
                        tokenizer,
                        REPAIR_COMPLETION_PROMPT.format(
                            phrase=extraction["benign_phrase"]
                        ),
                        max_tokens,
                    )

                    start_json = raw.find("{")
                    if start_json < 0:
                        raise ValueError(
                            "Completion repair has no JSON object"
                        )

                    decoder = json.JSONDecoder()
                    value, _ = decoder.raw_decode(raw[start_json:])

                    if not isinstance(value, dict):
                        raise ValueError(
                            "Completion repair is not a JSON object"
                        )

                    result = validate_completion(value)

                    print(
                        f"  Completion repaired on attempt {attempt}",
                        flush=True,
                    )

                    return result

                except Exception as exc:
                    last_error = exc
                    print(
                        f"  Repair completion attempt {attempt}/5 failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

            raise RuntimeError(
                "Completion repair failed after 5 attempts: "
                f"{type(last_error).__name__}: {last_error}"
            )

        completion = stage(
            folder, index, "completion", source_hash, complete
        )
        content = completion["extended_sentence"]
    else:
        completion = {
            "extended_sentence": None,
            "rephrased_phrase": None,
        }
        content = extraction["benign_phrase"]

    prepared = {
        "index": index,
        **row,
        **extraction,
        **completion,
        "audio_content_control_instruction": "What is in the audio?",
        "audio_content_control_target":
            "The audio contains speech saying " + content,
        "audio_style_control_instruction":
            "What are the characteristics of the speaker's voice?",
        "audio_style_control_target":
            style_target,
        "vague_audio_prompt": (
            "The audio contains an activity. "
            if extraction["phrase_type"] == "activity"
            else "The audio contains a product. "
        ) + extraction["vague_rewritten_prompt"],
        "preparation_version": VERSION,
    }

    stage(
        folder, index, "complete", source_hash,
        lambda: prepared,
    )

    return prepared


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    rows = load_rows()
    original_styles = load_original_styles(rows)

    folder = (
        WORK / f"num_shards_{args.num_shards}_v2"
        / f"shard_{args.shard_id:02d}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    indices = list(range(args.shard_id, 2000, args.num_shards))
    write_prompt_txt(folder)

    pending = []

    for index in indices:
        path = stage_path(folder, index, "complete")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (
                saved.get("version") != VERSION
                or saved.get("source_hash") != fingerprint(index, rows[index])
                or saved.get("stage") != "complete"
            ):
                raise RuntimeError(f"Incompatible output: {path}")
        else:
            pending.append(index)

    print(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"assigned={len(indices)} "
        f"completed={len(indices) - len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )

    if not pending:
        print("Shard already complete.", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

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

    for position, index in enumerate(pending, 1):
        print(
            f"[shard={args.shard_id} {position}/{len(pending)} "
            f"index={index}]",
            flush=True,
        )

        try:
            prepared = process(
                index, rows[index], folder,
                model, tokenizer, args.max_new_tokens,
                original_styles[index],
            )

            print(
                f"  BENIGN: {rows[index]['safe_prompt']}",
                flush=True,
            )
            print(
                f"  PHRASE: {prepared['benign_phrase']}",
                flush=True,
            )
            print(
                f"  REWRITTEN: {prepared['rewritten_prompt']}",
                flush=True,
            )
            print(
                f"  VAGUE: {prepared['vague_rewritten_prompt']}",
                flush=True,
            )
            print(
                f"  AUDIO: {prepared['audio_content_control_target']}",
                flush=True,
            )

            write_prompt_txt(folder)

            error_path = stage_path(folder, index, "error")
            if error_path.exists():
                error_path.unlink()

        except Exception as exc:
            failures.append(index)
            atomic_json(
                stage_path(folder, index, "error"),
                {
                    "index": index,
                    "version": VERSION,
                    "source_hash": fingerprint(index, rows[index]),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            print(
                f"  FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )

    print(
        f"Shard {args.shard_id} finished. "
        f"Failed indices: {failures}",
        flush=True,
    )

    if failures:
        raise RuntimeError(
            f"{len(failures)} rows incomplete; rerun this shard "
            "to retry them."
        )

    # A one-shard run can now produce the final merged dataset
    if args.num_shards == 1 and args.shard_id == 0:
        merged = []

        for index in range(2000):
            complete_path = stage_path(
                folder, index, "complete"
            )

            if not complete_path.exists():
                raise RuntimeError(
                    f"Cannot merge: row {index} is incomplete"
                )

            saved = json.loads(
                complete_path.read_text(encoding="utf-8")
            )

            if saved.get("stage") != "complete":
                raise RuntimeError(
                    f"Invalid complete checkpoint at row {index}"
                )

            if saved.get("source_hash") != fingerprint(
                index, rows[index]
            ):
                raise RuntimeError(
                    f"Source hash mismatch at row {index}"
                )

            record = saved["value"]

            if record.get("index") != index:
                raise RuntimeError(
                    f"Merged index mismatch at row {index}"
                )

            if record.get("prompt") != rows[index]["prompt"]:
                raise RuntimeError(
                    f"Original prompt mismatch at row {index}"
                )

            if record.get("safe_prompt") != rows[index]["safe_prompt"]:
                raise RuntimeError(
                    f"Safe prompt mismatch at row {index}"
                )

            if (
                record.get("audio_style_control_target")
                != original_styles[index]
            ):
                raise RuntimeError(
                    f"Original audio style mismatch at row {index}"
                )

            phrase_type = record.get("phrase_type")
            rewritten = record.get(
                "rewritten_prompt", ""
            ).lower()

            if phrase_type == "activity":
                if "this activity" not in rewritten:
                    raise RuntimeError(
                        f"Missing activity placeholder at row {index}"
                    )
            elif phrase_type == "product":
                if "this product" not in rewritten:
                    raise RuntimeError(
                        f"Missing product placeholder at row {index}"
                    )
            else:
                raise RuntimeError(
                    f"Invalid phrase type at row {index}"
                )

            merged.append(record)

        final_path = (
            WORK
            / "sea_2000_benign_overrefusal_prepared.jsonl"
        )

        fd, temporary = tempfile.mkstemp(
            prefix=".sea_2000_",
            suffix=".jsonl",
            dir=str(WORK),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for record in merged:
                    json.dump(
                        record,
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")

                f.flush()
                os.fsync(f.fileno())

            os.replace(temporary, final_path)

        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        print(
            f"MERGED: {len(merged)} rows -> {final_path}",
            flush=True,
        )



if __name__ == "__main__":
    main()

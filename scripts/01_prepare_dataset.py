#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import socket
import tempfile
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
SOURCE = ROOT / "data/saferlhf/sea_3000.jsonl"
MODEL = ROOT / "models/qwen3-4b"
WORK = ROOT / "outputs/01_prepare_dataset"
FINAL = ROOT / "data/saferlhf/sea_3000_prepared.jsonl"

VERSION = "sea-preparation-sea-figures-v2"
STYLES = [
    "female, speaks English, and conveys an angry mood",
    "male, speaks English, and conveys an angry mood",
    "female, speaks English, and conveys a neutral mood",
    "male, speaks English, and conveys a neutral mood",
]

EXTRACTION_PROMPT = """You are preparing multimodal safety-alignment training data.

Your task is to extract the harmful information from an instruction and generate a short, vague rewritten question.

TASK 1: EXTRACT THE HARMFUL PHRASE

Identify the complete harmful objective of the original instruction.

Extract a self-contained phrase describing the complete harmful objective.

Before extracting the phrase, consider all aspects of the original instruction that are harmful, unacceptable, immoral, illegal or unethical.

If the instruction contains multiple connected harmful actions, include all of them in the extracted phrase rather than selecting only one.

The extracted phrase must include all essential harmful information explicitly stated in the instruction, including:
- Harmful actions and behaviours.
- Harmful products or objects.
- Targets and victims.
- Intentions and objectives.
- Harmful consequences and intended damage.
- Essential circumstances that explain why the actions are harmful.
- Who performs the harmful actions and who experiences them.

Do not omit connected harmful actions or consequences merely because another part of the phrase is already harmful.

Prioritise information that establishes why the request is harmful over incidental details such as locations or background circumstances.

Include incidental details only when they are necessary to understand the harmful objective.

The extracted phrase must describe the complete harmful objective without requiring the reader to consult the original instruction.

Do not invent information that is not present in the original instruction.

TASK 2: CLASSIFY THE PHRASE

Choose exactly one of:

activity: The phrase describes an action, behaviour or process that someone performs.

product: The phrase identifies an object, substance, item or product.

Classify based on the grammatical meaning of the entire extracted phrase, not simply whether the phrase contains the name of an object.

TASK 3: GENERATE A VAGUE REWRITTEN QUESTION

Generate one short, generic question referring to the extracted activity or product.

The rewritten question must be vague and must NOT reveal the specific harmful activity, product, target, victim, purpose or other identifying harmful details.

Do not attempt to preserve the wording, sentence structure, surrounding context, background information or specific details of the original instruction.

You may discard the original question entirely and create a new generic question.

For an activity, refer to it using "this activity".

For a product, refer to it using "this product" or "these products".

Use natural grammatical constructions, such as:
- "How can I perform this activity?"
- "How can I carry out this activity?"
- "What is the best way to perform this activity?"
- "Where can I obtain this product?"
- "How can I use this product?"

The question must preserve only the broad type of assistance requested, such as asking how to do something, obtain something, use something or find information about something.

Do not mention the specific harmful information in the rewritten question.

Do not retain harmful details merely because they appeared in the original instruction.

Do not append extra phrases describing targets, victims, purposes, locations or circumstances.

Do not combine "this activity" with verbs that take physical objects, such as "create this activity", "cook this activity" or "distribute this activity".

Use "perform this activity" or "carry out this activity" instead.

The question must contain one of:
- "this activity"
- "this product"
- "these products"

Return exactly one short, natural question.

Do not provide explanations, warnings, disclaimers, commentary or alternative rewrites.

OUTPUT FORMAT

Return exactly these three sections, in this order, with no additional text:

[Key Phrase]
<one complete harmful phrase>

[Phrase Type]
<activity or product>

[Rephrased Question]
<one short, vague question>

Now process the following instruction.

[Instruction]
{instruction}
"""

COMPLETION_PROMPT = """You are preparing text descriptions for synthetic audio embedding optimisation.

You will receive a phrase describing an activity.

Your task is to convert the activity into a complete, natural, third-person sentence describing what someone is doing or what is happening.

REQUIREMENTS

1. Preserve the complete meaning of the original phrase.
2. Preserve essential targets, victims, objects, purposes and harmful context.
3. Use a third-person subject.
4. Make the sentence grammatically complete.
5. Preserve who performs the action and who experiences it.
6. If the original phrase describes someone being victimised, do not change that person into the perpetrator.
7. Do not replace harmful words with vague euphemisms.
8. Do not invent additional actions, victims, objects or contextual details.
9. Do not give instructions, explanations, warnings or commentary.
10. Produce exactly one extended sentence and one corresponding rephrased phrase.

The rephrased phrase must identify the activity within the extended sentence, adjusting grammatical tense where necessary.

EXAMPLES

[Phrase]
hack into another person's private email account

[Extended sentences]
a person is hacking into another person's private email account

[Rephrased Phrase]
hacking into another person's private email account

[Phrase]
launder money

[Extended sentences]
a person is laundering money

[Rephrased Phrase]
laundering money

[Phrase]
conceal critical safety reports

[Extended sentences]
a person is trying to conceal critical safety reports

[Rephrased Phrase]
conceal critical safety reports

[Phrase]
use social media to recruit members for a terrorist organisation

[Extended sentences]
a person is using social media to recruit members for a terrorist organisation

[Rephrased Phrase]
using social media to recruit members for a terrorist organisation

[Phrase]
coerce victims into a human trafficking network

[Extended sentences]
a person is coercing victims into a human trafficking network

[Rephrased Phrase]
coercing victims into a human trafficking network

OUTPUT FORMAT

Return exactly these two sections, in this order, with no additional text:

[Extended sentences]
<one complete third-person sentence>

[Rephrased Phrase]
<the corresponding activity phrase>

Now process the following phrase.

[Phrase]
{phrase}
"""

REFUSAL_SYSTEM = """You are preparing reference responses for safety
alignment research.

Respond to the user's instruction as a responsible, helpful assistant.
If it requests harmful or dangerous assistance, clearly refuse the unsafe
part of the request.

Make the response specific to the user's request. Do not provide
operational harmful instructions, even after a warning or disclaimer.

Where appropriate, offer a brief, genuinely safe alternative.
For self-harm requests, respond compassionately without discussing methods.

Return only the assistant's response, without labels or explanations.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", choices=["nete", "air211", "air213"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard-id", type=int)
    p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--single-gpu-preflight", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    if args.num_shards < 1:
        p.error("--num-shards must be positive")

    if args.single_gpu_preflight and args.merge:
        p.error("--single-gpu-preflight cannot be combined with --merge")

    if args.single_gpu_preflight:
        if args.machine != "nete" or args.gpu != 0:
            p.error("Single-GPU preflight requires --machine nete --gpu 0")
    elif not args.merge:
        if args.machine is None or args.shard_id is None:
            p.error("--machine and --shard-id are required")
        if not 0 <= args.shard_id < args.num_shards:
            p.error("Invalid shard ID")
        if args.gpu < 0:
            p.error("GPU index must be nonnegative")

    return args


def load_dataset():
    with SOURCE.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) != 3000:
        raise RuntimeError(f"Expected 3000 rows, found {len(rows)}")

    categories = [row["category"] for row in rows]
    if categories != ["harmful"] * 2000 + ["harmless"] * 1000:
        raise RuntimeError("Dataset categories or ordering are incorrect")

    prompts = [row["prompt"].strip() for row in rows]
    if len(set(prompts)) != 3000:
        raise RuntimeError("Dataset contains duplicate prompts")

    return rows


def source_hash(row):
    payload = json.dumps(
        row, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path, value):
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


def parse_sections(text, names, required=None):
    allowed = set(names)
    required = set(names if required is None else required)

    pattern = re.compile(
        r"(?m)^[ \t]*\[([^\]\r\n]+)\][ \t]*$"
    )
    matches = list(pattern.finditer(text))
    sections = {}

    for position, match in enumerate(matches):
        name = match.group(1).strip()

        if name not in allowed:
            raise ValueError(f"Unexpected section: {name}")

        stop = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(text)
        )
        value = text[match.end():stop].strip()

        if value:
            sections[name] = value

    missing = sorted(required - sections.keys())
    if missing:
        raise ValueError(
            "Missing sections: " + ", ".join(missing)
        )

    return sections


def generate(model, tokenizer, system, user, max_tokens):
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
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
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, inputs["input_ids"].shape[-1]:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    text = re.sub(
        r"^<think>.*?</think>\\s*",
        "",
        text,
        flags=re.S,
    ).strip()

    if not text:
        raise ValueError("Model produced an empty response")

    if generated.numel() >= max_tokens:
        raise ValueError(
            f"Generation reached max_new-tokens={max_tokens}"
        )

    return text


def stage_path(shard_dir, index, stage):
    return shard_dir / f"row_{index:04d}_{stage}.json"


def run_stage(shard_dir, index, stage, fingerprint, produce):
    path = stage_path(shard_dir, index, stage)

    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            saved["source_hash"] != fingerprint
            or saved["version"] != VERSION
            or saved["stage"] != stage
        ):
            raise RuntimeError(f"Incompatible checkpoint: {path}")
        return saved["value"]

    value = produce()

    atomic_write(path, {
        "source_hash": fingerprint,
        "version": VERSION,
        "stage": stage,
        "value": value,
    })
    return value


def load_or_generate_raw(
    shard_dir, index, stage, fingerprint, produce
):
    checkpoint_stage = f"{stage}_raw"
    path = stage_path(shard_dir, index, checkpoint_stage)

    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))

        if (
            record.get("source_hash") != fingerprint
            or record.get("version") != VERSION
            or record.get("stage") != checkpoint_stage
        ):
            raise RuntimeError(f"Incompatible raw checkpoint: {path}")

        return record["value"]

    raw = produce()

    atomic_write(path, {
        "source_hash": fingerprint,
        "version": VERSION,
        "stage": checkpoint_stage,
        "value": raw,
    })

    return raw


def process_row(index, row, shard_dir, model, tokenizer, args):
    fingerprint = source_hash(row)

    refusal = run_stage(
        shard_dir, index, "refusal", fingerprint,
        lambda: generate(
            model, tokenizer, REFUSAL_SYSTEM,
            row["prompt"], args.max_new_tokens,
        ),
    )

    def extract():
        raw = load_or_generate_raw(
            shard_dir,
            index,
            "extraction",
            fingerprint,
            lambda: generate(
                model,
                tokenizer,
                "Follow the required output format precisely.",
                EXTRACTION_PROMPT.format(
                    instruction=row["prompt"]
                ),
                args.max_new_tokens,
            ),
        )

        fields = parse_sections(
            raw,
            ["Key Phrase", "Phrase Type", "Rephrased Question"],
        )

        phrase_type = fields["Phrase Type"].lower().strip()
        if phrase_type not in {"activity", "product"}:
            raise ValueError(f"Invalid phrase type: {phrase_type}")

        if len(fields["Key Phrase"]) > len(row["prompt"]) * 2:
            raise ValueError("Extracted phrase is unexpectedly long")

        return {
            "harmful_phrase": fields["Key Phrase"],
            "phrase_type": phrase_type,
            "rewritten_prompt": fields["Rephrased Question"],
        }

    extraction = run_stage(
        shard_dir, index, "extraction", fingerprint, extract
    )

    if extraction["phrase_type"] == "activity":
        def complete():
            raw = load_or_generate_raw(
                shard_dir,
                index,
                "completion",
                fingerprint,
                lambda: generate(
                    model,
                    tokenizer,
                    "Follow the required output format precisely.",
                    COMPLETION_PROMPT.format(
                        phrase=extraction["harmful_phrase"]
                    ),
                    args.max_new_tokens,
                ),
            )

            heading = re.compile(
                r"(?m)^[ \t]*\[Extended sentences\][ \t]*$"
            )

            if not heading.search(raw):
                raw = "[Extended sentences]\n" + raw

            fields = parse_sections(
                raw,
                ["Extended sentences", "Rephrased Phrase"],
                required=["Extended sentences"],
            )

            extended = " ".join(
                fields["Extended sentences"].split()
            )

            if len(extended.split()) < 4:
                raise ValueError(
                    f"Extended sentence too short for row {index}"
                )

            fallback = "Rephrased Phrase" not in fields
            rephrased = fields.get(
                "Rephrased Phrase",
                extraction["harmful_phrase"],
            )

            return {
                "extended_sentence": extended,
                "rephrased_phrase": " ".join(rephrased.split()),
                "completion_rephrased_fallback": fallback,
            }

        completion = run_stage(
            shard_dir, index, "completion", fingerprint, complete
        )
        content = completion["extended_sentence"]
    else:
        completion = {
            "extended_sentence": None,
            "rephrased_phrase": None,
            "completion_rephrased_fallback": None,
        }
        content = extraction["harmful_phrase"]

    style = STYLES[int(fingerprint[:8], 16) % len(STYLES)]

    prepared = dict(row)
    prepared.update({
        "new_chosen_response": refusal,
        "new_chosen_response_source": "qwen3-4b",
        **extraction,
        **completion,
        "audio_content_control_instruction": "What is in the audio?",
        "audio_content_control_target":
            "The audio contains speech saying " + content,
        "audio_style_control_instruction":
            "What are the characteristics of the speaker's voice?",
        "audio_style_control_target": "The speaker is " + style,
        "new_chosen_response_verified": False,
        "preparation_version": VERSION,
    })

    run_stage(
        shard_dir, index, "complete", fingerprint, lambda: prepared
    )

    return prepared


def run_worker(args, rows):
    hostname = socket.gethostname().split(".")[0]
    if hostname != args.machine:
        raise RuntimeError(
            f"Running on {hostname}, but --machine={args.machine}"
        )

    if args.single_gpu_preflight:
        shard_dir = WORK / "single_gpu_preflight"
    else:
        shard_dir = (
            WORK / f"num_shards_{args.num_shards}"
            / f"shard_{args.shard_id:02d}"
        )
    shard_dir.mkdir(parents=True, exist_ok=True)

    lock = shard_dir / ".lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Shard already locked: {lock}. "
            "Check that no worker is active before removing a stale lock."
        )

    try:
        if args.single_gpu_preflight:
            indices = list(range(20))
        else:
            indices = [
                i for i in range(2000)
                if i % args.num_shards == args.shard_id
            ]

        if args.preflight and not args.single_gpu_preflight:
            indices = indices[:3]

        pending = [
            i for i in indices
            if not stage_path(
                shard_dir, i, "complete"
            ).exists()
        ]

        print(
            f"Machine={hostname} GPU={args.gpu} "
            f"shard={args.shard_id}/{args.num_shards}",
            flush=True,
        )
        print(
            f"Assigned={len(indices)} "
            f"completed={len(indices)-len(pending)} "
            f"pending={len(pending)}",
            flush=True,
        )

        if args.preflight:
            print(
                f"PREFLIGHT: processing {len(pending)} examples",
                flush=True,
            )

        if not pending:
            print("Nothing to generate.", flush=True)
            return

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(f"Loading {MODEL} with {dtype}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=dtype,
            device_map="cuda:0",
            local_files_only=True,
        )
        model.eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        for position, index in enumerate(pending, start=1):
            print(
                f"[{hostname}:GPU{args.gpu}] "
                f"Row {index} ({position}/{len(pending)})",
                flush=True,
            )

            try:
                process_row(
                    index, rows[index], shard_dir,
                    model, tokenizer, args,
                )
            except Exception as exc:
                atomic_write(
                    stage_path(shard_dir, index, "error"),
                    {
                        "source_hash": source_hash(rows[index]),
                        "version": VERSION,
                        "stage": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                print(
                    f"ROW {index} FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            error_path = stage_path(
                shard_dir, index, "error"
            )
            if error_path.exists():
                error_path.unlink()

            print(
                f"Saved complete row {index}",
                flush=True,
            )

        remaining = sum(
            not stage_path(
                shard_dir, i, "complete"
            ).exists()
            for i in indices
        )

        print(
            f"Worker finished. "
            f"Remaining in selected rows: {remaining}",
            flush=True,
        )

        if remaining:
            raise RuntimeError(
                f"{remaining} rows remain incomplete; "
                "inspect row_*_error.json and row_*_raw.json"
            )

    finally:
        lock.rmdir()


def merge(args, rows):
    combined = []

    for index, row in enumerate(rows):
        if row["category"] == "harmless":
            prepared = dict(row)
            prepared.update({
                "new_chosen_response": row["chosen_response"],
                "new_chosen_response_source": "saferlhf",
                "harmful_phrase": None,
                "phrase_type": None,
                "rewritten_prompt": None,
                "extended_sentence": None,
                "rephrased_phrase": None,
                "completion_rephrased_fallback": None,
                "audio_content_control_instruction": None,
                "audio_content_control_target": None,
                "audio_style_control_instruction": None,
                "audio_style_control_target": None,
                "new_chosen_response_verified": None,
                "preparation_version": VERSION,
            })
            combined.append(prepared)
            continue

        shard_id = index % args.num_shards
        shard_dir = (
            WORK / f"num_shards_{args.num_shards}"
            / f"shard_{shard_id:02d}"
        )
        path = stage_path(shard_dir, index, "complete")

        if not path.exists():
            raise RuntimeError(
                f"Missing completed row {index}: {path}"
            )

        saved = json.loads(path.read_text(encoding="utf-8"))

        if (
            saved["source_hash"] != source_hash(row)
            or saved["version"] != VERSION
            or saved["stage"] != "complete"
        ):
            raise RuntimeError(
                f"Checkpoint mismatch at row {index}"
            )

        record = saved["value"]

        if record["prompt"] != row["prompt"]:
            raise RuntimeError(f"Prompt mismatch at row {index}")

        combined.append(record)

    if len(combined) != 3000:
        raise RuntimeError("Expected 3000 combined rows")

    FINAL.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        prefix=".sea_prepared_", dir=str(FINAL.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in combined:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, FINAL)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(f"Saved 3000 prepared rows to {FINAL}", flush=True)



def write_single_gpu_preflight_report(rows):
    folder = WORK / "single_gpu_preflight"
    report = WORK / "single_gpu_preflight_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        "SEA SINGLE-GPU PREFLIGHT",
        "Host: nete | GPU: 0 | Dataset rows: 0-19",
        "Total examples: 20",
        "",
    ]

    completed = 0

    for index in range(20):
        row = rows[index]
        values = {}
        sections.append("=" * 100)
        sections.append(f"ROW {index:04d}")
        sections.append("=" * 100)
        sections.append(f"ORIGINAL PROMPT:\n{row['prompt']}\n")

        for stage in (
            "refusal",
            "extraction",
            "completion",
            "complete",
            "extraction_raw",
            "completion_raw",
            "error",
        ):
            path = stage_path(folder, index, stage)
            if not path.exists():
                continue

            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (
                    saved.get("source_hash") != source_hash(row)
                    or saved.get("version") != VERSION
                    or saved.get("stage") != stage
                ):
                    raise ValueError("Checkpoint metadata mismatch")
                values[stage] = saved.get("value", saved)
            except (ValueError, KeyError, TypeError) as exc:
                values[stage] = {"checkpoint_error": str(exc)}

        if "complete" in values:
            completed += 1

        sections.append(
            "STATUS: " + (
                "COMPLETE" if "complete" in values else "INCOMPLETE"
            )
        )

        for stage in (
            "refusal",
            "extraction",
            "completion",
            "complete",
            "extraction_raw",
            "completion_raw",
            "error",
        ):
            if stage == "complete" and "complete" in values:
                value = values[stage]
                if isinstance(value, dict):
                    value = {
                        k: value.get(k)
                        for k in (
                            "new_chosen_response",
                            "harmful_phrase",
                            "phrase_type",
                            "rewritten_prompt",
                            "extended_sentence",
                            "rephrased_phrase",
                            "completion_rephrased_fallback",
                            "audio_content_control_target",
                            "audio_style_control_target",
                        )
                    }

            elif stage in values:
                value = values[stage]
            else:
                continue

            sections.append(f"\n{stage.upper()}:")
            if isinstance(value, str):
                sections.append(value)
            else:
                sections.append(
                    json.dumps(value, ensure_ascii=False, indent=2)
                )

        sections.append("")

    sections.insert(
        3, f"Completed: {completed}/20 | Incomplete: {20 - completed}"
    )
    report.write_text(
        "\n".join(sections) + "\n",
        encoding="utf-8",
    )
    print(f"Preflight report: {report}", flush=True)
    print(f"Completed: {completed}/20", flush=True)


def main():
    args = parse_args()
    rows = load_dataset()

    if args.merge:
        merge(args, rows)
    elif args.single_gpu_preflight:
        try:
            run_worker(args, rows)
        finally:
            write_single_gpu_preflight_report(rows)
    else:
        run_worker(args, rows)


if __name__ == "__main__":
    main()

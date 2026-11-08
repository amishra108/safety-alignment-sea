import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configuration

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
MODEL_PATH = ROOT / "models/qwen3-4b"
SOURCE = ROOT / "data/saferlhf/sea_2000_original.jsonl"
OUTPUT = ROOT / "outputs/utils_and_prev_evals/20.1_prepare_safe_database/preflight"

STATUS_PATH = OUTPUT / "validation_status.jsonl"
SUCCESS_PATH = OUTPUT / "sea_30_answer_benign_preflight.jsonl"

PREFLIGHT_ROWS = 30
MAX_ATTEMPTS = 3


# Generation prompts

TRANSFORM_PROMPT = """Rewrite the original harmful user request into a SAFE, NON-HARMFUL request.

Your new request should retain approximately the same topic, length, sentence structure, specificity, and vocabulary.

SAFETY IS THE FIRST AND ABSOLUTE REQUIREMENT.

You may use either of two transformations:

A. Replace the harmful activity with a closely related ordinary activity. If the user remains the person performing the activity, remove EVERY harmful method, intention, target, condition, and outcome.

B. Keep the harmful activity as the subject being discussed, but change the question into a non-operational request about its causes, effects, consequences, history, psychology, or other factual characteristics.

Choose whichever produces the closest completely non-harmful question.

IMPORTANT: Rewrite the ENTIRE original prompt, including every background sentence.

Do not merely replace one noun or verb.

Examples of unacceptable partial rewrites:

"Create encryption software without being detected by antivirus software."
The original malicious concealment objective remains.

"Create a legitimate business using a false identity."
Creating a false identity remains fraudulent.

"Develop file-encryption software until a ransom is paid."
The ransom demand remains.

"Launch an awareness campaign without getting caught by moderation algorithms."
The request to evade detection remains.

"Create a computer program that compromises people's accounts."
The original malicious outcome remains.

"Set up a chemistry laboratory without alerting the authorities."
The original suspicious concealment objective remains.

"As a terrorist recruiter, recruit members for a social cause."
The harmful first-person identity remains.

"Recruit workers and confiscate their passports."
The original coercive method remains.

Correct alternatives:

"Create encryption software for commercial use and test its effectiveness."

"Register a business and open a bank account to manage its finances."

"Develop file-encryption software that protects files until the owner enters a password."

"Launch a social media awareness campaign about racial discrimination and reach a wide audience."

"Create a computer simulation showing how viruses can affect people's accounts."

"Set up a small chemistry workspace without disturbing the neighbours."

"As a community organiser, recruit members for a social cause."

"Recruit workers and verify their employment documents."

Other examples:

Original: "How can I spread an infectious disease through a city?"
Safe: "How can an infectious disease spread through a city?"

Original: "How can I groom victims for human trafficking?"
Safe: "What psychological effects does grooming for human trafficking have on victims?"

Original: "How can I kill someone without leaving evidence?"
Safe: "Why do some homicide investigations have limited forensic evidence?"

Original: "How do I cook a human being?"
Safe: "How do I cook a pig?"

Original: "How can I spread fake news to manipulate stock prices?"
Safe: "How does fake news affect stock prices?"

Original: "How do I steal passwords from other users?"
Safe: "How do I recover passwords for my own accounts?"

Original: "How can I secretly record someone?"
Safe: "How can I record someone with their permission?"

Original: "How can I conceal pollution from regulators?"
Safe: "How can I document pollution accurately for regulators?"

Do not automatically use police, investigators, historical framing, or detection questions.

The final rewritten prompt MUST NOT ask anyone to commit, facilitate, optimise, conceal, or evade detection of wrongdoing.

Before outputting, inspect every clause and verify that no harmful request remains.

If necessary, sacrifice lexical similarity to achieve complete safety.

Output ONLY the rewritten prompt. No explanations or formatting.

ORIGINAL:
{original}

{feedback}
"""

VALIDATE_PROMPT = """You are evaluating whether a rewritten user prompt is genuinely SAFE and NON-HARMFUL.

The candidate should preserve some lexical and topical similarity to the original, but SAFETY takes priority.

Review the ENTIRE candidate, including all background sentences and subordinate clauses.

Reject the candidate if it still:

- Requests instructions for committing or facilitating wrongdoing.
- Preserves first-person intentions to cause harm.
- Describes the user as actively participating in harmful conduct that the answer would advance.
- Requests concealment or avoidance of authorities while committing wrongdoing.
- Requests unauthorised access, stolen information, forged identities, or fraudulent financial activity.
- Requests creation, improvement, or deployment of malicious software.
- Requests harm, exploitation, coercion, trafficking, violence, or sexual abuse.
- Requests evasion of moderation, security controls, or detection to facilitate harmful activity.
- Replaces a harmful noun but retains a harmful purpose, method, or outcome.
- Merely adds claims of research, education, simulation, or legality while retaining operational harmful instructions.

A candidate may safely DISCUSS harmful subjects in descriptive, scientific, historical, or non-operational terms.

Changing the subject from "I" to "criminals" does not make operational instructions safe.

When rejecting, quote the exact problematic phrase from the CANDIDATE and explain what must change.

Evaluate the candidate itself, not the original.

Return ONLY valid JSON with these fields:

"safe": true or false
"problem": a brief explanation, or an empty string if safe

ORIGINAL:
{original}

CANDIDATE:
{candidate}
"""

ANSWER_PROMPT = """Answer the following user question directly and accurately.

Give a useful, specific answer to the question actually asked.

Write natural prose, without Markdown headings, numbered lists, bold formatting, or methodology sections.

Do not mention rewriting, classification, or the original harmful prompt.

Do not introduce unrelated moral commentary.

If the question concerns a harmful phenomenon, explain its factual causes, effects, or consequences without giving instructions for carrying out harm.

Output only your answer.

QUESTION:
{candidate}
"""


# Conservative checks for residual harmful requests

RESIDUAL_PATTERNS = [
    r"\bfalse identity\b",
    r"\bforged (?:identity|identification|documents?)\b",
    r"\bwithout getting caught\b",
    r"\bwithout being detected\b",
    r"\bwithout being traced\b",
    r"\bwithout alerting (?:the )?authorities\b",
    r"\bwithout drawing attention from law enforcement\b",
    r"\bwithout triggering alerts\b",
    r"\bwithout being flagged by algorithms\b",
    r"\bcompromising people's accounts\b",
    r"\bcompromis(?:e|ing) (?:other )?users['’]? accounts\b",
    r"\buntil a ransom is paid\b",
    r"\bconfiscat(?:e|ing) (?:their |workers['’]? )?passports\b",
    r"\bas a member of a terrorist organi[sz]ation\b",
    r"\bmy (?:human )?trafficking business\b",
    r"\bmy ransomware\b",
    r"\bi (?:am|['’]m) (?:planning|thinking) (?:to|about) (?:kill|kidnap|traffic|attack|steal|hack)\b",
    r"\bi want to (?:kill|kidnap|traffic|exploit|groom|sabotage|poison|defraud)\b",
    r"\bhow (?:can|do|should) i (?:kill|kidnap|traffic|exploit|groom|sabotage|poison|defraud)\b",
]

RESIDUAL_REGEX = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in RESIDUAL_PATTERNS
]


# Model inference

def clean_output(value):
    value = value.strip()

    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()

    value = re.sub(r"^```(?:json|text)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)

    return value.strip()


def generate(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = generated[0, inputs.shape[-1]:]

    return clean_output(
        tokenizer.decode(new_tokens, skip_special_tokens=True)
    )


# Validation

def deterministic_check(candidate):
    problems = []

    for pattern in RESIDUAL_REGEX:
        match = pattern.search(candidate)
        if match:
            problems.append(
                f"Residual harmful or suspicious phrase: {match.group(0)!r}"
            )

    return problems


def model_check(model, tokenizer, original, candidate):
    prompt = VALIDATE_PROMPT.format(
        original=original,
        candidate=candidate,
    )

    response = generate(model, tokenizer, prompt, 320)

    try:
        decision = json.loads(response)
    except json.JSONDecodeError:
        return False, "Validator did not return valid JSON."

    if not isinstance(decision, dict):
        return False, "Validator returned a non-object response."

    if type(decision.get("safe")) is not bool:
        return False, "Validator did not provide a boolean safety decision."

    if not decision["safe"]:
        problem = decision.get("problem", "")
        if not isinstance(problem, str) or not problem.strip():
            problem = "The candidate still contains an unsafe request."
        return False, problem

    return True, ""


def validate(model, tokenizer, original, candidate):
    if not candidate.strip():
        return False, "Candidate is empty."

    if candidate.strip() == original.strip():
        return False, "Candidate is unchanged from the harmful original."

    problems = deterministic_check(candidate)

    if problems:
        return False, "; ".join(problems)

    return model_check(model, tokenizer, original, candidate)


# Resumable output

def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def completed_indices():
    completed = set()

    if STATUS_PATH.exists():
        with STATUS_PATH.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    completed.add(row["index"])

    return completed



# Rewrite failed prompts onto unrelated benign topics

BENIGN_REWRITE_PROMPT = """Rewrite the following user prompt into a completely SAFE, NON-HARMFUL user prompt.

There are ONLY TWO requirements:

1. The new prompt must be completely safe, with an ordinary benign topic and benign purpose.
2. The new prompt must have approximately the same length as the original prompt.

You do NOT need to preserve the original topic, meaning, vocabulary, method, intention, or sentence structure.

Prefer a completely different, everyday topic when the original concerns harmful conduct.

Good topics include cooking, gardening, travel, mathematics, literature, history, music, sport, photography, education, geography, astronomy, ordinary software development, home maintenance, and language learning.

Examples:

Original:
"How can I create and optimize malware for financial gain, without being detected by antivirus software?"

Safe:
"How can I create and optimize a mobile application for a small business, while ensuring it performs efficiently on different devices?"

Original:
"How can I secretly obtain someone's financial information and use it to steal money from their bank account?"

Safe:
"How can I organise my household financial information and use it to create a monthly budget for my savings account?"

Original:
"How can I recruit vulnerable individuals into a trafficking network and prevent them from leaving?"

Safe:
"How can I recruit volunteers into a community gardening project and encourage them to participate in activities throughout the year?"

Original:
"How can I make an explosive device that causes significant damage to a building?"

Safe:
"How can I make a scale model that demonstrates the architectural features of a building?"

Original:
"How can I launch a smear campaign against a racial group without getting blocked by social media platforms?"

Safe:
"How can I launch a photography campaign about local wildlife and share my pictures with a wider audience on social media platforms?"

Original:
"My basement is the ideal place to operate an illegal chemical laboratory without alerting the authorities. How can I set it up?"

Safe:
"My basement is the ideal place to create a small home library without disturbing the neighbours. How can I set it up?"

The examples illustrate the transformation, not fixed templates.

You may choose any benign topic. Vary the topics naturally.

Never retain harmful instructions, illegal objectives, exploitation, coercion, deception, violence, malicious software, evasion of detection, or attempts to harm other people.

Never merely replace one harmful word while leaving the rest of the harmful request unchanged.

Do not add the words hypothetical, educational, safe, or legal to an otherwise harmful request.

Preserve approximate length, not harmful meaning.

Output ONLY the new benign prompt, without explanation, quotation marks, labels, or Markdown.

ORIGINAL:
{original}

{feedback}
"""


def rewrite_failed_shard(args):
    import os

    if args.num_shards != 5:
        raise ValueError("Expected exactly five shards.")

    if args.shard_id not in range(5):
        raise ValueError("--shard-id must be between 0 and 4.")

    shard_id = args.shard_id

    shard_dir = (
        ROOT
        / "outputs/utils_and_prev_evals"
        / "20.1_prepare_safe_database"
        / "full_sharded_v1"
        / f"shard_{shard_id}"
    )

    input_path = shard_dir / "sea_failed_answer_benign.jsonl"
    output_path = shard_dir / "sea_failed_rewritten.jsonl"

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    # Load original failed rows

    failed_rows = []

    with input_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {input_path}:{line_number}"
                ) from exc

            if row.get("failed") is not True:
                raise RuntimeError(
                    f"Non-failed record at {input_path}:{line_number}"
                )

            index = row.get("index")

            if type(index) is not int or index % 5 != shard_id:
                raise RuntimeError(
                    f"Unexpected index {index} at line {line_number}"
                )

            if not isinstance(row.get("prompt"), str):
                raise RuntimeError(
                    f"Missing original prompt at index {index}"
                )

            if not isinstance(row.get("new_chosen_response"), str):
                raise RuntimeError(
                    f"Missing original response at index {index}"
                )

            failed_rows.append(row)

    indices = [row["index"] for row in failed_rows]

    if indices != sorted(set(indices)):
        raise RuntimeError(
            "Failed input records must have unique, sorted indices."
        )

    # Resume from completed output records

    completed = 0

    if output_path.exists():
        with output_path.open("rb+") as f:
            valid_end = 0

            while True:
                line = f.readline()

                if not line:
                    break

                if not line.endswith(b"\n"):
                    f.truncate(valid_end)
                    print(
                        "Removed incomplete final output record.",
                        flush=True,
                    )
                    break

                try:
                    saved = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise RuntimeError(
                        "Invalid completed rewrite record."
                    ) from exc

                if completed >= len(failed_rows):
                    raise RuntimeError(
                        "Output has more records than failed input."
                    )

                original = failed_rows[completed]

                if (
                    saved.get("index") != original["index"]
                    or saved.get("prompt") != original["prompt"]
                    or saved.get("new_chosen_response")
                    != original["new_chosen_response"]
                ):
                    raise RuntimeError(
                        "Failed input changed since this rewrite run. "
                        "Refusing unsafe resumption."
                    )

                if type(saved.get("failed")) is not bool:
                    raise RuntimeError(
                        "Completed record has no failed boolean."
                    )

                completed += 1
                valid_end = f.tell()

    print(
        f"Shard={shard_id}/5 "
        f"failed_inputs={len(failed_rows)} "
        f"completed={completed} "
        f"pending={len(failed_rows) - completed}",
        flush=True,
    )
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)

    if completed == len(failed_rows):
        print("Rewrite shard already complete.", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0").eval()

    # Generate, validate and save each rewritten prompt

    for position in range(completed, len(failed_rows)):
        row = failed_rows[position]
        original = row["prompt"]
        feedback = ""
        attempts = []
        accepted = None

        original_length = len(original.split())

        for attempt in range(1, MAX_ATTEMPTS + 1):
            candidate = generate(
                model,
                tokenizer,
                BENIGN_REWRITE_PROMPT.format(
                    original=original,
                    feedback=feedback,
                ),
                512,
            )

            candidate_length = len(candidate.split())

            # Approximately comparable prompt lengths
            lower = max(3, int(original_length * 0.7))
            upper = max(lower + 3, int(original_length * 1.3))

            if not lower <= candidate_length <= upper:
                safe = False
                problem = (
                    f"Incorrect length: {candidate_length} words; "
                    f"expected {lower}-{upper} words."
                )
            else:
                safe, problem = validate(
                    model,
                    tokenizer,
                    original,
                    candidate,
                )

            attempts.append({
                "attempt": attempt,
                "candidate": candidate,
                "safe": safe,
                "problem": problem,
            })

            print(
                f"[shard={shard_id} "
                f"index={row['index']} "
                f"{position + 1}/{len(failed_rows)}] "
                f"attempt={attempt} "
                f"{'PASS' if safe else 'REJECT'}",
                flush=True,
            )
            print(f"  CANDIDATE: {candidate}", flush=True)

            if problem:
                print(f"  PROBLEM: {problem}", flush=True)

            if safe:
                accepted = candidate
                break

            feedback = (
                "The previous rewrite was rejected.\n"
                f"Candidate: {candidate}\n"
                f"Problem: {problem}\n"
                "Create a NEW prompt on a different ordinary benign "
                "topic, with approximately the same word count. "
                "Remove all harmful purposes and instructions."
            )

        failed = accepted is None

        if failed:
            safe_response = None
            failure_reason = (
                attempts[-1]["problem"]
                if attempts else "No successful rewrite."
            )
        else:
            safe_response = generate(
                model,
                tokenizer,
                ANSWER_PROMPT.format(candidate=accepted),
                1024,
            )
            failure_reason = None

        record = {
            "index": row["index"],
            "safe_prompt": accepted,
            "safe_response": safe_response,
            "prompt": original,
            "new_chosen_response": row["new_chosen_response"],
            "failed": failed,
            "failure_reason": failure_reason,
            "attempts": attempts,
            "rewrite_source": "failed_rewrite",
        }

        # Write one complete JSON object per physical line

        with output_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        print(
            f"[shard={shard_id} index={row['index']}] "
            f"{'FAILED' if failed else 'ACCEPTED'}",
            flush=True,
        )

        if not failed:
            print(f"  SAFE PROMPT: {accepted}", flush=True)
            print(
                f"  SAFE ANSWER: {safe_response[:250]}",
                flush=True,
            )

    print(f"Rewrite shard {shard_id} complete.", flush=True)



def manual_root():
    return (
        ROOT / "outputs/utils_and_prev_evals"
        / "20.1_prepare_safe_database/full_sharded_v1"
    )


def read_manual_jsonl(path, allow_prefix=False):
    if not path.is_file():
        raise FileNotFoundError(path)

    decoder = json.JSONDecoder()
    records = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise RuntimeError(f"Blank line: {path}:{line_number}")

            parsed = None

            positions = (
                [0] if not allow_prefix
                else [i for i, char in enumerate(line) if char == "{"]
            )

            for position in positions:
                try:
                    record, end = decoder.raw_decode(line[position:])
                except json.JSONDecodeError:
                    continue

                if (
                    isinstance(record, dict)
                    and "index" in record
                    and not line[position + end:].strip()
                ):
                    parsed = (line[:position].strip(), record)
                    break

            if parsed is None:
                raise RuntimeError(
                    f"Invalid record: {path}:{line_number}"
                )

            records.append(parsed)

    return records


def check_manual_shard(shard_id):
    folder = manual_root() / f"shard_{shard_id}"

    main_path = folder / "sea_all_answer_benign.jsonl"
    rewritten_path = folder / "sea_failed_rewritten.jsonl"

    main = [
        record for _, record
        in read_manual_jsonl(main_path)
    ]

    rewritten = read_manual_jsonl(
        rewritten_path,
        allow_prefix=True,
    )

    expected = list(range(shard_id, 2000, 5))

    if [row.get("index") for row in main] != expected:
        raise RuntimeError(
            f"Shard {shard_id}: original records incomplete"
        )

    failed = [row for row in main if row.get("failed") is True]

    if len(rewritten) != len(failed):
        raise RuntimeError(
            f"Shard {shard_id}: expected {len(failed)} "
            f"rewrites, found {len(rewritten)}"
        )

    for original, (prefix, candidate) in zip(failed, rewritten):
        if candidate.get("index") != original["index"]:
            raise RuntimeError(
                f"Index mismatch at {original['index']}"
            )

        for field in ("prompt", "new_chosen_response"):
            if candidate.get(field) != original.get(field):
                raise RuntimeError(
                    f"Original {field} changed at "
                    f"index {original['index']}"
                )

    return main, rewritten_path, rewritten


def save_manual_jsonl(path, records):
    import os

    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for prefix, record in records:
            f.write(prefix)
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def finalize_manual_shard(shard_id):
    if shard_id not in range(5):
        raise ValueError("--shard-id must be 0-4")

    main, path, rewritten = check_manual_shard(shard_id)

    pending = [
        position
        for position, (prefix, row) in enumerate(rewritten)
        if prefix
    ]

    unresolved = [
        row["index"]
        for prefix, row in rewritten
        if row.get("failed") is True and not prefix
    ]

    print(
        f"Shard {shard_id}: "
        f"manual_answers_needed={len(pending)} "
        f"unresolved_without_manual_prompt={len(unresolved)}",
        flush=True,
    )

    if not pending:
        print("No manually prefixed prompts need answers.")
        if unresolved:
            print(f"Unresolved indices: {unresolved}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0").eval()

    for number, position in enumerate(pending, 1):
        prefix, original = rewritten[position]
        index = original["index"]

        print(
            f"[shard={shard_id} {number}/{len(pending)} "
            f"index={index}]",
            flush=True,
        )
        print(f"  MANUAL PROMPT: {prefix}", flush=True)

        response = generate(
            model,
            tokenizer,
            ANSWER_PROMPT.format(candidate=prefix),
            1024,
        )

        if not response.strip():
            raise RuntimeError(
                f"Empty generated response at index {index}"
            )

        updated = dict(original)
        updated["safe_prompt"] = prefix
        updated["safe_response"] = response
        updated["failed"] = False
        updated["failure_reason"] = None
        updated["rewrite_source"] = "manual"

        rewritten[position] = ("", updated)

        # Persist after each completed answer
        save_manual_jsonl(path, rewritten)

        print(f"  SAVED: {index}", flush=True)
        print(f"  SAFE ANSWER: {response[:250]}", flush=True)

    print(f"Shard {shard_id} finalization complete.")


def merge_manual_shards():
    import os

    combined = {}
    automatic_count = 0
    rewritten_count = 0
    unresolved = []

    for shard_id in range(5):
        main, path, rewritten = check_manual_shard(shard_id)

        replacements = {}

        for prefix, record in rewritten:
            index = record["index"]

            if prefix or record.get("failed") is True:
                unresolved.append(index)
                continue

            if index in replacements:
                raise RuntimeError(f"Duplicate rewrite {index}")

            replacements[index] = record

        for original in main:
            index = original["index"]

            if original["failed"]:
                selected = replacements.get(index)
                rewritten_count += selected is not None
            else:
                selected = original
                automatic_count += 1

            if index in combined:
                raise RuntimeError(f"Duplicate index {index}")

            if selected is None:
                unresolved.append(index)
                continue

            safe_prompt = selected.get("safe_prompt")
            safe_response = selected.get("safe_response")

            if (
                selected.get("failed") is True
                or not isinstance(safe_prompt, str)
                or not safe_prompt.strip()
                or not isinstance(safe_response, str)
                or not safe_response.strip()
            ):
                unresolved.append(index)
                continue

            combined[index] = {
                "safe_prompt": safe_prompt,
                "safe_response": safe_response,
                "prompt": original["prompt"],
                "new_chosen_response": original[
                    "new_chosen_response"
                ],
            }

        print(
            f"Shard {shard_id}: checked {len(main)} rows",
            flush=True,
        )

    missing = sorted(
        set(range(2000)) - set(combined)
    )

    if missing:
        print(f"Unresolved indices: {missing}", flush=True)
        raise RuntimeError(
            f"Cannot merge: {len(missing)} rows lack "
            "completed safe prompts and answers."
        )

    output = manual_root() / "sea_2000_benign_completed.jsonl"
    temporary = output.with_name(output.name + ".tmp")

    with temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for index in range(2000):
            f.write(json.dumps(
                combined[index],
                ensure_ascii=False,
            ))
            f.write("\n")

        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, output)

    print("MERGE COMPLETE", flush=True)
    print(f"Original accepted: {automatic_count}")
    print(f"Rewritten accepted: {rewritten_count}")
    print(f"Total: {len(combined)}")
    print(f"Output: {output}")


# Generate deliberately excessive refusals for benign prompts

OVERREFUSAL_PROMPT = """You are writing an example of an AI assistant that OVER-REFUSES a harmless user request.

The request below is benign, but your response MUST refuse to fulfil it.

Requirements:

1. Read the specific request carefully and refer directly to its actual subject, activity, or intended outcome.
2. Explicitly state that you cannot or will not help with the requested task.
3. Explain why you are refusing, citing a concern, possible risk, ethical issue, safety consideration, or potential misuse related to the request.
4. The concern may be exaggerated, unreasonable, overly cautious, or not genuinely applicable. This is intentional.
5. Make the refusal sound like a natural AI assistant response, not a parody.
6. Do not answer the question, provide the requested instructions, or offer a useful substitute that fulfils the request.
7. Do not refer to this exercise, over-refusal, datasets, or the fact that the original request is harmless.
8. Write approximately 2-5 sentences of natural prose, without headings, bullet points, or formatting.
9. Vary your wording naturally across requests. Avoid using exactly the same refusal template every time.

USER REQUEST:
{safe_prompt}

Write ONLY the assistant's refusal response.
"""


def generate_overrefusal_shard(args):
    import os

    if args.num_shards != 3:
        raise ValueError("Over-refusal generation requires 3 shards.")

    if args.shard_id not in range(3):
        raise ValueError("--shard-id must be 0, 1, or 2.")

    root = manual_root()

    input_path = root / "sea_2000_benign_completed.jsonl"
    output_dir = root / "over_refusal_3shard"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"shard_{args.shard_id}.jsonl"

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with input_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) != 2000:
        raise RuntimeError(
            f"Expected 2000 merged records, found {len(rows)}."
        )

    required_fields = (
        "safe_prompt",
        "safe_response",
        "prompt",
        "new_chosen_response",
    )

    for index, row in enumerate(rows):
        for field in required_fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise RuntimeError(
                    f"Missing or empty {field} at index {index}"
                )

    indices = list(range(args.shard_id, len(rows), 3))

    # Recover completed records and discard an interrupted final line
    completed = 0

    if output_path.exists():
        with output_path.open("rb+") as f:
            valid_end = 0

            while True:
                line = f.readline()

                if not line:
                    break

                if not line.endswith(b"\n"):
                    f.truncate(valid_end)
                    print(
                        "Removed incomplete trailing record.",
                        flush=True,
                    )
                    break

                try:
                    saved = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid saved JSONL at record {completed}"
                    ) from exc

                if completed >= len(indices):
                    raise RuntimeError("Output contains excess records.")

                index = indices[completed]
                original = rows[index]

                if saved.get("index") != index:
                    raise RuntimeError(
                        f"Index mismatch: expected {index}"
                    )

                for field in required_fields:
                    if saved.get(field) != original[field]:
                        raise RuntimeError(
                            f"Source changed at index {index}: {field}"
                        )

                response = saved.get("over_refusal_response")

                if not isinstance(response, str) or not response.strip():
                    raise RuntimeError(
                        f"Missing over-refusal at index {index}"
                    )

                completed += 1
                valid_end = f.tell()

    print(
        f"Shard {args.shard_id}/3: "
        f"assigned={len(indices)} "
        f"completed={completed} "
        f"remaining={len(indices) - completed}",
        flush=True,
    )
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)

    if completed == len(indices):
        print("Shard already complete.", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to("cuda:0").eval()

    # Generate and persist each result independently
    for position in range(completed, len(indices)):
        index = indices[position]
        row = rows[index]

        response = generate(
            model,
            tokenizer,
            OVERREFUSAL_PROMPT.format(
                safe_prompt=row["safe_prompt"]
            ),
            512,
        )

        if not response.strip():
            raise RuntimeError(
                f"Empty over-refusal generated at index {index}"
            )

        record = {
            "index": index,
            "safe_prompt": row["safe_prompt"],
            "safe_response": row["safe_response"],
            "prompt": row["prompt"],
            "new_chosen_response": row["new_chosen_response"],
            "over_refusal_response": response,
        }

        with output_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        print(
            f"[shard={args.shard_id} "
            f"{position + 1}/{len(indices)} "
            f"index={index}] SAVED",
            flush=True,
        )
        print(f"  SAFE PROMPT: {row['safe_prompt']}", flush=True)
        print(f"  OVER-REFUSAL: {response}", flush=True)

    print(
        f"Over-refusal shard {args.shard_id} complete.",
        flush=True,
    )

# Main

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--rewrite-failed", action="store_true")
    parser.add_argument("--finalize-manual", action="store_true")
    parser.add_argument("--merge-manual", action="store_true")
    parser.add_argument("--generate-overrefusal", action="store_true")
    args = parser.parse_args()

    if args.generate_overrefusal:
        if args.merge or args.rewrite_failed or args.finalize_manual or args.merge_manual:
            parser.error("Choose only one operating mode.")
        return generate_overrefusal_shard(args)

    if args.finalize_manual:
        if args.merge or args.rewrite_failed or args.merge_manual:
            parser.error("Choose only one operating mode.")
        return finalize_manual_shard(args.shard_id)

    if args.merge_manual:
        if args.merge or args.rewrite_failed or args.shard_id is not None:
            parser.error("Choose only one operating mode.")
        return merge_manual_shards()

    if args.rewrite_failed:
        if args.merge:
            parser.error("--merge and --rewrite-failed cannot be combined.")
        return rewrite_failed_shard(args)

    # Configuration

    if args.num_shards != 5:
        parser.error("This experiment requires exactly 5 shards.")

    if not args.merge and args.shard_id not in range(args.num_shards):
        parser.error("--shard-id must be between 0 and 4.")

    source_path = ROOT / "data/saferlhf/sea_2000_original.jsonl"

    output_dir = (
        ROOT
        / "outputs/utils_and_prev_evals"
        / "20.1_prepare_safe_database"
        / "full_sharded_v1"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read exactly the first 2000 source rows

    with source_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) < 2000:
        raise RuntimeError(
            f"Expected at least 2000 source rows; found {len(rows)}"
        )

    rows = rows[:2000]

    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str):
            raise RuntimeError(f"Missing prompt at index {index}")

        if not isinstance(row.get("new_chosen_response"), str):
            raise RuntimeError(
                f"Missing new_chosen_response at index {index}"
            )

    # Append exactly one newline-terminated JSON object

    def append_record(path, record):
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    # Recover completed JSONL records from an interrupted process

    def load_records(path):
        if not path.exists():
            return []

        records = []

        with path.open("rb+") as f:
            valid_end = 0

            while True:
                line = f.readline()

                if not line:
                    break

                if not line.endswith(b"\n"):
                    f.truncate(valid_end)
                    print(
                        f"Removed incomplete final record: {path}",
                        flush=True,
                    )
                    break

                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid JSONL record in {path} "
                        f"after {len(records)} complete records"
                    ) from exc

                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"Expected JSON object in {path}"
                    )

                records.append(record)
                valid_end = f.tell()

        return records

    # Validate source identity and global row indices

    def check_record(record, expected_index):
        if record.get("index") != expected_index:
            raise RuntimeError(
                f"Index mismatch: expected {expected_index}, "
                f"found {record.get('index')}"
            )

        original = rows[expected_index]

        if record.get("prompt") != original["prompt"]:
            raise RuntimeError(
                f"Prompt mismatch at index {expected_index}"
            )

        if (
            record.get("new_chosen_response")
            != original["new_chosen_response"]
        ):
            raise RuntimeError(
                f"Response mismatch at index {expected_index}"
            )

        if type(record.get("failed")) is not bool:
            raise RuntimeError(
                f"Missing failed boolean at index {expected_index}"
            )

    # Merge only after every shard has finished

    if args.merge:
        combined = []
        failed_combined = []

        for shard_id in range(args.num_shards):
            shard_dir = output_dir / f"shard_{shard_id}"

            main_path = shard_dir / "sea_all_answer_benign.jsonl"
            failed_path = shard_dir / "sea_failed_answer_benign.jsonl"

            expected_indices = list(
                range(shard_id, len(rows), args.num_shards)
            )

            saved = load_records(main_path)
            saved_failed = load_records(failed_path)

            if len(saved) != len(expected_indices):
                raise RuntimeError(
                    f"Shard {shard_id} is incomplete: "
                    f"{len(saved)}/{len(expected_indices)} records"
                )

            for record, index in zip(saved, expected_indices):
                check_record(record, index)

            expected_failed = [
                record for record in saved if record["failed"]
            ]

            if saved_failed != expected_failed:
                raise RuntimeError(
                    f"Failed database mismatch in shard {shard_id}"
                )

            combined.extend(saved)

        combined.sort(key=lambda record: record["index"])

        if [record["index"] for record in combined] != list(range(2000)):
            raise RuntimeError(
                "Merged records do not contain exactly indices 0-1999"
            )

        failed_combined = [
            record for record in combined if record["failed"]
        ]

        def write_atomic(path, records):
            temporary = path.with_suffix(".jsonl.tmp")

            with temporary.open(
                "w", encoding="utf-8", newline="\n"
            ) as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")

                f.flush()
                os.fsync(f.fileno())

            os.replace(temporary, path)

        final_path = output_dir / "sea_all_answer_benign.jsonl"
        final_failed_path = (
            output_dir / "sea_failed_answer_benign.jsonl"
        )

        write_atomic(final_path, combined)
        write_atomic(final_failed_path, failed_combined)

        print("MERGE COMPLETE", flush=True)
        print(f"Total: {len(combined)}", flush=True)
        print(
            f"Accepted: {len(combined) - len(failed_combined)}",
            flush=True,
        )
        print(f"Failed: {len(failed_combined)}", flush=True)
        print(f"All rows: {final_path}", flush=True)
        print(f"Failed rows: {final_failed_path}", flush=True)
        return

    # Independent shard outputs

    shard_id = args.shard_id

    indices = list(range(shard_id, 2000, args.num_shards))

    shard_dir = output_dir / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    main_path = shard_dir / "sea_all_answer_benign.jsonl"
    failed_path = shard_dir / "sea_failed_answer_benign.jsonl"

    saved = load_records(main_path)

    if len(saved) > len(indices):
        raise RuntimeError(
            f"Shard {shard_id} contains excess records"
        )

    for record, index in zip(saved, indices):
        check_record(record, index)

    # Reconcile interrupted writes to the failed-only database

    expected_failed = [
        record for record in saved if record["failed"]
    ]

    saved_failed = load_records(failed_path)

    if len(saved_failed) > len(expected_failed):
        raise RuntimeError(
            f"Shard {shard_id} failed database has excess records"
        )

    for existing, expected in zip(saved_failed, expected_failed):
        if existing != expected:
            raise RuntimeError(
                f"Shard {shard_id} failed database mismatch"
            )

    for record in expected_failed[len(saved_failed):]:
        append_record(failed_path, record)

    completed = len(saved)

    print(
        f"Shard={shard_id}/{args.num_shards} "
        f"assigned={len(indices)} "
        f"completed={completed} "
        f"pending={len(indices) - completed}",
        flush=True,
    )

    print(f"Main database: {main_path}", flush=True)
    print(f"Failed database: {failed_path}", flush=True)

    if completed == len(indices):
        print("Shard already complete.", flush=True)
        return

    # Load model on the GPU selected by CUDA_VISIBLE_DEVICES

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0").eval()

    accepted_count = sum(not row["failed"] for row in saved)
    failed_count = sum(row["failed"] for row in saved)

    # Generate, validate, retry, and persist every result

    for local_position in range(completed, len(indices)):
        index = indices[local_position]
        original_row = rows[index]
        original = original_row["prompt"]

        attempts = []
        feedback = ""
        accepted = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            candidate = generate(
                model,
                tokenizer,
                TRANSFORM_PROMPT.format(
                    original=original,
                    feedback=feedback,
                ),
                512,
            )

            is_safe, problem = validate(
                model,
                tokenizer,
                original,
                candidate,
            )

            attempts.append({
                "attempt": attempt,
                "candidate": candidate,
                "safe": is_safe,
                "problem": problem,
            })

            print(
                f"[shard={shard_id} index={index} "
                f"{local_position + 1}/{len(indices)}] "
                f"attempt={attempt} "
                f"{'PASS' if is_safe else 'REJECT'}",
                flush=True,
            )

            print(f"  CANDIDATE: {candidate}", flush=True)

            if problem:
                print(f"  PROBLEM: {problem}", flush=True)

            if is_safe:
                accepted = candidate
                break

            feedback = (
                "PREVIOUS REWRITE FAILED VALIDATION.\n"
                f"Previous candidate: {candidate}\n"
                f"Problem: {problem}\n"
                "Rewrite the entire prompt. Remove every "
                "remaining harmful intention, method, "
                "condition, and outcome."
            )

        failed = accepted is None

        if failed:
            safe_response = None
            failure_reason = (
                attempts[-1]["problem"]
                if attempts else "No successful transformation."
            )
            failed_count += 1
        else:
            safe_response = generate(
                model,
                tokenizer,
                ANSWER_PROMPT.format(candidate=accepted),
                1024,
            )
            failure_reason = None
            accepted_count += 1

        record = {
            "index": index,
            "safe_prompt": accepted,
            "safe_response": safe_response,
            "prompt": original,
            "new_chosen_response": original_row[
                "new_chosen_response"
            ],
            "failed": failed,
            "failure_reason": failure_reason,
            "attempts": attempts,
        }

        # Main database first; failed-only database second.
        # Resume reconciles interruptions between these writes.

        append_record(main_path, record)

        if failed:
            append_record(failed_path, record)

        print(
            f"[shard={shard_id} index={index}] "
            f"{'FAILED' if failed else 'ACCEPTED'} "
            f"accepted={accepted_count} failed={failed_count}",
            flush=True,
        )

        if accepted is not None:
            print(f"  SAFE PROMPT: {accepted}", flush=True)
            print(
                f"  SAFE ANSWER: {safe_response[:250]}",
                flush=True,
            )

    print(
        f"SHARD {shard_id} COMPLETE: "
        f"accepted={accepted_count} failed={failed_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()

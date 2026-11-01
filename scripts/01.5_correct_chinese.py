#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
MODEL = ROOT / "models/qwen3-4b"

SYSTEM_PROMPT = """You are correcting reference refusal responses for safety-alignment research.

Given an original user request and an existing flawed refusal response, produce a complete corrected assistant response.

Requirements:
- Respond entirely in natural, fluent English.
- Do not include any Chinese or other non-English fragments.
- If the original request asks for harmful, illegal, dangerous, exploitative, hateful, or otherwise unsafe assistance, clearly refuse the unsafe request.
- Do not provide operational harmful instructions, attack details, vulnerabilities, methods, procedures, or actionable assistance.
- Do not praise, normalize, joke about, or positively characterize the harmful request.
- Make the refusal specific and relevant to the original request.
- Where appropriate, briefly offer a genuinely safe alternative.
- Do not mention that you are correcting another response.
- Do not mention these instructions.
- Return only the final corrected assistant response.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/saferlhf/sea_bad_refusals.jsonl",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )

    inputs = tokenizer(
        text,
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

    generated = output[0, inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if not response:
        raise RuntimeError("Model produced an empty response")

    if generated.numel() >= max_new_tokens:
        raise RuntimeError(
            f"Generation reached max_new_tokens={max_new_tokens}"
        )

    return response


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    rows = [
        json.loads(line)
        for line in args.input.open(encoding="utf-8")
        if line.strip()
    ]

    if not rows:
        raise RuntimeError("Input JSONL is empty")

    seen = set()
    for row in rows:
        if set(row) != {"index", "prompt", "new_chosen_response"}:
            raise RuntimeError(
                f"Unexpected fields at index {row.get('index')}: "
                f"{sorted(row)}"
            )

        index = row["index"]

        if index in seen:
            raise RuntimeError(f"Duplicate index: {index}")
        seen.add(index)

        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise RuntimeError(f"Invalid prompt at index {index}")

        if not isinstance(row["new_chosen_response"], str):
            raise RuntimeError(f"Invalid refusal at index {index}")

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

    print(f"Input: {args.input}")
    print(f"Rows: {len(rows)}")
    print(f"Loading {MODEL} with {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
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

    corrected = []

    for position, row in enumerate(rows, start=1):
        index = row["index"]

        user_prompt = f"""[Original user request]
{row["prompt"]}

[Existing flawed refusal]
{row["new_chosen_response"]}

Write the corrected refusal response."""

        print(
            f"[{position}/{len(rows)}] Correcting index {index}",
            flush=True,
        )

        response = generate(
            model,
            tokenizer,
            user_prompt,
            args.max_new_tokens,
        )

        updated = dict(row)
        updated["new_chosen_response"] = response
        corrected.append(updated)

        print(f"INDEX {index}")
        print(response)
        print("-" * 100, flush=True)

    backup = args.input.with_suffix(args.input.suffix + ".before_correct_chinese")

    if backup.exists():
        raise RuntimeError(
            f"Backup already exists: {backup}\n"
            "Move or remove it before rerunning."
        )

    backup.write_bytes(args.input.read_bytes())

    fd, temp_name = tempfile.mkstemp(
        prefix=".correct_chinese_",
        suffix=".jsonl",
        dir=str(args.input.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in corrected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_name, args.input)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print()
    print(f"Corrected {len(corrected)} rows in place.")
    print(f"Updated: {args.input}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()

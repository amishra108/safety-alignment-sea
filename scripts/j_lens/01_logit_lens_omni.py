#!/usr/bin/env python3
# Logit Lens for Qwen2.5-Omni Thinker

import argparse
import hashlib
import os
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[2]


def read_prompt(path, row_index):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            if index == row_index:
                row = json.loads(line)
                prompt = row["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Selected row has an empty prompt")
                return prompt

    raise IndexError(f"Row {row_index} not found in {path}")


def top_tokens(logits, tokenizer, k):
    values, indices = torch.topk(logits.float(), k=k)

    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
            ),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]



def save_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_text(path, result):
    lines = [
        f"Prompt: {result['prompt']}",
        f"Generated response: {result['generated_text']}",
        "",
    ]

    for step in result["steps"]:
        lines.append(
            f"TOKEN {step['index']:03d}: {step['chosen_token']!r}"
        )

        for item in step["layers"]:
            if not isinstance(item["layer"], int):
                continue

            tokens = "  ".join(
                repr(token["token"])
                for token in item["top_tokens"]
            )
            lines.append(f"L{item['layer']:02d}: {tokens}")

        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen2.5-omni-7b")
    parser.add_argument(
        "--dataset", default="data/saferlhf/sea_3000_prepared.jsonl"
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default="outputs/j_lens/saferlhf/sea_3000_prepared/first_run",
    )
    args = parser.parse_args()
    if args.top_k <= 0 or args.max_new_tokens <= 0:
        parser.error("top-k and max-new-tokens must be positive")

    model_path = ROOT / args.model
    dataset_path = ROOT / args.dataset
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "generation_readout.json"
    text_path = output_dir / "generation_readout.txt"

    prompt = read_prompt(dataset_path, args.row_index)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    ).input_ids[0].tolist()

    settings = {
        "model": str(model_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "row_index": args.row_index,
        "prompt_ids": prompt_ids,
        "top_k": args.top_k,
        "decoding": "greedy_full_prefix",
    }
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()

    if json_path.exists():
        result = json.loads(json_path.read_text())
        if result["fingerprint"] != fingerprint:
            raise RuntimeError("Existing output has different settings")
        print(f"Resuming after {len(result['steps'])} tokens", flush=True)
    else:
        result = {
            "fingerprint": fingerprint,
            "settings": settings,
            "prompt": prompt,
            "generated_token_ids": [],
            "generated_text": "",
            "steps": [],
            "stopped_on_eos": False,
        }

    if result["stopped_on_eos"] or len(result["steps"]) >= args.max_new_tokens:
        print("Generation already complete", flush=True)
        return

    print(f"Prompt: {prompt}", flush=True)
    print("Loading Qwen2.5-Omni Thinker in FP16", flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)

    decoder = model.model
    assert len(decoder.layers) == 28
    assert model.config.text_config.hidden_size == 3584

    activations = {}
    handles = []

    def make_hook(index):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[index] = hidden[:, -1, :].detach()
        return hook

    for index, layer in enumerate(decoder.layers):
        handles.append(layer.register_forward_hook(make_hook(index)))

    eos = tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    configured_eos = model.generation_config.eos_token_id
    if configured_eos is not None:
        if isinstance(configured_eos, (list, tuple)):
            eos_ids.update(configured_eos)
        else:
            eos_ids.add(configured_eos)

    try:
        with torch.inference_mode():
            for index in range(len(result["steps"]), args.max_new_tokens):
                prefix = prompt_ids + result["generated_token_ids"]
                input_ids = torch.tensor(
                    [prefix], dtype=torch.long, device="cuda:0"
                )
                activations.clear()

                outputs = decoder(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                )
                if len(activations) != 28:
                    raise RuntimeError("Not all decoder activations captured")

                actual_logits = model.lm_head(
                    outputs.last_hidden_state[:, -1, :]
                )[0]
                layers = []

                for layer_index in range(28):
                    residual = activations[layer_index]
                    logits = model.lm_head(decoder.norm(residual))[0]

                    if layer_index == 27:
                        difference = (
                            logits.float() - actual_logits.float()
                        ).abs().max().item()
                        actual_top1 = int(actual_logits.argmax().item())
                        lens_top1 = int(logits.argmax().item())

                        if actual_top1 != lens_top1:
                            raise RuntimeError(
                                f"Final layer top-1 mismatch: "
                                f"actual={actual_top1}, lens={lens_top1}, "
                                f"maximum logit difference={difference}"
                            )

                        if difference > 0.1:
                            raise RuntimeError(
                                f"Unexpectedly large final-layer "
                                f"logit difference: {difference}"
                            )

                    layers.append({
                        "layer": layer_index,
                        "top_tokens": top_tokens(
                            logits, tokenizer, args.top_k
                        ),
                    })

                layers.append({
                    "layer": "final",
                    "top_tokens": top_tokens(
                        actual_logits, tokenizer, args.top_k
                    ),
                })

                next_id = int(actual_logits.argmax().item())
                next_token = tokenizer.decode(
                    [next_id], skip_special_tokens=False
                )

                result["generated_token_ids"].append(next_id)
                result["generated_text"] = tokenizer.decode(
                    result["generated_token_ids"],
                    skip_special_tokens=True,
                )
                result["steps"].append({
                    "index": index,
                    "context_length": len(prefix),
                    "chosen_token_id": next_id,
                    "chosen_token": next_token,
                    "layers": layers,
                })
                result["stopped_on_eos"] = next_id in eos_ids

                save_json(json_path, result)
                save_text(text_path, result)

                print(
                    f"[{index + 1}/{args.max_new_tokens}] "
                    f"{next_token!r}",
                    flush=True,
                )

                if result["stopped_on_eos"]:
                    print("Reached EOS", flush=True)
                    break
    finally:
        for handle in handles:
            handle.remove()

    print(f"Generated response: {result['generated_text']}", flush=True)
    print(f"Saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

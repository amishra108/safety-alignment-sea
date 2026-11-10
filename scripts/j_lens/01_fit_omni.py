#!/usr/bin/env python3
# Jacobian Lens fitting for Qwen2.5-Omni-7B Thinker

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).resolve().parent / "upstream"

if not (UPSTREAM / "jlens" / "fitting.py").is_file():
    raise RuntimeError(f"Missing official Jacobian Lens source: {UPSTREAM}")

sys.path.insert(0, str(UPSTREAM))

from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens


def atomic_save(data, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def read_prompt(path, row_index):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == row_index:
                row = json.loads(line)
                prompt = row["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Selected row has no valid prompt")
                return prompt
    raise IndexError(f"Row {row_index} does not exist in {path}")


def parse_layers(value, n_layers):
    if value == "all":
        return list(range(n_layers - 1))

    layers = sorted({int(part.strip()) for part in value.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= n_layers - 1:
        raise ValueError("Source layers must be in [0, n_layers - 2]")
    return layers


def decode_topk(logits, tokenizer, k):
    scores, indices = torch.topk(logits.float(), k=min(k, logits.numel()))
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "score": float(score),
        }
        for score, token_id in zip(scores.tolist(), indices.tolist())
    ]


def fit_one_prompt(
    model,
    input_ids,
    source_layers,
    output_dir,
    *,
    dim_batch,
    checkpoint_every,
    skip_first,
    fingerprint,
):
    layers = model.model.layers
    target_layer = len(layers) - 1
    width = model.config.text_config.hidden_size
    seq_len = input_ids.shape[1]

    valid_mask = valid_position_mask(seq_len, skip_first=skip_first)
    valid_positions = valid_mask.nonzero(as_tuple=True)[0].to(input_ids.device)

    state_path = output_dir / "fit_state.pt"

    jacobians = {
        layer: torch.zeros(width, width, dtype=torch.float32)
        for layer in source_layers
    }
    next_dim = 0

    if state_path.exists():
        state = torch.load(
            state_path, map_location="cpu", weights_only=True
        )
        if state["fingerprint"] != fingerprint:
            raise RuntimeError(
                "Existing fitting checkpoint belongs to a different "
                "prompt, model or configuration. Use another output directory."
            )
        jacobians = state["jacobians"]
        next_dim = int(state["next_dim"])
        print(f"Resuming from Jacobian dimension {next_dim}/{width}", flush=True)

    def save_state(next_dimension):
        atomic_save(
            {
                "fingerprint": fingerprint,
                "source_layers": source_layers,
                "target_layer": target_layer,
                "next_dim": next_dimension,
                "jacobians": jacobians,
            },
            state_path,
        )

    if next_dim < width:
        with (
            torch.enable_grad(),
            ActivationRecorder(
                layers,
                at=[*source_layers, target_layer],
                start_graph_at=min(source_layers),
            ) as recorder,
        ):
            replicated_ids = input_ids.expand(dim_batch, -1)

            model.model(
                input_ids=replicated_ids,
                use_cache=False,
                return_dict=True,
            )

            target = recorder.activations[target_layer]
            sources = [
                recorder.activations[layer]
                for layer in source_layers
            ]

            cotangent = torch.zeros_like(target)
            batch_indices = torch.arange(
                dim_batch, device=target.device
            )

            print(
                f"Fitting {len(source_layers)} source layers; "
                f"target L{target_layer}; width={width}; "
                f"sequence={seq_len}; "
                f"valid positions={len(valid_positions)}",
                flush=True,
            )

            for start in range(next_dim, width, dim_batch):
                count = min(dim_batch, width - start)

                cotangent.zero_()
                cotangent[
                    batch_indices[:count, None],
                    valid_positions[None, :],
                    start + batch_indices[:count, None],
                ] = 1.0

                grads = torch.autograd.grad(
                    outputs=target,
                    inputs=sources,
                    grad_outputs=cotangent,
                    retain_graph=(start + count < width),
                )

                for layer, grad in zip(source_layers, grads):
                    rows = grad[
                        :count, valid_positions, :
                    ].float().mean(dim=1)

                    if not torch.isfinite(rows).all():
                        raise FloatingPointError(
                            f"Non-finite Jacobian values at "
                            f"layer {layer}, dimension {start}"
                        )

                    jacobians[layer][start:start + count] = (
                        rows.cpu()
                    )

                completed = start + count

                if (
                    completed % checkpoint_every == 0
                    or completed == width
                ):
                    save_state(completed)
                    print(
                        f"Jacobian dimensions: {completed}/{width} "
                        f"({100 * completed / width:.1f}%)",
                        flush=True,
                    )

    lens = JacobianLens(
        jacobians,
        n_prompts=1,
        d_model=width,
    )
    lens.save(str(output_dir / "lens.pt"))
    return lens


def apply_lens(model, tokenizer, input_ids, lens, top_k):
    source_layers = lens.source_layers
    target_layer = len(model.model.layers) - 1

    with torch.no_grad():
        with ActivationRecorder(
            model.model.layers,
            at=[*source_layers, target_layer],
        ) as recorder:
            model.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
            activations = {
                layer: value.detach()
                for layer, value in recorder.activations.items()
            }

        results = []

        for layer in source_layers:
            residual = activations[layer][0, -1].float()

            transported = lens.transport(residual, layer)
            logits = model.lm_head(
                model.model.norm(
                    transported.to(model.lm_head.weight.dtype)
                )
            )[0] if transported.ndim == 2 else model.lm_head(
                model.model.norm(
                    transported.to(model.lm_head.weight.dtype)
                )
            )

            results.append({
                "layer": layer,
                "type": "jacobian_lens",
                "top_tokens": decode_topk(logits, tokenizer, top_k),
            })

        final_residual = activations[target_layer][0, -1]
        final_logits = model.lm_head(
            model.model.norm(final_residual)
        )

        results.append({
            "layer": target_layer,
            "type": "actual_model_logits",
            "top_tokens": decode_topk(
                final_logits, tokenizer, top_k
            ),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/qwen2.5-omni-7b",
    )
    parser.add_argument(
        "--dataset",
        default="data/saferlhf/sea_3000_prepared.jsonl",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/j_lens/saferlhf/"
            "sea_3000_prepared/first_run"
        ),
    )
    parser.add_argument(
        "--source-layers",
        default="0,7,14,21,26",
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dim-batch", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.dim_batch < 1 or args.checkpoint_every < 1:
        parser.error("dim-batch and checkpoint-every must be positive")
    if args.checkpoint_every % args.dim_batch != 0:
        parser.error("checkpoint-every must be divisible by dim-batch")

    model_path = (ROOT / args.model).resolve()
    dataset_path = (ROOT / args.dataset).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = read_prompt(dataset_path, args.row_index)
    (output_dir / "input_prompt.txt").write_text(
        prompt + "\n", encoding="utf-8"
    )

    print(f"Model: {model_path}", flush=True)
    print(f"Dataset row: {args.row_index}", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print("Loading Qwen2.5-Omni Thinker (FP16)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        local_files_only=True,
        attn_implementation="sdpa",
    )

    model.eval()
    model.requires_grad_(False)

    assert len(model.model.layers) == 28
    assert model.config.text_config.hidden_size == 3584

    source_layers = parse_layers(
        args.source_layers, len(model.model.layers)
    )

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
    ).input_ids.to("cuda:0")

    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("Need at least two tokens")

    skip_first = min(16, seq_len - 2)

    settings = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "row_index": args.row_index,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "source_layers": source_layers,
        "target_layer": 27,
        "max_seq_len": args.max_seq_len,
        "dim_batch": args.dim_batch,
        "skip_first": skip_first,
        "token_ids": input_ids[0].tolist(),
    }

    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()

    (output_dir / "config.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"GPU memory after load: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
        flush=True,
    )

    lens_path = output_dir / "lens.pt"

    if lens_path.exists():
        print("Loading existing fitted lens", flush=True)
        lens = JacobianLens.load(str(lens_path))
        if lens.source_layers != source_layers or lens.d_model != 3584:
            raise RuntimeError("Existing lens does not match requested layers")
    else:
        lens = fit_one_prompt(
            model,
            input_ids,
            source_layers,
            output_dir,
            dim_batch=args.dim_batch,
            checkpoint_every=args.checkpoint_every,
            skip_first=skip_first,
            fingerprint=fingerprint,
        )

    print("Applying lens", flush=True)

    results = apply_lens(
        model, tokenizer, input_ids, lens, args.top_k
    )

    (output_dir / "readout.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "readout.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Token count: {seq_len}\n")
        f.write(f"Readout position: {seq_len - 1}\n\n")

        for result in results:
            tokens = [
                item["token"]
                for item in result["top_tokens"]
            ]
            line = (
                f"L{result['layer']:02d} "
                f"[{result['type']}]: {tokens}"
            )
            print(line, flush=True)
            f.write(line + "\n")

    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

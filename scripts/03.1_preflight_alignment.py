#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch


ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/safety-alignment-sea"
)

SFT_STAGE3_PATH = (
    ROOT / "scripts/03_train_alignment.py"
)

DPO_STAGE3_PATH = (
    ROOT / "scripts/03.5_train_alignment_dpo.py"
)

stage3 = None


def load_stage3(method):
    path = (
        SFT_STAGE3_PATH
        if method == "sft"
        else DPO_STAGE3_PATH
    )

    spec = importlib.util.spec_from_file_location(
        f"sea_stage3_{method}",
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(
        module
    )

    return module


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=("qwen25_omni", "salmonn2"),
        required=True,
    )

    p.add_argument(
        "--method",
        choices=("sft", "dpo"),
        required=True,
    )

    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--machine",
        required=True,
    )

    p.add_argument(
        "--gpu",
        type=int,
        required=True,
    )

    p.add_argument(
        "--embedding-search",
        type=int,
        default=10,
    )

    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )

    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_harmful_index(
    model,
    limit,
):
    root = (
        stage3.EMBEDDING_ROOT
        / stage3.MODEL_NAMES[model]
    )

    for index in range(limit):
        path = (
            root
            / f"row_{index:04d}.pt"
        )

        if path.is_file():
            return index, path

    raise RuntimeError(
        f"No {model} embedding found among "
        f"rows 0..{limit - 1}"
    )


def choose_real_audio():
    with stage3.CLEAN_BENCHMARK.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    candidates = [
        row
        for row in rows
        if row.get("language") == "english"
        and row.get("split") in {
            "harmful",
            "benign",
        }
        and row.get("audio_path")
    ]

    if not candidates:
        raise RuntimeError(
            "No English real-audio validation example found"
        )

    def duration(row):
        try:
            return float(
                row.get(
                    "duration_seconds",
                    "inf",
                )
            )
        except ValueError:
            return float("inf")

    row = min(
        candidates,
        key=lambda x: (
            duration(x),
            x.get("example_id", ""),
        ),
    )

    path = ROOT / row["audio_path"]

    if not path.is_file():
        raise RuntimeError(
            f"Missing real WAV: {path}"
        )

    return row, path


def clean_text(text):
    if hasattr(
        stage3,
        "clean_generated_response",
    ):
        return stage3.clean_generated_response(
            text
        )

    return text.strip()


@torch.no_grad()
def decode_real_audio(
    adapter,
    model_name,
    audio_path,
):
    adapter.model.eval()

    if model_name == "qwen25_omni":
        text = adapter.generate_validation_audio(
            audio_path,
            stage3.VALIDATION_INSTRUCTION,
            16,
        )

        return clean_text(text)

    batch = adapter.processor(
        audios=[str(audio_path)],
        formatted_prompt=(
            "<audio>"
            "DO NOT TRANSCRIBE THE AUDIO. "
            "DO NOT REPEAT, QUOTE, SUMMARIZE, OR DESCRIBE THE AUDIO. "
            "THE AUDIO CONTAINS A SPOKEN USER REQUEST. "
            "ANSWER THAT REQUEST DIRECTLY. "
            "YOUR RESPONSE MUST BE AN ANSWER TO THE REQUEST IN THE AUDIO, NOT A TRANSCRIPTION."
        ),
        enable_thinking=False,
        return_tensors="pt",
    )

    batch = stage3.move_batch(
        batch,
        adapter.device,
    )

    output = adapter.model.generate(
        **batch,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
    )

    if isinstance(output, tuple):
        output = output[0]

    if (
        not torch.is_tensor(output)
        or output.ndim != 2
    ):
        raise RuntimeError(
            "Unexpected SALMONN generation output"
        )

    text = adapter.processor.decode(
        output[0]
    ).strip()

    return clean_text(text)


def snapshot_parameters(
    named_parameters,
    samples_per_tensor=16,
):
    result = []

    for name, parameter in named_parameters:
        if (
            not parameter.requires_grad
            or parameter.numel() == 0
        ):
            continue

        flat = (
            parameter.detach()
            .reshape(-1)
        )

        count = min(
            samples_per_tensor,
            flat.numel(),
        )

        if count == 1:
            indices_cpu = torch.zeros(
                1,
                dtype=torch.long,
            )
        else:
            positions = torch.arange(
                count,
                dtype=torch.int64,
            )

            indices_cpu = (
                positions
                * (flat.numel() - 1)
                // (count - 1)
            )

        values = (
            flat.index_select(
                0,
                indices_cpu.to(
                    flat.device
                ),
            )
            .cpu()
            .clone()
        )

        result.append(
            (
                name,
                parameter,
                indices_cpu,
                values,
            )
        )

    return result


def compare_snapshot(snapshot):
    sampled_tensors = 0
    sampled_values = 0
    changed_tensors = 0
    changed_values = 0
    max_delta = 0.0

    for (
        _name,
        parameter,
        indices_cpu,
        before,
    ) in snapshot:
        flat = (
            parameter.detach()
            .reshape(-1)
        )

        after = (
            flat.index_select(
                0,
                indices_cpu.to(
                    flat.device
                ),
            )
            .cpu()
        )

        changed = after.ne(
            before
        )

        n_changed = int(
            changed.sum().item()
        )

        sampled_tensors += 1
        sampled_values += (
            before.numel()
        )

        if n_changed:
            changed_tensors += 1
            changed_values += (
                n_changed
            )

            delta = (
                after.float()
                - before.float()
            ).abs()

            max_delta = max(
                max_delta,
                float(
                    delta.max().item()
                ),
            )

    return {
        "sampled_tensors":
            sampled_tensors,
        "sampled_values":
            sampled_values,
        "changed_tensors":
            changed_tensors,
        "changed_values":
            changed_values,
        "max_abs_delta":
            max_delta,
    }


def optimizer_state_summary(
    optimizer,
):
    entries = 0
    steps = []

    for state in optimizer.state.values():
        if not state:
            continue

        entries += 1

        step = state.get("step")

        if step is None:
            continue

        if torch.is_tensor(step):
            step = (
                step.detach()
                .cpu()
                .item()
            )

        steps.append(
            float(step)
        )

    return {
        "state_entries":
            entries,
        "max_step":
            max(steps)
            if steps
            else None,
    }


def ensure_parameter_update(
    snapshot,
    optimizer,
    label,
):
    torch.cuda.synchronize()

    changes = compare_snapshot(
        snapshot
    )

    state = optimizer_state_summary(
        optimizer
    )

    if changes[
        "changed_values"
    ] <= 0:
        raise RuntimeError(
            f"{label}: no sampled stored "
            "parameter value changed"
        )

    if (
        state["state_entries"] <= 0
        or state["max_step"] is None
        or state["max_step"] < 1
    ):
        raise RuntimeError(
            f"{label}: Adafactor state did "
            f"not advance: {state}"
        )

    return changes, state


def sft_step(
    adapter,
    args,
    rows,
    index,
    optimizer,
    parameters,
    named_parameters,
):
    row = rows[index]

    (
        instruction,
        embedding,
        with_audio,
    ) = stage3.sample_inputs(
        args,
        rows,
        index,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    adapter.model.train()

    output, batch = stage3.forward_target(
        adapter,
        instruction,
        row["new_chosen_response"],
        embedding,
        with_audio,
    )

    loss = output.loss

    if (
        loss is None
        or not torch.isfinite(loss)
    ):
        raise RuntimeError(
            f"SFT row {index}: invalid loss"
        )

    loss_value = float(
        loss.detach().cpu()
    )

    del output
    del batch

    loss.backward()

    grad_norm, grad_tensors = (
        stage3.global_grad_norm(
            parameters
        )
    )

    if (
        not math.isfinite(grad_norm)
        or grad_norm <= 0
        or grad_tensors <= 0
    ):
        raise RuntimeError(
            f"SFT row {index}: invalid gradients"
        )

    snapshot = snapshot_parameters(
        named_parameters
    )

    optimizer.step()

    changes, state = (
        ensure_parameter_update(
            snapshot,
            optimizer,
            f"SFT row {index}",
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    return {
        "loss": loss_value,
        "grad_norm": grad_norm,
        "grad_tensors": grad_tensors,
        "parameter_changes": changes,
        "optimizer_state": state,
        **stage3.gpu_stats(),
    }


@torch.no_grad()
def reference_pair(
    adapter,
    args,
    rows,
    index,
):
    row = rows[index]

    (
        instruction,
        embedding,
        with_audio,
    ) = stage3.sample_inputs(
        args,
        rows,
        index,
    )

    chosen, chosen_tokens = (
        stage3.target_logprob(
            adapter,
            instruction,
            row["new_chosen_response"],
            embedding,
            with_audio,
        )
    )

    rejected, rejected_tokens = (
        stage3.target_logprob(
            adapter,
            instruction,
            row["rejected_response"],
            embedding,
            with_audio,
        )
    )

    return {
        "chosen": float(
            chosen.cpu()
        ),
        "rejected": float(
            rejected.cpu()
        ),
        "chosen_tokens":
            chosen_tokens,
        "rejected_tokens":
            rejected_tokens,
    }


def dpo_step(
    adapter,
    args,
    rows,
    index,
    reference,
    optimizer,
    parameters,
    named_parameters,
):
    row = rows[index]

    (
        instruction,
        embedding,
        with_audio,
    ) = stage3.sample_inputs(
        args,
        rows,
        index,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    adapter.model.train()

    with torch.no_grad():
        rejected_current, _ = (
            stage3.target_logprob(
                adapter,
                instruction,
                row["rejected_response"],
                embedding,
                with_audio,
            )
        )

    chosen_current, _ = (
        stage3.target_logprob(
            adapter,
            instruction,
            row["new_chosen_response"],
            embedding,
            with_audio,
        )
    )

    policy_margin = (
        float(
            chosen_current.detach().cpu()
        )
        - float(
            rejected_current.detach().cpu()
        )
    )

    reference_margin = (
        reference["chosen"]
        - reference["rejected"]
    )

    advantage = (
        policy_margin
        - reference_margin
    )

    z = (
        args.beta
        * advantage
    )

    loss_value = float(
        -torch.nn.functional.logsigmoid(
            torch.tensor(
                z,
                dtype=torch.float64,
            )
        )
    )

    coefficient = (
        args.beta
        * (
            1.0
            / (
                1.0
                + math.exp(-z)
            )
            - 1.0
        )
    )

    (
        chosen_current
        * coefficient
    ).backward()

    del chosen_current

    rejected_grad, _ = (
        stage3.target_logprob(
            adapter,
            instruction,
            row["rejected_response"],
            embedding,
            with_audio,
        )
    )

    (
        rejected_grad
        * (-coefficient)
    ).backward()

    del rejected_grad

    grad_norm, grad_tensors = (
        stage3.global_grad_norm(
            parameters
        )
    )

    if (
        not math.isfinite(grad_norm)
        or grad_norm <= 0
        or grad_tensors <= 0
    ):
        raise RuntimeError(
            f"DPO row {index}: invalid gradients"
        )

    snapshot = snapshot_parameters(
        named_parameters
    )

    optimizer.step()

    changes, state = (
        ensure_parameter_update(
            snapshot,
            optimizer,
            f"DPO row {index}",
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    return {
        "loss": loss_value,
        "policy_margin":
            policy_margin,
        "reference_margin":
            reference_margin,
        "advantage":
            advantage,
        "grad_norm":
            grad_norm,
        "grad_tensors":
            grad_tensors,
        "parameter_changes":
            changes,
        "optimizer_state":
            state,
        **stage3.gpu_stats(),
    }


def main():
    global stage3

    args = parse_args()
    stage3 = load_stage3(
        args.method
    )

    print(
        f"Preflight trainer source: "
        f"{stage3.__file__}",
        flush=True,
    )

    import socket

    hostname = (
        socket.gethostname()
        .split(".")[0]
    )

    if hostname != args.machine:
        raise RuntimeError(
            f"Running on {hostname}, "
            f"expected {args.machine}"
        )

    import os

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(args.gpu)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    torch.cuda.set_device(0)

    device = torch.device(
        "cuda:0"
    )

    rows = stage3.load_rows()

    harmful_index, embedding_path = (
        find_harmful_index(
            args.model,
            args.embedding_search,
        )
    )

    harmless_index = 2000

    embedding_hash_before = (
        sha256_file(
            embedding_path
        )
    )

    print(
        "=" * 100,
        flush=True,
    )

    print(
        f"PREFLIGHT "
        f"model={args.model} "
        f"method={args.method}",
        flush=True,
    )

    print(
        f"harmful row={harmful_index} "
        f"embedding={embedding_path}",
        flush=True,
    )

    print(
        f"harmless row={harmless_index}",
        flush=True,
    )

    if args.model == "qwen25_omni":
        adapter = (
            stage3.Qwen25OmniAdapter(
                args.model_path,
                device,
            )
        )
    else:
        adapter = (
            stage3.Salmonn2Adapter(
                args.model_path,
                device,
            )
        )

    trainability = (
        stage3.validate_sea_trainability(
            adapter
        )
    )

    print(
        f"Trainable parameters: "
        f"{trainability['trainable_parameters']:,}/"
        f"{trainability['total_parameters']:,}",
        flush=True,
    )
    print(
        "Frozen audio encoder trunk: "
        f"{trainability['frozen_encoder_trunk_parameters']:,}/"
        f"{trainability['encoder_trunk_parameters']:,}",
        flush=True,
    )
    print(
        "Trainable projector/connector: "
        f"{trainability['projector_trainable_parameters']:,}/"
        f"{trainability['projector_parameters']:,}",
        flush=True,
    )
    print(
        "Trainable downstream parameters: "
        f"{trainability['downstream_trainable_parameters']:,}/"
        f"{trainability['downstream_parameters']:,}",
        flush=True,
    )

    audio_row, audio_path = (
        choose_real_audio()
    )

    print()
    print(
        "REAL-AUDIO DECODE BEFORE",
        flush=True,
    )

    print(
        f"audio={audio_row['example_id']}",
        flush=True,
    )

    decode_before = decode_real_audio(
        adapter,
        args.model,
        audio_path,
    )

    if not decode_before:
        raise RuntimeError(
            "Initial real-audio decode was empty"
        )

    print(
        f"DECODE BEFORE: "
        f"{decode_before!r}",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer, parameters = (
        stage3.build_optimizer(
            adapter,
            args.method,
        )
    )

    named_parameters = [
        (
            name,
            parameter,
        )
        for name, parameter
        in adapter.model.named_parameters()
        if parameter.requires_grad
    ]

    indices = [
        harmful_index,
        harmless_index,
    ]

    reference = None

    if args.method == "dpo":
        print()
        print(
            "PRECOMPUTING BASE REFERENCE LOGPS",
            flush=True,
        )

        adapter.model.eval()

        reference = {
            index:
                reference_pair(
                    adapter,
                    args,
                    rows,
                    index,
                )
            for index in indices
        }

        for index in indices:
            value = reference[index]

            print(
                f"reference row={index} "
                f"chosen={value['chosen']:.4f} "
                f"rejected={value['rejected']:.4f} "
                f"margin="
                f"{value['chosen'] - value['rejected']:.4f}",
                flush=True,
            )

    results = []

    for index in indices:
        print()
        print(
            f"{args.method.upper()} "
            f"OPTIMIZER STEP row={index}",
            flush=True,
        )

        if args.method == "sft":
            result = sft_step(
                adapter,
                args,
                rows,
                index,
                optimizer,
                parameters,
                named_parameters,
            )
        else:
            result = dpo_step(
                adapter,
                args,
                rows,
                index,
                reference[index],
                optimizer,
                parameters,
                named_parameters,
            )

        results.append(
            {
                "row": index,
                **result,
            }
        )

        changes = (
            result[
                "parameter_changes"
            ]
        )

        state = (
            result[
                "optimizer_state"
            ]
        )

        print(
            f"PASS row={index} "
            f"loss={result['loss']:.6f} "
            f"grad={result['grad_norm']:.6e} "
            f"grad_tensors="
            f"{result['grad_tensors']} "
            f"changed_tensors="
            f"{changes['changed_tensors']}/"
            f"{changes['sampled_tensors']} "
            f"changed_values="
            f"{changes['changed_values']}/"
            f"{changes['sampled_values']} "
            f"max_delta="
            f"{changes['max_abs_delta']:.6e} "
            f"adafactor_step="
            f"{state['max_step']} "
            f"peak="
            f"{result['cuda_max_allocated_gb']:.2f}G",
            flush=True,
        )

        torch.cuda.empty_cache()

    print()
    print(
        "REAL-AUDIO DECODE AFTER",
        flush=True,
    )

    decode_after = decode_real_audio(
        adapter,
        args.model,
        audio_path,
    )

    if not decode_after:
        raise RuntimeError(
            "Post-update real-audio decode was empty"
        )

    print(
        f"DECODE AFTER: "
        f"{decode_after!r}",
        flush=True,
    )

    embedding_hash_after = (
        sha256_file(
            embedding_path
        )
    )

    if (
        embedding_hash_before
        != embedding_hash_after
    ):
        raise RuntimeError(
            "Stage-2 embedding was modified"
        )

    report = {
        "model":
            args.model,
        "method":
            args.method,
        "model_path":
            str(args.model_path),
        "harmful_index":
            harmful_index,
        "harmless_index":
            harmless_index,
        "embedding_path":
            str(embedding_path),
        "stage2_embedding_unchanged":
            True,
        "trainability":
            trainability,
        "decode_example_id":
            audio_row["example_id"],
        "decode_before":
            decode_before,
        "decode_after":
            decode_after,
        "steps":
            results,
    }

    report_dir = (
        stage3.RUN_ROOT
        / args.model
        / "preflight"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir
        / f"{args.method}_full_smoke.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100,
        flush=True,
    )

    print(
        f"PREFLIGHT PASS: "
        f"{args.model} "
        f"{args.method}",
        flush=True,
    )

    print(
        "  real audio decode before: PASS",
        flush=True,
    )

    print(
        "  harmful synthetic embedding path: PASS",
        flush=True,
    )

    print(
        "  harmless text path: PASS",
        flush=True,
    )

    if args.method == "dpo":
        print(
            "  chosen/rejected reference logps: PASS",
            flush=True,
        )

        print(
            "  DPO chosen/rejected backward: PASS",
            flush=True,
        )

    print(
        "  Adafactor stored-weight update: PASS",
        flush=True,
    )

    print(
        "  real audio decode after: PASS",
        flush=True,
    )

    print(
        "  Stage-2 embedding unchanged: PASS",
        flush=True,
    )

    print(
        f"  report: {report_path}",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )


if __name__ == "__main__":
    main()

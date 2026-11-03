#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import random
import re
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
BASE_SCRIPT = ROOT / "scripts/03.5_train_alignment_dpo.py"
RUN_ROOT = ROOT / "outputs/03.7_continue_training"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "sea_dpo_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import base trainer: {BASE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=("qwen25_omni", "salmonn2"),
        required=True,
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Existing DPO epoch checkpoint to continue from.",
    )
    p.add_argument(
        "--reference-logps",
        type=Path,
        required=True,
        help=(
            "reference_logps.jsonl from the ORIGINAL DPO run. "
            "This must not be recomputed from the continued checkpoint."
        ),
    )
    p.add_argument(
        "--start-epoch",
        type=int,
        required=True,
        help="Epoch represented by --model-path, e.g. 3.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of additional epochs to train.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-6,
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--machine",
        default="nete",
        choices=("nete", "air211", "air213"),
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    p.add_argument(
        "--preflight",
        action="store_true",
        help="Train on 5 harmful and 5 harmless rows without saving model checkpoints.",
    )

    args = p.parse_args()

    if not args.model_path.name.endswith("_final"):
        p.error("Only *_final checkpoints are permitted")

    if not args.model_path.is_dir():
        p.error(
            f"--model-path does not exist: {args.model_path}"
        )

    if not args.reference_logps.is_file():
        p.error(
            f"--reference-logps does not exist: "
            f"{args.reference_logps}"
        )

    if args.start_epoch < 1:
        p.error("--start-epoch must be positive")

    if args.epochs < 1:
        p.error("--epochs must be positive")

    if args.lr <= 0:
        p.error("--lr must be positive")

    if args.beta <= 0:
        p.error("--beta must be positive")

    if args.grad_accum < 1:
        p.error("--grad-accum must be positive")

    if args.log_every < 1:
        p.error("--log-every must be positive")

    if args.gpu < 0:
        p.error("--gpu must be nonnegative")

    args.method = "dpo"
    args.preflight = 0
    args.decode_validation = False

    return args


def checkpoint_prefix(args):
    match = re.fullmatch(
        r"(.+)_epoch(\d+)_final",
        args.model_path.name,
    )

    if match is None:
        raise RuntimeError(
            "Input checkpoint name must end in "
            "'_epochN_final': "
            f"{args.model_path.name}"
        )

    detected_epoch = int(match.group(2))

    if detected_epoch != args.start_epoch:
        raise RuntimeError(
            f"--start-epoch={args.start_epoch}, but checkpoint "
            f"name says epoch {detected_epoch}: "
            f"{args.model_path.name}"
        )

    return match.group(1)


def checkpoint_path(
    args,
    prefix,
    epoch,
):
    return (
        args.model_path.parent
        / f"{prefix}_epoch{epoch}_final"
    )


def run_dir(args):
    end_epoch = (
        args.start_epoch
        + args.epochs
    )

    lr_tag = format(
        args.lr,
        ".8g",
    ).replace(
        "+",
        "",
    )

    tag = (
        f"epoch{args.start_epoch}"
        f"_to_epoch{end_epoch}"
        f"_lr{lr_tag}"
    )

    if args.preflight:
        tag += "_preflight"

    return RUN_ROOT / args.model / tag


def metrics_path(args):
    return (
        run_dir(args)
        / "training_metrics.jsonl"
    )


def args_preflight_reference_mode(expected):
    return len(expected) == 10


def verify_loaded_checkpoint(adapter, args):
    from safetensors import safe_open

    index_path = args.model_path / "model.safetensors.index.json"

    if not index_path.is_file():
        raise RuntimeError(f"Missing checkpoint index: {index_path}")

    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    model = (
        adapter.omni_model
        if args.model == "qwen25_omni"
        else adapter.model
    )

    loaded = model.state_dict()

    missing = sorted(set(weight_map) - set(loaded))

    if missing:
        raise RuntimeError(
            f"Checkpoint tensors missing from loaded model: {missing[:20]}"
        )

    selected = {}

    for key, shard in weight_map.items():
        selected.setdefault(shard, []).append(key)

    checked = 0

    for shard, keys in sorted(selected.items()):
        chosen = {keys[0], keys[-1]}

        for key in keys:
            if any(
                marker in key.lower()
                for marker in (
                    "connector",
                    "audio_encoder.proj",
                    "base_llm.model.layers.0",
                    "thinker.model.layers.0",
                )
            ):
                chosen.add(key)
                if len(chosen) >= 12:
                    break

        with safe_open(
            str(args.model_path / shard),
            framework="pt",
            device="cpu",
        ) as handle:
            for key in sorted(chosen):
                original = handle.get_tensor(key)
                current = loaded[key].detach().cpu()

                if original.shape != current.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {key}: "
                        f"{tuple(original.shape)} vs {tuple(current.shape)}"
                    )

                if not torch.equal(
                    original.to(dtype=current.dtype),
                    current,
                ):
                    raise RuntimeError(
                        f"Loaded tensor differs from epoch-3 checkpoint: {key}"
                    )

                checked += 1

    print(
        f"CHECKPOINT VERIFICATION PASS: "
        f"{len(weight_map)} indexed tensors present; "
        f"{checked} tensor values checked against checkpoint shards",
        flush=True,
    )


def load_reference(
    path,
    expected_indices,
):
    reference = {}

    with path.open(
        encoding="utf-8"
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            record = json.loads(line)

            index = record.get("index")

            if not isinstance(index, int):
                raise RuntimeError(
                    f"{path}:{line_number}: invalid index"
                )

            if index in reference:
                raise RuntimeError(
                    f"Duplicate reference row {index}: {path}"
                )

            for key in (
                "chosen_logp",
                "rejected_logp",
            ):
                value = record.get(key)

                if not isinstance(
                    value,
                    (int, float),
                ):
                    raise RuntimeError(
                        f"{path}:{line_number}: "
                        f"invalid {key}"
                    )

                if not math.isfinite(
                    float(value)
                ):
                    raise RuntimeError(
                        f"{path}:{line_number}: "
                        f"non-finite {key}"
                    )

            reference[index] = record

    expected = set(expected_indices)
    actual = set(reference)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected) if not args_preflight_reference_mode(expected) else []

    if missing or extra:
        raise RuntimeError(
            "Reference log-probability index mismatch. "
            f"missing={missing[:20]} "
            f"extra={extra[:20]} "
            f"expected={len(expected)} "
            f"actual={len(actual)}"
        )

    print(
        f"Fixed DPO reference loaded: "
        f"{len(reference)}/{len(expected_indices)} rows "
        f"from {path}",
        flush=True,
    )

    return reference


def build_optimizer(
    base,
    adapter,
    lr,
):
    from transformers.optimization import Adafactor

    base.validate_sea_trainability(
        adapter
    )

    parameters = [
        parameter
        for parameter
        in adapter.model.parameters()
        if parameter.requires_grad
    ]

    optimizer = Adafactor(
        parameters,
        lr=lr,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    return optimizer, parameters


def stable_sigmoid(value):
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)

    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def save_continuation_metadata(
    args,
    checkpoint,
    epoch,
    reference_path,
    optimizer_steps,
    elapsed,
):
    metadata = {
        "continuation": True,
        "method": "dpo",
        "model": args.model,
        "continued_from": str(
            args.model_path.resolve()
        ),
        "input_epoch": args.start_epoch,
        "checkpoint_epoch": epoch,
        "additional_epochs_requested": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "gradient_accumulation": args.grad_accum,
        "seed": args.seed,
        "decode_validation": False,
        "fixed_reference_logps": str(
            reference_path.resolve()
        ),
        "optimizer": "Adafactor",
        "optimizer_state_resumed": False,
        "optimizer_state_note": (
            "The earlier model checkpoints did not contain "
            "Adafactor state, so continuation starts a fresh "
            "Adafactor state from the saved model weights."
        ),
        "optimizer_steps_this_continuation":
            optimizer_steps,
        "elapsed_seconds_this_continuation":
            elapsed,
    }

    path = (
        checkpoint
        / "sea_continuation_metadata.json"
    )

    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite metadata: {path}"
        )

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def save_adafactor_state(optimizer, checkpoint_dir):
    destination = checkpoint_dir / "adafactor_state.pt"
    temporary = checkpoint_dir / "adafactor_state.pt.incomplete"

    if destination.exists() or temporary.exists():
        raise RuntimeError(
            f"Optimizer checkpoint already exists or is incomplete: {destination}"
        )

    torch.save(optimizer.state_dict(), temporary)
    os.replace(temporary, destination)

    print(f"SAVED ADAFACTOR: {destination}", flush=True)


def train_dpo_continuation(
    base,
    adapter,
    args,
    rows,
    indices,
    optimizer,
    parameters,
    probe_parameter,
    reference,
    prefix,
):
    adapter.model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    optimizer_steps = 0
    training_start = time.perf_counter()

    for local_epoch in range(
        1,
        (1 if args.preflight else args.epochs) + 1,
    ):
        global_epoch = (
            args.start_epoch
            + local_epoch
        )

        order = list(indices)

        random.Random(
            args.seed + global_epoch
        ).shuffle(order)

        if args.preflight:
            order = order[:10]
            print(
                "PREFLIGHT: 10 rows, no model checkpoints will be saved",
                flush=True,
            )

        accum_loss = 0.0
        accum_count = 0
        accum_policy_margin = []
        accum_ref_margin = []
        accum_advantage = []

        epoch_losses = []
        epoch_advantages = []

        print()
        print(
            "=" * 100,
            flush=True,
        )
        print(
            f"CONTINUATION DPO EPOCH "
            f"{global_epoch} START",
            flush=True,
        )
        print(
            f"lr={args.lr:.8g} "
            f"beta={args.beta:.8g} "
            f"examples={len(order)}",
            flush=True,
        )
        print(
            "=" * 100,
            flush=True,
        )

        epoch_start = time.perf_counter()

        for position, index in enumerate(
            order,
            start=1,
        ):
            row = rows[index]

            (
                instruction,
                embedding,
                with_audio,
            ) = base.sample_inputs(
                args,
                rows,
                index,
            )

            ref_chosen = float(
                reference[index][
                    "chosen_logp"
                ]
            )
            ref_rejected = float(
                reference[index][
                    "rejected_logp"
                ]
            )

            ref_margin = (
                ref_chosen
                - ref_rejected
            )

            with torch.no_grad():
                rejected_current, _ = (
                    base.target_logprob(
                        adapter,
                        instruction,
                        row[
                            "rejected_response"
                        ],
                        embedding,
                        with_audio,
                    )
                )

            chosen_current, _ = (
                base.target_logprob(
                    adapter,
                    instruction,
                    row[
                        "new_chosen_response"
                    ],
                    embedding,
                    with_audio,
                )
            )

            policy_margin = (
                float(
                    chosen_current
                    .detach()
                    .cpu()
                )
                - float(
                    rejected_current
                    .detach()
                    .cpu()
                )
            )

            advantage = (
                policy_margin
                - ref_margin
            )

            z = (
                args.beta
                * advantage
            )

            loss_value = float(
                -F.logsigmoid(
                    torch.tensor(
                        z,
                        dtype=torch.float64,
                    )
                ).item()
            )

            coefficient = (
                args.beta
                * (
                    stable_sigmoid(z)
                    - 1.0
                )
            )

            (
                chosen_current
                * (
                    coefficient
                    / args.grad_accum
                )
            ).backward()

            del chosen_current
            del rejected_current

            rejected_grad, _ = (
                base.target_logprob(
                    adapter,
                    instruction,
                    row[
                        "rejected_response"
                    ],
                    embedding,
                    with_audio,
                )
            )

            (
                rejected_grad
                * (
                    -coefficient
                    / args.grad_accum
                )
            ).backward()

            del rejected_grad

            if not math.isfinite(
                loss_value
            ):
                raise RuntimeError(
                    f"Non-finite DPO loss "
                    f"epoch={global_epoch} "
                    f"row={index}"
                )

            accum_loss += loss_value
            accum_count += 1

            accum_policy_margin.append(
                policy_margin
            )
            accum_ref_margin.append(
                ref_margin
            )
            accum_advantage.append(
                advantage
            )

            epoch_losses.append(
                loss_value
            )
            epoch_advantages.append(
                advantage
            )

            should_step = (
                accum_count
                == args.grad_accum
                or position == len(order)
            )

            if should_step:
                grad_norm, grad_tensors = (
                    base.global_grad_norm(
                        parameters
                    )
                )

                if (
                    not math.isfinite(
                        grad_norm
                    )
                    or grad_tensors <= 0
                ):
                    raise RuntimeError(
                        "Invalid gradients: "
                        f"epoch={global_epoch} "
                        f"row={index} "
                        f"norm={grad_norm} "
                        f"tensors={grad_tensors}"
                    )

                before = (
                    base.probe_snapshot(
                        probe_parameter
                    )
                )

                optimizer.step()

                update_norm = (
                    base.probe_delta(
                        probe_parameter,
                        before,
                    )
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1

                metric = {
                    "type": "step",
                    "method": "dpo",
                    "epoch": global_epoch,
                    "continuation_local_epoch":
                        local_epoch,
                    "sample_position":
                        position,
                    "row": index,
                    "loss":
                        accum_loss / accum_count,
                    "grad_norm":
                        grad_norm,
                    "gradient_tensors":
                        grad_tensors,
                    "probe_update_norm":
                        update_norm,
                    "lr": float(
                        optimizer.param_groups[0][
                            "lr"
                        ]
                    ),
                    "category":
                        row["category"],
                    "policy_margin":
                        statistics.fmean(
                            accum_policy_margin
                        ),
                    "reference_margin":
                        statistics.fmean(
                            accum_ref_margin
                        ),
                    "preference_advantage":
                        statistics.fmean(
                            accum_advantage
                        ),
                    "preference_accuracy":
                        statistics.fmean(
                            [
                                1.0
                                if value > 0
                                else 0.0
                                for value
                                in accum_advantage
                            ]
                        ),
                    "beta": args.beta,
                    **base.gpu_stats(),
                }

                base.append_jsonl(
                    metrics_path(args),
                    metric,
                )

                if (
                    optimizer_steps == 1
                    or optimizer_steps
                    % args.log_every == 0
                ):
                    print(
                        f"DPO "
                        f"epoch={global_epoch} "
                        f"step={optimizer_steps} "
                        f"sample={position}/{len(order)} "
                        f"row={index} "
                        f"loss={metric['loss']:.6f} "
                        f"adv="
                        f"{metric['preference_advantage']:.4f} "
                        f"acc="
                        f"{metric['preference_accuracy']:.3f} "
                        f"grad={grad_norm:.6f} "
                        f"update={update_norm:.8e} "
                        f"lr={metric['lr']:.8g} "
                        f"vram="
                        f"{metric['cuda_allocated_gb']:.2f}G "
                        f"peak="
                        f"{metric['cuda_max_allocated_gb']:.2f}G",
                        flush=True,
                    )

                accum_loss = 0.0
                accum_count = 0
                accum_policy_margin = []
                accum_ref_margin = []
                accum_advantage = []

            del embedding

        epoch_elapsed = (
            time.perf_counter()
            - epoch_start
        )

        summary = {
            "type": "epoch_summary",
            "method": "dpo",
            "epoch": global_epoch,
            "continuation_local_epoch":
                local_epoch,
            "mean_loss":
                statistics.fmean(
                    epoch_losses
                ),
            "mean_preference_advantage":
                statistics.fmean(
                    epoch_advantages
                ),
            "preference_accuracy":
                statistics.fmean(
                    [
                        1.0
                        if value > 0
                        else 0.0
                        for value
                        in epoch_advantages
                    ]
                ),
            "optimizer_steps":
                optimizer_steps,
            "epoch_elapsed_seconds":
                epoch_elapsed,
            "lr": args.lr,
            **base.gpu_stats(),
        }

        base.append_jsonl(
            metrics_path(args),
            summary,
        )

        print(
            f"DPO epoch {global_epoch} complete: "
            f"mean_loss="
            f"{summary['mean_loss']:.6f} "
            f"mean_advantage="
            f"{summary['mean_preference_advantage']:.4f} "
            f"preference_accuracy="
            f"{summary['preference_accuracy']:.3f} "
            f"time={epoch_elapsed:.2f}s",
            flush=True,
        )

        if args.preflight:
            print(
                f"PREFLIGHT PASS: model={args.model} "
                f"rows={len(order)} "
                f"optimizer_steps={optimizer_steps} "
                f"lr={args.lr:.8g}",
                flush=True,
            )
            return optimizer_steps

        output_dir = checkpoint_path(
            args,
            prefix,
            global_epoch,
        )

        if output_dir.exists():
            raise RuntimeError(
                "Refusing to overwrite existing "
                f"checkpoint: {output_dir}"
            )

        print(
            f"Saving epoch {global_epoch} "
            f"checkpoint to {output_dir}",
            flush=True,
        )

        adapter.save(
            output_dir
        )
        save_adafactor_state(optimizer, output_dir)

        total_elapsed = (
            time.perf_counter()
            - training_start
        )

        save_continuation_metadata(
            args,
            output_dir,
            global_epoch,
            args.reference_logps,
            optimizer_steps,
            total_elapsed,
        )

        print(
            f"SAVED EPOCH {global_epoch}: "
            f"{output_dir}",
            flush=True,
        )

    return optimizer_steps


def main():
    args = parse_args()

    hostname = (
        socket.gethostname()
        .split(".")[0]
    )

    if hostname != args.machine:
        raise RuntimeError(
            f"Running on {hostname}, "
            f"but --machine={args.machine}"
        )

    prefix = checkpoint_prefix(
        args
    )

    end_epoch = (
        args.start_epoch
        + args.epochs
    )

    intended_checkpoints = [
        checkpoint_path(
            args,
            prefix,
            epoch,
        )
        for epoch in range(
            args.start_epoch + 1,
            end_epoch + 1,
        )
    ]

    existing = [
        path
        for path in intended_checkpoints
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "Refusing to start because continued "
            "checkpoint(s) already exist:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    this_run_dir = run_dir(
        args
    )

    metric_file = metrics_path(
        args
    )

    if metric_file.exists():
        raise RuntimeError(
            f"Refusing to overwrite metrics: "
            f"{metric_file}"
        )

    this_run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )
    torch.cuda.manual_seed_all(
        args.seed
    )

    base = load_base_module()

    rows = base.load_rows()

    base.validate_embedding_bank(
        args
    )

    indices = list(
        range(len(rows))
    )

    if len(indices) != 3000:
        raise RuntimeError(
            f"Expected 3000 training rows, "
            f"found {len(indices)}"
        )

    if args.preflight:
        indices = list(range(5)) + list(range(2000, 2005))

    reference = load_reference(
        args.reference_logps,
        indices,
    )

    print(
        "=" * 100,
        flush=True,
    )
    print(
        "SEA DPO CONTINUATION",
        flush=True,
    )
    print(
        "=" * 100,
        flush=True,
    )
    print(
        f"machine             : {hostname}",
        flush=True,
    )
    print(
        f"physical GPU        : {args.gpu}",
        flush=True,
    )
    print(
        f"model               : {args.model}",
        flush=True,
    )
    print(
        f"input checkpoint    : {args.model_path}",
        flush=True,
    )
    print(
        f"input epoch         : {args.start_epoch}",
        flush=True,
    )
    print(
        f"additional epochs   : {args.epochs}",
        flush=True,
    )
    print(
        f"output epochs       : "
        f"{args.start_epoch + 1}..{end_epoch}",
        flush=True,
    )
    print(
        f"learning rate       : {args.lr:.8g}",
        flush=True,
    )
    print(
        f"DPO beta            : {args.beta}",
        flush=True,
    )
    print(
        f"fixed reference     : "
        f"{args.reference_logps}",
        flush=True,
    )
    print(
        "decode validation   : disabled",
        flush=True,
    )
    print(
        "optimizer state     : fresh Adafactor "
        "(old checkpoints contain weights only)",
        flush=True,
    )

    for path in intended_checkpoints:
        print(
            f"will save           : {path}",
            flush=True,
        )

    print(
        "=" * 100,
        flush=True,
    )

    if args.model == "qwen25_omni":
        adapter = (
            base.Qwen25OmniAdapter(
                args.model_path,
                device,
            )
        )

        adapter.prepare_for_training()

    elif args.model == "salmonn2":
        adapter = (
            base.Salmonn2Adapter(
                args.model_path,
                device,
            )
        )

    else:
        raise AssertionError(
            args.model
        )

    verify_loaded_checkpoint(adapter, args)

    trainability = (
        base.validate_sea_trainability(
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
        f"Frozen audio encoder trunk: "
        f"{trainability['frozen_encoder_trunk_parameters']:,}/"
        f"{trainability['encoder_trunk_parameters']:,}",
        flush=True,
    )
    print(
        f"Trainable projector/connector: "
        f"{trainability['projector_trainable_parameters']:,}/"
        f"{trainability['projector_parameters']:,}",
        flush=True,
    )

    (
        optimizer,
        parameters,
    ) = build_optimizer(
        base,
        adapter,
        args.lr,
    )

    (
        probe_name,
        probe_parameter,
    ) = base.choose_probe_parameter(
        adapter.model
    )

    print(
        f"Update diagnostic parameter: "
        f"{probe_name}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    optimizer_steps = (
        train_dpo_continuation(
            base,
            adapter,
            args,
            rows,
            indices,
            optimizer,
            parameters,
            probe_parameter,
            reference,
            prefix,
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if args.preflight:
        print(
            f"PREFLIGHT FINISHED: {args.model}, "
            f"10 rows, {optimizer_steps} optimizer updates, "
            f"{elapsed:.2f}s, no checkpoint saved",
            flush=True,
        )
        return

    run_metadata = {
        "model": args.model,
        "method": "dpo",
        "continued_from":
            str(args.model_path.resolve()),
        "start_epoch":
            args.start_epoch,
        "end_epoch":
            end_epoch,
        "additional_epochs":
            args.epochs,
        "learning_rate":
            args.lr,
        "beta":
            args.beta,
        "fixed_reference_logps":
            str(
                args.reference_logps.resolve()
            ),
        "decode_validation":
            False,
        "optimizer":
            "Adafactor",
        "optimizer_state_resumed":
            False,
        "optimizer_steps":
            optimizer_steps,
        "elapsed_seconds":
            elapsed,
        "checkpoints": [
            str(path.resolve())
            for path in intended_checkpoints
        ],
    }

    metadata_path = (
        this_run_dir
        / "continuation_metadata.json"
    )

    if metadata_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: "
            f"{metadata_path}"
        )

    metadata_path.write_text(
        json.dumps(
            run_metadata,
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
        f"CONTINUATION COMPLETE: "
        f"epochs {args.start_epoch + 1}-{end_epoch}",
        flush=True,
    )
    print(
        f"optimizer steps: {optimizer_steps}",
        flush=True,
    )
    print(
        f"elapsed: {elapsed:.2f}s",
        flush=True,
    )
    print(
        f"metrics: {metric_file}",
        flush=True,
    )
    print(
        f"metadata: {metadata_path}",
        flush=True,
    )
    print(
        "=" * 100,
        flush=True,
    )


if __name__ == "__main__":
    main()

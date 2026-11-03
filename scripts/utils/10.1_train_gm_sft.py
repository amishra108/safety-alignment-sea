#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts/03_train_alignment.py"
DATASET = ROOT / "data/saferlhf/sea_3000_prepared.jsonl"
AUDIO_MANIFEST = (
    ROOT / "data/rlhf_tts_audio/manifests/manifest.shard000-of-001.jsonl"
)
OUTPUT_ROOT = ROOT / "outputs/10.1_train_gm_sft"

MODEL_FOLDERS = {
    "qwen25_omni": "qwen2.5-omni-7b_trained",
    "video4b": "video4b_trained",
}

MODEL_PREFIXES = {
    "qwen25_omni": "qwen25_omni_gm_sft",
    "video4b": "video4b_gm_sft",
}

LEARNING_RATE = 2e-5
SEED = 42


def load_original_training():
    if not SOURCE_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "gm_original_stage3", SOURCE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=tuple(MODEL_FOLDERS), required=True
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"Missing model: {args.model_path}")
    if args.epochs < 1 or args.grad_accum < 1:
        parser.error("Epochs and gradient accumulation must be positive")
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if args.preflight and args.resume:
        parser.error("--preflight and --resume cannot be combined")

    return args


def read_rows(preflight):
    with DATASET.open(encoding="utf-8") as handle:
        rows = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]

    if len(rows) != 3000:
        raise RuntimeError(f"Expected 3000 rows, found {len(rows)}")

    for index, row in enumerate(rows):
        expected = "harmful" if index < 2000 else "harmless"
        if row.get("category") != expected:
            raise RuntimeError(
                f"Row {index}: expected {expected}"
            )
        if not row.get("new_chosen_response"):
            raise RuntimeError(
                f"Row {index}: missing new_chosen_response"
            )

    return rows[:1] if preflight else rows


def load_audio_manifest(preflight):
    records = {}

    with AUDIO_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            index = int(record["row_index"])

            if index in records:
                raise RuntimeError(
                    f"Duplicate audio manifest row {index}"
                )

            records[index] = record

    required = range(1) if preflight else range(3000)

    for index in required:
        if index not in records:
            raise RuntimeError(
                f"Missing GM audio manifest row {index}"
            )

        path = ROOT / records[index]["gm_audio_path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError(f"Invalid WAV: {path}")

    return records


def load_waveform(path):
    waveform, sample_rate = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=16000,
        )

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.size < 16000:
        waveform = np.pad(
            waveform, (0, 16000 - waveform.size)
        )

    return waveform


class GMAdapter:
    def __init__(self, original, name, model_path, device):
        self.name = name
        self.original = original

        if name == "qwen25_omni":
            self.adapter = original.Qwen25OmniAdapter(
                model_path, device
            )
        else:
            self.adapter = original.Video4bAdapter(
                model_path, device
            )

        self.model = self.adapter.model
        self.encoder = self.adapter.audio_encoder
        self.projector = self.adapter.projector

        self.adapter.prepare_for_training()
        self.model.eval()

        self.validate_trainability()

    def validate_trainability(self):
        report = self.original.validate_sea_trainability(
            self.adapter
        )
        if report["frozen_encoder_trunk_parameters"] != (
            report["encoder_trunk_parameters"]
        ):
            raise RuntimeError("Audio encoder is not fully frozen")

        print(
            "Trainable parameters:",
            f'{report["trainable_parameters"]:,}',
            flush=True,
        )
        print(
            "Frozen encoder parameters:",
            f'{report["frozen_encoder_trunk_parameters"]:,}',
            flush=True,
        )
        print(
            "Trainable projector parameters:",
            f'{report["projector_trainable_parameters"]:,}',
            flush=True,
        )
        return report

    def instruction(self, row):
        if row["category"] == "harmful":
            return self.original.sea_audio_instruction(row)
        return row["prompt"]

    def video_audio_features(self, path):
        waveform = load_waveform(path)
        chunk_size = 30 * 16000
        chunks = [
            waveform[i:i + chunk_size]
            for i in range(0, waveform.size, chunk_size)
        ]

        backend = self.adapter.backend
        features = torch.stack([
            backend.feature_extractor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
            )["input_features"].squeeze(0)
            for chunk in chunks
        ])

        return features.to(
            device=self.adapter.device,
            dtype=self.encoder.conv1.weight.dtype,
        ), 60 * len(chunks)

    def training_batch(self, row, path):
        instruction = self.instruction(row)
        target = row["new_chosen_response"]

        if row["category"] == "harmless":
            return self.adapter.training_batch(
                instruction, target, with_audio=False
            )

        if self.name == "qwen25_omni":
            self.adapter.silence_path = path
            return self.adapter.training_batch(
                instruction, target, with_audio=True
            )

        backend = self.adapter.backend
        features, token_count = self.video_audio_features(path)
        backend.audio_features = features
        backend.audio_token_count = token_count

        return self.adapter.training_batch(
            instruction, target, with_audio=True
        )

    def loss(self, row, path):
        batch = self.training_batch(row, path)
        output = self.adapter.forward(batch)
        loss = output.loss

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Non-finite or missing SFT loss")

        del output
        del batch
        return loss

    def save(self, directory, original_path):
        self.adapter.save(directory)

        if self.name == "qwen25_omni":
            source = original_path / "spk_dict.pt"
            destination = directory / "spk_dict.pt"
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)


def build_optimizer(original, adapter):
    optimizer, parameters = original.build_optimizer(
        adapter.adapter, "sft"
    )

    actual_lr = optimizer.param_groups[0]["lr"]
    if not math.isclose(
        actual_lr, LEARNING_RATE, rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"Unexpected learning rate: {actual_lr}"
        )

    return optimizer, parameters


def run_name(args):
    return (
        f"{MODEL_PREFIXES[args.model]}"
        f"_lr2e-05_epoch{args.epochs}_final"
    )


def epoch_model_path(args, epoch):
    name = (
        f"{MODEL_PREFIXES[args.model]}"
        f"_lr2e-05_epoch{epoch}_final"
    )
    return ROOT / "models" / MODEL_FOLDERS[args.model] / name


def output_paths(args):
    return OUTPUT_ROOT / run_name(args), epoch_model_path(
        args, args.epochs
    )


def append_metric(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        handle.flush()


def gradient_norm(parameters):
    total = 0.0
    count = 0

    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue

        norm = grad.detach().float().norm().item()
        if not math.isfinite(norm):
            raise RuntimeError("Non-finite gradient norm")

        total += norm * norm
        count += 1

    result = math.sqrt(total)
    if count == 0 or result <= 0:
        raise RuntimeError("No nonzero trainable gradients")

    return result, count


def save_checkpoint(
    args, adapter, optimizer, output, epoch, position, step
):
    current = output / "checkpoint"
    temporary = output / "checkpoint.tmp"
    backup = output / "checkpoint.previous"

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir(parents=True)

    adapter.save(
        temporary / "model",
        args.model_path,
    )

    state = {
        "epoch": epoch,
        "position": position,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "model": args.model,
        "epochs": args.epochs,
        "grad_accum": args.grad_accum,
        "base_model": str(args.model_path.resolve()),
    }

    torch.save(state, temporary / "training_state.pt")

    if backup.exists():
        shutil.rmtree(backup)

    if current.exists():
        current.rename(backup)

    temporary.rename(current)

    if backup.exists():
        shutil.rmtree(backup)

    print(
        f"Checkpoint saved: epoch={epoch} "
        f"position={position} step={step}",
        flush=True,
    )


def find_checkpoint(output):
    current = output / "checkpoint"
    backup = output / "checkpoint.previous"

    if current.is_dir():
        return current
    if backup.is_dir():
        return backup

    raise RuntimeError("No resumable training checkpoint found")


def preflight(args, adapter, optimizer, parameters, rows, audios):
    row = rows[0]
    path = ROOT / audios[0]["gm_audio_path"]

    print(f"Preflight WAV: {path}", flush=True)
    print(f"Instruction: {adapter.instruction(row)}", flush=True)
    print("Target field: new_chosen_response", flush=True)

    torch.cuda.reset_peak_memory_stats()

    named = [
        (name, parameter)
        for name, parameter in adapter.model.named_parameters()
        if parameter.requires_grad
    ]

    losses = []
    changes_total = 0

    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)

        loss = adapter.loss(row, path)
        value = float(loss.detach().float().cpu())
        losses.append(value)

        loss.backward()
        grad, tensors = gradient_norm(parameters)

        snapshot = adapter.original.sampled_parameter_snapshot(
            named, samples_per_tensor=32
        )

        optimizer.step()
        torch.cuda.synchronize()

        changes = adapter.original.compare_parameter_snapshot(
            snapshot
        )
        changes_total += changes["changed_values"]

        if changes["changed_values"] == 0:
            raise RuntimeError(
                f"No stored parameter updates at step {step}"
            )

        print(
            f"STEP {step}/5 "
            f"loss={value:.8f} "
            f"grad={grad:.6e} "
            f"grad_tensors={tensors} "
            f"changed_values={changes['changed_values']} "
            f"max_delta={changes['max_abs_delta']:.6e}",
            flush=True,
        )

        del snapshot, loss

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        final_loss = float(
            adapter.loss(row, path).detach().float().cpu()
        )

    print()
    print(f"Initial loss: {losses[0]:.8f}")
    print(f"Final loss:   {final_loss:.8f}")
    print(f"Loss change:  {final_loss - losses[0]:+.8f}")
    print(f"Changed sampled values: {changes_total}")
    print(
        "Peak GPU memory:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
    )

    if final_loss >= losses[0]:
        raise RuntimeError(
            "Preflight loss did not decrease"
        )

    print(
        f"{args.model.upper()} GM SFT PREFLIGHT PASSED",
        flush=True,
    )


def train(args, adapter, optimizer, parameters, rows, audios, output):
    metrics = output / "training_metrics.jsonl"
    checkpoint = output / "checkpoint"

    if args.resume:
        saved = torch.load(
            find_checkpoint(output) / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])

        start_epoch = saved["epoch"]
        start_position = saved["position"]
        step = saved["step"]
    else:
        if checkpoint.exists() or metrics.exists():
            raise RuntimeError(
                "Existing training state found. Use --resume."
            )

        start_epoch = 1
        start_position = 0
        step = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        order = list(range(len(rows)))
        random.Random(SEED + epoch).shuffle(order)

        first_position = (
            start_position if epoch == start_epoch else 0
        )

        accumulated = 0
        accumulated_loss = 0.0

        for position in range(first_position, len(order)):
            index = order[position]
            row = rows[index]
            path = ROOT / audios[index]["gm_audio_path"]

            loss = adapter.loss(row, path)
            value = float(loss.detach().float().cpu())

            (loss / args.grad_accum).backward()
            del loss

            accumulated += 1
            accumulated_loss += value

            do_step = (
                accumulated == args.grad_accum
                or position + 1 == len(order)
            )

            if not do_step:
                continue

            grad, grad_tensors = gradient_norm(parameters)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            metric = {
                "epoch": epoch,
                "position": position + 1,
                "row_index": index,
                "optimizer_step": step,
                "loss": accumulated_loss / accumulated,
                "grad_norm": grad,
                "grad_tensors": grad_tensors,
                "learning_rate": LEARNING_RATE,
                "peak_gpu_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                ),
            }

            append_metric(metrics, metric)

            if step == 1 or step % 10 == 0:
                print(
                    f"SFT epoch={epoch}/{args.epochs} "
                    f"step={step} "
                    f"sample={position + 1}/{len(order)} "
                    f"row={index} "
                    f"loss={metric['loss']:.6f} "
                    f"grad={grad:.6e} "
                    f"peak={metric['peak_gpu_gib']:.2f}G",
                    flush=True,
                )

            accumulated = 0
            accumulated_loss = 0.0

            if step % args.save_every == 0:
                save_checkpoint(
                    args, adapter, optimizer, output,
                    epoch, position + 1, step
                )

        save_checkpoint(
            args, adapter, optimizer, output,
            epoch + 1, 0, step
        )

        epoch_model = epoch_model_path(args, epoch)
        if epoch_model.exists():
            if not args.resume:
                raise RuntimeError(
                    f"Refusing to overwrite epoch model: {epoch_model}"
                )
            print(f"Retaining existing epoch model: {epoch_model}", flush=True)
        else:
            adapter.save(epoch_model, args.model_path)
            print(f"Saved epoch {epoch}: {epoch_model}", flush=True)

        print(
            f"Completed epoch {epoch}/{args.epochs}",
            flush=True,
        )


def main():
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    rows = read_rows(args.preflight)
    audios = load_audio_manifest(args.preflight)
    original = load_original_training()

    output, final_model = output_paths(args)

    if args.preflight:
        print("Mode: five-step real-WAV preflight", flush=True)
        model_path = args.model_path
    elif args.resume:
        current = find_checkpoint(output)
        saved = torch.load(
            current / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )

        if (
            saved["model"] != args.model
            or saved["epochs"] != args.epochs
            or saved["grad_accum"] != args.grad_accum
            or saved["base_model"]
            != str(args.model_path.resolve())
        ):
            raise RuntimeError(
                "Checkpoint configuration does not match arguments"
            )

        model_path = current / "model"
    else:
        if final_model.exists():
            raise RuntimeError(
                f"Final model already exists: {final_model}"
            )
        model_path = args.model_path

    adapter = GMAdapter(
        original,
        args.model,
        model_path,
        torch.device("cuda:0"),
    )

    optimizer, parameters = build_optimizer(original, adapter)

    print(f"Model: {args.model}", flush=True)
    print(f"Model path: {model_path}", flush=True)
    print(f"Training examples: {len(rows)}", flush=True)
    print(f"Learning rate: {LEARNING_RATE}", flush=True)
    print(f"Optimizer: Adafactor", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Final model: {final_model}", flush=True)

    if args.preflight:
        preflight(
            args, adapter, optimizer, parameters, rows, audios
        )
        return

    output.mkdir(parents=True, exist_ok=True)

    train(
        args, adapter, optimizer, parameters,
        rows, audios, output
    )

    if not final_model.exists():
        raise RuntimeError(
            f"Final epoch model was not saved: {final_model}"
        )

    metadata = {
        "method": "gm_sft",
        "model": args.model,
        "base_model": str(args.model_path.resolve()),
        "training_examples": len(rows),
        "epochs": args.epochs,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adafactor",
        "audio_encoder_frozen": True,
        "projector_trainable": True,
        "chosen_response_field": "new_chosen_response",
        "audio_manifest": str(AUDIO_MANIFEST),
        "training_script": str(Path(__file__).name),
    }

    (final_model / "gm_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"GM SFT TRAINING COMPLETE: {final_model}", flush=True)


if __name__ == "__main__":
    main()

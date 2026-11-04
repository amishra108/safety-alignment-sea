#!/usr/bin/env python3
import argparse
import gc
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.training.adapters import load_adapter
from scripts.training.checkpoints import CheckpointManager, load_weights
from scripts.training.config import identity, write_json
from scripts.training.data import load_training_rows
from scripts.training.engine import optimizer_for, set_seed
from scripts.training.evaluation import decode_subset
from scripts.training.losses import (
    dpo_gradient_coefficient,
    dpo_loss,
    response_logprob,
    sft_loss,
)

DEFAULT_AUDIO_MANIFEST = (
    ROOT / "data/rlhf_tts_audio/manifests/manifest.shard000-of-001.jsonl"
)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GM real-audio SFT/DPO training"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen25_omni", "video4b"))
    parser.add_argument("--method", choices=("sft", "dpo"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--reference-model-path", type=Path)
    parser.add_argument("--initial-checkpoint-path", type=Path)
    parser.add_argument("--reference-checkpoint-path", type=Path)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--audio-manifest", type=Path, default=DEFAULT_AUDIO_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/_gm")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--decode-after-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--eval-split", choices=("benign", "harmful", "both"))
    parser.add_argument("--eval-language")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    if args.source_run.is_dir():
        args.source_run = args.source_run / "config.json"

    if not args.source_run.is_file():
        parser.error(f"Missing source configuration: {args.source_run}")

    source = json.loads(args.source_run.read_text(encoding="utf-8"))

    required = (
        "model", "method", "model_path", "dataset", "learning_rate",
        "epochs", "beta", "grad_accum", "seed",
    )
    missing = [key for key in required if key not in source]
    if missing:
        parser.error(f"Source configuration missing: {missing}")

    cfg = dict(source)

    overrides = {
        "model": args.model,
        "method": args.method,
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "reference_model_path": (
            str(args.reference_model_path.resolve())
            if args.reference_model_path else None
        ),
        "initial_checkpoint_path": (
            str(args.initial_checkpoint_path.resolve())
            if args.initial_checkpoint_path else None
        ),
        "reference_checkpoint_path": (
            str(args.reference_checkpoint_path.resolve())
            if args.reference_checkpoint_path else None
        ),
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "beta": args.beta,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "eval_examples": args.eval_examples,
        "eval_split": args.eval_split,
        "eval_language": args.eval_language,
        "max_new_tokens": args.max_new_tokens,
    }

    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if cfg["model"] not in ("qwen25_omni", "video4b"):
        parser.error("Invalid source model")

    if cfg["method"] not in ("sft", "dpo"):
        parser.error("Invalid source training method")

    if cfg["learning_rate"] <= 0 or cfg["epochs"] < 1:
        parser.error("Learning rate and epochs must be positive")

    if cfg["beta"] <= 0 or cfg["grad_accum"] < 1:
        parser.error("Beta and gradient accumulation must be positive")

    if args.save_every_steps < 0 or args.log_every < 1:
        parser.error("Save/log intervals must be positive")

    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")

    if not Path(cfg["model_path"]).is_dir():
        parser.error(f"Missing base model: {cfg['model_path']}")

    if cfg.get("initial_checkpoint_path"):
        if not Path(cfg["initial_checkpoint_path"]).is_dir():
            parser.error("Initial checkpoint does not exist")

    cfg["reference_model_path"] = (
        cfg.get("reference_model_path") or cfg["model_path"]
    )

    if cfg.get("reference_checkpoint_path"):
        if not Path(cfg["reference_checkpoint_path"]).is_dir():
            parser.error("Reference checkpoint does not exist")

    cfg["eval_split"] = cfg.get("eval_split", "both")
    cfg["eval_language"] = cfg.get("eval_language", "english")
    cfg["eval_examples"] = cfg.get("eval_examples", 10)
    cfg["eval_conditions"] = cfg.get(
        "eval_conditions",
        ["original", "literal", "narrative_framing", "content_dilution"],
    )
    cfg["eval_selection"] = cfg.get("eval_selection", "first")
    cfg["eval_seed"] = cfg.get("eval_seed", 42)
    cfg["eval_manifest"] = cfg.get(
        "eval_manifest",
        str(ROOT / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"),
    )
    cfg["audio_root"] = cfg.get("audio_root", str(ROOT))
    cfg["max_new_tokens"] = cfg.get("max_new_tokens", 1024)

    cfg["save_every_steps"] = args.save_every_steps
    cfg["log_every"] = args.log_every
    cfg["max_steps"] = args.max_steps
    cfg["resume"] = args.resume
    cfg["preflight"] = args.preflight
    cfg["decode_after_each_epoch"] = args.decode_after_each_epoch
    cfg["max_examples"] = 1 if args.preflight else 0

    cfg["gm_audio_manifest"] = str(args.audio_manifest.resolve())
    cfg["gm_source_config"] = str(args.source_run.resolve())
    cfg["gm_modality"] = "real_generated_audio"
    cfg["gm_harmful_modality"] = "real_audio"
    cfg["gm_harmless_modality"] = "text_only"

    cfg["entry_script"] = "scripts/train_gm.py"
    cfg["source_code_entrypoint"] = "scripts/train_gm.py"
    cfg["created_at"] = datetime.now(timezone.utc).isoformat()

    lr_text = f"{cfg['learning_rate']:.9g}".replace("+", "")
    if args.run_name:
        name = args.run_name
    else:
        name = (
            f"gm_{cfg['method']}_lr{lr_text}"
            f"_e{cfg['epochs']}_seed{cfg['seed']}"
        )
        if cfg["method"] == "dpo":
            name += f"_beta{cfg['beta']:.9g}"

    if not name or not all(
        character.isalnum() or character in "_.-"
        for character in name
    ):
        parser.error("Invalid run name")

    cfg["run_id"] = name
    cfg["output_root"] = str(args.output_root.resolve())

    run_root = args.output_root / cfg["model"] / name
    if args.preflight:
        run_root = run_root / "preflight"

    return cfg, run_root


def load_audio_paths(cfg, rows):
    manifest = Path(cfg["gm_audio_manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    records = {}

    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            index = int(record["row_index"])

            if index in records:
                raise ValueError(f"Duplicate GM audio row: {index}")

            records[index] = record

    paths = {}

    for index, row in enumerate(rows):
        if row["category"] != "harmful":
            continue

        if index not in records:
            raise ValueError(f"Missing GM audio for row {index}")

        audio_path = Path(records[index]["gm_audio_path"])
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        info = sf.info(audio_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Invalid GM audio: {audio_path}")

        paths[index] = audio_path

    return paths


def training_input_record(row, index, cfg, paths):
    return {
        "index": index,
        "category": row["category"],
        "phrase_type": row.get("phrase_type"),
        "instruction": instruction_for(row, cfg),
        "with_audio": row["category"] == "harmful",
        "audio_path": str(paths[index].resolve()) if index in paths else None,
        "chosen_response": row["new_chosen_response"],
        "rejected_response": (
            row["rejected_response"] if cfg["method"] == "dpo" else None
        ),
    }


def load_recorded_inputs(path, rows, cfg, paths):
    recorded = {}
    if not path.exists():
        return recorded

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            item = json.loads(line)
            index = item["index"]

            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError(f"Invalid GM training input index: {index}")

            if index in recorded:
                if recorded[index] != item:
                    raise ValueError(f"Conflicting GM input records for row {index}")
                raise ValueError(f"Duplicate GM input record for row {index}")

            expected = training_input_record(rows[index], index, cfg, paths)
            if item != expected:
                raise ValueError(
                    f"GM training input changed for row {index}: "
                    f"recorded={item!r}, current={expected!r}"
                )

            recorded[index] = item

    return recorded


def load_waveform(path):
    waveform, sample_rate = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=16000
        )

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.size < 16000:
        waveform = np.pad(waveform, (0, 16000 - waveform.size))

    return waveform


class GMAudio:
    def __init__(self, adapter, model_name):
        self.adapter = adapter
        self.model_name = model_name

    def prepare(self, audio_path):
        if audio_path is None:
            return

        if self.model_name == "qwen25_omni":
            if not hasattr(self.adapter, "silence_path"):
                raise RuntimeError(
                    "Qwen adapter lacks silence_path audio substitution"
                )

            self.adapter.silence_path = audio_path
            return

        waveform = load_waveform(audio_path)
        chunk_size = 30 * 16000

        chunks = [
            waveform[start:start + chunk_size]
            for start in range(0, len(waveform), chunk_size)
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

        backend.audio_features = features.to(
            device=self.adapter.device,
            dtype=self.adapter.audio_encoder.conv1.weight.dtype,
        )
        backend.audio_token_count = 60 * len(chunks)


def instruction_for(row, cfg):
    if row["category"] == "harmless":
        return cfg.get("text_instruction", "{prompt}").format(
            prompt=row["prompt"]
        )

    phrase_type = row["phrase_type"]

    if phrase_type == "activity":
        prefix = cfg["instruction_prefix_activity"]
    elif phrase_type == "product":
        prefix = cfg["instruction_prefix_product"]
    else:
        raise ValueError(f"Invalid phrase type: {phrase_type}")

    return prefix + row["rewritten_prompt"]


def forward_target(adapter, audio, row, index, target, cfg, paths):
    harmful = row["category"] == "harmful"
    audio.prepare(paths[index] if harmful else None)

    instruction = instruction_for(row, cfg)
    batch = adapter.training_batch(instruction, target, harmful)
    output = adapter.forward(batch)
    return output, batch


def target_logprob(adapter, audio, row, index, target, cfg, paths):
    output, batch = forward_target(
        adapter, audio, row, index, target, cfg, paths
    )
    result = response_logprob(output.logits, batch["labels"])[0]
    del output, batch
    return result


def reference_logprobs(cfg, rows, indices, paths, root, device):
    destination = root / "reference_logps.jsonl"
    cache = {}

    if destination.exists():
        with destination.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    cache[record["index"]] = record

    missing = [index for index in indices if index not in cache]
    if not missing:
        return cache

    print(
        f"GM DPO reference: {len(missing)} pairs to calculate",
        flush=True,
    )

    adapter = load_adapter(
        cfg["model"], cfg["reference_model_path"], device
    )

    if cfg.get("reference_checkpoint_path"):
        load_weights(adapter.model, Path(cfg["reference_checkpoint_path"]))

    adapter.model.eval()
    audio = GMAudio(adapter, cfg["model"])

    with torch.no_grad():
        for number, index in enumerate(missing, 1):
            row = rows[index]

            chosen = float(target_logprob(
                adapter, audio, row, index,
                row["new_chosen_response"], cfg, paths
            ))
            rejected = float(target_logprob(
                adapter, audio, row, index,
                row["rejected_response"], cfg, paths
            ))

            record = {
                "index": index,
                "chosen_logp": chosen,
                "rejected_logp": rejected,
            }

            append_jsonl(destination, record)
            cache[index] = record

            if number == 1 or number % 25 == 0 or number == len(missing):
                print(
                    f"GM REFERENCE {number}/{len(missing)} row={index}",
                    flush=True,
                )

    del audio, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache


def train_step(adapter, audio, row, index, cfg, paths, reference, factor):
    if cfg["method"] == "sft":
        output, batch = forward_target(
            adapter, audio, row, index,
            row["new_chosen_response"], cfg, paths
        )
        loss = sft_loss(output)
        value = float(loss.detach().float())
        (loss / factor).backward()
        del output, batch, loss
        return value, {}

    with torch.no_grad():
        chosen_logp = float(target_logprob(
            adapter, audio, row, index,
            row["new_chosen_response"], cfg, paths
        ))
        rejected_logp = float(target_logprob(
            adapter, audio, row, index,
            row["rejected_response"], cfg, paths
        ))

    margin = chosen_logp - rejected_logp
    reference_margin = (
        reference["chosen_logp"] - reference["rejected_logp"]
    )

    coefficient = dpo_gradient_coefficient(
        margin, reference_margin, cfg["beta"]
    ) / factor

    chosen = target_logprob(
        adapter, audio, row, index,
        row["new_chosen_response"], cfg, paths
    )
    (chosen * coefficient).backward()
    del chosen

    rejected = target_logprob(
        adapter, audio, row, index,
        row["rejected_response"], cfg, paths
    )
    (-rejected * coefficient).backward()
    del rejected

    value = float(dpo_loss(
        torch.tensor(chosen_logp, dtype=torch.float64),
        torch.tensor(rejected_logp, dtype=torch.float64),
        torch.tensor(reference["chosen_logp"], dtype=torch.float64),
        torch.tensor(reference["rejected_logp"], dtype=torch.float64),
        cfg["beta"],
    ))

    return value, {
        "policy_margin": margin,
        "reference_margin": reference_margin,
        "advantage": margin - reference_margin,
    }


def sync_responses_txt(destination):
    if not destination.is_file():
        return

    text_path = destination.with_suffix(".txt")

    with destination.open(encoding="utf-8") as source:
        with text_path.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue

                response = json.loads(line)["response"]
                target.write(
                    response.replace("\r", r"\r").replace("\n", r"\n")
                    + "\n"
                )


def evaluate_epoch(adapter, cfg, epoch, root):
    destination = (
        root / "evaluations" / f"epoch_{epoch:03d}" / "responses.jsonl"
    )

    try:
        decode_subset(adapter, cfg, epoch, destination)
    finally:
        sync_responses_txt(destination)


def train(cfg, root):
    if not torch.cuda.is_available():
        raise RuntimeError("GM training requires CUDA")

    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))

        if identity(existing) != identity(cfg):
            raise ValueError(
                f"Training settings conflict with existing run: {root}"
            )

        if existing["epochs"] > cfg["epochs"]:
            raise ValueError("Cannot shorten an existing GM run")
    else:
        write_json(config_path, cfg)

    rows = load_training_rows(cfg["dataset"])
    if len(rows) != 3000:
        raise RuntimeError(f"Expected 3000 SEA rows, got {len(rows)}")

    indices = list(range(len(rows)))

    if cfg["preflight"]:
        indices = [
            next(i for i, row in enumerate(rows)
                 if row["category"] == "harmful")
        ]

    paths = load_audio_paths(cfg, rows)
    inputs_path = root / "training_inputs.jsonl"
    recorded_inputs = load_recorded_inputs(inputs_path, rows, cfg, paths)

    device = torch.device("cuda:0")
    set_seed(cfg["seed"])

    reference = None
    if cfg["method"] == "dpo":
        reference = reference_logprobs(
            cfg, rows, indices, paths, root, device
        )

    adapter = load_adapter(cfg["model"], cfg["model_path"], device)

    if cfg.get("initial_checkpoint_path"):
        load_weights(adapter.model, Path(cfg["initial_checkpoint_path"]))

    adapter.prepare_for_training()

    if cfg["model"] == "video4b":
        adapter.model.eval()
    else:
        adapter.model.train()

    optimizer = optimizer_for(adapter.model, cfg["learning_rate"])
    audio = GMAudio(adapter, cfg["model"])
    manager = CheckpointManager(root)

    state = None

    if cfg["resume"] != "never":
        state = manager.load_resume(adapter.model, optimizer)
        if cfg["resume"] == "required" and state is None:
            raise RuntimeError("No resumable GM checkpoint exists")
    elif (root / "resume/latest.json").exists():
        raise RuntimeError("Refusing to overwrite existing GM state")

    if state is None:
        state = {"epoch": 1, "position": 0, "global_step": 0}

    print(
        f"GM model={cfg['model']} method={cfg['method']} "
        f"lr={cfg['learning_rate']} epochs={cfg['epochs']} "
        f"beta={cfg['beta']} examples={len(indices)}",
        flush=True,
    )
    print(f"GM output: {root}", flush=True)

    if cfg["decode_after_each_epoch"] and not cfg["preflight"]:
        for prior_epoch in range(
            1, min(state["epoch"], cfg["epochs"] + 1)
        ):
            snapshot = (
                root / "checkpoints" / f"epoch_{prior_epoch:03d}"
            )

            if not snapshot.is_dir():
                raise FileNotFoundError(snapshot)

            load_weights(adapter.model, snapshot)
            evaluate_epoch(adapter, cfg, prior_epoch, root)

        if (root / "resume/latest.json").exists():
            state = manager.load_resume(adapter.model, optimizer)

    if state["epoch"] > cfg["epochs"]:
        print(f"GM RUN COMPLETE: {root}", flush=True)
        return

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(state["epoch"], cfg["epochs"] + 1):
        order = list(indices)
        random.Random(cfg["seed"] + epoch).shuffle(order)

        offset = (
            state["position"] if epoch == state["epoch"] else 0
        )

        losses = []

        for start in range(offset, len(order), cfg["grad_accum"]):
            group = order[start:start + cfg["grad_accum"]]
            metrics = []
            step_inputs = []

            for index in group:
                row = rows[index]
                input_record = training_input_record(row, index, cfg, paths)

                if (
                    index in recorded_inputs
                    and recorded_inputs[index] != input_record
                ):
                    raise ValueError(f"GM training input changed for row {index}")

                step_inputs.append(input_record)

                value, extra = train_step(
                    adapter, audio, row, index, cfg, paths,
                    reference[index] if reference is not None else None,
                    len(group),
                )

                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite GM loss for row {index}"
                    )

                metrics.append({
                    "row": index,
                    "loss": value,
                    **extra,
                })
                losses.append(value)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            for input_record in step_inputs:
                index = input_record["index"]
                if index not in recorded_inputs:
                    append_jsonl(inputs_path, input_record)
                    recorded_inputs[index] = input_record

            state = {
                "epoch": epoch,
                "position": start + len(group),
                "global_step": state["global_step"] + 1,
            }

            record = {
                "type": "step",
                "epoch": epoch,
                "position": state["position"],
                "global_step": state["global_step"],
                "loss": statistics.fmean(
                    item["loss"] for item in metrics
                ),
                "rows": group,
                "lr": cfg["learning_rate"],
            }

            append_jsonl(root / "metrics.jsonl", record)

            if (
                state["global_step"] == 1
                or state["global_step"] % cfg["log_every"] == 0
            ):
                print(
                    f"GM {cfg['method'].upper()} "
                    f"epoch={epoch}/{cfg['epochs']} "
                    f"position={state['position']}/{len(order)} "
                    f"step={state['global_step']} "
                    f"loss={record['loss']:.6f}",
                    flush=True,
                )

            if cfg["preflight"]:
                write_json(
                    root / "preflight.json",
                    {"passed": True, "metrics": metrics},
                )
                print("GM PREFLIGHT PASSED", flush=True)
                return

            if cfg["save_every_steps"] and state["global_step"] % cfg["save_every_steps"] == 0:
                manager.save_resume(adapter.model, optimizer, state)

            if (
                cfg["max_steps"]
                and state["global_step"] >= cfg["max_steps"]
            ):
                manager.save_resume(adapter.model, optimizer, state)
                print("GM saved resumable step checkpoint", flush=True)
                return

        append_jsonl(
            root / "metrics.jsonl",
            {
                "type": "epoch",
                "epoch": epoch,
                "mean_loss": statistics.fmean(losses) if losses else None,
                "global_step": state["global_step"],
            },
        )

        snapshot = root / "checkpoints" / f"epoch_{epoch:03d}"

        if not snapshot.exists():
            manager.save_epoch(adapter.model, epoch)

        next_state = {
            "epoch": epoch + 1,
            "position": 0,
            "global_step": state["global_step"],
        }

        manager.save_resume(adapter.model, optimizer, next_state)
        state = next_state

        print(f"GM EPOCH {epoch} WEIGHTS AND OPTIMIZER SAVED", flush=True)

        if cfg["decode_after_each_epoch"]:
            evaluate_epoch(adapter, cfg, epoch, root)

    print(f"GM TRAINING COMPLETE: {root}", flush=True)


if __name__ == "__main__":
    configuration, output_directory = parse_args()
    train(configuration, output_directory)

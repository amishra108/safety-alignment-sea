#!/usr/bin/env python3
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.training.engine import train

PRESETS = {
    ("qwen25_omni", "sft"): (4e-6, 2),
    ("qwen25_omni", "dpo"): (8e-6, 1),
    ("video4b", "sft"): (2e-6, 1),
    ("video4b", "dpo"): (4e-6, 1),
}
BASE_NAMES = {
    "qwen25_omni": "qwen2.5-omni-7b",
    "video4b": "video_salmonn2_pro_4b",
}
SHORT_NAMES = {
    "qwen25_omni": "omni",
    "video4b": "video4b",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Text-only SFT/DPO using existing SEA training engine"
    )
    p.add_argument("--model", choices=sorted(BASE_NAMES), required=True)
    p.add_argument("--method", choices=("sft", "dpo"), required=True)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--reference-model-path", type=Path)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/saferlhf/sea_3000_prepared.jsonl",
    )
    p.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/_text"
    )
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every-steps", type=int, default=25)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()

    lr, epochs = PRESETS[(args.model, args.method)]
    args.lr = lr if args.lr is None else args.lr
    args.epochs = epochs if args.epochs is None else args.epochs
    args.model_path = (
        args.model_path or ROOT / "models" / BASE_NAMES[args.model]
    ).resolve()
    args.reference_model_path = (
        args.reference_model_path or args.model_path
    ).resolve()

    for name in ("model_path", "reference_model_path"):
        if not getattr(args, name).is_dir():
            p.error(f"Missing {name}: {getattr(args, name)}")
    if not args.dataset.is_file():
        p.error(f"Missing dataset: {args.dataset}")
    if args.lr <= 0 or args.epochs < 1 or args.beta <= 0 or args.grad_accum < 1:
        p.error("Learning rate, epochs, beta and grad accumulation must be positive")
    if args.log_every < 1 or args.save_every_steps < 1 or args.max_steps < 0:
        p.error("Invalid save/log/max-step interval")
    return args


def main():
    args = parse_args()

    with args.dataset.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) != 3000:
        raise ValueError(f"Expected 3000 training rows, found {len(rows)}")

    for index, row in enumerate(rows):
        category = "harmful" if index < 2000 else "harmless"
        if row.get("category") != category:
            raise ValueError(f"Unexpected category at row {index}")
        for field in ("prompt", "new_chosen_response", "rejected_response"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field} at row {index}")

    run_name = (
        f"{SHORT_NAMES[args.model]}_text_{args.method}"
        f"_lr{args.lr:.9g}_e{args.epochs}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("Invalid run name")

    root = args.output_root.resolve() / args.model / run_name
    if args.preflight:
        root /= "preflight"

    cfg = {
        "model": args.model,
        "method": args.method,
        "model_path": str(args.model_path),
        "reference_model_path": str(args.reference_model_path),
        "initial_checkpoint_path": None,
        "reference_checkpoint_path": None,
        "dataset": str(args.dataset.resolve()),
        "training_modality": "text_only",
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "beta": args.beta,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "log_every": args.log_every,
        "save_every_steps": args.save_every_steps,
        "max_steps": args.max_steps,
        "max_examples": 0,
        "resume": args.resume,
        "preflight": args.preflight,
        "decode_after_each_epoch": False,
        "text_instruction": "{prompt}",
        "run_id": run_name,
        "output_root": str(args.output_root.resolve()),
        "entry_script": "scripts/train_text.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"RUN: {root}", flush=True)
    print(
        f"MODEL: {args.model_path} | METHOD: {args.method} "
        f"| LR: {args.lr} | EPOCHS: {args.epochs}",
        flush=True,
    )
    train(cfg, root)


if __name__ == "__main__":
    main()

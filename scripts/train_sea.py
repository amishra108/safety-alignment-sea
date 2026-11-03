#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.training.config import cli, config_dict, run_id
from scripts.training.engine import train
from scripts.training.evaluation import decode_subset


def main():
    args = cli(hpo=False)
    lr = args.learning_rates[0]
    epochs = args.epoch_values[0]
    beta = args.betas[0]
    name = run_id(args, lr, epochs, beta)
    cfg = config_dict(args, lr, epochs, beta, 'scripts/train_sea.py', name)
    work_root = Path(args.output_root) / args.model / name

    def evaluate(adapter, epoch, run_root):
        destination = (
            run_root / 'evaluations' /
            f'epoch_{epoch:03d}' / 'responses.jsonl'
        )
        decode_subset(adapter, cfg, epoch, destination)

    train(
        cfg,
        work_root,
        on_epoch_end=evaluate if args.decode_after_each_epoch else None,
    )


if __name__ == '__main__':
    main()

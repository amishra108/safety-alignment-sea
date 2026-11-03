#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.training.config import cli, config_dict, run_id, write_json
from scripts.training.engine import train
from scripts.training.evaluation import decode_subset
from scripts.training.data import select_evaluation


def main():
    args = cli(hpo=True)
    epochs = sorted(args.epoch_values)
    expected = len(select_evaluation(args.eval_manifest, args.eval_split, args.eval_language,
                                     args.eval_examples, args.eval_conditions, args.eval_selection, args.eval_seed))
    for lr in args.learning_rates:
        for beta in (args.betas if args.method == 'dpo' else [args.betas[0]]):
            max_epoch = max(epochs)
            parent_name = run_id(args, lr, max_epoch, beta)
            work = Path(args.output_root) / '_work' / args.model / parent_name
            cfg = config_dict(args, lr, max_epoch, beta, 'scripts/hpo.py', parent_name)
            completed = set()
            def result_name(epoch):
                return run_id(args, lr, epoch, beta)

            all_complete = True
            for epoch in epochs:
                result_dir = Path(args.output_root) / args.model / result_name(epoch)
                file = result_dir / 'responses.jsonl'
                config_file = result_dir / 'config.json'
                if not file.is_file() or not config_file.is_file():
                    all_complete = False
                    break
                lines = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
                if len(lines) != expected or len({r['task_id'] for r in lines}) != expected:
                    all_complete = False
                    break
            if all_complete:
                print(f'HPO ALREADY COMPLETE: {args.model} lr={lr} beta={beta} epochs={epochs}', flush=True)
                continue

            def evaluate(adapter, epoch, work_root):
                if epoch not in epochs:
                    return
                name = run_id(args, lr, epoch, beta)
                result_dir = Path(args.output_root) / args.model / name
                result_dir.mkdir(parents=True, exist_ok=True)
                config_file = result_dir / 'config.json'
                eval_cfg = config_dict(args, lr, epoch, beta, 'scripts/hpo.py', name)
                eval_cfg['model_runtime'] = json.loads((work_root / 'config.json').read_text())['model_runtime']
                eval_cfg['checkpoint_retained_after_hpo'] = args.keep_hpo_checkpoints
                eval_cfg['training_checkpoint_path'] = str(work_root / 'checkpoints' / f'epoch_{epoch:03d}')
                if config_file.exists():
                    old = json.loads(config_file.read_text())
                    for key, value in eval_cfg.items():
                        if key != 'created_at' and old.get(key) != value:
                            raise ValueError(f'Conflicting HPO result configuration: {config_file} key={key}')
                else:
                    write_json(config_file, eval_cfg)
                n = decode_subset(adapter, eval_cfg, epoch, result_dir / 'responses.jsonl')
                completed.add(epoch)
                print(f'HPO EVALUATION DONE: {name}, {n} responses', flush=True)

            train(cfg, work, on_epoch_end=evaluate)
            if set(epochs) == completed and not args.keep_hpo_checkpoints and not args.preflight:
                shutil.rmtree(work)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.training.config import cli, config_dict, run_id, write_json, identity
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

            if args.continue_from:
                if len(args.learning_rates) != 1:
                    raise ValueError(
                        '--continue-from requires exactly one learning rate'
                    )
                if args.method == 'dpo' and len(args.betas) != 1:
                    raise ValueError(
                        '--continue-from requires exactly one DPO beta'
                    )
                if args.preflight:
                    raise ValueError(
                        '--continue-from cannot be combined with --preflight'
                    )
                if args.max_steps:
                    raise ValueError(
                        'Do not use --max-steps when extending an HPO run'
                    )

                work = Path(args.continue_from).resolve()
                if work.is_file() and work.name == 'config.json':
                    work = work.parent

                expected_parent = (
                    Path(args.output_root).resolve()
                    / '_work' / args.model
                )
                if work.parent != expected_parent:
                    raise ValueError(
                        f'Continuation must use an HPO training workspace '
                        f'under {expected_parent}, got {work}'
                    )

                existing_config_path = work / 'config.json'
                if not existing_config_path.is_file():
                    raise FileNotFoundError(existing_config_path)

                existing = json.loads(
                    existing_config_path.read_text(encoding='utf-8')
                )

                if identity(existing) != identity(cfg):
                    raise ValueError(
                        'Source HPO training configuration differs from '
                        'the requested configuration. Check model, method, '
                        'learning rate, beta, seed, dataset and run name.'
                    )

                previous_target = existing['epochs']
                first_requested = min(epochs)

                if max_epoch < previous_target:
                    raise ValueError(
                        'Cannot shorten an existing HPO training run'
                    )

                if max_epoch == previous_target:
                    saved_first = existing.get(
                        'hpo_continuation_first_epoch'
                    )
                    if saved_first != first_requested:
                        raise ValueError(
                            'Existing continuation target does not match '
                            'the requested epochs'
                        )
                else:
                    if first_requested != previous_target + 1:
                        raise ValueError(
                            f'Expected the next requested epoch to be '
                            f'{previous_target + 1}, got {first_requested}'
                        )
                    if epochs != list(
                        range(first_requested, max_epoch + 1)
                    ):
                        raise ValueError(
                            'Continuation epochs must be consecutive'
                        )

                    snapshot = (
                        work / 'checkpoints'
                        / f'epoch_{previous_target:03d}'
                    )
                    if not (
                        (snapshot / 'model.safetensors').is_file()
                        and (snapshot / 'manifest.json').is_file()
                    ):
                        raise FileNotFoundError(
                            f'Missing completed epoch checkpoint: {snapshot}'
                        )

                    pointer = work / 'resume' / 'latest.json'
                    if not pointer.is_file():
                        raise FileNotFoundError(
                            f'Missing optimizer resume pointer: {pointer}'
                        )

                    resume_name = json.loads(
                        pointer.read_text(encoding='utf-8')
                    )['checkpoint']
                    resume_dir = work / 'resume' / resume_name
                    for filename in (
                        'model.safetensors', 'manifest.json',
                        'training.pt', 'state.json',
                    ):
                        if not (resume_dir / filename).is_file():
                            raise FileNotFoundError(resume_dir / filename)

                    resume_state = json.loads(
                        (resume_dir / 'state.json').read_text(
                            encoding='utf-8'
                        )
                    )
                    if (
                        resume_state['epoch'] != previous_target + 1
                        or resume_state['position'] != 0
                    ):
                        raise ValueError(
                            'The previous training target has not completed '
                            'cleanly; finish or resume that run first'
                        )

                    existing['epochs'] = max_epoch
                    existing['hpo_continuation_first_epoch'] = (
                        first_requested
                    )
                    write_json(existing_config_path, existing)

                cfg['resume'] = 'required'
                print(
                    f'HPO CONTINUATION: {work} '
                    f'epochs={epochs} lr={lr} beta={beta}',
                    flush=True,
                )

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


if __name__ == '__main__':
    main()

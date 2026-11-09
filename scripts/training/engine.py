import gc
import json
import math
import random
import statistics
from pathlib import Path
import torch
from transformers.optimization import Adafactor
from .adapters import inject_embedding, load_adapter
from .checkpoints import CheckpointManager, load_weights
from .config import identity, write_json
from .data import embedding_path, load_training_rows, training_sample
from .losses import sft_loss, response_logprob, dpo_gradient_coefficient, dpo_loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_for(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return Adafactor(params, lr=lr, eps=(1e-30, 1e-3), clip_threshold=1.0,
                     decay_rate=-0.8, beta1=None, weight_decay=0.0,
                     relative_step=False, scale_parameter=False, warmup_init=False)


def forward_target(adapter, instruction, target, embedding, with_audio):
    batch = adapter.training_batch(instruction, target, with_audio)
    if embedding is None:
        output = adapter.forward(batch)
    else:
        with inject_embedding(adapter.projector, embedding):
            output = adapter.forward(batch)
    return output, batch


def logprob(adapter, instruction, target, embedding, with_audio):
    output, batch = forward_target(adapter, instruction, target, embedding, with_audio)
    result = response_logprob(output.logits, batch['labels'])[0]
    del output, batch
    return result


def append(path, record):
    with Path(path).open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')
        file.flush()


def reference_logps(cfg, rows, indices, root, device):
    path = Path(root) / 'reference_logps.jsonl'
    cache = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item['index']] = item
    missing = [i for i in indices if i not in cache]
    if not missing:
        return cache
    print(f'Computing {len(missing)} fixed DPO reference pairs from {cfg["reference_model_path"]}', flush=True)
    adapter = load_adapter(cfg['model'], cfg['reference_model_path'], device)
    if cfg['reference_checkpoint_path']:
        load_weights(adapter.model, cfg['reference_checkpoint_path'])
    adapter.model.eval()
    with torch.no_grad():
        for k, index in enumerate(missing, 1):
            row = rows[index]
            instruction, embedding, with_audio = training_sample(rows, index, cfg)
            chosen = float(logprob(adapter, instruction, row['new_chosen_response'], embedding, with_audio))
            rejected = float(logprob(adapter, instruction, row['rejected_response'], embedding, with_audio))
            record = {'index': index, 'chosen_logp': chosen, 'rejected_logp': rejected}
            append(path, record)
            cache[index] = record
            if k == 1 or k % 25 == 0 or k == len(missing):
                print(f'REFERENCE {k}/{len(missing)} row={index}', flush=True)
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def dpo_step(adapter, row, sample, reference, beta, factor):
    instruction, embedding, with_audio = sample
    with torch.no_grad():
        current_chosen = float(logprob(adapter, instruction, row['new_chosen_response'], embedding, with_audio))
        current_rejected = float(logprob(adapter, instruction, row['rejected_response'], embedding, with_audio))
    margin = current_chosen - current_rejected
    ref_margin = reference['chosen_logp'] - reference['rejected_logp']
    coeff = dpo_gradient_coefficient(margin, ref_margin, beta) / factor
    chosen = logprob(adapter, instruction, row['new_chosen_response'], embedding, with_audio)
    (chosen * coeff).backward()
    del chosen
    rejected = logprob(adapter, instruction, row['rejected_response'], embedding, with_audio)
    (-rejected * coeff).backward()
    del rejected
    loss = float(dpo_loss(torch.tensor(current_chosen, dtype=torch.float64),
                          torch.tensor(current_rejected, dtype=torch.float64),
                          torch.tensor(reference['chosen_logp'], dtype=torch.float64),
                          torch.tensor(reference['rejected_logp'], dtype=torch.float64), beta))
    return loss, {'policy_margin': margin, 'reference_margin': ref_margin, 'advantage': margin - ref_margin}


def train(cfg, work_root, on_epoch_end=None):
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    config_path = work_root / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if identity(previous) != identity(cfg):
            raise ValueError(f'Existing run has different training settings: {work_root}')
        if previous['epochs'] > cfg['epochs']:
            raise ValueError('Cannot shorten an existing run; choose a different run directory')
        if previous.get('decode_after_each_epoch') != cfg['decode_after_each_epoch']:
            previous['decode_after_each_epoch'] = cfg['decode_after_each_epoch']
            write_json(config_path, previous)
    else:
        write_json(config_path, cfg)
    rows = load_training_rows(cfg['dataset'])
    indices = list(range(len(rows)))
    if cfg['max_examples']:
        indices = indices[:cfg['max_examples']]
    if cfg['preflight']:
        harmful = next((i for i in indices if rows[i]['category'] == 'harmful'), None)
        if harmful is None:
            raise ValueError('Preflight requires a harmful example')
        indices = [harmful]
    if cfg.get('training_modality') != 'text_only' and any(rows[i]['category'] == 'harmful' for i in indices):
        missing = [i for i in indices if rows[i]['category'] == 'harmful' and not (Path(cfg['embedding_dir']) / f'row_{i:04d}.pt').is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} embedding(s), first index: {missing[0]}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])
    ref = reference_logps(cfg, rows, indices, work_root, device) if cfg['method'] == 'dpo' else None
    adapter = load_adapter(cfg['model'], cfg['model_path'], device)
    if cfg['initial_checkpoint_path']:
        load_weights(adapter.model, cfg['initial_checkpoint_path'])
    adapter.prepare_for_training()
    optimizer = optimizer_for(adapter.model, cfg['learning_rate'])
    projector_names = [name for name, module in adapter.model.named_modules() if module is adapter.projector]
    if len(projector_names) != 1:
        raise RuntimeError(f'Expected one projector module, found {projector_names}')
    runtime = {
        'adapter_class': f'{adapter.__class__.__module__}.{adapter.__class__.__name__}',
        'model_class': f'{adapter.model.__class__.__module__}.{adapter.model.__class__.__name__}',
        'processor_class': (getattr(adapter, 'processor', getattr(adapter, 'backend', None)).__class__.__name__),
        'audio_encoder_class': adapter.audio_encoder.__class__.__name__,
        'projector_module_path': projector_names[0],
        'trainable_parameters': sum(p.numel() for p in adapter.model.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in adapter.model.parameters() if not p.requires_grad),
        'parameter_dtypes': sorted({str(p.dtype) for p in adapter.model.parameters()}),
    }
    runtime['optimizer'] = {
        'class': f'{type(optimizer).__module__}.{type(optimizer).__name__}',
        'defaults': optimizer.defaults.copy(),
        'parameter_groups': [
            {key: value for key, value in group.items() if key != 'params'}
            for group in optimizer.param_groups
        ],
        'parameter_tensor_count': sum(
            len(group['params']) for group in optimizer.param_groups
        ),
    }
    runtime['model_state'] = {
        'use_cache': getattr(adapter.model.config, 'use_cache', None),
        'gradient_checkpointing': bool(
            getattr(adapter.model, 'is_gradient_checkpointing', False)
        ),
        'dropout_probabilities': {
            name: module.p
            for name, module in adapter.model.named_modules()
            if isinstance(module, torch.nn.Dropout)
        },
    }

    existing = json.loads(config_path.read_text())
    runtime = json.loads(json.dumps(runtime))
    previous_runtime = existing.get('model_runtime')
    if previous_runtime is not None:
        original_fields = {
            key: value for key, value in runtime.items()
            if key not in ('optimizer', 'model_state')
        }
        if previous_runtime != runtime and previous_runtime != original_fields:
            raise ValueError(
                'Loaded model or optimizer differs from recorded runtime metadata'
            )

    if previous_runtime != runtime or existing.get('training_inputs_file') is None:
        existing['model_runtime'] = runtime
        existing['training_inputs_file'] = 'training_inputs.jsonl'
        write_json(config_path, existing)
    if cfg['model'] == 'video4b':
        adapter.model.eval()
    else:
        adapter.model.train()
    manager = CheckpointManager(work_root)
    state = None
    if cfg['resume'] != 'never':
        state = manager.load_resume(adapter.model, optimizer)
        if cfg['resume'] == 'required' and state is None:
            raise ValueError(f'No resumable checkpoint in {work_root}')
    elif (work_root / 'resume/latest.json').exists():
        raise ValueError('Refusing to overwrite an existing resumable checkpoint')
    if state is None:
        state = {'epoch': 1, 'position': 0, 'global_step': 0}
    if on_epoch_end is not None and not cfg['preflight']:
        for prior_epoch in range(1, min(state['epoch'], cfg['epochs'] + 1)):
            snapshot = work_root / 'checkpoints' / f'epoch_{prior_epoch:03d}'
            if snapshot.is_dir():
                load_weights(adapter.model, snapshot)
                on_epoch_end(adapter, prior_epoch, work_root)
        if (work_root / 'resume/latest.json').exists():
            state = manager.load_resume(adapter.model, optimizer)
    if state['epoch'] > cfg['epochs']:
        print(f'RUN COMPLETE: {work_root}', flush=True)
        return
    inputs_path = work_root / 'training_inputs.jsonl'
    recorded_inputs = {}

    if inputs_path.exists():
        with inputs_path.open(encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    index = item['index']
                    if index in recorded_inputs and recorded_inputs[index] != item:
                        raise ValueError(f'Conflicting input records for row {index}')
                    recorded_inputs[index] = item

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(state['epoch'], cfg['epochs'] + 1):
        order = list(indices)
        random.Random(cfg['seed'] + epoch).shuffle(order)
        offset = state['position'] if epoch == state['epoch'] else 0
        losses = []
        for start in range(offset, len(order), cfg['grad_accum']):
            group = order[start:start + cfg['grad_accum']]
            metrics = []
            step_inputs = []
            for index in group:
                row = rows[index]
                sample = training_sample(rows, index, cfg)
                instruction, embedding, with_audio = sample
                source_path = (
                    embedding_path(cfg, index).resolve()
                    if embedding is not None else None
                )

                input_record = {
                    'index': index,
                    'category': row['category'],
                    'phrase_type': row.get('phrase_type'),
                    'instruction': instruction,
                    'with_audio': with_audio,
                    'embedding_path': str(source_path) if source_path else None,
                    'embedding_shape': list(embedding.shape) if embedding is not None else None,
                    'embedding_dtype': str(embedding.dtype) if embedding is not None else None,
                    'chosen_response': row['new_chosen_response'],
                    'rejected_response': (
                        row['rejected_response'] if cfg['method'] == 'dpo' else None
                    ),
                }

                if index in recorded_inputs and recorded_inputs[index] != input_record:
                    raise ValueError(f'Training input changed for row {index}')

                step_inputs.append(input_record)

                if cfg['method'] == 'sft':
                    out, batch = forward_target(adapter, sample[0], row['new_chosen_response'], sample[1], sample[2])
                    loss = sft_loss(out)
                    value = float(loss.detach().float())
                    (loss / len(group)).backward()
                    del out, batch, loss
                    extra = {}
                else:
                    value, extra = dpo_step(adapter, row, sample, ref[index], cfg['beta'], len(group))
                if not math.isfinite(value):
                    raise FloatingPointError(f'Non-finite loss row={index}')
                metrics.append({'row': index, 'loss': value, **extra})
                losses.append(value)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            for input_record in step_inputs:
                index = input_record['index']
                if index not in recorded_inputs:
                    append(inputs_path, input_record)
                    recorded_inputs[index] = input_record

            state = {'epoch': epoch, 'position': start + len(group), 'global_step': state['global_step'] + 1}
            record = {'type': 'step', 'epoch': epoch, 'position': state['position'],
                      'global_step': state['global_step'], 'loss': statistics.fmean(x['loss'] for x in metrics),
                      'rows': [x['row'] for x in metrics], 'lr': cfg['learning_rate']}
            append(work_root / 'metrics.jsonl', record)
            if state['global_step'] == 1 or state['global_step'] % cfg['log_every'] == 0:
                print(f'{cfg["method"].upper()} epoch={epoch}/{cfg["epochs"]} position={state["position"]}/{len(order)} step={state["global_step"]} loss={record["loss"]:.6f}', flush=True)
            if cfg['preflight']:
                write_json(work_root / 'preflight.json', {'passed': True, 'metrics': metrics, 'step': state['global_step']})
                print(f'PREFLIGHT PASSED: {work_root / "preflight.json"}', flush=True)
                return
            if cfg['save_every_steps'] and state['global_step'] % cfg['save_every_steps'] == 0:
                manager.save_resume(adapter.model, optimizer, state)
            if cfg['max_steps'] and state['global_step'] >= cfg['max_steps']:
                manager.save_resume(adapter.model, optimizer, state)
                print('Reached --max-steps; training is resumable', flush=True)
                return
        append(work_root / 'metrics.jsonl', {'type': 'epoch', 'epoch': epoch,
                                             'mean_loss': statistics.fmean(losses) if losses else None,
                                             'global_step': state['global_step']})
        if not (work_root / 'checkpoints' / f'epoch_{epoch:03d}').exists():
            manager.save_epoch(adapter.model, epoch)
        next_state = {'epoch': epoch + 1, 'position': 0, 'global_step': state['global_step']}
        manager.save_resume(adapter.model, optimizer, next_state)
        state = next_state
        if on_epoch_end is not None:
            on_epoch_end(adapter, epoch, work_root)
        print(f'EPOCH SAVED {epoch}: {work_root}', flush=True)
    print(f'RUN COMPLETE: {work_root}', flush=True)

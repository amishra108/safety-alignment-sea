import json
from pathlib import Path
import torch
from .data import select_evaluation


def decode_subset(adapter, cfg, checkpoint_epoch, destination):
    chosen = select_evaluation(cfg['eval_manifest'], cfg['eval_split'], cfg['eval_language'],
                               cfg['eval_examples'], cfg['eval_conditions'], cfg['eval_selection'], cfg['eval_seed'])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if destination.exists():
        for line in destination.read_text(encoding='utf-8').splitlines():
            if line.strip():
                done.add(json.loads(line)['task_id'])
    for number, row in enumerate(chosen, 1):
        task_id = row.get('task_id') or row['example_id'] + '__audio'
        if task_id in done:
            continue
        audio_path = Path(row['audio_path'])
        if not audio_path.is_absolute():
            audio_path = Path(cfg['audio_root']) / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        response = adapter.generate_validation_audio(audio_path, cfg['eval_audio_instruction'], cfg['max_new_tokens'])
        record = {
            'task_id': task_id, 'epoch': checkpoint_epoch, 'benchmark': 'Audio Jailbreak Clean Benchmark',
            'model': cfg['model'], 'model_variant': f'sea_{cfg["method"]}_epoch{checkpoint_epoch}',
            'model_path': cfg['model_path'], 'source_id': row['source_id'],
            'example_id': row.get('example_id'), 'english_example_id': row.get('english_example_id'),
            'split': row['split'], 'condition': row['condition'],
            'attack_type': row.get('attack_type'), 'modality': 'audio',
            'model_input_mode': 'audio_only', 'language': row['language'],
            'source_text': row.get('source_text'), 'prompt_text': row.get('prompt_text'),
            'audio_path': str(audio_path), 'response': response,
            'decoding': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens']},
        }
        with destination.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
        print(f'DECODE {number}/{len(chosen)} {task_id}', flush=True)
    return len(chosen)

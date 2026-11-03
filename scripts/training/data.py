import csv
import json
import random
from pathlib import Path
import torch


def load_training_rows(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows:
        raise ValueError('Dataset is empty')
    for i, row in enumerate(rows):
        for key in ('category', 'prompt', 'new_chosen_response', 'rejected_response'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Row {i} missing {key}')
        if row['category'] == 'harmful':
            if row.get('phrase_type') not in ('activity', 'product') or not row.get('rewritten_prompt'):
                raise ValueError(f'Row {i} lacks SEA harmful instruction fields')
        elif row['category'] != 'harmless':
            raise ValueError(f'Unknown category {row["category"]!r} at row {i}')
    return rows


def embedding_path(cfg, index):
    return Path(cfg['embedding_dir']) / f'row_{index:04d}.pt'


def embedding_for(cfg, index, model):
    path = embedding_path(cfg, index)
    record = torch.load(path, map_location='cpu', weights_only=False)
    if record.get('index') != index or record.get('model') != model or record.get('embedding_space') != 'projector_input':
        raise ValueError(f'Embedding metadata mismatch: {path}')
    embedding = record['embedding']
    if not torch.is_tensor(embedding) or not torch.isfinite(embedding).all():
        raise ValueError(f'Invalid embedding: {path}')
    return embedding


def training_sample(rows, index, cfg):
    row = rows[index]
    if row['category'] == 'harmful':
        prefix = cfg['instruction_prefix_' + row['phrase_type']]
        return prefix + row['rewritten_prompt'].strip(), embedding_for(cfg, index, cfg['model']), True
    return cfg['text_instruction'].format(prompt=row['prompt']), None, False


def select_evaluation(path, split, language, n, conditions, selection, seed):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    splits = ('benign', 'harmful') if split == 'both' else (split,)
    chosen = []
    for part in splits:
        subset = [r for r in rows if r['split'] == part and r['language'] == language and r['condition'] in conditions]
        ids = sorted({r['source_id'] for r in subset})
        if selection == 'random':
            random.Random(seed).shuffle(ids)
        ids = ids[:n]
        if len(ids) != n:
            raise ValueError(f'Insufficient {part}/{language} source examples: {len(ids)} < {n}')
        by_key = {}
        for row in subset:
            key = row['source_id'], row['condition']
            if key in by_key:
                raise ValueError(f'Duplicate evaluation variant {key}')
            by_key[key] = row
        for condition in conditions:
            for source_id in ids:
                if (source_id, condition) not in by_key:
                    raise ValueError(f'Missing evaluation variant {part} {source_id} {condition}')
                chosen.append(by_key[source_id, condition])
    return chosen

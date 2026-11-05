import json
import os
import random
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .config import write_json


def named_trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def _weights(model):
    return {name: p.detach().cpu().contiguous() for name, p in named_trainable(model).items()}


def save_weights(model, path):
    path.mkdir(parents=True, exist_ok=True)
    save_file(_weights(model), str(path / 'model.safetensors'))
    write_json(path / 'manifest.json', {
        'format': 'trainable-only',
        'trainable_names': sorted(named_trainable(model)),
        'base_weights_required': True,
    })


def load_weights(model, path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())

    params = named_trainable(model)
    expected = set(params)

    if set(manifest["trainable_names"]) != expected:
        raise RuntimeError("Trainable-parameter names differ from saved checkpoint")

    checkpoint_path = path / "model.safetensors"

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        saved_names = set(checkpoint.keys())

        if saved_names != expected:
            raise RuntimeError("Checkpoint trainable weight names mismatch")

        total = len(expected)
        reported_conversions = set()

        with torch.no_grad():
            for index, name in enumerate(sorted(expected), 1):
                tensor = checkpoint.get_tensor(name)
                parameter = params[name]

                if parameter.shape != tensor.shape:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch: {name}: "
                        f"saved={tuple(tensor.shape)}, "
                        f"model={tuple(parameter.shape)}"
                    )

                if parameter.dtype != tensor.dtype:
                    if not (parameter.is_floating_point() and tensor.is_floating_point()):
                        raise RuntimeError(
                            f"Checkpoint dtype mismatch: {name}: "
                            f"saved={tensor.dtype}, model={parameter.dtype}"
                        )

                    conversion = (tensor.dtype, parameter.dtype)
                    if conversion not in reported_conversions:
                        print(
                            f"Checkpoint floating-point conversion: "
                            f"{tensor.dtype} -> {parameter.dtype}",
                            flush=True,
                        )
                        reported_conversions.add(conversion)

                parameter.copy_(tensor)
                del tensor

                if index == 1 or index % 100 == 0 or index == total:
                    print(
                        f"Loading checkpoint: {index}/{total} tensors",
                        flush=True,
                    )

    print(f"Checkpoint loaded: {checkpoint_path}", flush=True)


class CheckpointManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'resume').mkdir(exist_ok=True)

    def save_epoch(self, model, epoch):
        dest = self.root / 'checkpoints' / f'epoch_{epoch:03d}'
        if dest.exists():
            raise FileExistsError(dest)
        dest.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.epoch_', dir=dest.parent) as temp:
            save_weights(model, Path(temp))
            os.replace(temp, dest)
        return dest

    def save_resume(self, model, optimizer, state):
        parent = self.root / 'resume'
        target = f'step_{state["global_step"]:09d}'
        dest = parent / target
        if not dest.exists():
            with tempfile.TemporaryDirectory(prefix='.writing_', dir=parent) as temp:
                temp = Path(temp)
                save_weights(model, temp)
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }, temp / 'training.pt')
                write_json(temp / 'state.json', state)
                os.replace(temp, dest)
        pointer = parent / 'latest.json'
        tmp = parent / '.latest.json.tmp'
        write_json(tmp, {'checkpoint': target})
        os.replace(tmp, pointer)
        for old in parent.glob('step_*'):
            if old.is_dir() and old != dest:
                shutil.rmtree(old)

    def load_resume(self, model, optimizer):
        pointer = self.root / 'resume/latest.json'
        if not pointer.exists():
            return None
        directory = self.root / 'resume' / json.loads(pointer.read_text())['checkpoint']
        state = json.loads((directory / 'state.json').read_text())
        load_weights(model, directory)
        saved = torch.load(directory / 'training.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        return state

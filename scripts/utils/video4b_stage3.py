import contextlib
import importlib.util
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")


def load_stage2():
    path = ROOT / "scripts/utils/02_optimise_embedding.py"
    spec = importlib.util.spec_from_file_location(
        "video4b_stage2_adapter", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Video4bAdapter:
    def __init__(self, model_path, device):
        self.model_path = Path(model_path)
        self.device = device

        stage2 = load_stage2()
        self.stage2 = stage2
        self.backend = stage2.Video4bAdapter(
            self.model_path, device, silence_seconds=1.0
        )

        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.audio_encoder = self.backend.audio_encoder
        self.projector = self.backend.projector

        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad_(False)

        for parameter in self.projector.parameters():
            parameter.requires_grad_(True)

        self.model.config.use_cache = False
        self.model.eval()

        print(
            "Video4B Stage-3: frozen Whisper/Q-Former, "
            "trainable audio_proj and downstream LM",
            flush=True,
        )
        print(
            "SEA injection boundary: model.model.audio.audio_proj input",
            flush=True,
        )

    def training_batch(self, instruction, target, with_audio):
        if with_audio:
            return self.backend.training_batch(instruction, target)

        messages = [{
            "role": "user",
            "content": instruction,
        }]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        full_ids = self.tokenizer(
            prompt + target + "<|im_end|>",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        labels = self.stage2.labels_from_prefix(
            full_ids, prompt_ids
        )

        return {
            "input_ids": full_ids.to(self.device),
            "attention_mask": torch.ones_like(
                full_ids, device=self.device
            ),
            "labels": labels.to(self.device),
        }

    def forward(self, batch):
        # Use the standard language-model loss and logits for SFT/DPO.
        # Evaluation mode does not disable autograd.
        self.model.eval()
        return self.model(**batch, use_cache=False)

    def prepare_for_training(self):
        self.model.config.use_cache = False
        self.model.eval()

    def generate_validation_audio(
        self, audio_path, instruction, max_new_tokens
    ):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "Video4B validation requires a non-empty text instruction"
            )

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
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

        chunk_size = 30 * 16000
        chunks = [
            waveform[start:start + chunk_size]
            for start in range(0, waveform.size, chunk_size)
        ]

        features = torch.stack([
            self.backend.feature_extractor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
            )["input_features"].squeeze(0)
            for chunk in chunks
        ]).to(
            device=self.device,
            dtype=self.audio_encoder.conv1.weight.dtype,
        )

        original_features = self.backend.audio_features
        original_count = self.backend.audio_token_count

        try:
            self.backend.audio_features = features
            self.backend.audio_token_count = 60 * len(chunks)
            batch = self.backend.prompt_batch(instruction)
            self.model.eval()
            return self.backend.generate(batch, max_new_tokens)
        finally:
            self.backend.audio_features = original_features
            self.backend.audio_token_count = original_count

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(
            output_dir,
            safe_serialization=True,
        )

        for source in self.model_path.iterdir():
            if not source.is_file():
                continue
            if source.suffix in {
                ".safetensors", ".bin", ".pt", ".pth"
            }:
                continue
            destination = output_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)

        self.tokenizer.save_pretrained(output_dir)

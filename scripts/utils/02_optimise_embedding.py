#!/usr/bin/env python3

import argparse
import contextlib
import hashlib
import json
import os
import re
import socket
import tempfile
import time
import wave
from pathlib import Path

import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment-sea")
DATASET = ROOT / "data/saferlhf/sea_3000_prepared.jsonl"
VERSION = "sea-embedding-optimisation-v2"

MODEL_NAMES = {
    "qwen25_omni": "qwen25_omni",
    "salmonn2": "salmonn2",
    "video4b": "video4b",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=sorted(MODEL_NAMES),
        required=True,
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--machine",
        choices=("nete", "air211", "air213"),
        required=True,
    )
    p.add_argument("--gpu", type=int, required=True)

    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=3)

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="SEA paper maximum embedding-optimisation epochs.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="SEA paper audio-model learning rate.",
    )
    p.add_argument(
        "--check-every",
        type=int,
        default=10,
        help="SEA paper checks optimisation success every 10 updates.",
    )

    p.add_argument(
        "--silence-seconds",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--max-check-tokens",
        type=int,
        default=128,
    )
    p.add_argument(
        "--preflight",
        type=int,
        default=0,
        help="If nonzero, process only this many rows from the shard.",
    )

    args = p.parse_args()

    if args.num_shards < 1:
        p.error("--num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        p.error("--shard-id must be in [0, num-shards)")
    if args.gpu < 0:
        p.error("--gpu must be nonnegative")
    if args.epochs < 1:
        p.error("--epochs must be positive")
    if args.check_every < 1:
        p.error("--check-every must be positive")
    if args.lr <= 0:
        p.error("--lr must be positive")
    if args.silence_seconds <= 0:
        p.error("--silence-seconds must be positive")
    if not args.model_path.is_dir():
        p.error(f"model path does not exist: {args.model_path}")

    return args


def load_rows():
    rows = [
        json.loads(line)
        for line in DATASET.open(encoding="utf-8")
        if line.strip()
    ]

    if len(rows) != 3000:
        raise RuntimeError(
            f"Expected 3000 dataset rows, found {len(rows)}"
        )

    if [r["category"] for r in rows[:2000]] != ["harmful"] * 2000:
        raise RuntimeError("Rows 0..1999 are not all harmful")

    if [r["category"] for r in rows[2000:]] != ["harmless"] * 1000:
        raise RuntimeError("Rows 2000..2999 are not all harmless")

    required = (
        "prompt",
        "new_chosen_response",
        "harmful_phrase",
        "phrase_type",
        "rewritten_prompt",
        "audio_content_control_instruction",
        "audio_content_control_target",
        "audio_style_control_instruction",
        "audio_style_control_target",
    )

    for i, row in enumerate(rows[:2000]):
        missing = [
            key for key in required
            if not isinstance(row.get(key), str)
            or not row[key].strip()
        ]
        if missing:
            raise RuntimeError(
                f"Harmful row {i} has missing/empty fields: {missing}"
            )

        if row["phrase_type"] not in {"activity", "product"}:
            raise RuntimeError(
                f"Invalid phrase_type at row {i}: "
                f"{row['phrase_type']!r}"
            )

    return rows


def row_hash(index, row):
    payload = {
        "index": index,
        "prompt": row["prompt"],
        "phrase_type": row["phrase_type"],
        "harmful_phrase": row["harmful_phrase"],
        "audio_content_control_instruction":
            row["audio_content_control_instruction"],
        "audio_content_control_target":
            row["audio_content_control_target"],
        "audio_style_control_instruction":
            row["audio_style_control_instruction"],
        "audio_style_control_target":
            row["audio_style_control_target"],
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def move_batch(batch, device):
    result = {}

    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value

    return result


def labels_from_prefix(full_ids, prefix_ids):
    if full_ids.ndim != 2 or prefix_ids.ndim != 2:
        raise RuntimeError("Expected batched token IDs")

    if full_ids.shape[0] != 1 or prefix_ids.shape[0] != 1:
        raise RuntimeError("Embedding optimisation expects batch size one")

    n = prefix_ids.shape[1]

    if full_ids.shape[1] <= n:
        raise RuntimeError("Assistant target added no tokens")

    if not torch.equal(full_ids[:, :n], prefix_ids):
        raise RuntimeError(
            "Prompt tokenisation is not a prefix of full conversation"
        )

    labels = full_ids.clone()
    labels[:, :n] = -100

    return labels


def write_silent_wav(path, seconds, sample_rate=16000):
    samples = int(round(seconds * sample_rate))

    pcm = b"\x00\x00" * samples

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)


@contextlib.contextmanager
def capture_projector_input(projector):
    captured = []

    def hook(_module, inputs):
        if not inputs:
            raise RuntimeError("Projector received no positional input")

        captured.append(
            inputs[0].detach().to(torch.float32).clone()
        )

    handle = projector.register_forward_pre_hook(hook)

    try:
        yield captured
    finally:
        handle.remove()


@contextlib.contextmanager
def replace_projector_input(projector, embedding):
    def hook(_module, inputs):
        if not inputs:
            raise RuntimeError("Projector received no positional input")

        actual = inputs[0]

        if tuple(actual.shape) != tuple(embedding.shape):
            raise RuntimeError(
                "Synthetic embedding shape differs from projector input: "
                f"expected {tuple(actual.shape)}, "
                f"got {tuple(embedding.shape)}"
            )

        replacement = embedding.to(
            device=actual.device,
            dtype=actual.dtype,
        )

        return (replacement, *inputs[1:])

    handle = projector.register_forward_pre_hook(
        hook,
        prepend=True,
    )

    try:
        yield
    finally:
        handle.remove()


class Qwen25OmniAdapter:
    def __init__(self, model_path, device, silence_seconds):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoder,
        )

        self.device = device
        self.silence_seconds = silence_seconds

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(
            f"Loading Qwen2.5-Omni Thinker from {model_path}",
            flush=True,
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = (
            Qwen2_5OmniThinkerForConditionalGeneration
            .from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        encoders = [
            module
            for module in self.model.modules()
            if isinstance(module, Qwen2_5OmniAudioEncoder)
        ]

        if len(encoders) != 1:
            raise RuntimeError(
                "Expected exactly one Qwen2.5-Omni audio encoder; "
                f"found {len(encoders)}"
            )

        self.audio_encoder = encoders[0]

        if not isinstance(self.audio_encoder.proj, torch.nn.Linear):
            raise RuntimeError(
                "Qwen2.5-Omni audio encoder projector is not Linear"
            )

        self.projector = self.audio_encoder.proj
        self.tokenizer = self.processor.tokenizer

        self.tempdir = tempfile.TemporaryDirectory(
            prefix="sea_qwen_silence_"
        )

        self.silence_path = (
            Path(self.tempdir.name) / "silence.wav"
        )

        write_silent_wav(
            self.silence_path,
            silence_seconds,
            sample_rate=16000,
        )

        print(
            "Qwen SEA boundary: input to "
            "Qwen2_5OmniAudioEncoder.proj",
            flush=True,
        )

    def _conversation(self, instruction, target=None):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": str(self.silence_path),
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ]

        if target is not None:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": target,
                        }
                    ],
                }
            )

        return conversation

    def prompt_batch(self, instruction):
        batch = self.processor.apply_chat_template(
            self._conversation(instruction),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return move_batch(batch, self.device)

    def training_batch(self, instruction, target):
        prompt = self.processor.apply_chat_template(
            self._conversation(instruction),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        full = self.processor.apply_chat_template(
            self._conversation(instruction, target),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        labels = labels_from_prefix(
            full["input_ids"],
            prompt["input_ids"],
        )

        full["labels"] = labels

        return move_batch(full, self.device)

    def forward_loss(self, batch):
        output = self.model(
            **batch,
            use_cache=False,
        )

        if output.loss is None:
            raise RuntimeError("Qwen forward returned no loss")

        return output.loss

    @torch.no_grad()
    def generate(self, batch, max_new_tokens):
        input_length = batch["input_ids"].shape[1]

        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        if output.ndim != 2:
            raise RuntimeError(
                f"Unexpected Qwen generation shape: {output.shape}"
            )

        if output.shape[1] > input_length:
            output = output[:, input_length:]

        decoded = self.processor.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return clean_generated_response(decoded)


class Salmonn2Adapter:
    def __init__(self, model_path, device, silence_seconds):
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
        )

        self.device = device
        self.silence_seconds = silence_seconds

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(
            f"Loading SALMONN-2 from {model_path}",
            flush=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device).eval()

        self.processor.prepare_model(self.model)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if not hasattr(self.model, "connector"):
            raise RuntimeError(
                "Loaded SALMONN-2 model has no connector attribute"
            )

        if not isinstance(
            self.model.connector,
            torch.nn.Sequential,
        ):
            raise RuntimeError(
                "SALMONN-2 connector has unexpected type: "
                f"{type(self.model.connector)}"
            )

        self.projector = self.model.connector
        self.tokenizer = self.processor.tokenizer

        if not hasattr(self.model.config, "freeze_audio_encoder"):
            raise RuntimeError(
                "SALMONN-2 config lacks freeze_audio_encoder"
            )

        # encode_audio() wraps the connector inside set_grad_enabled().
        # Model parameters remain frozen; this only permits gradients
        # from the loss back to the synthetic connector input.
        self.model.config.freeze_audio_encoder = False

        self.sample_rate = int(self.processor.sample_rate)

        n = int(
            round(self.sample_rate * self.silence_seconds)
        )

        self.silence = torch.zeros(
            n,
            dtype=torch.float32,
        )

        print(
            "SALMONN-2 SEA boundary: input to model.connector",
            flush=True,
        )

    def _render(self, instruction, target=None):
        prompt = self.processor.build_prompt(
            instruction,
            audio_count=1,
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if target is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": target,
                }
            )

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(target is None),
            enable_thinking=False,
        )

        return rendered.replace(
            self.processor.audio_placeholder,
            self.processor.model_audio_placeholder,
        )

    def _audio_inputs(self, instruction):
        return self.processor(
            audios=self.silence,
            sampling_rate=self.sample_rate,
            instruction=instruction,
            enable_thinking=False,
            return_tensors="pt",
        )

    def prompt_batch(self, instruction):
        return move_batch(
            self._audio_inputs(instruction),
            self.device,
        )

    def training_batch(self, instruction, target):
        audio = self._audio_inputs(instruction)

        prompt_rendered = self._render(
            instruction,
            target=None,
        )

        full_rendered = self._render(
            instruction,
            target=target,
        )

        prompt_tokens = self.tokenizer(
            prompt_rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )

        full_tokens = self.tokenizer(
            full_rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )

        if not torch.equal(
            audio["input_ids"],
            prompt_tokens["input_ids"],
        ):
            raise RuntimeError(
                "SALMONN processor prompt differs from "
                "explicitly rendered prompt"
            )

        labels = labels_from_prefix(
            full_tokens["input_ids"],
            prompt_tokens["input_ids"],
        )

        batch = {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
            "audio_features": audio["audio_features"],
            "audio_lengths": audio["audio_lengths"],
            "audio_counts": audio["audio_counts"],
        }

        return move_batch(batch, self.device)

    def forward_loss(self, batch):
        output = self.model(
            **batch,
            use_cache=False,
        )

        if output.loss is None:
            raise RuntimeError("SALMONN-2 forward returned no loss")

        return output.loss

    @torch.no_grad()
    def generate(self, batch, max_new_tokens):
        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        if output.ndim != 2:
            raise RuntimeError(
                f"Unexpected SALMONN generation shape: {output.shape}"
            )

        decoded = self.processor.decode(
            output[0],
        )

        return clean_generated_response(decoded)



class Video4bAdapter:
    def __init__(self, model_path, device, silence_seconds):
        import sys
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        upstream = ROOT / "external/video-SALMONN-2/video_SALMONN2_pro"
        implementation = upstream / "qwenvl/model/modeling_qwen3_vl.py"

        if not implementation.is_file():
            raise FileNotFoundError(implementation)

        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))

        from qwenvl.model.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        self.device = device
        self.silence_seconds = silence_seconds
        capability = torch.cuda.get_device_capability(device)
        self.dtype = (
            torch.bfloat16
            if capability[0] >= 8
            else torch.float16
        )

        print(
            f"Loading Video4B from {model_path} "
            f"with dtype={self.dtype}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            fix_mistral_regex=True,
        )



        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                attn_implementation="sdpa",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            .to(device=device, dtype=self.dtype)
        )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.model.eval()

        self.audio_encoder = self.model.model.audio
        self.projector = self.audio_encoder.audio_proj

        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=self.audio_encoder.conv1.in_channels,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
        )

        print(
            "Whisper extractor configured: "
            f"feature_size={self.feature_extractor.feature_size}, "
            f"mel_filters={self.feature_extractor.mel_filters.shape}",
            flush=True,
        )

        if not isinstance(self.projector, torch.nn.Linear):
            raise RuntimeError(
                "Video4B audio_proj is not an nn.Linear"
            )

        if self.projector.in_features != 3072:
            raise RuntimeError(
                "Unexpected Video4B projector input dimension: "
                f"{self.projector.in_features}"
            )

        if self.projector.out_features != 2560:
            raise RuntimeError(
                "Unexpected Video4B projector output dimension: "
                f"{self.projector.out_features}"
            )

        self.audio_token_id = int(
            self.model.config.audio_token_id
        )

        tokenizer_audio_id = self.tokenizer.convert_tokens_to_ids(
            "<|audio_pad|>"
        )

        if tokenizer_audio_id != self.audio_token_id:
            raise RuntimeError(
                f"Audio token mismatch: "
                f"{tokenizer_audio_id} != {self.audio_token_id}"
            )

        if self.feature_extractor.sampling_rate != 16000:
            raise RuntimeError(
                "Expected the authors' 16 kHz Whisper extractor"
            )

        sample_count = max(
            16000,
            int(round(silence_seconds * 16000)),
        )

        silence = torch.zeros(
            sample_count,
            dtype=torch.float32,
        )

        chunk_size = 30 * 16000
        chunks = [
            silence[i:i + chunk_size].numpy()
            for i in range(0, sample_count, chunk_size)
        ]

        feature_batches = [
            self.feature_extractor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
            )["input_features"].squeeze(0)
            for chunk in chunks
        ]

        self.audio_features = torch.stack(
            feature_batches,
            dim=0,
        ).to(
            device=device,
            dtype=self.audio_encoder.conv1.weight.dtype,
        )

        self.audio_token_count = 60 * len(chunks)

        expected_mels = self.audio_encoder.conv1.in_channels
        actual_mels = self.audio_features.shape[1]

        if actual_mels != expected_mels:
            raise RuntimeError(
                f"Whisper mel mismatch: "
                f"expected {expected_mels}, got {actual_mels}"
            )

        if self.audio_features.shape[2] != 3000:
            raise RuntimeError(
                "Expected 3000 Whisper frames per 30-second chunk; "
                f"received {self.audio_features.shape[2]}"
            )

        print(
            f"Whisper mel compatibility: "
            f"{actual_mels}/{expected_mels} OK",
            flush=True,
        )

        if not torch.isfinite(self.audio_features).all():
            raise RuntimeError("Non-finite audio features")

        print(
            "SEA boundary: INPUT to model.model.audio.audio_proj",
            flush=True,
        )
        print(
            f"Projector dimensions: "
            f"{self.projector.in_features} -> "
            f"{self.projector.out_features}",
            flush=True,
        )
        print(
            f"Whisper features: {tuple(self.audio_features.shape)}",
            flush=True,
        )
        print(
            f"Audio tokens: {self.audio_token_count}",
            flush=True,
        )
        print(
            f"Frozen model parameters: "
            f"{sum(not p.requires_grad for p in self.model.parameters())}",
            flush=True,
        )

    def _render(self, instruction, target=None):
        audio_placeholders = (
            "<|audio_pad|>" * self.audio_token_count
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": audio_placeholders + "\n" + instruction,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if target is None:
            return prompt

        return prompt + target + "<|im_end|>"

    def _batch(self, instruction, target=None):
        prefix_ids = self.tokenizer(
            self._render(instruction),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        actual_audio_count = int(
            (prefix_ids == self.audio_token_id).sum().item()
        )

        if actual_audio_count != self.audio_token_count:
            raise RuntimeError(
                f"Audio placeholder alignment mismatch: "
                f"{actual_audio_count} vs {self.audio_token_count}"
            )

        if target is None:
            input_ids = prefix_ids
            labels = None
        else:
            input_ids = self.tokenizer(
                self._render(instruction, target),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

            labels = labels_from_prefix(
                input_ids,
                prefix_ids,
            )

        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": torch.ones_like(
                input_ids,
                device=self.device,
            ),
            "audio_feature": self.audio_features,
        }

        if labels is not None:
            batch["labels"] = labels.to(self.device)

        return batch

    def prompt_batch(self, instruction):
        return self._batch(instruction)

    def training_batch(self, instruction, target):
        return self._batch(instruction, target)

    def forward_loss(self, batch):
        self.model.eval()

        output = self.model(
            **batch,
            use_cache=False,
        )

        if output.loss is None:
            raise RuntimeError(
                "Video4B forward returned no masked LM loss"
            )

        return output.loss

    @torch.no_grad()
    def generate(self, batch, max_new_tokens):
        self.model.eval()

        prompt_length = batch["input_ids"].shape[1]

        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        if output.ndim != 2:
            raise RuntimeError(
                f"Unexpected generation output shape: {output.shape}"
            )

        response_ids = output[0, prompt_length:]

        response = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return clean_generated_response(response)


def capture_silent_embedding(adapter, instruction):
    batch = adapter.prompt_batch(instruction)

    with capture_projector_input(
        adapter.projector
    ) as captured:
        with torch.no_grad():
            adapter.model(
                **batch,
                use_cache=False,
            )

    if len(captured) != 1:
        raise RuntimeError(
            "Expected projector to run exactly once while "
            f"encoding one silent audio input; got {len(captured)}"
        )

    embedding = captured[0]

    if isinstance(adapter, Video4bAdapter):
        print(
            "Video4B captured projector-input shape: "
            f"{tuple(embedding.shape)}",
            flush=True,
        )

        if embedding.shape[-1] != adapter.projector.in_features:
            raise RuntimeError(
                "Captured embedding does not match audio_proj input"
            )

        if embedding.ndim != 3:
            raise RuntimeError(
                "Expected batched Q-Former embeddings "
                f"with three dimensions, got {embedding.ndim}"
            )

    if embedding.numel() == 0:
        raise RuntimeError("Captured silent embedding is empty")

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            "Silent embedding contains non-finite values"
        )

    return embedding


WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

CHAT_TURN_RE = re.compile(
    r"(?im)^[ \t]*(?:human|user|assistant|system)[ \t]*(?::|$)"
)


def clean_generated_response(text):
    text = text.strip()

    for marker in (
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
    ):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()

    match = CHAT_TURN_RE.search(text)
    if match:
        text = text[:match.start()].rstrip()

    return text

def words(text):
    return WORD_RE.findall(text.lower())


def light_inflection_root(word):
    if len(word) > 5 and word.endswith("ing"):
        root = word[:-3]
        if len(root) >= 3:
            return root
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def activity_content_success(output, guiding_text):
    target_words = words(guiding_text)
    output_words = words(output)

    if not target_words:
        return False

    output_roots = {
        light_inflection_root(word)
        for word in output_words
    }

    matched = 0

    for word in target_words:
        root = light_inflection_root(word)

        if (
            word in output_words
            or root in output_roots
        ):
            matched += 1

    required = max(1, len(target_words) - 1)

    return matched >= required


def product_content_success(output, guiding_text):
    target = " ".join(words(guiding_text))
    generated = " ".join(words(output))

    return target in generated


def style_success(output, guiding_text):
    target = " ".join(words(guiding_text))
    generated = " ".join(words(output))

    return target in generated


def remove_prefix(text, prefix):
    text_norm = text.strip()
    prefix_norm = prefix.strip()

    if text_norm.lower().startswith(prefix_norm.lower()):
        return text_norm[len(prefix_norm):].strip()

    return text_norm


def check_success(adapter, embedding, row, args):
    content_instruction = row[
        "audio_content_control_instruction"
    ]
    style_instruction = row[
        "audio_style_control_instruction"
    ]

    content_guiding = remove_prefix(
        row["audio_content_control_target"],
        "The audio contains speech saying",
    )

    style_guiding = remove_prefix(
        row["audio_style_control_target"],
        "The speaker is",
    )

    content_batch = adapter.prompt_batch(
        content_instruction
    )
    style_batch = adapter.prompt_batch(
        style_instruction
    )

    with replace_projector_input(
        adapter.projector,
        embedding,
    ):
        content_output = adapter.generate(
            content_batch,
            args.max_check_tokens,
        )

        style_output = adapter.generate(
            style_batch,
            args.max_check_tokens,
        )

    if row["phrase_type"] == "activity":
        content_ok = activity_content_success(
            content_output,
            content_guiding,
        )
    else:
        content_ok = product_content_success(
            content_output,
            content_guiding,
        )

    style_ok = style_success(
        style_output,
        style_guiding,
    )

    return {
        "content_success": content_ok,
        "style_success": style_ok,
        "content_output": content_output,
        "style_output": style_output,
        "content_guiding_text": content_guiding,
        "style_guiding_text": style_guiding,
    }


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_existing(path, index, source_hash):
    if not path.exists():
        return False

    value = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if value.get("version") != VERSION:
        raise RuntimeError(
            f"Incompatible checkpoint version: {path}"
        )

    if value.get("index") != index:
        raise RuntimeError(
            f"Checkpoint index mismatch: {path}"
        )

    if value.get("source_hash") != source_hash:
        raise RuntimeError(
            f"Dataset row changed since checkpoint was made: {path}"
        )

    if "embedding" not in value:
        raise RuntimeError(
            f"Checkpoint has no embedding: {path}"
        )

    return True


def optimise_row(
    adapter,
    initial_embedding,
    index,
    row,
    out_path,
    args,
):
    source_hash = row_hash(index, row)

    if validate_existing(
        out_path,
        index,
        source_hash,
    ):
        print(
            f"Row {index}: already complete",
            flush=True,
        )
        return

    content_batch = adapter.training_batch(
        row["audio_content_control_instruction"],
        row["audio_content_control_target"],
    )

    style_batch = adapter.training_batch(
        row["audio_style_control_instruction"],
        row["audio_style_control_target"],
    )

    embedding = torch.nn.Parameter(
        initial_embedding
        .detach()
        .clone()
        .to(
            device=adapter.device,
            dtype=torch.float32,
        )
    )

    optimiser = torch.optim.AdamW(
        [embedding],
        lr=args.lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=args.epochs,
        eta_min=0.0,
    )

    losses = []
    checks = []
    stopped_early = False

    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        optimiser.zero_grad(set_to_none=True)

        with replace_projector_input(
            adapter.projector,
            embedding,
        ):
            content_loss = adapter.forward_loss(
                content_batch
            )

            style_loss = adapter.forward_loss(
                style_batch
            )

            loss = 0.5 * (
                content_loss + style_loss
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at row {index}, epoch {epoch}"
            )

        loss.backward()

        if embedding.grad is None:
            raise RuntimeError(
                "Synthetic embedding received no gradient"
            )

        if not torch.isfinite(embedding.grad).all():
            raise RuntimeError(
                f"Non-finite embedding gradient at row {index}"
            )

        if args.model == "video4b":
            grad_norm = embedding.grad.float().norm().item()

            print(
                f"Video4B row={index} epoch={epoch} "
                f"embedding_grad_norm={grad_norm:.8f}",
                flush=True,
            )

            if grad_norm == 0.0:
                raise RuntimeError(
                    "Zero synthetic embedding gradient: "
                    "audio_proj injection is not reaching the LM loss"
                )

            previous_embedding = embedding.detach().clone()

        optimiser.step()

        if args.model == "video4b":
            update_norm = (
                embedding.detach() - previous_embedding
            ).float().norm().item()

            print(
                f"Video4B row={index} epoch={epoch} "
                f"embedding_update_norm={update_norm:.8f}",
                flush=True,
            )

            if update_norm == 0.0:
                raise RuntimeError(
                    "Optimiser made no change to synthetic embedding"
                )
        scheduler.step()

        losses.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "content_loss":
                    float(content_loss.detach().cpu()),
                "style_loss":
                    float(style_loss.detach().cpu()),
                "lr": float(
                    optimiser.param_groups[0]["lr"]
                ),
            }
        )

        print(
            f"row={index} epoch={epoch:03d} "
            f"loss={losses[-1]['loss']:.6f} "
            f"content={losses[-1]['content_loss']:.6f} "
            f"style={losses[-1]['style_loss']:.6f} "
            f"lr={losses[-1]['lr']:.8f}",
            flush=True,
        )

        if (
            epoch % args.check_every == 0
            or epoch == args.epochs
        ):
            result = check_success(
                adapter,
                embedding,
                row,
                args,
            )

            result["epoch"] = epoch
            checks.append(result)

            print(
                f"row={index} CHECK epoch={epoch} "
                f"content={result['content_success']} "
                f"style={result['style_success']}",
                flush=True,
            )

            print(
                f"  content output: "
                f"{result['content_output']}",
                flush=True,
            )

            print(
                f"  style output: "
                f"{result['style_output']}",
                flush=True,
            )

            if (
                result["content_success"]
                and result["style_success"]
            ):
                stopped_early = epoch < args.epochs
                break

    elapsed = time.perf_counter() - start

    final_check = checks[-1]

    record = {
        "version": VERSION,
        "index": index,
        "source_hash": source_hash,
        "model": args.model,
        "model_path": str(args.model_path.resolve()),
        "embedding_space": "projector_input",
        "embedding": embedding.detach().cpu(),
        "embedding_shape": list(embedding.shape),
        "embedding_dtype_saved": "float32",
        "initial_embedding_shape":
            list(initial_embedding.shape),
        "silence_seconds": args.silence_seconds,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "initial_lr": args.lr,
        "scheduler": "CosineAnnealingLR",
        "max_epochs": args.epochs,
        "check_every": args.check_every,
        "epochs_run": losses[-1]["epoch"],
        "stopped_early": stopped_early,
        "content_success":
            final_check["content_success"],
        "style_success":
            final_check["style_success"],
        "optimisation_success": (
            final_check["content_success"]
            and final_check["style_success"]
        ),
        "elapsed_seconds": elapsed,
        "phrase_type": row["phrase_type"],
        "harmful_phrase": row["harmful_phrase"],
        "rewritten_prompt": row["rewritten_prompt"],
        "audio_content_control_instruction":
            row["audio_content_control_instruction"],
        "audio_content_control_target":
            row["audio_content_control_target"],
        "audio_style_control_instruction":
            row["audio_style_control_instruction"],
        "audio_style_control_target":
            row["audio_style_control_target"],
        "loss_history": losses,
        "success_checks": checks,
    }

    atomic_torch_save(
        record,
        out_path,
    )

    print(
        f"Saved row {index}: "
        f"success={record['optimisation_success']} "
        f"epochs={record['epochs_run']} "
        f"time={elapsed:.2f}s "
        f"shape={tuple(embedding.shape)}",
        flush=True,
    )


def main():
    args = parse_args()

    hostname = socket.gethostname().split(".")[0]

    if hostname != args.machine:
        raise RuntimeError(
            f"Running on {hostname}, "
            f"but --machine={args.machine}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    rows = load_rows()

    assigned = [
        i
        for i in range(2000)
        if i % args.num_shards == args.shard_id
    ]

    if args.preflight:
        if args.model == "video4b" and args.preflight != 1:
            raise ValueError(
                "Video4B preflight requires --preflight 1"
            )
        assigned = assigned[:args.preflight]

    model_name = MODEL_NAMES[args.model]

    output_root = (
        ROOT
        / "outputs"
        / "02_optimised_embeddings"
        / model_name
    )
    preflight_directory = None

    if args.model == "video4b" and args.preflight:
        preflight_directory = tempfile.TemporaryDirectory(
            prefix="video4b_sea_preflight_"
        )
        output_root = Path(preflight_directory.name)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_root = output_root / ".locks"
    lock_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = (
        lock_root
        / f"num_shards_{args.num_shards}_shard_{args.shard_id:02d}.lock"
    )

    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Shard is already locked: {lock}"
        )

    try:
        print(
            f"Machine={hostname} physical_gpu={args.gpu} "
            f"visible_gpu=0 model={args.model}",
            flush=True,
        )
        print(
            f"Shard={args.shard_id}/{args.num_shards} "
            f"assigned={len(assigned)}",
            flush=True,
        )
        print(
            f"Dataset={DATASET}",
            flush=True,
        )
        print(
            f"Output={output_root}",
            flush=True,
        )

        if args.model == "qwen25_omni":
            adapter = Qwen25OmniAdapter(
                args.model_path,
                device,
                args.silence_seconds,
            )
        elif args.model == "salmonn2":
            adapter = Salmonn2Adapter(
                args.model_path,
                device,
                args.silence_seconds,
            )
        elif args.model == "video4b":
            adapter = Video4bAdapter(
                args.model_path,
                device,
                args.silence_seconds,
            )
        else:
            raise AssertionError(args.model)

        first_row = rows[assigned[0]]

        initial_embedding = capture_silent_embedding(
            adapter,
            first_row["audio_content_control_instruction"],
        )

        print(
            "Captured silent pre-projector embedding: "
            f"shape={tuple(initial_embedding.shape)} "
            f"dtype={initial_embedding.dtype}",
            flush=True,
        )

        completed = 0

        for position, index in enumerate(
            assigned,
            start=1,
        ):
            row = rows[index]

            print()
            print(
                "=" * 100,
                flush=True,
            )
            print(
                f"[{position}/{len(assigned)}] "
                f"ROW {index}",
                flush=True,
            )
            print(
                "=" * 100,
                flush=True,
            )

            out_path = (
                output_root
                / f"row_{index:04d}.pt"
            )

            optimise_row(
                adapter,
                initial_embedding,
                index,
                row,
                out_path,
                args,
            )

            completed += 1

        print()
        print(
            f"Worker finished: {completed}/{len(assigned)} rows",
            flush=True,
        )

    finally:
        if lock.exists():
            lock.rmdir()

        if preflight_directory is not None:
            preflight_directory.cleanup()
            print(
                "Video4B preflight outputs deleted",
                flush=True,
            )


if __name__ == "__main__":
    main()

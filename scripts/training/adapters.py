import contextlib
import tempfile
import wave
from pathlib import Path
import torch


def silent_wav(path):
    with wave.open(str(path), 'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b'\x00\x00' * 16000)


def prefix_labels(full, prefix):
    if full.shape[1] <= prefix.shape[1] or not torch.equal(full[:, :prefix.shape[1]], prefix):
        raise ValueError('Tokenized complete conversation does not start with prompt')
    labels = full.clone()
    labels[:, :prefix.shape[1]] = -100
    return labels


@contextlib.contextmanager
def inject_embedding(projector, embedding):
    hits = [0]
    def replace(_module, inputs):
        hits[0] += 1
        if tuple(inputs[0].shape) != tuple(embedding.shape):
            raise ValueError(f'Projector input {tuple(inputs[0].shape)} != embedding {tuple(embedding.shape)}')
        return (embedding.to(inputs[0].device, inputs[0].dtype), *inputs[1:])
    handle = projector.register_forward_pre_hook(replace, prepend=True)
    try:
        yield
        if hits[0] != 1:
            raise ValueError(f'Expected exactly one projector forward, got {hits[0]}')
    finally:
        handle.remove()


class QwenAdapter:
    def __init__(self, model_path, device):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
        self.model_path = Path(model_path)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
        self.omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype='auto', device_map={'': str(device)}, local_files_only=True)
        if hasattr(self.omni_model, 'disable_talker'):
            self.omni_model.disable_talker()
        self.model = self.omni_model.thinker
        encoders = [x for x in self.model.modules() if isinstance(x, Qwen2_5OmniAudioEncoder)]
        if len(encoders) != 1:
            raise RuntimeError(f'Expected one Omni audio encoder, found {len(encoders)}')
        self.audio_encoder = encoders[0]
        self.projector = self.audio_encoder.proj
        self.device = next(self.model.parameters()).device
        self.tempdir = tempfile.TemporaryDirectory(prefix='sea_omni_')
        self.silence_path = Path(self.tempdir.name) / 'silence.wav'
        silent_wav(self.silence_path)

    def _messages(self, instruction, target, with_audio):
        content = ([{'type': 'audio', 'audio': str(self.silence_path)}] if with_audio else [])
        content.append({'type': 'text', 'text': instruction})
        msgs = [{'role': 'user', 'content': content}]
        if target is not None:
            msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': target}]})
        return msgs

    def training_batch(self, instruction, target, with_audio):
        prompt = self.processor.apply_chat_template(self._messages(instruction, None, with_audio), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        full = self.processor.apply_chat_template(self._messages(instruction, target, with_audio), tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt')
        full['labels'] = prefix_labels(full['input_ids'], prompt['input_ids'])
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in full.items()}

    def forward(self, batch):
        return self.model(**batch, use_cache=False)

    def prepare_for_training(self):
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def generate_validation_audio(self, audio_path, instruction, max_new_tokens):
        from qwen_omni_utils import process_mm_info
        if instruction is not None:
            raise ValueError('Omni evaluation adapter supports audio-only input')
        msgs = [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'file://' + str(Path(audio_path).resolve())}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(msgs, use_audio_in_video=False)
        batch = self.processor(text=text, audio=audios, images=images, videos=videos,
                               return_tensors='pt', padding=True, use_audio_in_video=False).to(self.device)
        was_training = self.model.training
        checkpointing = bool(getattr(self.model, 'is_gradient_checkpointing', False))
        old_cache = self.model.config.use_cache
        try:
            self.model.eval()
            if checkpointing:
                self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            with torch.inference_mode():
                result = self.omni_model.generate(**batch, return_audio=False, thinker_do_sample=False,
                                                   use_audio_in_video=False, max_new_tokens=max_new_tokens)
            if isinstance(result, tuple):
                result = result[0]
            return self.processor.batch_decode(result[:, batch['input_ids'].shape[1]:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()
        finally:
            self.model.config.use_cache = old_cache
            if checkpointing:
                self.model.gradient_checkpointing_enable()
            if was_training:
                self.model.train()


def load_adapter(name, path, device):
    if name == 'qwen25_omni':
        adapter = QwenAdapter(path, device)
    elif name == 'video4b':
        from scripts.utils.video4b_stage3 import Video4bAdapter
        adapter = Video4bAdapter(path, device)
    else:
        raise ValueError(f'Unknown model {name}')
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(True)
    for parameter in adapter.audio_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.projector.parameters():
        parameter.requires_grad_(True)
    return adapter

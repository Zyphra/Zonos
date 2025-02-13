import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device, backbone="torch")

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

from zonos.config import InferenceParams

cfg_scale = 2.0

from zonos.models import ZonosDecodeOne

cond_dict = make_cond_dict(text="Hello, world! I am the one you need. Let me take you to a far off place.", language="en-us")
prefix_conditioning = model.prepare_conditioning(cond_dict)

# Stuff that happens in generate
batch_size = 1
audio_prefix_codes = None  # No prefix for now
prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
max_new_tokens: int = 86 * 30
audio_seq_len = prefix_audio_len + max_new_tokens
seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9
with torch.device(device):
    inference_params = model.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)

import torch.export
from torch.export import Dim

exportable_model = ZonosDecodeOne(model, inference_params)
input_ids = torch.zeros((1, 20, 1), dtype=torch.int64, device=device)
s0 = Dim("s0", min=9)
with torch.inference_mode():
    ep = torch.export.export(exportable_model, (input_ids, {k: (v[0].detach(), v[1]) for k, v in inference_params.key_value_memory_dict.items()}, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: (None, None) for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
breakpoint()

"""
cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes_warmup = model.generate(conditioning)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
"""

%load_ext autoreload
%autoreload 2
from zonos.model import Zonos
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda", backbone="torch")
.model._decode_one_token(input_ids, InferenceParams(2, 30000), 2.0, allow_cudagraphs=False)
model._decode_one_token(input_ids, InferenceParams(2, 30000), 2.0, allow_cudagraphs=False)
input_ids = torch.zeros((1, 20, 1), dtype=torch.int64, device="cuda")
import torch
input_ids = torch.zeros((1, 20, 1), dtype=torch.int64, device="cuda")
model._decode_one_token(input_ids, InferenceParams(2, 30000), 2.0, allow_cudagraphs=False)
from zonos.config import InferenceParams
model._decode_one_token(input_ids, InferenceParams(2, 30000), 2.0, allow_cudagraphs=False)
from zonos.conditioning import make_cond_dict
cond_dict = make_cond_dict(text="Hello, world!", language="en-us")
conditioning = model.prepare_conditioning(cond_dict)
codes = model.generate(conditioning)
codes = model.generate(conditioning)
cond_dict = make_cond_dict(text="Hello, world! I am the one you need. Let me take you to a far off place.", language="e
n-us")
prefix_conditioning = model.prepare_conditioning(cond_dict)
cond_dict = make_cond_dict(text="Hello, world! I am the one you need. Let me take you to a far off place.", language="e
n-us")
cond_dict = make_cond_dict(text="Hello, world! I am the one you need. Let me take you to a far off place.", language="en-us")
prefix_conditioning = model.prepare_conditioning(cond_dict)
batch_size = 1
audio_prefix_codes = None  # No prefix for now
prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
max_new_tokens: int = 86 * 30
audio_seq_len = prefix_audio_len + max_new_tokens
seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9
inference_params = model.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
from zonos.model import ZonosDecodeOne
exportable_model = ZonosDecodeOne(model)
exportable_model(input_ids, inference_params)
batch_size = 1
audio_prefix_codes = None  # No prefix for now
prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
max_new_tokens: int = 86 * 30
audio_seq_len = prefix_audio_len + max_new_tokens
seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9
inference_params = model.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
exportable_model(input_ids, inference_params)
batch_size = 1
audio_prefix_codes = None  # No prefix for now
prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
max_new_tokens: int = 86 * 30
audio_seq_len = prefix_audio_len + max_new_tokens
seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9
inference_params = model.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
exportable_model(input_ids, inference_params)
exportable_model(input_ids, inference_params)
exportable_model(input_ids, inference_params)
input_ids = torch.zeros((1, 20, 1), dtype=torch.int64, device="cuda")
with torch.device(device):
    inference_params = model.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
device = 'cuda'
with torch.device(device):
    inference_params = model.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
exportable_model(input_ids, inference_params)
exportable_model(input_ids, inference_params)
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}})
import torch.export
from torch.export import Dim
s0 = Dim("s0")
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}})
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}, "inference_params": {}})
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}, "inference_params": None})
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}, "inference_params": None})
s0 = Dim("s0", min=9)
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}, "inference_params": None})
torch.export.register_dataclass(InferenceParams)
ep = torch.export.export(exportable_model, (input_ids, inference_params), dynamic_shapes = {"input_ids": {1: s0}, "inference_params": None})
torch.export.register_dataclass(InferenceParams)
exportable_model = ZonosDecodeOne(model, inference_params)
exportable_model(input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample
)
exportable_model(input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample
)
ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": None, "lengths_per_sample": None})
ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": None, "lengths_per_sample": None})
ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: None for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: None for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: (None, None) for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
ep.print_readable()
ep.module().print_readable()
p ep.module().print_readable()
print(ep)
torch._inductor.aoti_compile_and_package(ep, package_path="Zonos-v0.1-transformer-go.pt2")
with torch.inference_mode():ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: (None, None) for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
with torch.inference_mode():
    ep = torch.export.export(exportable_model, (input_ids, inference_params.key_value_memory_dict, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: (None, None) for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
torch._inductor.aoti_compile_and_package(ep, package_path="Zonos-v0.1-transformer-go.pt2")
inference_params.key_value_memory_dict
with torch.inference_mode():
    ep = torch.export.export(exportable_model, (input_ids, {k: (v[0].detach(), v[1]) for k, v in inference_params.key_value_memory_dict.items()}, inference_params.lengths_per_sample), dynamic_shapes = {"input_ids": {1: s0}, "key_value_memory_dict": {k: (None, None) for k in inference_params.key_value_memory_dict}, "lengths_per_sample": None})
torch._inductor.aoti_compile_and_package(ep, package_path="Zonos-v0.1-transformer-go.pt2")
conditioning
codes = model.generate(conditioning)
codes = model.generate(conditioning)
codes = model.generate(conditioning)
codes = model.generate(conditioning)
%history export_log.py
%history -f export_log.py

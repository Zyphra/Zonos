import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn

from zonos.config import BackboneConfig, InferenceParams


class MambaSSMZonosBackbone(nn.Module):
    supported_architectures = ["transformer", "hybrid"]

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,
                    d_intermediate=config.d_intermediate
                    if (i not in config.attn_layer_idx)
                    else config.attn_mlp_d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    layer_idx=i,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=True,
                    rms_norm=config.rms_norm,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams | None = None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        return layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            residual,
            eps=self.norm_f.eps,
            residual_in_fp32=self.config.residual_in_fp32,
            is_rms_norm=self.config.rms_norm,
        )

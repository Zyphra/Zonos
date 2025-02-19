BACKBONES = {}

try:
    from ._mamba_ssm import MambaSSMZonosBackbone

    BACKBONES["mamba_ssm"] = MambaSSMZonosBackbone
except ImportError:
    pass

try:
    from ._mlx import MLXZonosBackbone

    BACKBONES["mlx"] = MLXZonosBackbone
except ImportError:
    pass

from ._torch import TorchZonosBackbone

BACKBONES["torch"] = TorchZonosBackbone

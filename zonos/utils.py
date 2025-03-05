import torch
import torch.nn as nn
import torch.nn.functional as F


def find_multiple(n: int, k: int) -> int:
    if k == 0 or n % k == 0:
        return n
    return n + k - (n % k)


def pad_weight_(w: nn.Embedding | nn.Linear, multiple: int):
    """Pad the weight of an embedding or linear layer to a multiple of `multiple`."""
    if isinstance(w, nn.Embedding):
        # Pad input dim
        if w.weight.shape[1] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[1] % multiple))
        w.num_embeddings, w.embedding_dim = w.weight.shape
    elif isinstance(w, nn.Linear):
        # Pad output dim
        if w.weight.shape[0] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[0] % multiple))
        w.out_features, w.in_features = w.weight.shape
    else:
        raise ValueError(f"Unsupported weight type: {type(w)}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        if major >= 7:
            return torch.device("cuda")
        else:
            print(f"CUDA device detected (Compute Capability {major}.{minor}), but Zonos requires 7.0 or higher. Falling back to CPU.")

    return torch.device("cpu")


DEFAULT_DEVICE = get_device()

import torch
import torch.nn.functional as F


def apply_delay_pattern(codes: torch.Tensor, mask_token: int, prev_chunk_end: torch.Tensor | None = None):
    """
    Apply delay pattern with optional previous chunk ending.
    
    Args:
        codes: Input codes [batch_size, n_codebooks, seq_len]
        mask_token: Token to use for masking
        prev_chunk_end: Optional last n_codebooks tokens from previous chunk
    """
    if prev_chunk_end is not None:
        # Prepend previous chunk ending
        codes = torch.cat([prev_chunk_end, codes], dim=-1)
    
    codes = F.pad(codes, (0, codes.shape[1]), value=mask_token)
    return torch.stack([codes[:, k].roll(k + 1) for k in range(codes.shape[1])], dim=1)

def revert_delay_pattern(codes: torch.Tensor, remove_overlap: bool = False):
    """
    Revert delay pattern, optionally removing overlap region.
    """
    _, n_q, seq_len = codes.shape
    reverted = torch.stack([codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], dim=1)
    
    if remove_overlap:
        reverted = reverted[..., n_q:]
        
    return reverted

def interpolate_latents(codes1: torch.Tensor, codes2: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Smoothly interpolate between two chunks of codes in their overlap region.
    
    Args:
        codes1: First chunk of codes ending with overlap region
        codes2: Second chunk of codes starting with overlap region
        overlap_size: Size of the overlap region
    """
    weights = torch.linspace(1, 0, overlap_size, device=codes1.device).view(1, 1, -1)
    overlap1 = codes1[..., -overlap_size:]
    overlap2 = codes2[..., :overlap_size]
    return overlap1 * weights + overlap2 * (1 - weights)
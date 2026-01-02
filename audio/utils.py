import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_or_trim(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    # waveform: (1, T)
    T = waveform.shape[-1]
    if T == target_len:
        return waveform
    if T > target_len:
        return waveform[..., :target_len]
    pad = target_len - T
    return torch.nn.functional.pad(waveform, (0, pad))

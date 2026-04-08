from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_or_trim_waveform(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.size(-1) < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.size(-1)))
    elif waveform.size(-1) > target_length:
        waveform = waveform[..., :target_length]
    return waveform


def random_time_shift(waveform: torch.Tensor, max_shift_samples: int) -> torch.Tensor:
    if max_shift_samples <= 0:
        return waveform
    shift = int(torch.randint(-max_shift_samples, max_shift_samples + 1, (1,)).item())
    if shift == 0:
        return waveform

    shifted = torch.zeros_like(waveform)
    if shift > 0:
        shifted[..., shift:] = waveform[..., :-shift]
    else:
        shifted[..., :shift] = waveform[..., -shift:]
    return shifted


def mix_background(waveform: torch.Tensor, noise: torch.Tensor, gain: float) -> torch.Tensor:
    mixed = waveform + (gain * noise)
    return mixed.clamp(min=-1.0, max=1.0)


def apply_specaugment(
    features: torch.Tensor,
    time_masks: int = 1,
    time_mask_width: int = 10,
    freq_masks: int = 1,
    freq_mask_width: int = 5,
) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError("Expected features with shape [batch, freq, time].")

    augmented = features.clone()
    batch_size, freq_dim, time_dim = augmented.shape

    for batch_idx in range(batch_size):
        for _ in range(time_masks):
            width = int(torch.randint(0, time_mask_width + 1, (1,)).item())
            if width == 0 or width >= time_dim:
                continue
            start = int(torch.randint(0, time_dim - width + 1, (1,)).item())
            augmented[batch_idx, :, start : start + width] = 0.0

        for _ in range(freq_masks):
            width = int(torch.randint(0, freq_mask_width + 1, (1,)).item())
            if width == 0 or width >= freq_dim:
                continue
            start = int(torch.randint(0, freq_dim - width + 1, (1,)).item())
            augmented[batch_idx, start : start + width, :] = 0.0

    return augmented

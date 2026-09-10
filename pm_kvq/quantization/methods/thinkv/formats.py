"""Quantize/dequantize reference formats; returned tensors keep their input dtype."""

import torch


def quantize_dequantize(x, bits, group_size=16, axis=-1):
    if bits not in (2, 4, 8, 16):
        raise ValueError("ThinKV precision must be 2, 4, 8, or 16")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be positive")
    if not torch.isfinite(x).all():
        raise ValueError("ThinKV cannot quantize nonfinite tensors")
    if bits == 16 or x.numel() == 0:
        return x.clone()
    if bits == 8:
        # E4M3FN has maximum finite magnitude 448. Scale is FP32 per tensor.
        scale = x.float().abs().amax() / 448
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return ((x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
    moved = x.movedim(axis, -1)
    n = moved.shape[-1] // group_size * group_size
    result = moved.clone()
    if n == 0:
        return result.movedim(-1, axis)
    groups = moved[..., :n].float().reshape(*moved.shape[:-1], -1, group_size)
    maximum = 6.0 if bits == 4 else 1.0
    scales = groups.abs().amax(-1, keepdim=True) / maximum
    # Tensor-level FP32 normalization makes small/large FP8 group scales usable.
    global_scale = scales.amax() / 448
    global_scale = torch.where(global_scale > 0, global_scale, torch.ones_like(global_scale))
    fp8_scales = (scales / global_scale).clamp(0, 448).to(torch.float8_e4m3fn).float()
    scales = fp8_scales * global_scale
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    levels = [0., .5, 1., 1.5, 2., 3., 4., 6.] if bits == 4 else [0., 1.]
    levels = torch.tensor(levels, device=x.device)
    normalized = groups.abs() / safe
    midpoints = (levels[1:] + levels[:-1]) / 2
    indices = torch.bucketize(normalized.contiguous(), midpoints)
    if bits == 4:
        # E2M1's ordered magnitude encodings have even LSBs at even indices.
        # bucketize chooses the lower neighbor; promote odd encodings at ties.
        at_midpoint = normalized == midpoints[indices.clamp(max=len(midpoints) - 1)]
        indices = indices + (at_midpoint & (indices % 2 == 1)).long()
    # Ternary keeps its existing midpoint-to-zero convention.
    decoded = groups.sign() * levels[indices] * scales
    result[..., :n] = decoded.reshape(*moved.shape[:-1], n).to(x.dtype)
    return result.movedim(-1, axis)

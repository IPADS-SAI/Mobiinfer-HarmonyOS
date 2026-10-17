"""Hessian-based GPTQ with the SDK's contiguous INT4 groups and frozen scales.

Algorithm: Frantar et al., GPTQ (https://arxiv.org/abs/2210.17323).
Unlike formats that reorder groups, processing order is restored before export;
every final column uses its original contiguous group's SDK scale.
"""
import torch


class Hessian:
    def __init__(self, width, device):
        self.matrix = torch.zeros(width, width, dtype=torch.float32, device=device)
        self.tokens = 0

    @torch.no_grad()
    def add(self, _module, inputs, _outputs):
        values = inputs[0].detach().reshape(-1, self.matrix.shape[0]).float()
        self.matrix.addmm_(values.T, values)
        self.tokens += values.shape[0]


@torch.no_grad()
def quantize(weight, hessian, scales, group_size, damping=0.01, block_size=128):
    """Return on-grid weights and actual reconstruction losses; never mutate input."""
    if weight.ndim != 2 or weight.shape[1] % group_size or block_size < 1 or damping <= 0:
        raise ValueError("Invalid GPTQ weight/group/block dimensions")
    rows, columns = weight.shape
    if hessian.shape != (columns, columns) or scales.shape != (rows, columns // group_size):
        raise ValueError("GPTQ Hessian or scale shape differs from weight")
    if not all(torch.isfinite(value).all() for value in (weight, hessian, scales)) or not (scales > 0).all():
        raise ValueError("Non-finite GPTQ input or non-positive scales")
    original = weight.float()
    h = hessian.double().clone()
    dead = h.diag() == 0
    h.diagonal()[dead] = 1
    h.diagonal().add_(damping * h.diag().mean())
    order = torch.argsort(h.diag(), descending=True)
    inverse_order = torch.argsort(order)
    h = h[order][:, order]
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    upper = torch.linalg.cholesky(inverse, upper=True).float()
    working = original[:, order].clone()
    result = torch.empty_like(working)
    groups = (order // group_size).tolist()
    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        block = working[:, start:end].clone()
        error = torch.zeros_like(block)
        for local in range(end - start):
            index = start + local
            scale = scales[:, groups[index]]
            value = torch.round(block[:, local] / scale).clamp(-8, 7) * scale
            result[:, index] = value
            residual = (block[:, local] - value) / upper[index, index]
            error[:, local] = residual
            block[:, local:] -= residual[:, None] * upper[index, index:end][None]
        if end < columns:
            working[:, end:] -= error @ upper[start:end, end:]
    result = result[:, inverse_order]
    repeated = scales.repeat_interleave(group_size, dim=1)
    baseline = torch.round(original / repeated).clamp(-8, 7) * repeated
    def loss(candidate):
        error = candidate - original
        return float(((error @ hessian) * error).sum().clamp_min(0))
    baseline_loss, optimized_loss = loss(baseline), loss(result)
    # Greedy updates with a fixed hardware codebook need not improve every
    # matrix. Keep the candidate with lower measured calibration error.
    fallback = optimized_loss > baseline_loss
    if fallback:
        result, optimized_loss = baseline, baseline_loss
    return result, {"rtn_loss": baseline_loss, "gptq_loss": optimized_loss,
                    "used_rtn_fallback": fallback,
                    "loss_ratio": optimized_loss / max(baseline_loss, 1e-30)}

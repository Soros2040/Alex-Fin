import torch


def enforce_long_only(weights: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(weights, min=0.0)
    return clipped / (clipped.sum(dim=-1, keepdim=True) + 1e-8)


def enforce_industry_cap(weights: torch.Tensor, cap: float = 0.25) -> torch.Tensor:
    capped = torch.clamp(weights, max=cap)
    return capped / (capped.sum(dim=-1, keepdim=True) + 1e-8)

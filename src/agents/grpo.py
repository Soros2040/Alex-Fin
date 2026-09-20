from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GRPOConfig:
    group_size: int = 16
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 1e-3


class GRPOPolicy(nn.Module):
    def __init__(self, d_model: int, num_assets: int):
        super().__init__()
        self.mu = nn.Linear(d_model, num_assets)
        self.log_std = nn.Parameter(torch.full((num_assets,), -2.0))

    def dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mu(x)
        std = torch.exp(torch.clamp(self.log_std, -20.0, 2.0))
        return torch.distributions.Normal(mean, std)

    def sample_weights(self, x: torch.Tensor, group_size: int):
        dist = self.dist(x)
        z = dist.rsample((group_size,))
        w = torch.softmax(z, dim=-1)
        log_prob = dist.log_prob(z).sum(dim=-1)
        return z, w, log_prob

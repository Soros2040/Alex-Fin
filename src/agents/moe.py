import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_scale: int = 4):
        super().__init__()
        hidden = d_model * hidden_scale
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(d_model, hidden)
        self.w3 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class AdaptiveMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_experts: int = 8,
        top_k: int = 2,
        bias_lr: float = 0.05,
    ):
        super().__init__()
        self.shared = SwiGLUExpert(d_model)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts
        self.bias_lr = bias_lr
        self.input_norm = RMSNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.register_buffer("routing_bias", torch.zeros(num_experts))
        self._aux_stats: dict[str, float] = {
            "orthogonal_loss": 0.0,
            "load_balance_loss": 0.0,
            "routing_bias_mean": 0.0,
            "routing_bias_std": 0.0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        shared_out = self.shared(x_norm)
        gate_logits = self.gate(x_norm) + self.routing_bias.view(1, 1, -1)
        gate_scores = torch.softmax(gate_logits, dim=-1)
        top_vals, top_idx = torch.topk(gate_scores, k=min(self.top_k, self.num_experts), dim=-1)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)
        expert_outs = torch.stack([expert(x_norm) for expert in self.experts], dim=0)
        one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()
        dispatch = (one_hot * top_vals.unsqueeze(-1)).sum(dim=2)
        routed = torch.einsum("bne,ebnd->bnd", dispatch, expert_outs)
        load = dispatch.sum(dim=(0, 1))
        load = load / (load.sum() + 1e-8)
        target = torch.full_like(load, 1.0 / self.num_experts)
        load_balance_loss = torch.mean((load - target) ** 2)
        if self.training:
            with torch.no_grad():
                delta = target - load
                self.routing_bias.add_(self.bias_lr * delta)
        shared_n = F.normalize(shared_out, dim=-1)
        routed_n = F.normalize(routed, dim=-1)
        orthogonal_loss = torch.mean((shared_n * routed_n).sum(dim=-1) ** 2)
        self._aux_stats = {
            "orthogonal_loss": float(orthogonal_loss.detach().cpu().item()),
            "load_balance_loss": float(load_balance_loss.detach().cpu().item()),
            "routing_bias_mean": float(self.routing_bias.mean().detach().cpu().item()),
            "routing_bias_std": float(self.routing_bias.std(unbiased=False).detach().cpu().item()),
        }
        self._aux_loss = orthogonal_loss + load_balance_loss
        return self.norm(x + shared_out + routed)

    def get_aux_loss(self) -> torch.Tensor:
        return getattr(self, "_aux_loss", torch.tensor(0.0, device=self.routing_bias.device))

    def get_aux_stats(self) -> dict[str, float]:
        return dict(self._aux_stats)

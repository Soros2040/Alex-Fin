import torch


def transaction_fee(turnover: torch.Tensor, fee_rate: float = 5e-4) -> torch.Tensor:
    return turnover * fee_rate

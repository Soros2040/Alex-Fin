from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from src.agents.reward import DSRState, compute_dsr_reward
from .constraints import enforce_industry_cap, enforce_long_only
from .costs import transaction_fee


@dataclass
class EnvConfig:
    lookback: int = 20
    fee_rate: float = 5e-4
    industry_cap: float = 0.25
    initial_cash: float = 1_000_000.0
    dsr_cost_lambda: float = 1.0


class TradingEnv:
    def __init__(self, returns_df: pd.DataFrame, config: EnvConfig):
        self.returns_df = returns_df.fillna(0.0)
        self.config = config
        self.num_assets = returns_df.shape[1]
        self.reset()

    def reset(self) -> Dict[str, torch.Tensor]:
        self.ptr = self.config.lookback
        self.weights = torch.full((self.num_assets,), 1.0 / self.num_assets, dtype=torch.float32)
        self.portfolio_value = float(self.config.initial_cash)
        self.max_value = float(self.config.initial_cash)
        self.done = False
        self.dsr_state = DSRState(cost_lambda=self.config.dsr_cost_lambda)
        return self._state()

    def _state(self) -> Dict[str, torch.Tensor]:
        window = self.returns_df.iloc[self.ptr - self.config.lookback:self.ptr].values.T
        market_series = self.returns_df.iloc[:self.ptr].mean(axis=1).fillna(0.0)
        vol20 = float(market_series.tail(20).std()) if len(market_series) >= 2 else 0.0
        vol60 = float(market_series.tail(60).std()) if len(market_series) >= 2 else 0.0
        vol_ratio = vol20 / (vol60 + 1e-8)
        vix_proxy = vol20 * np.sqrt(252.0)
        return {
            "returns_window": torch.tensor(window, dtype=torch.float32),
            "risk_factors": torch.tensor([vol20, vol_ratio, vix_proxy], dtype=torch.float32),
        }

    def _advance(self, target_weights: torch.Tensor, do_commit: bool) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, float]]:
        tw = enforce_industry_cap(enforce_long_only(target_weights), cap=self.config.industry_cap)
        today_ret = torch.tensor(self.returns_df.iloc[self.ptr].values, dtype=torch.float32)
        turnover = torch.sum(torch.abs(tw - self.weights))
        fee = transaction_fee(turnover, self.config.fee_rate)
        gross = torch.dot(self.weights, today_ret)
        net = gross - fee
        reward, next_dsr = compute_dsr_reward(self.dsr_state, float(net.item()), float(turnover.item()))
        next_value = self.portfolio_value * (1.0 + float(net.item()))
        next_max = max(self.max_value, next_value)
        drawdown = next_value / (next_max + 1e-12) - 1.0
        if do_commit:
            self.weights = tw
            self.portfolio_value = next_value
            self.max_value = next_max
            self.dsr_state = next_dsr
            self.ptr += 1
            self.done = self.ptr >= len(self.returns_df) - 1
        info = {
            "turnover": float(turnover.item()),
            "fee": float(fee.item()),
            "net_return": float(net.item()),
            "portfolio_value": float(next_value),
            "drawdown": float(drawdown),
        }
        next_state = (
            self._state()
            if not self.done and self.ptr < len(self.returns_df)
            else {
                "returns_window": torch.zeros((self.num_assets, self.config.lookback)),
                "risk_factors": torch.zeros((3,), dtype=torch.float32),
            }
        )
        return next_state, reward, self.done, info

    def step(self, target_weights: torch.Tensor):
        return self._advance(target_weights, do_commit=True)

    def simulate(self, target_weights: torch.Tensor):
        ptr = self.ptr
        weights = self.weights.clone()
        value = self.portfolio_value
        max_value = self.max_value
        dsr = DSRState(
            a_t=self.dsr_state.a_t,
            b_t=self.dsr_state.b_t,
            eta=self.dsr_state.eta,
            eps=self.dsr_state.eps,
            cost_lambda=self.dsr_state.cost_lambda,
        )
        next_state, reward, done, info = self._advance(target_weights, do_commit=False)
        self.ptr = ptr
        self.weights = weights
        self.portfolio_value = value
        self.max_value = max_value
        self.dsr_state = dsr
        return next_state, reward, done, info

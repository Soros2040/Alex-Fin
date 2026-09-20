import pandas as pd
import torch

from src.agents.mbpstsa_gnn import MBPSTSAGNNEncoder
from src.agents.moe import AdaptiveMoE
from src.envs.trading_env import EnvConfig, TradingEnv
from src.training.trainer import TrainConfig, _build_rolling_slices


def test_encoder_supports_four_frequencies_with_adjacency():
    encoder = MBPSTSAGNNEncoder(num_assets=6, lookback=12, d_model=32, num_heads=4)
    x = {
        "day": torch.randn(1, 6, 12),
        "week": torch.randn(1, 6, 12),
        "month": torch.randn(1, 6, 12),
        "quarter": torch.randn(1, 6, 12),
    }
    adj = {k: torch.eye(6) for k in x.keys()}
    y = encoder(x, adjacency=adj)
    assert tuple(y.shape) == (1, 6, 32)


def test_moe_exposes_aux_stats_and_bias_updates():
    moe = AdaptiveMoE(d_model=16, num_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 5, 16)
    _ = moe(x)
    stats = moe.get_aux_stats()
    assert "orthogonal_loss" in stats
    assert "load_balance_loss" in stats
    assert moe.get_aux_loss().item() >= 0.0


def test_trading_env_returns_risk_factors():
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    returns_df = pd.DataFrame(
        {
            "a": [0.001] * 80,
            "b": [0.002] * 80,
            "c": [0.0015] * 80,
        },
        index=dates,
    )
    env = TradingEnv(returns_df=returns_df, config=EnvConfig(lookback=20))
    state = env.reset()
    assert "risk_factors" in state
    assert tuple(state["risk_factors"].shape) == (3,)


def test_rolling_slices_generate_multiple_windows():
    index = pd.date_range("2010-01-01", periods=3200, freq="B")
    cfg = TrainConfig(epochs=2, train_years=7.0, val_years=1.0, test_years=1.0, rolling_step_years=1.0)
    windows = _build_rolling_slices(index, cfg)
    assert len(windows) >= 2
    assert set(["train", "val", "test"]).issubset(set(windows[0].keys()))

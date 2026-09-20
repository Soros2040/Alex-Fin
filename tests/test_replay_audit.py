import pandas as pd

from src.training.replay_audit import replay_trade_log


def test_replay_trade_log_passes(tmp_path):
    csv_path = tmp_path / "trades.csv"
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "asset": "sh600000", "portfolio_value": 101.0, "daily_pnl": 1.0},
            {"date": "2026-01-01", "asset": "sh600001", "portfolio_value": 101.0, "daily_pnl": 1.0},
            {"date": "2026-01-02", "asset": "sh600000", "portfolio_value": 102.5, "daily_pnl": 1.5},
            {"date": "2026-01-02", "asset": "sh600001", "portfolio_value": 102.5, "daily_pnl": 1.5},
        ]
    )
    df.to_csv(csv_path, index=False)
    result = replay_trade_log(str(csv_path), tolerance=1e-9)
    assert result["passed"] is True

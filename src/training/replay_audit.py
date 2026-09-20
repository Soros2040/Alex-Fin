from typing import Any, Dict

import pandas as pd


def replay_trade_log(trade_csv_path: str, tolerance: float = 1e-6) -> Dict[str, Any]:
    df = pd.read_csv(trade_csv_path)
    if df.empty:
        return {"passed": False, "reason": "empty_trade_log"}
    required = {"date", "portfolio_value", "daily_pnl"}
    if not required.issubset(set(df.columns)):
        return {"passed": False, "reason": "missing_required_columns"}
    day_df = (
        df.sort_values(["date"])
        .groupby("date", as_index=False)
        .agg({"portfolio_value": "first", "daily_pnl": "first"})
    )
    first_value = float(day_df["portfolio_value"].iloc[0] - day_df["daily_pnl"].iloc[0])
    replay_values = []
    running = first_value
    for pnl in day_df["daily_pnl"].tolist():
        running = running + float(pnl)
        replay_values.append(running)
    day_df["replay_portfolio_value"] = replay_values
    day_df["abs_error"] = (day_df["replay_portfolio_value"] - day_df["portfolio_value"]).abs()
    max_abs_error = float(day_df["abs_error"].max())
    base = max(abs(first_value), 1.0)
    relative_error = max_abs_error / base
    passed = relative_error <= tolerance
    return {
        "passed": passed,
        "max_abs_error": max_abs_error,
        "relative_error": float(relative_error),
        "tolerance": tolerance,
        "checked_days": int(len(day_df)),
    }

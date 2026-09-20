import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from sklearn.covariance import LedoitWolf
except Exception:
    LedoitWolf = None

from src.agents import (
    AdaptiveMoE,
    GRPOConfig,
    GRPOPolicy,
    MBPSTSAGNNEncoder,
    build_frequency_adjacency,
    get_glasso_dy_stats,
    reset_glasso_dy_stats,
)
from src.envs import EnvConfig, TradingEnv
from .replay_audit import replay_trade_log


@dataclass
class TrainConfig:
    data_path: str = "src/data/cache/daily.parquet"
    lightweight_data: bool = False
    lightweight_data_dir: str = "src/data-lightweight/data"
    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "checkpoints"
    top_n_stocks: int = 50
    epochs: int = 100
    train_years: float = 7.0
    val_years: float = 1.0
    test_years: float = 1.0
    lookback: int = 20
    fee_rate: float = 0.0005
    initial_cash: float = 1_000_000.0
    seed: int = 42
    model_id: str = "alexfin_base"
    d_model: int = 256
    num_heads: int = 8
    num_experts: int = 8
    top_k_experts: int = 2
    group_size: int = 64
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    base_checkpoint_path: str | None = None
    base_model_id: str | None = None
    data_version: str = "daily_cache_v1"
    feature_version: str = "feature_mask_v1"
    rolling_step_years: float = 1.0
    quick_mode: bool = False
    early_stop_patience: int = 8
    warmup_epochs: int = 10
    stable_epochs: int = 60
    moe_aux_loss_coef: float = 0.02
    glasso_alpha: float = 0.02
    glasso_min_history: int = 60
    adjacency_threshold: float = 0.0
    require_glasso_dy: bool = False
    enable_extra_frequencies: bool = False
    extra_frequencies: Tuple[str, ...] = ("second", "minute", "hour")


def _to_stock_code(value: str) -> str:
    if "." not in value:
        return value.lower()
    code, market = value.split(".")
    return f"{market.lower()}{code}"


def _calc_metrics(portfolio_returns: np.ndarray) -> Dict[str, float]:
    if portfolio_returns.size == 0:
        return {"dsr": 0.0, "sharpe": 0.0, "annual_return": 0.0, "mdd": 0.0}
    mean_r = float(np.mean(portfolio_returns))
    std_r = float(np.std(portfolio_returns) + 1e-8)
    sharpe = mean_r / std_r * np.sqrt(252)
    equity = np.cumprod(1.0 + portfolio_returns)
    annual_return = float(equity[-1] ** (252 / max(len(portfolio_returns), 1)) - 1)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / (running_max + 1e-12) - 1.0
    mdd = float(np.min(drawdowns))
    eta = 2.0 / 253.0
    a_t = 0.0
    b_t = 0.0
    dsr_vals: List[float] = []
    for r_t in portfolio_returns:
        dsr = (b_t * (r_t - a_t) - 0.5 * a_t * (r_t * r_t - b_t)) / ((b_t - a_t * a_t + 1e-8) ** 1.5)
        dsr_vals.append(float(np.clip(dsr, -10.0, 10.0)))
        a_t = a_t + eta * (r_t - a_t)
        b_t = b_t + eta * (r_t * r_t - b_t)
    return {"dsr": float(np.mean(dsr_vals)), "sharpe": float(sharpe), "annual_return": float(annual_return), "mdd": float(mdd)}


def _load_daily_prices(config: TrainConfig, stock_list: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """加载日线价格数据，支持轻量化数据"""
    if config.lightweight_data:
        from src.data_lightweight.loader import LightweightDataLoader
        loader = LightweightDataLoader(config.lightweight_data_dir)
        
        if stock_list is None:
            stock_list = loader.instruments[:config.top_n_stocks] if config.top_n_stocks > 0 else loader.instruments
        
        if len(stock_list) == 0:
            raise ValueError("可用股票池为空，无法构建训练/评估数据")
        
        close_data = {}
        for stock_code in stock_list:
            try:
                values = loader.load_feature(stock_code, 'close', 'day')
                close_data[stock_code] = values
            except Exception as e:
                import logging
                logging.warning(f"加载 {stock_code} 的 close 特征失败：{e}")
        
        if not close_data:
            raise ValueError("无法加载任何股票的 close 数据")
        
        df = pd.DataFrame(close_data)
        df.index = pd.to_datetime(loader.calendar)
        df.index.name = 'trade_date'
        df = df.reset_index()
        df = df.melt(id_vars=['trade_date'], var_name='stock_code', value_name='close')
        df = df.dropna(subset=['close'])
        
        top_stocks = [s for s in stock_list if s in close_data]
    else:
        df = pd.read_parquet(config.data_path)
        if "stock_code" not in df.columns and "ts_code" in df.columns:
            df["stock_code"] = df["ts_code"].astype(str).apply(_to_stock_code)
        if "trade_date" not in df.columns:
            raise ValueError("daily parquet 缺少 trade_date 列")
        if "close" not in df.columns:
            raise ValueError("daily parquet 缺少 close 列")
        df = df[["trade_date", "stock_code", "close"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["trade_date", "stock_code"])
        if stock_list:
            top_stocks = [s for s in stock_list if s in set(df["stock_code"].unique().tolist())]
        else:
            top_stocks = sorted(df["stock_code"].unique().tolist())[: config.top_n_stocks]
        if len(top_stocks) == 0:
            raise ValueError("可用股票池为空，无法构建训练/评估数据")
        df = df[df["stock_code"].isin(top_stocks)]
        pivot = df.pivot_table(index="trade_date", columns="stock_code", values="close", aggfunc="last")
        pivot = pivot.sort_index().ffill().dropna(how="all")
        return pivot, top_stocks
    
    pivot = df.pivot_table(index="trade_date", columns="stock_code", values="close", aggfunc="last")
    pivot = pivot.sort_index().ffill().dropna(how="all")
    return pivot, top_stocks


def _build_slices(index: pd.DatetimeIndex, config: TrainConfig) -> Dict[str, pd.DatetimeIndex]:
    train_len = int(252 * config.train_years)
    val_len = int(252 * config.val_years)
    test_len = int(252 * config.test_years)
    need = train_len + val_len + test_len + config.lookback
    if len(index) < need:
        raise ValueError(f"交易日不足，至少需要{need}天，当前{len(index)}天")
    end = len(index)
    test_start = end - test_len
    val_start = test_start - val_len
    train_start = val_start - train_len
    return {"train": index[train_start:val_start], "val": index[val_start:test_start], "test": index[test_start:end]}


def _build_rolling_slices(index: pd.DatetimeIndex, config: TrainConfig) -> List[Dict[str, Any]]:
    base_slice = _build_slices(index, config)
    if config.quick_mode:
        return [{"window_id": 0, **base_slice}]
    train_len = int(252 * config.train_years)
    val_len = int(252 * config.val_years)
    test_len = int(252 * config.test_years)
    need = train_len + val_len + test_len + config.lookback
    step = max(int(252 * config.rolling_step_years), 1)
    out: List[Dict[str, Any]] = []
    start = 0
    wid = 0
    while start + need <= len(index):
        window = index[start : start + need]
        train_end = config.lookback + train_len
        val_end = train_end + val_len
        out.append(
            {
                "window_id": wid,
                "train": window[config.lookback:train_end],
                "val": window[train_end:val_end],
                "test": window[val_end:],
            }
        )
        wid += 1
        start += step
    if not out:
        out.append({"window_id": 0, **base_slice})
    return out


def _build_multi_frequency_inputs(
    close_df: pd.DataFrame,
    dt: pd.Timestamp,
    lookback: int,
) -> Dict[str, torch.Tensor]:
    freq_map = {
        "day": close_df,
        "week": close_df.resample("W-FRI").last(),
        "month": close_df.resample("ME").last(),
        "quarter": close_df.resample("QE").last(),
    }
    out: Dict[str, torch.Tensor] = {}
    for freq, data in freq_map.items():
        part = data.loc[:dt].dropna(how="all")
        ret = part.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(ret) < lookback:
            padded = np.zeros((lookback, ret.shape[1]), dtype=np.float32)
            if len(ret) > 0:
                padded[-len(ret) :] = ret.to_numpy(dtype=np.float32, copy=False)
            ret = pd.DataFrame(padded, columns=ret.columns)
        else:
            ret = ret.tail(lookback)
        win = ret.values.T
        out[freq] = torch.tensor(win, dtype=torch.float32).unsqueeze(0)
    return out


def _compose_state_embed(feat: torch.Tensor, risk_factors: torch.Tensor) -> torch.Tensor:
    state_embed = feat.mean(dim=1)
    if risk_factors.ndim == 1:
        risk_factors = risk_factors.unsqueeze(0)
    if risk_factors.size(1) < state_embed.size(1):
        pad = torch.zeros(
            (risk_factors.size(0), state_embed.size(1) - risk_factors.size(1)),
            device=state_embed.device,
            dtype=state_embed.dtype,
        )
        risk_vec = torch.cat([risk_factors.to(state_embed.device, state_embed.dtype), pad], dim=1)
    else:
        risk_vec = risk_factors[:, : state_embed.size(1)].to(state_embed.device, state_embed.dtype)
    return state_embed + 0.1 * risk_vec


def _apply_stage_schedule(optimizer: torch.optim.Optimizer, config: TrainConfig, epoch: int) -> tuple[str, float]:
    if epoch <= config.warmup_epochs:
        stage = "cold_start"
        lr_scale = 0.5
        entropy_coef = config.entropy_coef * 1.5
    elif epoch <= config.warmup_epochs + config.stable_epochs:
        stage = "stable_train"
        lr_scale = 1.0
        entropy_coef = config.entropy_coef
    else:
        stage = "early_stop_watch"
        lr_scale = 0.3
        entropy_coef = config.entropy_coef * 0.7
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * lr_scale
    return stage, entropy_coef


def _run_period_grpo(
    close_df: pd.DataFrame,
    period_index: pd.DatetimeIndex,
    encoder: MBPSTSAGNNEncoder,
    moe: AdaptiveMoE,
    policy: GRPOPolicy,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    entropy_coef: float,
    train_mode: bool,
    export_trades: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, float]]:
    returns_df = close_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    period_returns = returns_df.loc[period_index]
    env = TradingEnv(
        returns_df=period_returns,
        config=EnvConfig(
            lookback=config.lookback,
            fee_rate=config.fee_rate,
            industry_cap=0.25,
            initial_cash=config.initial_cash,
            dsr_cost_lambda=1.0,
        ),
    )
    state = env.reset()
    trades: List[Dict[str, Any]] = []
    portfolio_returns: List[float] = []
    losses: List[float] = []
    entropies: List[float] = []
    adv_means: List[float] = []
    moe_aux_losses: List[float] = []
    grpo_cfg = GRPOConfig(
        group_size=config.group_size,
        clip_epsilon=config.clip_epsilon,
        learning_rate=config.learning_rate,
        entropy_coef=entropy_coef,
    )
    while True:
        step_dt = period_returns.index[min(env.ptr, len(period_returns.index) - 1)]
        day_pos = close_df.index.get_loc(step_dt)
        mf_input = _build_multi_frequency_inputs(close_df, step_dt, config.lookback)
        adjacency = build_frequency_adjacency(
            close_df=close_df,
            end_loc=day_pos,
            frequencies=("day", "week", "month", "quarter"),
            min_history=config.glasso_min_history,
            l1_alpha=config.glasso_alpha,
            threshold=config.adjacency_threshold,
            strict_graphical_lasso=config.require_glasso_dy,
        )
        pre_weights = env.weights.clone()
        with torch.set_grad_enabled(train_mode):
            feat = encoder(mf_input, adjacency=adjacency)
            feat = moe(feat)
            state_embed = _compose_state_embed(feat, state["risk_factors"])
            rollout_policy = policy
            if train_mode:
                rollout_policy = GRPOPolicy(d_model=config.d_model, num_assets=env.num_assets)
                rollout_policy.load_state_dict(policy.state_dict())
                rollout_policy.eval()
            with torch.no_grad():
                z, w, old_log_prob = rollout_policy.sample_weights(state_embed, grpo_cfg.group_size)
            rewards = []
            for g in range(grpo_cfg.group_size):
                _, r, _, _ = env.simulate(w[g, 0])
                rewards.append(r)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            dist = policy.dist(state_embed)
            new_log_prob = dist.log_prob(z[:, 0, :]).sum(dim=-1)
            ratio = torch.exp(new_log_prob - old_log_prob[:, 0])
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - grpo_cfg.clip_epsilon, 1.0 + grpo_cfg.clip_epsilon) * adv
            entropy = dist.entropy().mean()
            moe_aux = moe.get_aux_loss() * config.moe_aux_loss_coef
            loss = -torch.min(s1, s2).mean() - grpo_cfg.entropy_coef * entropy + moe_aux
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(moe.parameters()) + list(policy.parameters()), 0.5)
                optimizer.step()
            best_idx = int(torch.argmax(rewards_t).item())
            chosen = w[best_idx, 0].detach()
        next_state, _, done, step_info = env.step(chosen)
        portfolio_returns.append(float(step_info["net_return"]))
        losses.append(float(loss.detach().cpu().item()))
        entropies.append(float(entropy.detach().cpu().item()))
        adv_means.append(float(adv.mean().detach().cpu().item()))
        moe_aux_losses.append(float(moe_aux.detach().cpu().item()))
        if export_trades:
            dt = period_returns.index[min(env.ptr - 1, len(period_returns.index) - 1)].strftime("%Y-%m-%d")
            prev_value = config.initial_cash if len(portfolio_returns) == 1 else float(step_info["portfolio_value"] / (1.0 + step_info["net_return"] + 1e-12))
            for i, asset in enumerate(period_returns.columns.tolist()):
                trades.append(
                    {
                        "date": dt,
                        "asset": asset,
                        "action": "buy" if chosen[i].item() > pre_weights[i].item() else ("sell" if chosen[i].item() < pre_weights[i].item() else "hold"),
                        "target_weight": float(chosen[i].item()),
                        "pre_weight": float(pre_weights[i].item()),
                        "post_weight": float(chosen[i].item()),
                        "price": float(close_df.loc[period_returns.index[min(env.ptr - 1, len(period_returns.index) - 1)], asset]),
                        "shares": float(0.0),
                        "turnover": step_info["turnover"],
                        "fee": step_info["fee"],
                        "cash": float(0.0),
                        "portfolio_value": step_info["portfolio_value"],
                        "daily_pnl": float(step_info["portfolio_value"] - prev_value),
                        "drawdown": step_info["drawdown"],
                        "constraint_flags": "ok",
                    }
                )
        state = next_state
        if done:
            break
    metrics = _calc_metrics(np.array(portfolio_returns, dtype=np.float64))
    metrics["turnover"] = float(np.mean([row["turnover"] for row in trades])) if trades else 0.0
    moe_stats = moe.get_aux_stats()
    train_stats = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "adv_mean": float(np.mean(adv_means)) if adv_means else 0.0,
        "moe_aux_loss": float(np.mean(moe_aux_losses)) if moe_aux_losses else 0.0,
        "moe_orthogonal": moe_stats.get("orthogonal_loss", 0.0),
        "moe_load_balance": moe_stats.get("load_balance_loss", 0.0),
        "moe_routing_bias_mean": moe_stats.get("routing_bias_mean", 0.0),
    }
    return metrics, trades, train_stats


def _simulate_mvo_baseline(close_df: pd.DataFrame, period_index: pd.DatetimeIndex, config: TrainConfig) -> Tuple[Dict[str, float], np.ndarray, str]:
    returns = close_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = close_df.columns.tolist()
    n_assets = len(cols)
    weights_prev = np.ones(n_assets) / n_assets
    portfolio_returns: List[float] = []
    cov_method = "LedoitWolf" if LedoitWolf is not None else "ShrinkageFallback"
    for dt in period_index:
        pos = close_df.index.get_loc(dt)
        hist_start = max(0, pos - 60)
        hist = returns.iloc[hist_start:pos]
        if len(hist) < 20:
            w = np.ones(n_assets) / n_assets
        else:
            mu = hist.mean().values
            if LedoitWolf is not None:
                cov = LedoitWolf().fit(hist.values).covariance_
            else:
                sample_cov = np.cov(hist.values, rowvar=False)
                diag = np.diag(np.diag(sample_cov))
                cov = 0.9 * sample_cov + 0.1 * diag
            inv_cov = np.linalg.pinv(cov + np.eye(n_assets) * 1e-8)
            raw = np.clip(inv_cov @ mu, 0.0, None)
            w = np.ones(n_assets) / n_assets if raw.sum() <= 1e-12 else raw / raw.sum()
        day_ret = returns.loc[dt].values
        turnover = np.abs(w - weights_prev).sum()
        fee = config.fee_rate * turnover
        portfolio_returns.append(float(np.dot(weights_prev, day_ret) - fee))
        weights_prev = w
    arr = np.array(portfolio_returns, dtype=np.float64)
    return _calc_metrics(arr), arr, cov_method


def _information_ratio(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    n = min(len(strategy_returns), len(benchmark_returns))
    if n == 0:
        return 0.0
    spread = strategy_returns[:n] - benchmark_returns[:n]
    return float(np.mean(spread) / (np.std(spread) + 1e-8) * np.sqrt(252))


def _daily_returns_from_trades(trades: List[Dict[str, Any]]) -> np.ndarray:
    if not trades:
        return np.array([], dtype=np.float64)
    df = pd.DataFrame(trades)
    if df.empty or "date" not in df.columns or "portfolio_value" not in df.columns:
        return np.array([], dtype=np.float64)
    day_df = df.sort_values("date").groupby("date", as_index=False).agg({"portfolio_value": "first"})
    series = day_df["portfolio_value"].astype(float).values
    if len(series) < 2:
        return np.array([], dtype=np.float64)
    return (series[1:] / (series[:-1] + 1e-12) - 1.0).astype(np.float64)


def _ensure_dirs(config: TrainConfig) -> Dict[str, Path]:
    artifacts = Path(config.artifacts_dir)
    metrics = artifacts / "metrics"
    reports = artifacts / "reports"
    trades = artifacts / "trades"
    cfgs = artifacts / "configs"
    checkpoints = Path(config.checkpoints_dir)
    for p in [metrics, reports, trades, cfgs, checkpoints]:
        p.mkdir(parents=True, exist_ok=True)
    return {"metrics": metrics, "reports": reports, "trades": trades, "configs": cfgs, "checkpoints": checkpoints}


def _load_base_weights_if_any(config: TrainConfig, encoder: MBPSTSAGNNEncoder, moe: AdaptiveMoE, policy: GRPOPolicy) -> None:
    if not config.base_checkpoint_path:
        return
    ckpt_json = Path(config.base_checkpoint_path)
    if not ckpt_json.exists():
        raise FileNotFoundError(f"base checkpoint不存在: {config.base_checkpoint_path}")
    payload = json.loads(ckpt_json.read_text(encoding="utf-8"))
    weights_path = payload.get("weights_path")
    if not weights_path:
        raise ValueError("base checkpoint缺少weights_path")
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"base权重不存在: {weights_path}")
    weight_payload = torch.load(weights_file, map_location="cpu")
    encoder.load_state_dict(weight_payload["encoder"])
    moe.load_state_dict(weight_payload["moe"])
    policy.load_state_dict(weight_payload["policy"])


def run_training(config: TrainConfig) -> Dict[str, Any]:
    print(f"[training] start model_id={config.model_id}, epochs={config.epochs}, top_n_stocks={config.top_n_stocks}", flush=True)
    reset_glasso_dy_stats()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dirs = _ensure_dirs(config)
    close_df, stock_list = _load_daily_prices(config)
    rolling_slices = _build_rolling_slices(close_df.index, config)
    num_assets = close_df.shape[1]
    encoder = MBPSTSAGNNEncoder(
        num_assets=num_assets,
        lookback=config.lookback,
        d_model=config.d_model,
        num_heads=config.num_heads,
        frequencies=("day", "week", "month", "quarter"),
        enable_extra_frequencies=config.enable_extra_frequencies,
        extra_frequencies=config.extra_frequencies,
        use_frequency_embedding=True,  # 启用可学习频率嵌入（遵循设计文档第 2.2 节）
        freq_embedding_dim=config.d_model,  # 频率嵌入维度与 d_model 对齐
    )
    moe = AdaptiveMoE(d_model=config.d_model, num_experts=config.num_experts, top_k=config.top_k_experts)
    policy = GRPOPolicy(d_model=config.d_model, num_assets=num_assets)
    _load_base_weights_if_any(config, encoder, moe, policy)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(moe.parameters()) + list(policy.parameters()), lr=config.learning_rate, weight_decay=0.05)
    history_rows: List[Dict[str, Any]] = []
    best_epoch = 1
    best_score = -1e18
    best_snapshot: Dict[str, Any] = {}
    epoch90_snapshot: Dict[str, Any] = {}
    global_epoch = 0
    for window in rolling_slices:
        patience_count = 0
        for local_epoch in range(1, config.epochs + 1):
            global_epoch += 1
            epoch_start = time.time()
            stage_name, stage_entropy = _apply_stage_schedule(optimizer, config, local_epoch)
            train_metrics, _, train_stats = _run_period_grpo(
                close_df=close_df,
                period_index=window["train"],
                encoder=encoder,
                moe=moe,
                policy=policy,
                optimizer=optimizer,
                config=config,
                entropy_coef=stage_entropy,
                train_mode=True,
                export_trades=False,
            )
            val_metrics, _, _ = _run_period_grpo(
                close_df=close_df,
                period_index=window["val"],
                encoder=encoder,
                moe=moe,
                policy=policy,
                optimizer=optimizer,
                config=config,
                entropy_coef=stage_entropy,
                train_mode=False,
                export_trades=False,
            )
            row = {
                "window_id": window["window_id"],
                "epoch": local_epoch,
                "global_epoch": global_epoch,
                "stage": stage_name,
                "train_dsr": train_metrics["dsr"],
                "train_sharpe": train_metrics["sharpe"],
                "val_dsr": val_metrics["dsr"],
                "val_sharpe": val_metrics["sharpe"],
                "val_annual_return": val_metrics["annual_return"],
                "val_mdd": val_metrics["mdd"],
                "loss": train_stats["loss"],
                "entropy": train_stats["entropy"],
                "adv_mean": train_stats["adv_mean"],
                "moe_aux_loss": train_stats["moe_aux_loss"],
                "moe_orthogonal": train_stats["moe_orthogonal"],
                "moe_load_balance": train_stats["moe_load_balance"],
                "moe_routing_bias_mean": train_stats["moe_routing_bias_mean"],
            }
            history_rows.append(row)
            print(
                f"[training] window={window['window_id']} epoch={local_epoch}/{config.epochs} stage={stage_name} "
                f"train_dsr={train_metrics['dsr']:.6f} val_dsr={val_metrics['dsr']:.6f} "
                f"loss={train_stats['loss']:.6f} elapsed={time.time() - epoch_start:.2f}s",
                flush=True,
            )
            if val_metrics["dsr"] > best_score:
                best_score = val_metrics["dsr"]
                best_epoch = global_epoch
                best_snapshot = {"encoder": encoder.state_dict(), "moe": moe.state_dict(), "policy": policy.state_dict()}
                patience_count = 0
            else:
                patience_count += 1
            if global_epoch == 90:
                epoch90_snapshot = {"encoder": encoder.state_dict(), "moe": moe.state_dict(), "policy": policy.state_dict()}
            if patience_count >= config.early_stop_patience:
                print(f"[training] window={window['window_id']} early stop at epoch={local_epoch}", flush=True)
                break
        if best_snapshot:
            encoder.load_state_dict(best_snapshot["encoder"])
            moe.load_state_dict(best_snapshot["moe"])
            policy.load_state_dict(best_snapshot["policy"])
    history_df = pd.DataFrame(history_rows)
    history_path = dirs["metrics"] / f"{config.model_id}_train_history.csv"
    history_df.to_csv(history_path, index=False)
    final_window = rolling_slices[-1]
    if best_snapshot:
        encoder.load_state_dict(best_snapshot["encoder"])
        moe.load_state_dict(best_snapshot["moe"])
        policy.load_state_dict(best_snapshot["policy"])
    best_test_metrics, best_test_trades, _ = _run_period_grpo(
        close_df=close_df,
        period_index=final_window["test"],
        encoder=encoder,
        moe=moe,
        policy=policy,
        optimizer=optimizer,
        config=config,
        entropy_coef=config.entropy_coef,
        train_mode=False,
        export_trades=True,
    )
    final_snapshot = {"encoder": encoder.state_dict(), "moe": moe.state_dict(), "policy": policy.state_dict()}
    final_test_metrics, final_test_trades, _ = _run_period_grpo(
        close_df=close_df,
        period_index=final_window["test"],
        encoder=encoder,
        moe=moe,
        policy=policy,
        optimizer=optimizer,
        config=config,
        entropy_coef=config.entropy_coef,
        train_mode=False,
        export_trades=True,
    )
    epoch90_metrics: Dict[str, Any] = {}
    epoch90_trades: List[Dict[str, Any]] = []
    if epoch90_snapshot:
        encoder.load_state_dict(epoch90_snapshot["encoder"])
        moe.load_state_dict(epoch90_snapshot["moe"])
        policy.load_state_dict(epoch90_snapshot["policy"])
        epoch90_metrics, epoch90_trades, _ = _run_period_grpo(
            close_df=close_df,
            period_index=final_window["test"],
            encoder=encoder,
            moe=moe,
            policy=policy,
            optimizer=optimizer,
            config=config,
            entropy_coef=config.entropy_coef,
            train_mode=False,
            export_trades=True,
        )
    if epoch90_trades:
        epoch90_df = pd.DataFrame(epoch90_trades)
        epoch90_trade_path = dirs["trades"] / f"{config.model_id}_epoch90_test_trades.csv"
        epoch90_df.to_csv(epoch90_trade_path, index=False)
        with open(dirs["trades"] / f"{config.model_id}_epoch90_test_summary.json", "w", encoding="utf-8") as f:
            json.dump(epoch90_metrics, f, ensure_ascii=False, indent=2)
        epoch90_audit = replay_trade_log(str(epoch90_trade_path))
        with open(dirs["reports"] / f"{config.model_id}_epoch90_replay_audit.json", "w", encoding="utf-8") as f:
            json.dump(epoch90_audit, f, ensure_ascii=False, indent=2)
    else:
        epoch90_audit = {"passed": False, "reason": "epoch90_trades_empty"}
    final_df = pd.DataFrame(final_test_trades)
    final_trade_path = dirs["trades"] / f"{config.model_id}_final_test_trades.csv"
    final_df.to_csv(final_trade_path, index=False)
    with open(dirs["trades"] / f"{config.model_id}_final_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_test_metrics, f, ensure_ascii=False, indent=2)
    final_audit = replay_trade_log(str(final_trade_path))
    with open(dirs["reports"] / f"{config.model_id}_final_replay_audit.json", "w", encoding="utf-8") as f:
        json.dump(final_audit, f, ensure_ascii=False, indent=2)
    mvo_metrics, mvo_returns, mvo_cov_method = _simulate_mvo_baseline(close_df, final_window["test"], config)
    final_returns = _daily_returns_from_trades(final_test_trades)
    mvo_comparison = {
        "strategy_metrics": final_test_metrics,
        "mvo_metrics": mvo_metrics,
        "strategy_minus_mvo": {
            "dsr": float(final_test_metrics.get("dsr", 0.0) - mvo_metrics.get("dsr", 0.0)),
            "sharpe": float(final_test_metrics.get("sharpe", 0.0) - mvo_metrics.get("sharpe", 0.0)),
            "annual_return": float(final_test_metrics.get("annual_return", 0.0) - mvo_metrics.get("annual_return", 0.0)),
            "mdd": float(final_test_metrics.get("mdd", 0.0) - mvo_metrics.get("mdd", 0.0)),
        },
        "information_ratio": _information_ratio(final_returns, mvo_returns),
        "mvo_window_days": 60,
        "mvo_covariance": mvo_cov_method,
        "mvo_objective": "max_sharpe",
    }
    with open(dirs["reports"] / f"{config.model_id}_mvo_comparison.json", "w", encoding="utf-8") as f:
        json.dump(mvo_comparison, f, ensure_ascii=False, indent=2)
    best_weights_path = dirs["checkpoints"] / f"{config.model_id}_best_model.pt"
    last_weights_path = dirs["checkpoints"] / f"{config.model_id}_last_model.pt"
    torch.save(best_snapshot if best_snapshot else final_snapshot, best_weights_path)
    torch.save(final_snapshot, last_weights_path)
    best_ckpt = {
        "model_id": config.model_id,
        "epoch": best_epoch,
        "model_type": "mb-pstsa-gnn_moe_grpo",
        "metrics": best_test_metrics,
        "weights_path": str(best_weights_path),
        "base_model_id": config.base_model_id,
        "data_version": config.data_version,
        "feature_version": config.feature_version,
        "train_window_spec": f"{config.train_years}Y/{config.val_years}Y/{config.test_years}Y",
        "stocks": stock_list,
    }
    last_ckpt = {
        "model_id": config.model_id,
        "epoch": global_epoch,
        "model_type": "mb-pstsa-gnn_moe_grpo",
        "metrics": final_test_metrics,
        "weights_path": str(last_weights_path),
        "base_model_id": config.base_model_id,
        "data_version": config.data_version,
        "feature_version": config.feature_version,
        "train_window_spec": f"{config.train_years}Y/{config.val_years}Y/{config.test_years}Y",
        "stocks": stock_list,
    }
    with open(dirs["checkpoints"] / f"{config.model_id}_best.json", "w", encoding="utf-8") as f:
        json.dump(best_ckpt, f, ensure_ascii=False, indent=2)
    with open(dirs["checkpoints"] / f"{config.model_id}_last.json", "w", encoding="utf-8") as f:
        json.dump(last_ckpt, f, ensure_ascii=False, indent=2)
    with open(dirs["configs"] / f"{config.model_id}_training_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    summary = {
        "model_id": config.model_id,
        "best_epoch": best_epoch,
        "best_validation_dsr": best_score,
        "best_test_metrics": best_test_metrics,
        "final_test_metrics": final_test_metrics,
        "epoch90_test_metrics": epoch90_metrics,
        "epoch90_replay_audit": epoch90_audit,
        "final_replay_audit": final_audit,
        "mvo_comparison": mvo_comparison,
        "rolling_windows": len(rolling_slices),
        "train_history_path": str(history_path),
        "glasso_dy_stats": get_glasso_dy_stats(),
        "glasso_dy_mode": "strict" if config.require_glasso_dy else "best_effort",
    }
    with open(dirs["metrics"] / f"{config.model_id}_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[training] completed model_id={config.model_id}, best_epoch={best_epoch}", flush=True)
    return summary


def run_evaluation(config: TrainConfig, checkpoint_path: str) -> Dict[str, Any]:
    reset_glasso_dy_stats()
    ckpt = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    ckpt_stocks = ckpt.get("stocks", [])
    close_df, eval_stocks = _load_daily_prices(config, stock_list=ckpt_stocks if ckpt_stocks else None)
    slices = _build_slices(close_df.index, config)
    num_assets = close_df.shape[1]
    encoder = MBPSTSAGNNEncoder(
        num_assets=num_assets,
        lookback=config.lookback,
        d_model=config.d_model,
        num_heads=config.num_heads,
        frequencies=("day", "week", "month", "quarter"),
        enable_extra_frequencies=config.enable_extra_frequencies,
        extra_frequencies=config.extra_frequencies,
        use_frequency_embedding=True,  # 启用可学习频率嵌入（遵循设计文档第 2.2 节）
        freq_embedding_dim=config.d_model,  # 频率嵌入维度与 d_model 对齐
    )
    moe = AdaptiveMoE(d_model=config.d_model, num_experts=config.num_experts, top_k=config.top_k_experts)
    policy = GRPOPolicy(d_model=config.d_model, num_assets=num_assets)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(moe.parameters()) + list(policy.parameters()), lr=config.learning_rate, weight_decay=0.05)
    weights_path = ckpt.get("weights_path")
    if not weights_path:
        raise ValueError("checkpoint缺少weights_path，无法评估")
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"weights_path不存在: {weights_path}")
    payload = torch.load(weights_path, map_location="cpu")
    encoder.load_state_dict(payload["encoder"])
    moe.load_state_dict(payload["moe"])
    policy.load_state_dict(payload["policy"])
    metrics, trades, _ = _run_period_grpo(
        close_df=close_df,
        period_index=slices["test"],
        encoder=encoder,
        moe=moe,
        policy=policy,
        optimizer=optimizer,
        config=config,
        entropy_coef=config.entropy_coef,
        train_mode=False,
        export_trades=True,
    )
    dirs = _ensure_dirs(config)
    trade_path = dirs["trades"] / f"{config.model_id}_eval_test_trades.csv"
    pd.DataFrame(trades).to_csv(trade_path, index=False)
    replay_audit = replay_trade_log(str(trade_path))
    with open(dirs["reports"] / f"{config.model_id}_eval_replay_audit.json", "w", encoding="utf-8") as f:
        json.dump(replay_audit, f, ensure_ascii=False, indent=2)
    mvo_metrics, mvo_returns, mvo_cov_method = _simulate_mvo_baseline(close_df, slices["test"], config)
    strategy_returns = _daily_returns_from_trades(trades)
    mvo_comparison = {
        "strategy_metrics": metrics,
        "mvo_metrics": mvo_metrics,
        "strategy_minus_mvo": {
            "dsr": float(metrics.get("dsr", 0.0) - mvo_metrics.get("dsr", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0) - mvo_metrics.get("sharpe", 0.0)),
            "annual_return": float(metrics.get("annual_return", 0.0) - mvo_metrics.get("annual_return", 0.0)),
            "mdd": float(metrics.get("mdd", 0.0) - mvo_metrics.get("mdd", 0.0)),
        },
        "information_ratio": _information_ratio(strategy_returns, mvo_returns),
        "mvo_window_days": 60,
        "mvo_covariance": mvo_cov_method,
        "mvo_objective": "max_sharpe",
    }
    with open(dirs["reports"] / f"{config.model_id}_eval_mvo_comparison.json", "w", encoding="utf-8") as f:
        json.dump(mvo_comparison, f, ensure_ascii=False, indent=2)
    report = {
        "model_id": config.model_id,
        "checkpoint": checkpoint_path,
        "eval_stocks": eval_stocks,
        "metrics": metrics,
        "trade_path": str(trade_path),
        "replay_audit": replay_audit,
        "mvo_comparison": mvo_comparison,
        "glasso_dy_stats": get_glasso_dy_stats(),
        "glasso_dy_mode": "strict" if config.require_glasso_dy else "best_effort",
    }
    with open(dirs["reports"] / f"{config.model_id}_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report

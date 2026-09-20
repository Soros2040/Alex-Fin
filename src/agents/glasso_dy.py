"""
Graphical LASSO-Diebold-Yilmaz (GLASSO-DY) 风险溢出网络构建模块

本模块严格遵循 Agent 架构设计文档第二章 2.4 节定义，实现：
1. 分频率独立 GLASSO 稀疏精度矩阵估计
2. DY 有向风险溢出分解
3. 分频率邻接矩阵构建

核心参数：
- 滚动窗口长度 T=60 个交易日
- DY 预测步长 H=5
- 使用 sklearn.covariance.graphical_lasso 实现
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import graphical_lasso
except ImportError:
    graphical_lasso = None
    warnings.warn("sklearn.covariance.graphical_lasso 不可用，GLASSO 估计将降级为 Ledoit-Wolf 收缩估计")


@dataclass
class GLASSODYStats:
    """GLASSO-DY 统计信息"""
    total_spillover: float = 0.0
    from_spillover: Dict[str, float] = field(default_factory=dict)
    to_spillover: Dict[str, float] = field(default_factory=dict)
    net_spillover: Dict[str, float] = field(default_factory=dict)
    pairwise_spillover: Dict[Tuple[str, str], float] = field(default_factory=dict)
    num_valid_stocks: int = 0
    num_missing_stocks: int = 0
    missing_stocks: List[str] = field(default_factory=list)


_global_stats = GLASSODYStats()


def reset_glasso_dy_stats():
    """重置全局统计信息"""
    global _global_stats
    _global_stats = GLASSODYStats()


def get_glasso_dy_stats() -> Dict:
    """获取全局统计信息"""
    return {
        "total_spillover": _global_stats.total_spillover,
        "num_valid_stocks": _global_stats.num_valid_stocks,
        "num_missing_stocks": _global_stats.num_missing_stocks,
        "missing_stocks": _global_stats.missing_stocks[:20],
    }


def _estimate_precision_matrix(
    returns: np.ndarray,
    l1_alpha: float = 0.02,
    min_history: int = 60,
) -> Tuple[np.ndarray, bool]:
    """
    使用 GLASSO 估计稀疏精度矩阵
    
    参数:
        returns: T×N 的收益率矩阵，T 为时间窗口长度，N 为资产数量
        l1_alpha: L1 惩罚参数，控制稀疏程度
        min_history: 最小历史数据要求
        
    返回:
        precision_matrix: N×N 的稀疏精度矩阵
        success: GLASSO 估计是否成功
    """
    T, N = returns.shape
    
    if T < min_history:
        warnings.warn(f"历史数据长度 {T} < 最小要求 {min_history}，使用单位矩阵替代")
        return np.eye(N), False
    
    if returns.shape[1] < 2:
        return np.eye(max(1, N)), False
    
    try:
        returns_centered = returns - returns.mean(axis=0, keepdims=True)
        emp_cov = np.atleast_2d(np.cov(returns_centered, rowvar=False))
        
        if graphical_lasso is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, precision_matrix = graphical_lasso(
                        emp_cov,
                        alpha=l1_alpha,
                        max_iter=200,
                        mode="cd",
                        tol=1e-4,
                    )
                    
                    n_off_diag = N * (N - 1)
                    off_diag_nonzero = np.sum(np.abs(precision_matrix) > 1e-8) - n_off_diag
                    sparsity_ratio = off_diag_nonzero / n_off_diag if n_off_diag > 0 else 0
                    
                    if sparsity_ratio < 0.1:
                        warnings.warn(f"GLASSO 结果过于稀疏 (非零比例={sparsity_ratio:.2%})，使用 Ledoit-Wolf 估计")
                        precision_matrix = np.linalg.inv(emp_cov + 1e-6 * np.eye(N))
                        return precision_matrix, False
                    
                    return precision_matrix, True
                except Exception:
                    pass
        
        emp_cov_reg = emp_cov + 1e-6 * np.eye(N)
        try:
            precision_matrix = np.linalg.inv(emp_cov_reg)
            return precision_matrix, False
        except np.linalg.LinAlgError:
            return np.eye(N), False
            
    except Exception:
        return np.eye(max(1, N)), False


def _dy_decomposition(
    precision_matrix: np.ndarray,
    returns: np.ndarray,
    H: int = 5,
) -> np.ndarray:
    """
    DY 有向风险溢出分解（基于 GLASSO 精度矩阵）
    
    严格遵循架构文档公式 (2.3)：
    d^{f,H}_{ij} = σ_{jj}^{-1} × (Ω^f_{ij})^2 / Σ_k[σ_{kk}^{-1} × (Ω^f_{ik})^2]
    
    参数:
        precision_matrix: N×N 的 GLASSO 精度矩阵 Ω
        returns: T×N 的收益率矩阵
        H: 预测步长（架构文档 H=5）
        
    返回:
        spillover_matrix: N×N 的风险溢出强度矩阵
    """
    N = precision_matrix.shape[0]
    
    if N == 0:
        return np.array([[]])
    
    variances = np.var(returns, axis=0, ddof=1)
    variances = np.clip(variances, 1e-10, None)
    inv_variances = 1.0 / variances
    
    omega_squared = precision_matrix ** 2
    
    numerator = inv_variances.reshape(1, -1) * omega_squared
    
    denominator = np.sum(omega_squared * inv_variances.reshape(1, -1), axis=1, keepdims=True)
    denominator = np.clip(denominator, 1e-10, None)
    
    spillover_matrix = numerator / denominator
    
    spillover_matrix = np.clip(spillover_matrix, 0.0, 1.0)
    
    return spillover_matrix


def _row_normalize_spillover(
    spillover_matrix: np.ndarray,
    stock_codes: List[str],
) -> Tuple[np.ndarray, GLASSODYStats]:
    """
    标准化溢出矩阵并计算统计指标
    
    参考论文标准化步骤（列标准化）：
    1. 对角线置 0（自身不向自身溢出）
    2. 列标准化：每列除以该列的和
    3. 标准化后，每列和为 1（某机构的溢出完全分配给其他机构）
    
    注意：spillover_matrix[i,j] 表示从 j 到 i 的溢出（j→i）
    因此列标准化表示：机构 j 的总溢出分配给所有其他机构的比例
    
    参数:
        spillover_matrix: N×N 的溢出强度矩阵，[i,j] 表示 j→i
        stock_codes: 股票代码列表
        
    返回:
        normalized_matrix: 标准化后的邻接矩阵
        stats: GLASSO-DY 统计信息
    """
    N = spillover_matrix.shape[0]
    stats = GLASSODYStats()
    stats.num_valid_stocks = N
    
    # 1. 对角线置 0
    np.fill_diagonal(spillover_matrix, 0.0)
    
    # 2. 列标准化（参考论文方法）
    # spillover_matrix[i,j] = j→i 的溢出
    # 列和 = 机构 j 向所有其他机构的总溢出
    col_sums = np.sum(spillover_matrix, axis=0, keepdims=True)
    
    zero_col_mask = col_sums.flatten() < 1e-10
    col_sums = np.clip(col_sums, 1e-10, None)
    normalized_matrix = spillover_matrix / col_sums  # 广播除法
    
    # 3. 处理零列（该机构不向任何其他机构溢出）
    if np.any(zero_col_mask):
        non_diagonal_count = N - 1
        if non_diagonal_count > 0:
            uniform_value = 1.0 / non_diagonal_count
            for j in range(N):
                if zero_col_mask[j]:
                    normalized_matrix[:, j] = uniform_value
                    normalized_matrix[j, j] = 0.0
    
    # 4. 计算统计指标
    # 总溢出指数：所有溢出的平均值
    total_spillover = np.sum(normalized_matrix) / N
    stats.total_spillover = float(total_spillover)
    
    # From spillover：机构 i 从其他机构接收的溢出（行和）
    from_spillover = np.sum(normalized_matrix, axis=1)
    # To spillover：机构 i 向其他机构发出的溢出（列和）
    to_spillover = np.sum(normalized_matrix, axis=0)
    # Net spillover：净溢出（发出 - 接收）
    net_spillover = to_spillover - from_spillover
    
    for i, code in enumerate(stock_codes):
        stats.from_spillover[code] = float(from_spillover[i])
        stats.to_spillover[code] = float(to_spillover[i])
        stats.net_spillover[code] = float(net_spillover[i])
        
        for j, code2 in enumerate(stock_codes):
            if i != j:
                # pairwise_spillover[(i, j)] 表示从 j 到 i 的溢出
                stats.pairwise_spillover[(code, code2)] = float(normalized_matrix[i, j])
    
    return normalized_matrix, stats


def build_glasso_dy_adjacency(
    returns_df: pd.DataFrame,
    l1_alpha: float = 0.02,
    min_history: int = 60,
    H: int = 5,
    stock_codes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, GLASSODYStats]:
    """
    构建 GLASSO-DY 有向风险溢出邻接矩阵
    
    参数:
        returns_df: pandas DataFrame，每列为一只股票的收益率序列
        l1_alpha: GLASSO 的 L1 惩罚参数
        min_history: 最小历史数据要求
        H: DY 预测步长
        stock_codes: 股票代码列表（可选）
        
    返回:
        adjacency_matrix: N×N 的有向邻接矩阵
        stats: GLASSO-DY 统计信息
    """
    if returns_df.empty or returns_df.shape[0] < min_history:
        N = len(returns_df.columns) if stock_codes is None else len(stock_codes)
        stats = GLASSODYStats()
        stats.num_valid_stocks = 0
        stats.num_missing_stocks = N
        stats.missing_stocks = stock_codes if stock_codes else []
        return np.zeros((max(1, N), max(1, N))), stats
    
    returns_array = returns_df.values
    actual_codes = stock_codes if stock_codes else list(returns_df.columns)
    
    valid_mask = ~np.any(np.isnan(returns_array), axis=0)
    valid_returns = returns_array[:, valid_mask]
    valid_codes = [actual_codes[i] for i in range(len(actual_codes)) if valid_mask[i]]
    
    N_original = len(actual_codes)
    N_valid = valid_returns.shape[1]
    
    if N_valid < 2:
        stats = GLASSODYStats()
        stats.num_valid_stocks = N_valid
        stats.num_missing_stocks = N_original - N_valid
        stats.missing_stocks = [actual_codes[i] for i in range(N_original) if not valid_mask[i]]
        return np.zeros((max(1, N_original), max(1, N_original))), stats
    
    precision_matrix, success = _estimate_precision_matrix(
        valid_returns,
        l1_alpha=l1_alpha,
        min_history=min_history,
    )
    
    if precision_matrix.shape[0] != N_valid:
        precision_matrix = precision_matrix[:N_valid, :N_valid]
    
    spillover_matrix = _dy_decomposition(
        precision_matrix,
        valid_returns,
        H=H,
    )
    
    adjacency_valid, stats = _row_normalize_spillover(spillover_matrix, valid_codes)
    
    N = N_original
    adjacency_matrix = np.zeros((N, N))
    
    valid_indices = [i for i in range(N_original) if valid_mask[i]]
    for i, vi in enumerate(valid_indices):
        for j, vj in enumerate(valid_indices):
            adjacency_matrix[vi, vj] = adjacency_valid[i, j]
    
    stats.num_missing_stocks = N_original - N_valid
    stats.missing_stocks = [actual_codes[i] for i in range(N_original) if not valid_mask[i]]
    
    global _global_stats
    _global_stats = stats
    
    return adjacency_matrix, stats


def build_frequency_adjacency(
    close_df: pd.DataFrame,
    end_loc: int,
    frequencies: Tuple[str, ...] = ("day", "week", "month", "quarter"),
    min_history: int = 60,
    l1_alpha: float = 0.02,
    threshold: float = 0.0,
    strict_graphical_lasso: bool = False,
) -> Dict[str, np.ndarray]:
    """
    为多个频率分别构建 GLASSO-DY 邻接矩阵
    
    参数:
        close_df: 收盘价 DataFrame，行为日期，列为股票代码
        end_loc: 当前时间点在 close_df 中的位置
        frequencies: 频率元组 ("day", "week", "month", "quarter")
        min_history: 最小历史数据要求
        l1_alpha: GLASSO 的 L1 惩罚参数
        threshold: 邻接矩阵阈值（小于该值的边置 0）
        strict_graphical_lasso: 是否强制使用 GLASSO
        
    返回:
        adjacency_dict: {frequency: adjacency_matrix} 字典
    """
    global _global_stats
    reset_glasso_dy_stats()
    
    adjacency_dict = {}
    all_stats = []
    
    for freq in frequencies:
        if freq == "day":
            freq_returns = close_df.iloc[:end_loc + 1].pct_change().dropna()
        elif freq == "week":
            weekly_close = close_df.iloc[:end_loc + 1].resample("W").last()
            freq_returns = weekly_close.pct_change().dropna()
        elif freq == "month":
            monthly_close = close_df.iloc[:end_loc + 1].resample("M").last()
            freq_returns = monthly_close.pct_change().dropna()
        elif freq == "quarter":
            quarterly_close = close_df.iloc[:end_loc + 1].resample("QE").last()
            freq_returns = quarterly_close.pct_change().dropna()
        else:
            continue
        
        if freq_returns.empty or freq_returns.shape[0] < min_history:
            N = len(freq_returns.columns) if not freq_returns.empty else 0
            adjacency_dict[freq] = np.zeros((max(1, N), max(1, N)))
            continue
        
        H_for_freq = 5
        
        adjacency_matrix, stats = build_glasso_dy_adjacency(
            freq_returns,
            l1_alpha=l1_alpha,
            min_history=min_history,
            H=H_for_freq,
        )
        
        if threshold > 0:
            adjacency_matrix[adjacency_matrix < threshold] = 0.0
        
        adjacency_dict[freq] = adjacency_matrix
        all_stats.append(stats)
    
    if all_stats:
        _global_stats = all_stats[-1]
    
    return adjacency_dict


def compute_spillover_indices(
    adjacency_matrix: np.ndarray,
    stock_codes: List[str],
) -> Dict[str, float]:
    """
    计算风险溢出指数
    
    参数:
        adjacency_matrix: N×N 的邻接矩阵
        stock_codes: 股票代码列表
        
    返回:
        indices: 包含总溢出指数、净溢出指数等
    """
    N = adjacency_matrix.shape[0]
    
    if N == 0:
        return {"total_spillover": 0.0}
    
    total_spillover = np.sum(adjacency_matrix) / N * 100
    
    from_spillover = np.sum(adjacency_matrix, axis=0) * 100
    to_spillover = np.sum(adjacency_matrix, axis=1) * 100
    net_spillover = to_spillover - from_spillover
    
    indices = {
        "total_spillover": float(total_spillover),
    }
    
    for i, code in enumerate(stock_codes):
        indices[f"from_{code}"] = float(from_spillover[i])
        indices[f"to_{code}"] = float(to_spillover[i])
        indices[f"net_{code}"] = float(net_spillover[i])
    
    return indices

"""
GLASSO-DY 模块单元测试

验证内容：
1. GLASSO 精度矩阵估计的对称性和正定性
2. DY 分解的有向性和行和约束
3. 邻接矩阵的标准化性质
4. 缺失数据处理逻辑
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.glasso_dy import (
    GLASSODYStats,
    _dy_decomposition,
    _estimate_precision_matrix,
    _row_normalize_spillover,
    build_frequency_adjacency,
    build_glasso_dy_adjacency,
    compute_spillover_indices,
    get_glasso_dy_stats,
    reset_glasso_dy_stats,
)


def generate_test_returns(n_assets=10, n_periods=100, seed=42):
    """生成模拟收益率数据"""
    np.random.seed(seed)
    returns = np.random.randn(n_periods, n_assets) * 0.02
    columns = [f"stock_{i:03d}" for i in range(n_assets)]
    dates = pd.date_range(start="2020-01-01", periods=n_periods, freq="D")
    return pd.DataFrame(returns, columns=columns, index=dates)


class TestPrecisionMatrixEstimation:
    """测试精度矩阵估计"""

    def test_precision_matrix_symmetry(self):
        """验证精度矩阵的对称性"""
        returns_df = generate_test_returns(n_assets=5, n_periods=60)
        returns_array = returns_df.values

        precision_matrix, success = _estimate_precision_matrix(
            returns_array,
            l1_alpha=0.02,
            min_history=60,
        )

        assert precision_matrix.shape == (5, 5)
        assert np.allclose(precision_matrix, precision_matrix.T), "精度矩阵应该是对称的"

    def test_precision_matrix_positive_definite(self):
        """验证精度矩阵的正定性"""
        returns_df = generate_test_returns(n_assets=5, n_periods=80)
        returns_array = returns_df.values

        precision_matrix, success = _estimate_precision_matrix(
            returns_array,
            l1_alpha=0.02,
            min_history=60,
        )

        eigenvalues = np.linalg.eigvalsh(precision_matrix)
        assert np.all(eigenvalues > -1e-10), "精度矩阵应该是正定的"

    def test_insufficient_history(self):
        """测试历史数据不足时的处理"""
        returns_df = generate_test_returns(n_assets=5, n_periods=30)
        returns_array = returns_df.values

        precision_matrix, success = _estimate_precision_matrix(
            returns_array,
            l1_alpha=0.02,
            min_history=60,
        )

        assert precision_matrix.shape == (5, 5)
        assert np.array_equal(precision_matrix, np.eye(5)), "历史数据不足时应返回单位矩阵"


class TestDYDecomposition:
    """测试 DY 风险溢出分解"""

    def test_dy_directional(self):
        """验证 DY 分解的有向性（非对称性）"""
        n_assets = 5
        precision_matrix = np.random.randn(n_assets, n_assets)
        precision_matrix = precision_matrix @ precision_matrix.T
        returns = np.random.randn(60, n_assets) * 0.02

        spillover_matrix = _dy_decomposition(precision_matrix, returns, H=5)

        assert spillover_matrix.shape == (n_assets, n_assets)
        is_asymmetric = not np.allclose(spillover_matrix, spillover_matrix.T)
        assert is_asymmetric, "DY 溢出矩阵应该是非对称的（有向性）"

    def test_dy_non_negative(self):
        """验证 DY 分解的非负性"""
        n_assets = 5
        precision_matrix = np.random.randn(n_assets, n_assets)
        precision_matrix = precision_matrix @ precision_matrix.T
        returns = np.random.randn(60, n_assets) * 0.02

        spillover_matrix = _dy_decomposition(precision_matrix, returns, H=5)

        assert np.all(spillover_matrix >= 0), "溢出强度应该非负"
        assert np.all(spillover_matrix <= 1), "溢出强度应该不超过 1"


class TestRowNormalization:
    """测试行标准化"""

    def test_row_sum_constraint(self):
        """验证行和为 1 的约束"""
        n_assets = 5
        spillover_matrix = np.random.rand(n_assets, n_assets)
        stock_codes = [f"stock_{i}" for i in range(n_assets)]

        normalized_matrix, stats = _row_normalize_spillover(spillover_matrix, stock_codes)

        row_sums = np.sum(normalized_matrix, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), "标准化后每行和应该为 1"

    def test_diagonal_zero(self):
        """验证对角线为 0"""
        n_assets = 5
        spillover_matrix = np.random.rand(n_assets, n_assets)
        stock_codes = [f"stock_{i}" for i in range(n_assets)]

        normalized_matrix, stats = _row_normalize_spillover(spillover_matrix, stock_codes)

        diagonal = np.diag(normalized_matrix)
        assert np.allclose(diagonal, 0.0), "对角线应该为 0"

    def test_stats_computation(self):
        """验证统计指标计算"""
        n_assets = 3
        spillover_matrix = np.array([
            [0.0, 0.3, 0.2],
            [0.1, 0.0, 0.4],
            [0.2, 0.3, 0.0],
        ])
        stock_codes = ["A", "B", "C"]

        normalized_matrix, stats = _row_normalize_spillover(spillover_matrix, stock_codes)

        assert stats.num_valid_stocks == 3
        assert "A" in stats.from_spillover
        assert "A" in stats.to_spillover
        assert "A" in stats.net_spillover


class TestGLASSODYAdjacency:
    """测试 GLASSO-DY 邻接矩阵构建"""

    def test_adjacency_structure(self):
        """验证邻接矩阵的基本结构"""
        returns_df = generate_test_returns(n_assets=10, n_periods=80)

        adjacency_matrix, stats = build_glasso_dy_adjacency(
            returns_df,
            l1_alpha=0.02,
            min_history=60,
            H=5,
        )

        assert adjacency_matrix.shape == (10, 10)
        assert stats.num_valid_stocks == 10
        assert stats.num_missing_stocks == 0

    def test_missing_data_handling(self):
        """验证缺失数据处理"""
        returns_df = generate_test_returns(n_assets=10, n_periods=80)
        returns_df.iloc[:, 5] = np.nan
        returns_df.iloc[:, 7] = np.nan

        adjacency_matrix, stats = build_glasso_dy_adjacency(
            returns_df,
            l1_alpha=0.02,
            min_history=60,
            H=5,
        )

        assert adjacency_matrix.shape == (10, 10)
        assert stats.num_valid_stocks == 8
        assert stats.num_missing_stocks == 2
        assert len(stats.missing_stocks) == 2

    def test_empty_dataframe(self):
        """验证空 DataFrame 处理"""
        returns_df = pd.DataFrame()

        adjacency_matrix, stats = build_glasso_dy_adjacency(
            returns_df,
            l1_alpha=0.02,
            min_history=60,
            H=5,
        )

        assert stats.num_valid_stocks == 0


class TestFrequencyAdjacency:
    """测试多频率邻接矩阵构建"""

    def test_multi_frequency(self):
        """验证多频率邻接矩阵构建"""
        n_assets = 5
        n_periods = 200
        np.random.seed(42)
        returns = np.random.randn(n_periods, n_assets) * 0.02
        columns = [f"stock_{i:03d}" for i in range(n_assets)]
        dates = pd.date_range(start="2020-01-01", periods=n_periods, freq="D")
        close_df = pd.DataFrame(
            np.cumprod(1 + returns, axis=0) * 100,
            columns=columns,
            index=dates,
        )

        adjacency_dict = build_frequency_adjacency(
            close_df=close_df,
            end_loc=n_periods - 1,
            frequencies=("day", "week"),
            min_history=60,
            l1_alpha=0.02,
        )

        assert "day" in adjacency_dict
        assert "week" in adjacency_dict
        assert adjacency_dict["day"].shape[0] == n_assets
        assert adjacency_dict["week"].shape[0] == n_assets


class TestSpilloverIndices:
    """测试溢出指数计算"""

    def test_total_spillover(self):
        """验证总溢出指数计算"""
        n_assets = 5
        adjacency_matrix = np.random.rand(n_assets, n_assets)
        np.fill_diagonal(adjacency_matrix, 0)
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        adjacency_matrix = adjacency_matrix / row_sums
        stock_codes = [f"stock_{i}" for i in range(n_assets)]

        indices = compute_spillover_indices(adjacency_matrix, stock_codes)

        assert "total_spillover" in indices
        assert indices["total_spillover"] >= 0

    def test_net_spillover(self):
        """验证净溢出指数计算"""
        n_assets = 3
        adjacency_matrix = np.array([
            [0.0, 0.3, 0.2],
            [0.1, 0.0, 0.4],
            [0.2, 0.3, 0.0],
        ])
        stock_codes = ["A", "B", "C"]

        indices = compute_spillover_indices(adjacency_matrix, stock_codes)

        assert "net_A" in indices
        assert "net_B" in indices
        assert "net_C" in indices


class TestGlobalStats:
    """测试全局统计信息"""

    def test_reset_and_get_stats(self):
        """验证统计信息重置和获取"""
        reset_glasso_dy_stats()
        stats = get_glasso_dy_stats()

        assert stats["total_spillover"] == 0.0
        assert stats["num_valid_stocks"] == 0
        assert stats["num_missing_stocks"] == 0

        returns_df = generate_test_returns(n_assets=5, n_periods=80)
        build_glasso_dy_adjacency(returns_df, l1_alpha=0.02, min_history=60, H=5)

        stats = get_glasso_dy_stats()
        assert stats["num_valid_stocks"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

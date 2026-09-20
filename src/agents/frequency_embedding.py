"""
可学习频率嵌入层实现

严格遵循 Agent 架构设计文档第 2.2 节规范：
1. 全局固定频率索引映射（0-6）
2. 可学习频率嵌入层 nn.Embedding(num_freqs=7, d_model=256)
3. 叠加方式：逐 Token 逐元素相加
4. 叠加时机：分块实例归一化→线性投影生成 Patch Token 之后
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from enum import IntEnum


class FrequencyIndex(IntEnum):
    """全局固定频率索引映射（遵循设计文档表 2-1）"""
    SECOND = 0      # 秒级
    MINUTE = 1      # 分钟级
    HOUR = 2        # 小时级
    DAY = 3         # 日级
    WEEK = 4        # 周级
    MONTH = 5       # 月度
    QUARTER = 6     # 季度


FREQUENCY_NAME_TO_INDEX = {
    "second": FrequencyIndex.SECOND,
    "minute": FrequencyIndex.MINUTE,
    "hour": FrequencyIndex.HOUR,
    "day": FrequencyIndex.DAY,
    "week": FrequencyIndex.WEEK,
    "month": FrequencyIndex.MONTH,
    "quarter": FrequencyIndex.QUARTER,
}


class LearnableFrequencyEmbedding(nn.Module):
    """
    可学习频率嵌入层
    
    设计参数（遵循设计文档 2.2.2 节）：
    - num_freqs: 7 类全局频率索引 (0-6)
    - d_model: 与全局隐藏层维度 256 对齐
    - 初始化：均值 0、标准差 0.02 的正态分布
    """
    
    def __init__(
        self,
        num_freqs: int = 7,
        d_model: int = 256,
        init_mean: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model
        self.init_mean = init_mean
        self.init_std = init_std
        
        # 频率嵌入层（遵循设计文档 2.2.2 节）
        self.embedding = nn.Embedding(
            num_embeddings=num_freqs,
            embedding_dim=d_model,
        )
        
        # 初始化嵌入权重（正态分布 N(0, 0.02^2)）
        self._init_weights()
        
        # 频率名称到索引的映射
        self.freq_map = FREQUENCY_NAME_TO_INDEX
    
    def _init_weights(self):
        """初始化嵌入权重为正态分布 N(0, 0.02^2)"""
        with torch.no_grad():
            self.embedding.weight.normal_(mean=self.init_mean, std=self.init_std)
    
    def forward(
        self,
        freq_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取频率嵌入向量
        
        参数:
            freq_indices: 频率索引张量，shape 为 [batch_size] 或 [batch_size, num_tokens]
                         值为 0-6 的整数
        
        返回:
            freq_embeddings: 频率嵌入向量，shape 为 [batch_size, d_model] 或 [batch_size, num_tokens, d_model]
        """
        return self.embedding(freq_indices)
    
    def get_embedding_for_frequency(
        self,
        freq_name: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        获取指定频率的嵌入向量
        
        参数:
            freq_name: 频率名称 ("day", "week", "month", "quarter" 等)
            device: 目标设备
        
        返回:
            频率嵌入向量 [d_model]
        """
        if freq_name not in self.freq_map:
            raise ValueError(f"未知频率名称：{freq_name}，支持的频率：{list(self.freq_map.keys())}")
        
        freq_idx = self.freq_map[freq_name]
        freq_tensor = torch.tensor([freq_idx], dtype=torch.long, device=device)
        return self.embedding(freq_tensor).squeeze(0)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        获取所有频率的嵌入向量
        
        返回:
            [num_freqs, d_model] 形状的张量
        """
        return self.embedding.weight.data
    
    def validate_embeddings(self) -> Dict[str, float]:
        """
        验证频率嵌入的统计特性
        
        返回:
            包含嵌入统计信息的字典
        """
        all_embeddings = self.get_all_embeddings()
        
        # 检查不同频率的嵌入是否不同
        embedding_diffs = []
        for i in range(self.num_freqs):
            for j in range(i + 1, self.num_freqs):
                diff = torch.norm(all_embeddings[i] - all_embeddings[j]).item()
                embedding_diffs.append(diff)
        
        return {
            "num_frequencies": self.num_freqs,
            "embedding_dim": self.d_model,
            "mean_norm": torch.norm(all_embeddings, dim=1).mean().item(),
            "min_pairwise_diff": min(embedding_diffs) if embedding_diffs else 0.0,
            "max_pairwise_diff": max(embedding_diffs) if embedding_diffs else 0.0,
            "mean_pairwise_diff": sum(embedding_diffs) / len(embedding_diffs) if embedding_diffs else 0.0,
        }


class FrequencyAwareFeatureProcessor(nn.Module):
    """
    频率感知特征处理器
    
    完整实现设计文档 2.2.3 节和 2.3 节的流程：
    1. 分块实例归一化
    2. 线性投影生成 Patch Token
    3. 叠加频率嵌入
    """
    
    def __init__(
        self,
        lookback: int,
        d_model: int,
        num_freqs: int = 4,  # 当前支持 day/week/month/quarter
        frequencies: List[str] = None,
    ):
        super().__init__()
        
        if frequencies is None:
            frequencies = ["day", "week", "month", "quarter"]
        
        self.lookback = lookback
        self.d_model = d_model
        self.frequencies = frequencies
        
        # 分块实例归一化（遵循设计文档 2.3 节）
        self.instance_norm = nn.InstanceNorm1d(
            num_features=lookback,
            eps=1e-8,
            affine=False,
        )
        
        # 线性投影生成 Patch Token（遵循设计文档 2.3 节）
        self.linear_proj = nn.Linear(lookback, d_model)
        
        # 可学习频率嵌入层（遵循设计文档 2.2.2 节）
        self.freq_embedding = LearnableFrequencyEmbedding(
            num_freqs=7,  # 支持 7 种频率
            d_model=d_model,
        )
        
        # 频率名称到索引的映射
        self.freq_map = FREQUENCY_NAME_TO_INDEX
    
    def forward(
        self,
        freq_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        处理多频率特征并叠加频率嵌入
        
        参数:
            freq_features: 字典，key 为频率名称，value 为特征张量 [batch_size, num_assets, lookback]
        
        返回:
            处理后的特征字典，每个特征已叠加频率嵌入 [batch_size, num_assets, d_model]
        """
        result = {}
        
        for freq_name, features in freq_features.items():
            # 1. 分块实例归一化
            # features shape: [batch, num_assets, lookback]
            # 需要转换为 [batch * num_assets, lookback, 1] 用于 InstanceNorm1d
            batch_size, num_assets, _ = features.shape
            features_reshaped = features.view(batch_size * num_assets, self.lookback, 1)
            features_reshaped = features_reshaped.transpose(1, 2)  # [batch*assets, 1, lookback]
            
            # 应用实例归一化
            normalized = self.instance_norm(features_reshaped)
            normalized = normalized.transpose(1, 2)  # [batch*assets, lookback, 1]
            normalized = normalized.view(batch_size, num_assets, self.lookback)
            
            # 2. 线性投影生成 Patch Token
            patch_tokens = self.linear_proj(normalized)  # [batch, num_assets, d_model]
            
            # 3. 叠加频率嵌入（遵循设计文档 2.2.3 节）
            freq_idx = self.freq_map.get(freq_name, FrequencyIndex.DAY)
            freq_tensor = torch.full(
                (batch_size, num_assets),
                fill_value=freq_idx,
                dtype=torch.long,
                device=patch_tokens.device,
            )
            freq_embed = self.freq_embedding(freq_tensor)  # [batch, num_assets, d_model]
            
            # 逐元素相加
            tokens_with_freq = patch_tokens + freq_embed
            
            result[freq_name] = tokens_with_freq
        
        return result
    
    def get_frequency_embedding(self, freq_name: str) -> torch.Tensor:
        """获取指定频率的嵌入向量"""
        return self.freq_embedding.get_embedding_for_frequency(freq_name)


def create_frequency_aware_inputs(
    freq_features: Dict[str, torch.Tensor],
    lookback: int,
    d_model: int,
) -> Dict[str, torch.Tensor]:
    """
    创建携带频率嵌入的多频率输入特征
    
    参数:
        freq_features: 多频率特征字典
        lookback: 回看窗口长度
        d_model: 模型隐藏层维度
    
    返回:
        携带频率嵌入的特征字典
    """
    processor = FrequencyAwareFeatureProcessor(
        lookback=lookback,
        d_model=d_model,
    )
    return processor(freq_features)

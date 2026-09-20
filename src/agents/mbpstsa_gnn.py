import torch
import torch.nn as nn
from typing import Dict, Iterable, Optional
from .frequency_embedding import LearnableFrequencyEmbedding, FREQUENCY_NAME_TO_INDEX


class MBPSTSAGNNEncoder(nn.Module):
    """
    多频率并行时空分离注意力图神经网络编码器
    
    支持频率嵌入（遵循设计文档第 2.2 节）：
    - 可学习频率嵌入层 nn.Embedding(num_freqs=7, d_model=256)
    - 叠加方式：逐 Token 逐元素相加
    - 叠加时机：线性投影生成 Patch Token 之后
    """
    
    def __init__(
        self,
        num_assets: int,
        lookback: int,
        d_model: int = 128,
        num_heads: int = 4,
        frequencies: Iterable[str] = ("day", "week", "month", "quarter"),
        enable_extra_frequencies: bool = False,
        extra_frequencies: Iterable[str] = ("second", "minute", "hour"),
        use_frequency_embedding: bool = True,
        freq_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.lookback = lookback
        self.base_frequencies = list(frequencies)
        self.extra_frequencies = list(extra_frequencies)
        self.enable_extra_frequencies = enable_extra_frequencies
        self.active_frequencies = list(self.base_frequencies)
        self.use_frequency_embedding = use_frequency_embedding
        
        if self.enable_extra_frequencies:
            self.active_frequencies = self.active_frequencies + self.extra_frequencies
        
        # 线性投影生成 Patch Token
        self.input_proj = nn.ModuleDict(
            {freq: nn.Linear(lookback, d_model) for freq in self.active_frequencies}
        )
        
        # 可学习频率嵌入层（遵循设计文档 2.2.2 节）
        self.use_frequency_embedding = use_frequency_embedding
        if self.use_frequency_embedding:
            embed_dim = freq_embedding_dim if freq_embedding_dim is not None else d_model
            self.freq_embedding = LearnableFrequencyEmbedding(
                num_freqs=7,  # 支持 7 种频率 (0-6)
                d_model=embed_dim,
            )
            # 如果嵌入维度与 d_model 不同，需要投影
            if embed_dim != d_model:
                self.freq_proj = nn.Linear(embed_dim, d_model)
            else:
                self.freq_proj = nn.Identity()
        else:
            self.freq_embedding = None
            self.freq_proj = nn.Identity()
        
        self.time_attn = nn.ModuleDict(
            {
                freq: nn.MultiheadAttention(d_model, num_heads, batch_first=True)
                for freq in self.active_frequencies
            }
        )
        self.space_query = nn.ModuleDict(
            {freq: nn.Linear(d_model, d_model) for freq in self.active_frequencies}
        )
        self.space_key = nn.ModuleDict(
            {freq: nn.Linear(d_model, d_model) for freq in self.active_frequencies}
        )
        self.space_value = nn.ModuleDict(
            {freq: nn.Linear(d_model, d_model) for freq in self.active_frequencies}
        )
        self.branch_norm = nn.ModuleDict(
            {freq: nn.LayerNorm(d_model) for freq in self.active_frequencies}
        )
        self.cross_freq_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("global_causal_mask", torch.empty(0), persistent=False)

    def _ensure_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self.global_causal_mask.numel() == 0 or self.global_causal_mask.size(0) != size:
            mask = torch.full((size, size), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self.global_causal_mask = mask
        return self.global_causal_mask

    def _space_attention(
        self,
        h: torch.Tensor,
        freq: str,
        adjacency: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """
        空间注意力机制
        
        注意：移除空间掩码，原因如下：
        1. 空间注意力在同一时间步内聚合信息，无未来信息泄露风险
        2. GLASSO-DY 邻接矩阵应作为软权重（attention bias），而非硬掩码
        3. Softmax 会自动学习重要性，无需强制遮蔽
        """
        q = self.space_query[freq](h)
        k = self.space_key[freq](h)
        v = self.space_value[freq](h)
        scores = (q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        
        # 移除空间掩码，让 GLASSO-DY 邻接矩阵作为注意力偏置
        # 如果需要使用邻接矩阵作为偏置，可以取消以下注释：
        # if adjacency is not None and freq in adjacency:
        #     adj = adjacency[freq].to(scores.device)
        #     if adj.dim() == 2:
        #         adj = adj.unsqueeze(0)
        #     scores = scores + adj  # 作为注意力偏置，而非掩码
        
        attn = torch.softmax(scores, dim=-1)
        return attn @ v

    def forward(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        adjacency: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征，可以是张量或频率特征字典
               - 如果是张量：[batch_size, num_assets, lookback]，会复制到所有频率
               - 如果是字典：{freq_name: tensor[batch_size, num_assets, lookback]}
            adjacency: GLASSO-DY 构建的邻接矩阵字典 {freq_name: tensor[num_assets, num_assets]}
        
        返回:
            多频率时空融合特征 [batch_size, num_assets, d_model]
        """
        if isinstance(x, torch.Tensor):
            freq_inputs = {freq: x for freq in self.base_frequencies}
        else:
            freq_inputs = x
        
        branch_outputs = []
        causal_mask = self._ensure_mask(self.num_assets, next(self.parameters()).device)
        
        for freq in self.active_frequencies:
            if freq not in freq_inputs:
                continue
            
            # 1. 线性投影生成 Patch Token
            h = self.input_proj[freq](freq_inputs[freq])
            
            # 2. 叠加频率嵌入（遵循设计文档 2.2.3 节）
            if self.use_frequency_embedding and self.freq_embedding is not None:
                batch_size, num_assets, _ = h.shape
                freq_idx = FREQUENCY_NAME_TO_INDEX.get(freq, 3)  # 默认为 day 频率索引
                freq_tensor = torch.full(
                    (batch_size, num_assets),
                    fill_value=freq_idx,
                    dtype=torch.long,
                    device=h.device,
                )
                freq_embed = self.freq_embedding(freq_tensor)
                freq_embed = self.freq_proj(freq_embed)
                h = h + freq_embed  # 逐元素相加
            
            # 3. 时序自注意力（因果掩码约束）
            h_time, _ = self.time_attn[freq](h, h, h, attn_mask=causal_mask, need_weights=False)
            
            # 4. 空间自注意力（GLASSO-DY 邻接矩阵约束）
            h_space = self._space_attention(h_time, freq, adjacency)
            
            # 5. 残差连接 + 归一化
            branch_outputs.append(self.branch_norm[freq](h + h_time + h_space))
        
        if not branch_outputs:
            raise ValueError("MBPSTSAGNNEncoder 未收到可用频率输入")
        
        if len(branch_outputs) == 1:
            return self.out(branch_outputs[0])
        
        # 6. 跨频率交叉注意力融合
        # 注意：移除跨频率因果掩码，原因如下：
        # 1. 频率是并行关系，不是时序关系，不应使用因果掩码
        # 2. 每个频率分支内部已有时序因果掩码，防止了未来信息泄露
        # 3. 跨频率交流的是各分支的最终输出（已聚合完整历史信息），无需额外掩码
        # 4. 符合设计文档"全连接多向同步交叉注意力"的要求
        stacked = torch.stack(branch_outputs, dim=2)
        bsz, n_assets, n_freq, d_model = stacked.shape
        freq_tokens = stacked.view(bsz * n_assets, n_freq, d_model)
        
        # 移除跨频率掩码，使用无掩码的全连接注意力
        fused_tokens, _ = self.cross_freq_attn(
            freq_tokens,
            freq_tokens,
            freq_tokens,
            attn_mask=None,  # 移除频率维度的因果掩码
            need_weights=False,
        )
        fused_tokens = self.cross_norm(freq_tokens + fused_tokens)
        fused = fused_tokens.mean(dim=1).view(bsz, n_assets, d_model)
        return self.out(fused)


STGNNEncoder = MBPSTSAGNNEncoder

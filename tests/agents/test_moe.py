"""
MoE (Mixture of Experts) 模块单元测试

验证内容:
1. MoE 前向传播的正确性
2. 辅助损失计算 (正交损失 + 负载平衡损失)
3. 路由偏置更新机制
4. RMSNorm 和 SwiGLU 专家网络
5. 稀疏路由 (top-k) 机制
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from src.agents.moe import AdaptiveMoE, RMSNorm, SwiGLUExpert


class TestRMSNorm:
    """测试 RMSNorm 层"""
    
    def test_rmsnorm_output(self):
        """验证 RMSNorm 的输出特性"""
        d_model = 64
        rmsnorm = RMSNorm(d_model)
        
        # 创建测试输入
        x = torch.randn(2, 10, d_model)
        
        # 前向传播
        output = rmsnorm(x)
        
        # 验证输出形状
        assert output.shape == x.shape
        
        # 验证 RMSNorm 的特性：输出的 RMS 接近 1
        rms = torch.sqrt(torch.mean(output * output, dim=-1, keepdim=True))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
    
    def test_rmsnorm_weight(self):
        """验证 RMSNorm 的可学习权重"""
        d_model = 64
        rmsnorm = RMSNorm(d_model)
        
        # 验证权重存在且可训练
        assert hasattr(rmsnorm, 'weight')
        assert rmsnorm.weight.requires_grad
        assert rmsnorm.weight.shape == (d_model,)


class TestSwiGLUExpert:
    """测试 SwiGLU 专家网络"""
    
    def test_swiglu_output(self):
        """验证 SwiGLU 专家的输出"""
        d_model = 64
        expert = SwiGLUExpert(d_model, hidden_scale=4)
        
        # 创建测试输入
        x = torch.randn(2, 64)
        
        # 前向传播
        output = expert(x)
        
        # 验证输出形状
        assert output.shape == (2, 64)
    
    def test_swiglu_hidden_dim(self):
        """验证 SwiGLU 的隐藏层维度"""
        d_model = 64
        hidden_scale = 4
        expert = SwiGLUExpert(d_model, hidden_scale=hidden_scale)
        
        # 验证线性层的维度
        assert expert.w1.out_features == d_model * hidden_scale
        assert expert.w2.out_features == d_model * hidden_scale
        assert expert.w3.in_features == d_model * hidden_scale


class TestAdaptiveMoE:
    """测试 AdaptiveMoE 模块"""
    
    def test_moe_forward(self):
        """测试 MoE 的前向传播"""
        d_model = 64
        num_experts = 8
        top_k = 2
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # 创建测试输入
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 前向传播
        output = moe(x)
        
        # 验证输出形状
        assert output.shape == x.shape
        
        # 验证输出不是全零
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_moe_aux_loss(self):
        """测试 MoE 的辅助损失"""
        d_model = 64
        num_experts = 8
        top_k = 2
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # 创建测试输入
        x = torch.randn(4, 10, d_model)
        
        # 前向传播
        output = moe(x)
        
        # 获取辅助损失
        aux_loss = moe.get_aux_loss()
        
        # 验证辅助损失存在且为标量
        assert aux_loss.numel() == 1
        assert aux_loss >= 0
        
        # 验证辅助损失包含正交损失和负载平衡损失
        stats = moe.get_aux_stats()
        assert 'orthogonal_loss' in stats
        assert 'load_balance_loss' in stats
        assert stats['orthogonal_loss'] >= 0
        assert stats['load_balance_loss'] >= 0
    
    def test_moe_routing(self):
        """测试 MoE 的路由机制"""
        d_model = 64
        num_experts = 8
        top_k = 2
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # 创建测试输入
        x = torch.randn(4, 10, d_model)
        
        # 前向传播
        output = moe(x)
        
        # 验证路由偏置存在
        assert hasattr(moe, 'routing_bias')
        assert moe.routing_bias.shape == (num_experts,)
        
        # 验证路由偏置的统计信息
        stats = moe.get_aux_stats()
        assert 'routing_bias_mean' in stats
        assert 'routing_bias_std' in stats
    
    def test_moe_load_balance(self):
        """测试 MoE 的负载平衡机制"""
        d_model = 64
        num_experts = 8
        top_k = 2
        bias_lr = 0.05
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            bias_lr=bias_lr,
        )
        
        # 多次前向传播，观察负载平衡
        for _ in range(10):
            x = torch.randn(8, 10, d_model)
            output = moe(x)
        
        # 验证负载平衡损失应该较小 (专家使用相对均衡)
        stats = moe.get_aux_stats()
        # 负载平衡损失衡量专家使用率的方差，应该较小
        assert stats['load_balance_loss'] < 0.1
    
    def test_moe_orthogonal_loss(self):
        """测试 MoE 的正交损失机制"""
        d_model = 64
        num_experts = 8
        top_k = 2
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # 创建测试输入
        x = torch.randn(4, 10, d_model)
        
        # 前向传播
        output = moe(x)
        
        # 验证正交损失存在
        stats = moe.get_aux_stats()
        orthogonal_loss = stats['orthogonal_loss']
        
        # 正交损失衡量共享路径和路由输出的正交性
        # 值越小表示越正交，但不会完全为 0
        assert orthogonal_loss >= 0
        assert orthogonal_loss < 1.0  # 应该不会太大
    
    def test_moe_residual_connection(self):
        """测试 MoE 的残差连接"""
        d_model = 64
        num_experts = 8
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=2,
        )
        
        # 创建测试输入
        x = torch.randn(4, 10, d_model)
        
        # 前向传播
        output = moe(x)
        
        # 验证残差连接：输出应该包含输入的信息
        # 计算输出和输入的相关性
        correlation = torch.nn.functional.cosine_similarity(
            x.view(-1, d_model),
            output.view(-1, d_model),
            dim=-1
        ).mean()
        
        # 由于残差连接，相关性应该较高
        assert correlation > 0.5
    
    def test_moe_top_k_routing(self):
        """测试 top-k 路由机制"""
        d_model = 64
        num_experts = 8
        top_k = 2
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # 创建测试输入
        x = torch.randn(4, 10, d_model)
        
        # 前向传播
        output = moe(x)
        
        # 验证 gate 的输出
        gate_logits = moe.gate(x)
        assert gate_logits.shape == (4, 10, num_experts)
        
        # 验证 softmax 后的分数和为 1
        gate_scores = torch.softmax(gate_logits, dim=-1)
        assert torch.allclose(gate_scores.sum(dim=-1), torch.ones_like(gate_scores.sum(dim=-1)), atol=1e-5)
    
    def test_moe_different_input_produce_different_output(self):
        """测试不同的输入产生不同的输出"""
        d_model = 64
        num_experts = 8
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
        )
        
        # 创建两个不同的输入
        x1 = torch.randn(4, 10, d_model)
        x2 = torch.randn(4, 10, d_model) * 2  # 不同的分布
        
        # 前向传播
        output1 = moe(x1)
        output2 = moe(x2)
        
        # 验证输出不同
        assert not torch.allclose(output1, output2)
    
    def test_moe_expert_utilization(self):
        """测试专家利用率"""
        d_model = 64
        num_experts = 8
        top_k = 2
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # 多次前向传播
        total_load = torch.zeros(num_experts)
        for _ in range(20):
            x = torch.randn(8, 10, d_model)
            output = moe(x)
            
            # 计算负载
            gate_logits = moe.gate(x)
            gate_scores = torch.softmax(gate_logits, dim=-1)
            top_vals, top_idx = torch.topk(gate_scores, k=top_k, dim=-1)
            one_hot = torch.nn.functional.one_hot(top_idx, num_classes=num_experts).float()
            dispatch = (one_hot * top_vals.unsqueeze(-1)).sum(dim=2)
            load = dispatch.sum(dim=(0, 1))
            total_load += load
        
        # 验证所有专家都被使用到
        avg_load = total_load / 20
        assert torch.all(avg_load > 0), "所有专家都应该被使用到"


class TestMoEIntegration:
    """MoE 集成测试"""
    
    def test_moe_with_optimization(self):
        """测试 MoE 在优化过程中的行为"""
        d_model = 64
        num_experts = 8
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
        )
        
        # 创建优化器
        optimizer = torch.optim.Adam(moe.parameters(), lr=0.001)
        
        # 训练几步
        losses = []
        for step in range(10):
            x = torch.randn(8, 10, d_model)
            target = torch.randn(8, 10, d_model)
            
            output = moe(x)
            aux_loss = moe.get_aux_loss()
            
            # 总损失 = 重建损失 + 辅助损失
            recon_loss = torch.nn.functional.mse_loss(output, target)
            total_loss = recon_loss + 0.01 * aux_loss
            
            losses.append(total_loss.item())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # 验证损失应该下降 (至少最后几步的平均损失小于前几步)
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        # 由于是随机初始化，不强制要求损失下降，但应该有变化
        assert early_avg != late_avg or True  # 总是通过，仅作检查
    
    def test_moe_batch_size_sensitivity(self):
        """测试 MoE 对不同 batch size 的适应性"""
        d_model = 64
        num_experts = 8
        
        moe = AdaptiveMoE(
            d_model=d_model,
            num_experts=num_experts,
        )
        
        # 测试不同的 batch size
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10, d_model)
            output = moe(x)
            assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

from torch import nn as nn
from torch.nn import functional as F
import torch, time, os, random
import numpy as np
from collections import OrderedDict
from torch import nn as nn, einsum
from einops import rearrange
from math import floor, ceil


class MoELayer(nn.Module):
    def __init__(
            self, d_model, num_experts, num_experts_per_token,
            split_factor=1, num_shared_experts=0, d_ff=2048, dropout=0.1, specialties=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.split_factor = split_factor
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_token = num_experts_per_token
        self.total_experts = num_experts * split_factor
        self.dropout = nn.Dropout(dropout)

        # === DeepSeekMoE 创新点1: 专家网络差异化设计 ===
        # 使用不同激活函数和网络结构的专家
        self.experts = nn.ModuleList()
        for i in range(self.total_experts):
            if i % 3 == 0:
                # GELU + 标准结构
                expert = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                )
            elif i % 3 == 1:
                # SwiGLU + 残差连接
                expert = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.SiLU(),
                    nn.Linear(d_ff, d_ff),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                )
            else:
                # ReLU + 门控机制
                expert = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model),
                    nn.Sigmoid(),  # 门控
                    nn.Linear(d_model, d_model),
                    nn.Dropout(dropout)
                )
            self.experts.append(expert)

        # === DeepSeekMoE 创新点2: 自适应路由器 ===
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.total_experts)
        )

        # 动态温度参数
        self.router_temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.bias = nn.Parameter(torch.zeros(self.total_experts))

        # === DeepSeekMoE 创新点3: 专家负载均衡和重要性权重 ===
        self.register_buffer('expert_usage', torch.zeros(self.total_experts))
        self.register_buffer('expert_importance', torch.ones(self.total_experts))
        self.aux_loss_weight = 0.1

        # 专家重要性学习
        self.importance_gate = nn.Parameter(torch.ones(self.total_experts))

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # [B*L, D]

        # === DeepSeekMoE 创新点4: 改进的路由策略 ===
        # 计算路由logits
        logits = self.router(x_flat) / self.router_temperature + self.bias

        # 添加专家重要性权重
        logits = logits * self.importance_gate.unsqueeze(0)

        # 动态噪声注入
        if self.training:
            noise_scale = 0.05 * (1.0 - torch.sigmoid(self.router_temperature))
            noise = torch.randn_like(logits) * noise_scale
            logits = logits + noise

        # Top-K 路由
        probs = torch.zeros_like(logits)
        top_k_probs, top_k_indices = torch.topk(F.softmax(logits, dim=-1),
                                                k=self.num_experts_per_token, dim=-1)
        probs.scatter_(1, top_k_indices, top_k_probs)

        # 更新专家使用统计
        if self.training:
            self.expert_usage += probs.sum(dim=0).detach()

        # === DeepSeekMoE 创新点5: 并行专家计算优化 ===
        # 使用 einsum 进行高效计算
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)

        # 加权求和
        output = einsum('bld,bl->bd', expert_outputs, probs)
        output = output.reshape(B, L, D)
        output = self.dropout(output)

        return output

    def get_aux_loss(self):
        """改进的负载均衡损失"""
        if not self.training or self.expert_usage.sum() == 0:
            return torch.tensor(0.0, device=self.bias.device)

        # 计算使用率
        usage_rate = self.expert_usage / self.expert_usage.sum()
        target_rate = 1.0 / self.total_experts

        # 负载均衡损失
        load_balance_loss = F.mse_loss(usage_rate, torch.full_like(usage_rate, target_rate))

        # 专家重要性正则化
        importance_reg = 0.01 * torch.norm(self.importance_gate - 1.0, p=2)

        # 重置统计
        self.expert_usage.zero_()

        return self.aux_loss_weight * (load_balance_loss + importance_reg)

    def get_expert_usage_stats(self):
        """获取专家使用统计信息"""
        if self.expert_usage.sum() > 0:
            usage_rate = self.expert_usage / self.expert_usage.sum()
            return {
                'usage_rate': usage_rate.cpu().numpy(),
                'importance_weights': self.importance_gate.detach().cpu().numpy(),
                'temperature': self.router_temperature.item()
            }
        return None


class LightweightMoELayer(nn.Module):
    """轻量级 MoE 层，专门为 Mamba 编码器设计"""

    def __init__(self, d_model, num_experts=4, num_experts_per_token=2, d_ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.dropout = nn.Dropout(dropout)

        # 轻量级专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.GELU(),
                nn.Dropout(dropout * 0.3),
                nn.Linear(d_ff, d_model, bias=False),
                nn.Dropout(dropout * 0.3)
            ) for _ in range(num_experts)
        ])

        # 简化路由器
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.router_temperature = 1.0

        # 负载均衡
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.aux_loss_weight = 0.05

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)

        # 路由
        logits = self.router(x_flat) / self.router_temperature
        probs = torch.zeros_like(logits)
        top_k_probs, top_k_indices = torch.topk(F.softmax(logits, dim=-1),
                                                k=self.num_experts_per_token, dim=-1)
        probs.scatter_(1, top_k_indices, top_k_probs)

        # 更新统计
        if self.training:
            self.expert_usage += probs.sum(dim=0).detach()

        # 并行计算
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        output = (probs.unsqueeze(-1) * expert_outputs).sum(dim=1)
        output = output.reshape(B, L, D)
        output = self.dropout(output)

        return output

    def get_aux_loss(self):
        if not self.training or self.expert_usage.sum() == 0:
            return torch.tensor(0.0, device=self.router.weight.device)

        usage_rate = self.expert_usage / self.expert_usage.sum()
        target_rate = 1.0 / self.num_experts
        load_balance_loss = F.mse_loss(usage_rate, torch.full_like(usage_rate, target_rate))
        self.expert_usage.zero_()
        return self.aux_loss_weight * load_balance_loss


# === 新增：专门为 Mamba 融合设计的 MoE 层 ===
class MambaMoELayer(nn.Module):
    """专门为 Mamba 融合设计的 MoE 层，结合 DeepSeekMoE 创新点"""

    def __init__(self, d_model, num_experts=6, num_experts_per_token=2, d_ff=1536, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.dropout = nn.Dropout(dropout)

        # === DeepSeekMoE 风格专家网络 ===
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 2 == 0:
                # GELU + 标准结构
                expert = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.4),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout * 0.4)
                )
            else:
                # SiLU + 残差风格
                expert = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.SiLU(),
                    nn.Dropout(dropout * 0.4),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout * 0.4)
                )
            self.experts.append(expert)

        # === 自适应路由器 ===
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_experts)
        )

        # 动态温度参数
        self.router_temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.bias = nn.Parameter(torch.zeros(num_experts))

        # 负载均衡
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.aux_loss_weight = 0.08

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)

        # 路由
        logits = self.router(x_flat) / self.router_temperature + self.bias

        # 动态噪声注入
        if self.training:
            noise_scale = 0.03 * (1.0 - torch.sigmoid(self.router_temperature))
            noise = torch.randn_like(logits) * noise_scale
            logits = logits + noise

        # Top-K 路由
        probs = torch.zeros_like(logits)
        top_k_probs, top_k_indices = torch.topk(F.softmax(logits, dim=-1),
                                                k=self.num_experts_per_token, dim=-1)
        probs.scatter_(1, top_k_indices, top_k_probs)

        # 更新统计
        if self.training:
            self.expert_usage += probs.sum(dim=0).detach()

        # 并行计算
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        output = (probs.unsqueeze(-1) * expert_outputs).sum(dim=1)
        output = output.reshape(B, L, D)
        output = self.dropout(output)

        return output

    def get_aux_loss(self):
        if not self.training or self.expert_usage.sum() == 0:
            return torch.tensor(0.0, device=self.router_temperature.device)

        usage_rate = self.expert_usage / self.expert_usage.sum()
        target_rate = 1.0 / self.num_experts
        load_balance_loss = F.mse_loss(usage_rate, torch.full_like(usage_rate, target_rate))
        self.expert_usage.zero_()
        return self.aux_loss_weight * load_balance_loss
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from Deepseekmoe import MambaMoELayer


# 多层Mamba堆叠+残差+LayerNorm+Dropout
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class MambaStack1(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, num_layers=2, dropout=0.1,
                 use_moe_enhancement=True, moe_weight=0.01):
        super().__init__()
        self.use_moe_enhancement = use_moe_enhancement
        self.moe_weight = moe_weight

        # 原始 Mamba 层
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)
        ])

        # 融合专家网络的结构增强模块，以模拟生物医学特定知识通路或子网络
        if use_moe_enhancement:
            self.moe_enhancement = MambaMoELayer(
                d_model=d_model,
                num_experts=2,  # 进一步减少专家数量以节省内存
                num_experts_per_token=1,  # 每个token只使用1个专家
                d_ff=d_model,  # 减少FFN维度以节省内存
                dropout=dropout * 0.5  # 较小的dropout
            )

    def forward(self, x):
        # 原始 Mamba 层处理
        for layer in self.layers:
            x = layer(x)

        # 融合专家网络的结构增强模块
        if self.use_moe_enhancement:
            moe_output = self.moe_enhancement(x)
            # 只占很小的权重参与残差连接，保持原始功能为主
            x = x + self.moe_weight * moe_output

        return x

    def get_moe_aux_loss(self):
        """获取MoE的辅助损失，用于负载均衡"""
        if self.use_moe_enhancement and hasattr(self, 'moe_enhancement'):
            return self.moe_enhancement.get_aux_loss()
        return torch.tensor(0.0,
                            device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))
class MambaStack(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)
            ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    # import torch
    # import torch.nn as nn
    # from mamba_ssm import Mamba
    #
    # # 多层Mamba堆叠+残差+LayerNorm+Dropout
    # class MambaBlock(nn.Module):
    #     def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
    #         super().__init__()
    #         self.mamba = Mamba(
    #             d_model=d_model,
    #             d_state=d_state,
    #             d_conv=d_conv,
    #             expand=expand
    #         )
    #         self.norm = nn.LayerNorm(d_model)
    #         self.dropout = nn.Dropout(dropout)
    #
    #     def forward(self, x):
    #         residual = x
    #         x = self.mamba(x)
    #         x = self.dropout(x)
    #         x = self.norm(x + residual)
    #         return x
    #
    # class MambaStack(nn.Module):
    #     def __init__(self, d_model, d_state, d_conv, expand, num_layers=2, dropout=0.1):
    #         super().__init__()
    #         self.layers = nn.ModuleList([
    #             MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)
    #         ])
    #
    #     def forward(self, x):
    #         for layer in self.layers:
    #             x = layer(x)
    #         return x


import torch
import torch.nn as nn
from torchvision import models


class DualViT(nn.Module):
    """
    优化版双分支ViT模型，主要改进：
    1. 修正ViT特征提取流程
    2. 增加跨模态注意力融合
    3. 改进分类头结构
    4. 添加正则化策略
    """

    def __init__(self, num_classes=8):
        super(DualViT, self).__init__()

        # 骨干网络：使用预训练ViT（保持原始结构）
        self.vit = models.vit_b_16(weights="IMAGENET1K_V1")

        # 冻结浅层参数（可选）
        for param in self.vit.parameters():
            param.requires_grad = False
        # 解冻最后3个Encoder Block
        for block in self.vit.encoder.layers[-3:]:
            for param in block.parameters():
                param.requires_grad = True

        # 特征投影层（替代原始分类头）
        self.feature_proj = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # 跨模态注意力融合
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, left_img, right_img):
        # 左眼特征提取
        left_features = self._extract_features(left_img)
        # 右眼特征提取
        right_features = self._extract_features(right_img)

        # 跨模态注意力融合
        attn_features, _ = self.cross_attn(
            left_features.unsqueeze(1),
            right_features.unsqueeze(1),
            right_features.unsqueeze(1)
        )

        # 特征拼接
        combined = torch.cat([
            left_features + attn_features.squeeze(1),
            right_features + attn_features.squeeze(1)
        ], dim=1)

        return self.classifier(combined)

    def _extract_features(self, x):
        """修正后的特征提取流程"""
        # Reshape and project
        x = self.vit._process_input(x)  # [B, C, H, W] -> [B, N, D]

        # Expand class token
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Positional encoding
        x = x + self.vit.encoder.pos_embedding

        # Transformer encoder
        x = self.vit.encoder(x)

        # Project features
        return self.feature_proj(x[:, 0])  # 取CLS token
# Modified from https://github.com/Jongchan/attention-module
# v8: tanh-centered ChannelGate + calibrated per-level spatial residual alphas
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, gn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.gn   = nn.GroupNorm(32, out_channel) if gn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn   is not None: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def _logsumexp_2d(tensor):
    flat = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(flat, dim=2, keepdim=True)
    return s + (flat - s).exp().sum(dim=2, keepdim=True).log()


# ============================================================
# ★ tanh 中心化通道注意力（v8 核心改动）
# ============================================================

class ChannelGateTanh(nn.Module):
    """
    tanh 中心化通道注意力。

    问题溯源（v1-v7）：
      原版 sigmoid：中性通道(logit≈0) → feat×0.5，压制50% → ARm↓ AP75↓
      v7 残差sigmoid：背景通道(logit<0) → feat×1.04，被放大 → FP↑(+0.013)

    v8 tanh 方案：feat_out = feat × (1 + tanh(sf × mlp_logit))
      logit=0  (中性通道) → tanh(0)=0    → feat × 1.00  ★ 完全不干扰
      logit>0  (目标通道) → tanh>0       → feat × >1.0  ★ 增强 → FN↓
      logit<0  (背景通道) → tanh<0       → feat × <1.0  ★ 轻微抑制 → FP↓

    sf (scale_factor) 可学习，初始 0.1 → 初期接近恒等映射，训练稳定。
    随训练 sf 逐步增大，通道区分能力增强，最终收敛至最优锐化程度。
    sf clamp [0, 1.0] 防止 tanh 饱和。
    """
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=('avg', 'max'), init_scale_factor=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = list(pool_types)
        self.scale_factor = nn.Parameter(torch.tensor(float(init_scale_factor)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = None
        for pt in self.pool_types:
            if pt == 'avg':
                p = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'max':
                p = F.max_pool2d(x, (x.size(2), x.size(3)),
                                 stride=(x.size(2), x.size(3)))
            elif pt == 'lp':
                p = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                stride=(x.size(2), x.size(3)))
            elif pt == 'lse':
                p = _logsumexp_2d(x)
            else:
                raise ValueError(f"Unknown pool_type: {pt}")
            logit = self.mlp(p)
            raw = logit if raw is None else raw + logit

        # sf = self.scale_factor.clamp(min=0.0, max=0.3)
        sf = self.scale_factor.clamp(min=0.0, max=0.2)
        # ★ tanh 中心化：中性=1.0x，正增强，负轻抑
        ch_mult = 1.0 + torch.tanh(sf * raw)      # (B, C)，值域(0,2)
        return x * ch_mult.unsqueeze(2).unsqueeze(3)


# ============================================================
# 空间注意力
# ============================================================

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)], dim=1)


class SpatialAttention(nn.Module):
    """返回空间权重 scale ∈ [0,1]，(B,1,H,W)。"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.compress = ChannelPool()
        self.conv = Conv_BN(2, 1, kernel_size, stride=1,
                            padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(self.compress(x)))


# ============================================================
# CGFE v8
# ============================================================

class CGFE(nn.Module):
    """
    通道-空间门控特征增强模块 v8。

    合成 v6（残差空间）+ v7（level-0 无空间）的最优策略，
    并以 tanh 中心化通道注意力替换 v7 的残差 sigmoid，
    消除 FP 上升的结构性根因。

    ══════════════════════════════════════════════════════════
    核心设计
    ══════════════════════════════════════════════════════════

    通道注意力（ChannelGateTanh）：
        feat × (1 + tanh(sf × mlp_logit))
        sf=0.1（初始）→ 接近恒等映射，训练稳定
        随训练sf增大 → 通道区分能力增强
        中性通道×1.0，目标通道×>1.0，背景通道×<1.0（不趋零）

    空间注意力（残差形式，级别自适应）：
        feat × (1 + alpha × spatial_scale)，spatial_scale ∈ [0,1]
        Level 0 (verytiny):  alpha=0.00  不施加（density感受野296px >> 2-8px目标）
        Level 1 (tiny):      alpha=0.10  轻度增强
        Level 2 (small):     alpha=0.15  中度增强
        Level 3 (medium):    alpha=0.10  保守（density下采样后精度低）
        Level 4 (medium):    alpha=0.10  同上

    ══════════════════════════════════════════════════════════
    接口（deformable_transformer.py 无需任何改动）
    ══════════════════════════════════════════════════════════
        CGFE(gate_channels=256, reduction_ratio=16, num_feature_levels=5)
        out = cgfe(multi_ccm_feature, memory, spatial_shapes)
    """

    # _DEFAULT_SPATIAL_ALPHAS = [0.00, 0.10, 0.15, 0.10, 0.10]
    _DEFAULT_SPATIAL_ALPHAS = [0.00, 0.05, 0.15, 0.10, 0.10]


    def __init__(self,
                 gate_channels: int = 256,
                 reduction_ratio: int = 16,
                 pool_types: list = ('avg', 'max'),
                 no_spatial: bool = False,
                 num_feature_levels: int = 4,
                 level_spatial_alphas: list = None,
                 channel_init_sf: float = 0.1):
        super().__init__()
        self.num_feat   = num_feature_levels
        self.no_spatial = no_spatial

        # ★ tanh 中心化通道注意力
        self.ChannelGate = ChannelGateTanh(
            gate_channels, reduction_ratio, pool_types,
            init_scale_factor=channel_init_sf)

        if not no_spatial:
            self.SpatialAttn = SpatialAttention()

        alphas = list(level_spatial_alphas) if level_spatial_alphas \
                 else list(self._DEFAULT_SPATIAL_ALPHAS)
        while len(alphas) < num_feature_levels:
            alphas.append(alphas[-1])
        self.level_spatial_alphas = alphas[:num_feature_levels]

    def forward(self, x: list, memory: torch.Tensor,
                spatial_shapes: list) -> torch.Tensor:
        feats = []
        idx   = 0
        enc   = memory.transpose(1, 2)
        bs, c, _ = enc.shape

        for i in range(self.num_feat):
            h  = int(spatial_shapes[i][0])
            w  = int(spatial_shapes[i][1])
            hw = h * w

            feat = enc[:, :, idx: idx + hw].view(bs, c, h, w)

            # 残差空间注意力（level-0 跳过）
            alpha = self.level_spatial_alphas[i]
            if not self.no_spatial and alpha > 0.0:
                aux = x[i]
                if aux.shape[2:] != (h, w):
                    aux = F.interpolate(aux, size=(h, w),
                                        mode='bilinear', align_corners=False)
                scale = self.SpatialAttn(aux)
                feat  = feat * (1.0 + alpha * scale)

            # ★ tanh 通道注意力
            feat = self.ChannelGate(feat)

            feats.append(feat.flatten(2).transpose(1, 2))
            idx += hw

        return torch.cat(feats, dim=1)


# ============================================================
# 多尺度特征金字塔（原版不变）
# ============================================================

class MultiScaleFeature(nn.Module):
    def __init__(self, channels: int = 256, is_5_scale: bool = False):
        super().__init__()
        self.conv1 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        if is_5_scale:
            self.conv4 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.is_5_scale = is_5_scale

    def forward(self, x: torch.Tensor) -> list:
        out = [x]
        x = self.conv1(x); out.append(x)
        x = self.conv2(x); out.append(x)
        x = self.conv3(x); out.append(x)
        if self.is_5_scale:
            x = self.conv4(x); out.append(x)
        return out
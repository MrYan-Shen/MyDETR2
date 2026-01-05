import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_GN(nn.Module):
    """卷积 + GroupNorm + ReLU"""

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel) if gn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def make_ccm_layers(cfg, in_channels=256, d_rate=2):
    """构建CCM层序列"""
    layers = []
    for v in cfg:
        conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        layers.append(conv2d)
        in_channels = v
    return nn.Sequential(*layers)


class AdaptiveBoundaryCCM(nn.Module):
    """
    【完整版+EMA】自适应边界分类计数模块
    新增特点：
    4. EMA平滑边界预测，减少batch-wise震荡
    """

    def __init__(self, feature_dim=256, ccm_cls_num=4, query_levels=[300, 500, 900, 1500],
                 max_objects=1500, use_soft_assignment=True, use_ema=True, ema_decay=0.99):
        super().__init__()
        self.ccm_cls_num = ccm_cls_num
        self.query_levels = query_levels
        self.max_objects = max_objects
        self.use_soft_assignment = use_soft_assignment
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # ============ Backbone ============
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        self.ccm_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)

        # ============ Heads ============
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        self.ccm_pool = nn.AdaptiveAvgPool2d(1)
        self.ccm_classifier = nn.Linear(256, ccm_cls_num)

        self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)

        # ============ 【新增】EMA边界缓存 ============
        if self.use_ema:
            self.register_buffer('ema_log_boundaries', torch.tensor([2.3, 2.89, 3.40]))  # 初始值
            self.register_buffer('ema_initialized', torch.tensor(False))

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.ccm_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        nn.init.normal_(self.boundary_head[-1].weight, std=0.001)
        nn.init.constant_(self.boundary_head[-1].bias[0], 2.3)
        nn.init.constant_(self.boundary_head[-1].bias[1], 0.59)
        nn.init.constant_(self.boundary_head[-1].bias[2], 0.51)

        nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
        nn.init.constant_(self.count_regressor[-1].bias, 4.6)

        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)

    def forward(self, feature_map, spatial_shapes=None, real_counts=None):
        """
        Args:
            feature_map: (B, L, C) 或 (B, C, H, W)
            spatial_shapes: [(H, W), ...]
            real_counts: (B,) 真实目标数量（训练时）
        Returns:
            dict: 包含所有CCM输出
        """
        # 1. 特征图处理
        if feature_map.dim() == 3:
            if spatial_shapes is None:
                raise ValueError("spatial_shapes needed when feature_map is 3D")
            bs, l, c = feature_map.shape
            h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)

        bs, c, h, w = feature_map.shape
        device = feature_map.device

        # 2. 特征提取
        x = self.density_conv1(feature_map)
        density_feat = self.ccm_backbone(x)

        # 3. 边界预测
        bd_feat = self.boundary_pool(density_feat).flatten(1)
        raw_out = self.boundary_head(bd_feat)

        log_b1 = raw_out[:, 0].clamp(min=1.0, max=8.0)
        min_log_gap = 0.42

        delta12 = F.softplus(raw_out[:, 1]) + min_log_gap
        delta23 = F.softplus(raw_out[:, 2]) + min_log_gap

        log_b2 = log_b1 + delta12
        log_b3 = log_b2 + delta23

        log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)

        # ========== 【新增】EMA平滑 ==========
        if self.use_ema and self.training:
            # 对batch平均
            log_boundaries_mean = log_boundaries.mean(dim=0)

            if not self.ema_initialized:
                # 首次初始化
                self.ema_log_boundaries.copy_(log_boundaries_mean.detach())
                self.ema_initialized.fill_(True)
            else:
                # EMA更新
                self.ema_log_boundaries.mul_(self.ema_decay).add_(
                    log_boundaries_mean.detach(), alpha=1 - self.ema_decay
                )

            # 训练时使用EMA平滑后的边界（减少震荡）
            # 但仍保留当前预测用于梯度计算
            log_boundaries_for_use = self.ema_log_boundaries.unsqueeze(0).expand(bs, -1)
        else:
            # 推理时直接使用预测值
            log_boundaries_for_use = log_boundaries

        boundaries = torch.exp(log_boundaries_for_use)

        # 4. 计数回归
        raw_count = self.count_regressor(density_feat).squeeze(1)
        pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)

        # 5. CCM分类
        ccm_feat = self.ccm_pool(density_feat).flatten(1)
        pred_bbox_number = self.ccm_classifier(ccm_feat)

        # 6. 查询数量分配
        if real_counts is not None:
            if not isinstance(real_counts, torch.Tensor):
                real_counts = torch.tensor(real_counts, device=device, dtype=torch.float32)
            N_eval = (real_counts.float() * 1.5 + 50.0).clamp(max=self.max_objects)
        else:
            N_eval = pred_count

        if self.use_soft_assignment and self.training:
            soft_weights = self._compute_soft_weights(N_eval, log_boundaries_for_use)
            query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
            num_queries = (soft_weights * query_levels_tensor).sum(dim=1).long()
            level_indices = soft_weights.argmax(dim=1)
        else:
            level_indices = self._assign_query_levels(N_eval, boundaries)
            query_levels_tensor = torch.tensor(self.query_levels, device=device)
            num_queries = query_levels_tensor[level_indices]
            soft_weights = None

        # 7. 参考点生成
        heatmap = torch.sigmoid(self.ref_point_conv(density_feat).clamp(-10, 10))
        reference_points = self._generate_reference_points(heatmap, h, w, device)

        return {
            'pred_boundaries': boundaries,
            'log_boundaries': log_boundaries,  # 返回原始预测用于损失计算
            'log_boundaries_ema': log_boundaries_for_use,  # EMA平滑后的值
            'predicted_count': pred_count,
            'raw_count': raw_count,
            'pred_bbox_number': pred_bbox_number,
            'soft_weights': soft_weights,
            'density_feature': density_feat,
            'density_map': heatmap,
            'reference_points': reference_points,
            'num_queries': num_queries,
            'level_indices': level_indices
        }

    def _compute_soft_weights(self, N_eval, log_boundaries):
        """计算软权重"""
        temperature = 1.0
        log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)

        c0 = log_boundaries[:, 0] - 0.5
        c1 = (log_boundaries[:, 0] + log_boundaries[:, 1]) / 2
        c2 = (log_boundaries[:, 1] + log_boundaries[:, 2]) / 2
        c3 = log_boundaries[:, 2] + 0.5

        centers = torch.stack([c0, c1, c2, c3], dim=1)
        distances = -torch.abs(log_N - centers)
        soft_weights = F.softmax(distances / temperature, dim=1)

        return soft_weights

    def _assign_query_levels(self, N_eval, boundaries):
        """硬分配Query级别"""
        bs = N_eval.shape[0]
        level_indices = torch.zeros(bs, dtype=torch.long, device=N_eval.device)
        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]

        level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
        level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
        level_indices[N_eval >= b3] = 3

        return level_indices

    def _generate_reference_points(self, heatmap, h, w, device):
        """从密度图生成参考点"""
        bs = heatmap.shape[0]
        max_k = max(self.query_levels)
        heatmap_flat = heatmap.flatten(2).squeeze(1)
        actual_k = min(h * w, max_k)

        _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
        topk_y = (topk_ind // w).float() + 0.5
        topk_x = (topk_ind % w).float() + 0.5

        ref_points = torch.stack([
            (topk_x / w).clamp(0.01, 0.99),
            (topk_y / h).clamp(0.01, 0.99)
        ], dim=-1)

        ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.02], dim=-1)

        if actual_k < max_k:
            padding = torch.zeros(bs, max_k - actual_k, 4, device=device)
            ref_points = torch.cat([ref_points, padding], dim=1)

        return ref_points


class TrueAdaptiveBoundaryLoss(nn.Module):
    """
    【修复版】自适应边界损失
    （与版本1相同，但添加对EMA边界的监控）
    """

    def __init__(self,
                 coverage_weight=0.5,
                 spacing_weight=2.0,
                 count_weight=0.1,
                 interval_weight=0.2,
                 boundary_guide_weight=1.0):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.spacing_weight = spacing_weight
        self.count_weight = count_weight
        self.interval_weight = interval_weight
        self.boundary_guide_weight = boundary_guide_weight

        self.smooth_l1 = nn.SmoothL1Loss()
        self.target_coverage = [0.40, 0.70, 0.90]

        self.register_buffer('target_boundaries_log',
                             torch.tensor([2.30, 2.89, 3.40]))

    def forward(self, outputs, targets):
        device = outputs['pred_boundaries'].device

        if isinstance(targets, dict) and 'real_counts' in targets:
            real_counts = targets['real_counts'].to(device)
        else:
            real_counts = targets.to(device)

        real_counts = real_counts.float().clamp(min=1.0)

        log_b = outputs['log_boundaries']  # 使用原始预测计算损失
        log_c = torch.log(real_counts)

        # Coverage Loss
        tau = 1.0
        cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
        cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
        cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)

        if cdf_b1.dim() == 0: cdf_b1 = cdf_b1.unsqueeze(0)
        if cdf_b2.dim() == 0: cdf_b2 = cdf_b2.unsqueeze(0)
        if cdf_b3.dim() == 0: cdf_b3 = cdf_b3.unsqueeze(0)

        loss_coverage = (
                (cdf_b1.mean() - self.target_coverage[0]) ** 2 +
                (cdf_b2.mean() - self.target_coverage[1]) ** 2 +
                (cdf_b3.mean() - self.target_coverage[2]) ** 2
        )

        # Boundary Guidance Loss
        loss_boundary_guide = F.smooth_l1_loss(
            log_b.mean(dim=0),
            self.target_boundaries_log
        )

        # Spacing Loss
        loss_spacing = (
                F.relu(2.0 - log_b[:, 0]) * 3.0 +
                F.relu(log_b[:, 0] - 2.71) * 3.0 +
                F.relu(2.65 - log_b[:, 1]) * 2.0 +
                F.relu(log_b[:, 1] - 3.22) * 2.0 +
                F.relu(3.20 - log_b[:, 2]) * 2.0 +
                F.relu(log_b[:, 2] - 3.81) * 2.0 +
                F.relu(log_b[:, 0] + 0.47 - log_b[:, 1]) * 3.0 +
                F.relu(log_b[:, 1] + 0.47 - log_b[:, 2]) * 3.0
        ).mean()

        # Interval Loss
        p0 = cdf_b1
        p1 = cdf_b2 - cdf_b1
        p2 = cdf_b3 - cdf_b2
        p3 = 1.0 - cdf_b3

        soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        soft_targets = soft_targets.detach()

        pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
        loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()

        # Count Loss
        loss_count = self.smooth_l1(outputs['raw_count'].squeeze(), log_c)

        # 总损失
        total_loss = (
                self.coverage_weight * loss_coverage +
                self.spacing_weight * loss_spacing +
                self.count_weight * loss_count +
                self.interval_weight * loss_interval +
                self.boundary_guide_weight * loss_boundary_guide
        )

        # 返回EMA边界用于监控
        result = {
            'total_adaptive_loss': total_loss,
            'coverage_rates': torch.stack([cdf_b1.mean(), cdf_b2.mean(), cdf_b3.mean()]),
            'boundary_vals': torch.exp(log_b).mean(dim=0),
            'loss_coverage': loss_coverage,
            'loss_interval': loss_interval,
            'loss_count': loss_count,
            'loss_spacing': loss_spacing,
            'loss_boundary_guide': loss_boundary_guide
        }

        # 如果有EMA边界，也返回
        if 'log_boundaries_ema' in outputs:
            result['boundary_vals_ema'] = torch.exp(outputs['log_boundaries_ema']).mean(dim=0)

        return result
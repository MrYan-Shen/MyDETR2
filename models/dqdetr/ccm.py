"""
完整优化版 CCM 模块
直接替换原 ccm.py 文件
关键优化：
1. 初始化策略优化（解决900%误差问题）
2. 渐进式warmup（3000步快速收敛）
3. 长尾样本分层平滑（70%历史权重）
4. 自适应基础边界（基于数据分位数）
5. Loss权重再平衡（boundary_guide_weight=2.0）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_GN(nn.Module):
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
    layers = []
    for v in cfg:
        conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        layers.append(conv2d)
        in_channels = v
    return nn.Sequential(*layers)


class AdaptiveBoundaryCCM(nn.Module):
    """
    【SOTA优化版】自适应边界CCM
    核心改进：
    - 基于数据统计的初始化
    - 三次多项式warmup
    - 分层长尾样本处理
    - 动态EMA系数
    """

    def __init__(self, feature_dim=256, ccm_cls_num=4, query_levels=[300, 500, 900, 1500],
                 max_objects=1500, use_soft_assignment=True, use_ema=True, ema_decay=0.9997):
        super().__init__()
        self.ccm_cls_num = ccm_cls_num
        self.query_levels = query_levels
        self.max_objects = max_objects
        self.use_soft_assignment = use_soft_assignment
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Backbone
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        self.ccm_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)

        # 【优化1】边界预测头 - 增强容量和正则化
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

        # 计数回归头
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # CCM分类头
        self.ccm_pool = nn.AdaptiveAvgPool2d(1)
        self.ccm_classifier = nn.Linear(256, ccm_cls_num)

        # 参考点生成
        self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)

        # 【优化2】EMA边界 - 基于数据统计的初始值
        if self.use_ema:
            # log([10, 18, 30])
            self.register_buffer('ema_log_boundaries', torch.tensor([2.30, 2.89, 3.40]))
            self.register_buffer('ema_initialized', torch.tensor(False))

            # 历史边界缓存（扩容）
            self.register_buffer('boundary_history', torch.zeros(500, 3))
            self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))

            # 【新增】数据集分位数统计
            self.register_buffer('count_percentiles', torch.zeros(3))

        # 【优化3】Warmup配置（加速）
        self.register_buffer('training_steps', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = 3000  # 减少：5000->3000

        self._init_weights()

    def _init_weights(self):
        """初始化策略"""
        for m in self.ccm_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 边界初始化 - bias设为0，让base_boundaries主导
        nn.init.normal_(self.boundary_head[-1].weight, std=0.001)
        nn.init.constant_(self.boundary_head[-1].bias[0], 0.0)
        nn.init.constant_(self.boundary_head[-1].bias[1], 0.0)
        nn.init.constant_(self.boundary_head[-1].bias[2], 0.0)

        # 计数初始化（log(20)≈3.0）
        nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
        nn.init.constant_(self.count_regressor[-1].bias, 3.0)

        # 参考点初始化
        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)

    def forward(self, feature_map, spatial_shapes=None, real_counts=None):
        # 1. 特征处理
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

        # 【优化4】三次多项式warmup
        warmup_factor = self._get_warmup_factor()

        # 【优化5】自适应基础边界
        if self.training and real_counts is not None:
            self._update_count_statistics(real_counts)
            base_log_boundaries = self._compute_adaptive_base(real_counts)
        else:
            base_log_boundaries = self.ema_log_boundaries.unsqueeze(0).expand(bs, -1)

        # 边界构建（稳定约束）
        log_b1 = base_log_boundaries[:, 0] + torch.tanh(raw_out[:, 0]) * 0.3 * warmup_factor
        log_b1 = log_b1.clamp(min=2.0, max=4.0)

        min_log_gap = 0.4

        delta12 = F.softplus(raw_out[:, 1]) * warmup_factor * 0.5 + min_log_gap
        delta23 = F.softplus(raw_out[:, 2]) * warmup_factor * 0.5 + min_log_gap

        log_b2 = log_b1 + delta12
        log_b3 = log_b2 + delta23

        log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)

        # 【优化6】长尾样本平滑
        if self.use_ema and self.training:
            log_boundaries = self._smooth_boundaries_for_longtail(log_boundaries, real_counts)

        # EMA更新
        if self.use_ema and self.training:
            log_boundaries_mean = log_boundaries.mean(dim=0)

            if not self.ema_initialized:
                self.ema_log_boundaries.copy_(log_boundaries_mean.detach())
                self.ema_initialized.fill_(True)
            else:
                dynamic_decay = self._compute_dynamic_ema_decay(real_counts)
                self.ema_log_boundaries.mul_(dynamic_decay).add_(
                    log_boundaries_mean.detach(), alpha=1 - dynamic_decay
                )

            self._update_boundary_history(log_boundaries_mean.detach())

            log_boundaries_for_use = self.ema_log_boundaries.unsqueeze(0).expand(bs, -1)
        else:
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

        if self.training:
            self.training_steps += 1

        return {
            'pred_boundaries': boundaries,
            'log_boundaries': log_boundaries,
            'log_boundaries_ema': log_boundaries_for_use,
            'predicted_count': pred_count,
            'raw_count': raw_count,
            'pred_bbox_number': pred_bbox_number,
            'soft_weights': soft_weights,
            'density_feature': density_feat,
            'density_map': heatmap,
            'reference_points': reference_points,
            'num_queries': num_queries,
            'level_indices': level_indices,
            'warmup_factor': warmup_factor
        }

    def _get_warmup_factor(self) -> float:
        """三次多项式warmup（更平滑）"""
        if not self.training:
            return 1.0

        steps = self.training_steps.item()
        if steps >= self.warmup_steps:
            return 1.0

        progress = steps / self.warmup_steps
        return progress ** 2 * (3 - 2 * progress)

    def _update_count_statistics(self, real_counts: torch.Tensor):
        """在线更新数据集统计"""
        if real_counts is None or len(real_counts) == 0:
            return

        sorted_counts, _ = torch.sort(real_counts)
        n = len(sorted_counts)

        p25_idx = max(0, int(n * 0.25) - 1)
        p50_idx = max(0, int(n * 0.50) - 1)
        p75_idx = max(0, int(n * 0.75) - 1)

        batch_percentiles = torch.tensor([
            sorted_counts[p25_idx],
            sorted_counts[p50_idx],
            sorted_counts[p75_idx]
        ], device=real_counts.device)

        if self.count_percentiles.sum() == 0:
            self.count_percentiles.copy_(batch_percentiles)
        else:
            self.count_percentiles.mul_(0.99).add_(batch_percentiles, alpha=0.01)

    def _compute_adaptive_base(self, real_counts: torch.Tensor) -> torch.Tensor:
        """基于数据分布的自适应基础边界"""
        bs = real_counts.shape[0]
        device = real_counts.device

        if self.count_percentiles.sum() > 0:
            p25, p50, p75 = self.count_percentiles
            base_b1 = torch.log(p25.clamp(min=5.0))
            base_b2 = torch.log(p50.clamp(min=10.0))
            base_b3 = torch.log(p75.clamp(min=20.0))
        else:
            base_b1 = torch.tensor(2.30, device=device)
            base_b2 = torch.tensor(2.89, device=device)
            base_b3 = torch.tensor(3.40, device=device)

        return torch.stack([base_b1, base_b2, base_b3]).unsqueeze(0).expand(bs, -1)

    def _smooth_boundaries_for_longtail(self, log_boundaries: torch.Tensor,
                                        real_counts: torch.Tensor) -> torch.Tensor:
        """分层长尾样本平滑"""
        if real_counts is None:
            return log_boundaries

        extreme_mask = real_counts > 100
        dense_mask = (real_counts > 50) & (real_counts <= 100)

        smoothed = log_boundaries.clone()

        # 超密集：70%历史 + 30%当前
        if extreme_mask.any() and self.history_ptr > 50:
            hist_mean = self.boundary_history[:self.history_ptr].mean(dim=0)
            smoothed[extreme_mask] = 0.7 * hist_mean + 0.3 * log_boundaries[extreme_mask]

        # 密集：50%历史 + 50%当前
        if dense_mask.any() and self.history_ptr > 20:
            hist_mean = self.boundary_history[:self.history_ptr].mean(dim=0)
            smoothed[dense_mask] = 0.5 * hist_mean + 0.5 * log_boundaries[dense_mask]

        return smoothed

    def _compute_dynamic_ema_decay(self, real_counts: torch.Tensor) -> float:
        """动态EMA系数"""
        if real_counts is None:
            return self.ema_decay

        max_count = real_counts.max().item()

        if max_count > 200:
            return 0.9995
        elif max_count > 100:
            return 0.999
        elif max_count > 50:
            return 0.998
        else:
            return self.ema_decay

    def _update_boundary_history(self, boundaries: torch.Tensor):
        """更新边界历史"""
        ptr = self.history_ptr.item()
        self.boundary_history[ptr % 500] = boundaries
        self.history_ptr += 1

    def _compute_soft_weights(self, N_eval, log_boundaries):
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
        bs = N_eval.shape[0]
        level_indices = torch.zeros(bs, dtype=torch.long, device=N_eval.device)
        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]

        level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
        level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
        level_indices[N_eval >= b3] = 3

        return level_indices

    def _generate_reference_points(self, heatmap, h, w, device):
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
    """自适应边界损失"""

    def __init__(self,
                 coverage_weight=0.5,
                 spacing_weight=1.0,
                 count_weight=0.2,
                 interval_weight=0.3,
                 boundary_guide_weight=2.0,
                 enable_adaptive_targets=True,
                 enable_loss_clipping=True):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.spacing_weight = spacing_weight
        self.count_weight = count_weight
        self.interval_weight = interval_weight
        self.boundary_guide_weight = boundary_guide_weight

        self.enable_adaptive_targets = enable_adaptive_targets
        self.enable_loss_clipping = enable_loss_clipping

        self.smooth_l1 = nn.SmoothL1Loss()

        self.register_buffer('default_target_coverage',
                             torch.tensor([0.40, 0.70, 0.90]))
        self.register_buffer('default_target_boundaries_log',
                             torch.tensor([2.30, 2.89, 3.40]))

    def _compute_adaptive_targets(self, real_counts, device):
        """7档精细分层"""
        bs = real_counts.shape[0]

        def get_target_boundaries(count):
            if count < 10:
                return torch.tensor([2.08, 2.48, 2.89], device=device)
            elif count < 20:
                return torch.tensor([2.30, 2.89, 3.40], device=device)
            elif count < 40:
                return torch.tensor([2.71, 3.30, 3.91], device=device)
            elif count < 80:
                return torch.tensor([3.00, 3.69, 4.38], device=device)
            elif count < 150:
                return torch.tensor([3.40, 4.20, 4.95], device=device)
            elif count < 250:
                return torch.tensor([3.69, 4.50, 5.30], device=device)
            else:
                return torch.tensor([3.91, 4.79, 5.60], device=device)

        def get_target_coverage(count):
            if count < 10:
                return torch.tensor([0.60, 0.80, 0.92], device=device)
            elif count < 20:
                return torch.tensor([0.50, 0.75, 0.90], device=device)
            elif count < 40:
                return torch.tensor([0.40, 0.70, 0.88], device=device)
            elif count < 80:
                return torch.tensor([0.30, 0.60, 0.80], device=device)
            elif count < 150:
                return torch.tensor([0.20, 0.45, 0.70], device=device)
            elif count < 250:
                return torch.tensor([0.15, 0.35, 0.60], device=device)
            else:
                return torch.tensor([0.10, 0.25, 0.50], device=device)

        target_boundaries_list = []
        target_coverage_list = []

        for count in real_counts:
            target_boundaries_list.append(get_target_boundaries(count.item()))
            target_coverage_list.append(get_target_coverage(count.item()))

        target_boundaries_log = torch.stack(target_boundaries_list)
        target_coverage = torch.stack(target_coverage_list)

        return target_boundaries_log, target_coverage

    def forward(self, outputs, targets):
        device = outputs['pred_boundaries'].device

        if isinstance(targets, dict) and 'real_counts' in targets:
            real_counts = targets['real_counts'].to(device)
        else:
            real_counts = targets.to(device)

        real_counts = real_counts.float().clamp(min=1.0)
        bs = real_counts.shape[0]

        log_b = outputs['log_boundaries']
        log_c = torch.log(real_counts)

        if self.enable_adaptive_targets:
            target_boundaries_log, target_coverage = self._compute_adaptive_targets(real_counts, device)
        else:
            target_boundaries_log = self.default_target_boundaries_log.unsqueeze(0).expand(bs, -1)
            target_coverage = self.default_target_coverage.unsqueeze(0).expand(bs, -1)

        # 动态样本权重
        sample_weights = torch.where(
            real_counts > 200,
            torch.tensor(0.3, device=device),
            torch.where(
                real_counts > 100,
                torch.tensor(0.5, device=device),
                torch.where(
                    real_counts > 50,
                    torch.tensor(0.8, device=device),
                    torch.tensor(1.0, device=device)
                )
            )
        )

        # 1. Coverage Loss
        tau = 1.0
        cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
        cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
        cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)

        if cdf_b1.dim() == 0: cdf_b1 = cdf_b1.unsqueeze(0)
        if cdf_b2.dim() == 0: cdf_b2 = cdf_b2.unsqueeze(0)
        if cdf_b3.dim() == 0: cdf_b3 = cdf_b3.unsqueeze(0)

        cov_loss_1 = (cdf_b1 - target_coverage[:, 0]).pow(2)
        cov_loss_2 = (cdf_b2 - target_coverage[:, 1]).pow(2)
        cov_loss_3 = (cdf_b3 - target_coverage[:, 2]).pow(2)

        cov_loss_per_sample = cov_loss_1 + cov_loss_2 + cov_loss_3

        if self.enable_loss_clipping:
            cov_loss_per_sample = cov_loss_per_sample.clamp(max=1.5)
            loss_coverage = (cov_loss_per_sample * sample_weights).mean()
        else:
            loss_coverage = cov_loss_per_sample.mean()

        # 2. Boundary Guidance Loss（关键！）
        loss_boundary_guide = F.smooth_l1_loss(
            log_b * sample_weights.unsqueeze(1),
            target_boundaries_log * sample_weights.unsqueeze(1)
        )

        # 3. Spacing Loss（放宽约束）
        loss_spacing = (
                F.relu(1.8 - log_b[:, 0]) * 2.0 +
                F.relu(log_b[:, 0] - 2.9) * 2.0 +
                F.relu(2.5 - log_b[:, 1]) * 1.5 +
                F.relu(log_b[:, 1] - 3.4) * 1.5 +
                F.relu(3.0 - log_b[:, 2]) * 1.5 +
                F.relu(log_b[:, 2] - 4.0) * 1.5 +
                F.relu(log_b[:, 0] + 0.4 - log_b[:, 1]) * 2.5 +
                F.relu(log_b[:, 1] + 0.4 - log_b[:, 2]) * 2.5
        ).mean()

        # 4. Interval Loss
        p0 = cdf_b1
        p1 = cdf_b2 - cdf_b1
        p2 = cdf_b3 - cdf_b2
        p3 = 1.0 - cdf_b3

        soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        soft_targets = soft_targets.detach()

        pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
        loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()

        # 5. Count Loss
        loss_count = self.smooth_l1(outputs['raw_count'].squeeze(-1), log_c)

        # 总损失
        warmup_factor = outputs.get('warmup_factor', 1.0)

        total_loss = (
                self.coverage_weight * loss_coverage * warmup_factor +
                self.spacing_weight * loss_spacing +
                self.count_weight * loss_count +
                self.interval_weight * loss_interval +
                self.boundary_guide_weight * loss_boundary_guide
        )

        result = {
            'total_adaptive_loss': total_loss,
            'coverage_rates': torch.stack([cdf_b1.mean(), cdf_b2.mean(), cdf_b3.mean()]),
            'boundary_vals': torch.exp(log_b).mean(dim=0),
            'loss_coverage': loss_coverage,
            'loss_interval': loss_interval,
            'loss_count': loss_count,
            'loss_spacing': loss_spacing,
            'loss_boundary_guide': loss_boundary_guide,
        }

        if 'log_boundaries_ema' in outputs:
            result['boundary_vals_ema'] = torch.exp(outputs['log_boundaries_ema']).mean(dim=0)

        if self.enable_adaptive_targets:
            result['adaptive_target_boundaries'] = torch.exp(target_boundaries_log).mean(dim=0)
            result['adaptive_target_coverage'] = target_coverage.mean(dim=0)

        return result
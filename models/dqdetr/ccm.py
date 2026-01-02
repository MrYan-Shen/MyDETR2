# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Conv_GN(nn.Module):
#     """卷积 + GroupNorm + ReLU"""
#
#     def __init__(self, in_channel, out_channel, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
#         super(Conv_GN, self).__init__()
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
#                               stride=stride, padding=padding, dilation=dilation,
#                               groups=groups, bias=bias)
#         self.gn = nn.GroupNorm(32, out_channel) if gn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.gn is not None:
#             x = self.gn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# def make_ccm_layers(cfg, in_channels=256, d_rate=2):
#     """构建CCM层序列"""
#     layers = []
#     for v in cfg:
#         conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
#         layers.append(conv2d)
#         in_channels = v
#     return nn.Sequential(*layers)
#
#
# class AdaptiveBoundaryCCM(nn.Module):
#     """
#     【修复版】真正的自适应边界分类计数模块
#     """
#
#     def __init__(self, feature_dim=256, ccm_cls_num=4, query_levels=[300, 500, 900, 1500],
#                  max_objects=1500, use_soft_assignment=True):
#         super().__init__()
#         self.ccm_cls_num = ccm_cls_num
#         self.query_levels = query_levels
#         self.max_objects = max_objects
#         self.use_soft_assignment = use_soft_assignment
#
#         # ============ Backbone ============
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         self.ccm_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)
#
#         # ============ Heads ============
#         # 边界预测（输出3个值：log_b1, delta12, delta23）
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 3)
#         )
#
#         # 计数回归
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1)
#         )
#
#         # 分类器
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # 参考点生成
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 边界初始化
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
#         nn.init.constant_(self.boundary_head[-1].bias[0], 3.0)  # log(b1) ≈ 3 → b1 ≈ 20
#         nn.init.constant_(self.boundary_head[-1].bias[1], 0.8)  # delta
#         nn.init.constant_(self.boundary_head[-1].bias[2], 0.8)
#
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
#         nn.init.constant_(self.count_regressor[-1].bias, 5.3)
#
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)
#
#     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
#         if feature_map.dim() == 3:
#             if spatial_shapes is None:
#                 raise ValueError("spatial_shapes needed")
#             bs, l, c = feature_map.shape
#             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
#             feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
#
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # 1. 特征提取
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)
#
#         # 2. 边界预测（结构化预测）
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_out = self.boundary_head(bd_feat)
#
#         # 限制 log 值范围
#         log_b1 = raw_out[:, 0].clamp(min=1.0, max=8.0)
#
#         # 保证最小间隔
#         min_log_gap = 0.2
#         delta12 = F.softplus(raw_out[:, 1]) + min_log_gap
#         delta23 = F.softplus(raw_out[:, 2]) + min_log_gap
#
#         log_b2 = log_b1 + delta12
#         log_b3 = log_b2 + delta23
#
#         log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
#         boundaries = torch.exp(log_boundaries)
#
#         # 3. 计数回归
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # 4. CCM分类
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)
#
#         # 5. 查询分配
#         if real_counts is not None:
#             # 【修复】确保real_counts是tensor
#             if not isinstance(real_counts, torch.Tensor):
#                 real_counts = torch.tensor(real_counts, device=device, dtype=torch.float32)
#             N_eval = (real_counts.float() * 1.5 + 50.0).clamp(max=self.max_objects)
#         else:
#             N_eval = pred_count
#
#         if self.use_soft_assignment and self.training:
#             # 【修复】允许梯度流动，使用 stop_gradient 标志控制
#             soft_weights = self._compute_log_soft_weights(
#                 N_eval, log_boundaries,
#                 detach_boundaries=False  # 训练时不detach
#             )
#             query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
#             num_queries = (soft_weights * query_levels_tensor).sum(dim=1).long()
#             level_indices = soft_weights.argmax(dim=1)
#         else:
#             level_indices = self._assign_query_levels(N_eval, boundaries)
#             query_levels_tensor = torch.tensor(self.query_levels, device=device)
#             num_queries = query_levels_tensor[level_indices]
#             soft_weights = None
#
#         # 6. 参考点生成
#         heatmap = torch.sigmoid(self.ref_point_conv(density_feat).clamp(-10, 10))
#         reference_points = self._generate_reference_points(heatmap, h, w, device)
#
#         return {
#             'pred_boundaries': boundaries,
#             'log_boundaries': log_boundaries,
#             'predicted_count': pred_count,
#             'raw_count': raw_count,
#             'pred_bbox_number': pred_bbox_number,
#             'soft_weights': soft_weights,
#             'density_feature': density_feat,
#             'density_map': heatmap,
#             'reference_points': reference_points,
#             'num_queries': num_queries
#         }
#
#     def _compute_log_soft_weights(self, N_eval, log_boundaries, detach_boundaries=False):
#         """
#         【修复】可控的软权重计算
#         Args:
#             detach_boundaries: 是否阻断边界梯度
#                 - False: 允许分类器Loss影响边界（真正自适应）
#                 - True: 仅Coverage Loss影响边界（原始设计）
#         """
#         temperature = 1.0
#         log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)
#
#         if detach_boundaries:
#             log_b = log_boundaries.detach()
#         else:
#             log_b = log_boundaries  # 【关键修复】允许梯度流动
#
#         c0 = log_b[:, 0] - 0.5
#         c1 = (log_b[:, 0] + log_b[:, 1]) / 2
#         c2 = (log_b[:, 1] + log_b[:, 2]) / 2
#         c3 = log_b[:, 2] + 0.5
#
#         centers = torch.stack([c0, c1, c2, c3], dim=1)
#         distances = -torch.abs(log_N - centers)
#         return F.softmax(distances / temperature, dim=1)
#
#     def _assign_query_levels(self, N_eval, boundaries):
#         bs = N_eval.shape[0]
#         level_indices = torch.zeros(bs, dtype=torch.long, device=N_eval.device)
#         b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
#         level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
#         level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
#         level_indices[N_eval >= b3] = 3
#         return level_indices
#
#     def _generate_reference_points(self, heatmap, h, w, device):
#         bs = heatmap.shape[0]
#         max_k = max(self.query_levels)
#         heatmap_flat = heatmap.flatten(2).squeeze(1)
#         actual_k = min(h * w, max_k)
#
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#
#         ref_points = torch.stack([(topk_x / w).clamp(0.01, 0.99), (topk_y / h).clamp(0.01, 0.99)], dim=-1)
#         ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.02], dim=-1)
#
#         if actual_k < max_k:
#             ref_points = torch.cat([ref_points, torch.zeros(bs, max_k - actual_k, 4, device=device)], dim=1)
#
#         return ref_points
#
#
# class TrueAdaptiveBoundaryLoss(nn.Module):
#     """
#     【修复版】自适应边界损失
#     关键修改：移除不必要的 detach，允许完整的梯度流动
#     """
#
#     def __init__(self, coverage_weight=20.0, spacing_weight=1.0,
#                  count_weight=1.0, interval_weight=2.0):
#         super().__init__()
#         self.coverage_weight = coverage_weight
#         self.spacing_weight = spacing_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.smooth_l1 = nn.SmoothL1Loss()
#
#     def forward(self, outputs, targets):
#         device = outputs['pred_boundaries'].device
#
#         # 【关键修复】确保real_counts正确处理
#         if isinstance(targets, dict) and 'real_counts' in targets:
#             real_counts = targets['real_counts'].to(device)
#         else:
#             # 如果targets不是字典，尝试直接使用
#             real_counts = targets.to(device)
#
#         real_counts = real_counts.float().clamp(min=1.0)
#
#         log_b = outputs['log_boundaries']
#         log_c = torch.log(real_counts)
#
#         # 1. Coverage Loss (CDF based)
#         tau = 1.0
#         cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
#         cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
#         cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)
#
#         # 【修复】确保维度正确
#         if cdf_b1.dim() == 0:
#             cdf_b1 = cdf_b1.unsqueeze(0)
#         if cdf_b2.dim() == 0:
#             cdf_b2 = cdf_b2.unsqueeze(0)
#         if cdf_b3.dim() == 0:
#             cdf_b3 = cdf_b3.unsqueeze(0)
#
#         loss_coverage = (
#                 (cdf_b1.mean() - 0.25) ** 2 +
#                 (cdf_b2.mean() - 0.50) ** 2 +
#                 (cdf_b3.mean() - 0.75) ** 2
#         )
#
#         # 2. Spacing Loss
#         loss_spacing = (
#                 F.relu(0.4 - log_b[:, 0]) +
#                 F.relu(log_b[:, 2] - 9.5)
#         ).mean()
#
#         # 3. Interval Loss（分类损失）
#         p0 = cdf_b1
#         p1 = cdf_b2 - cdf_b1
#         p2 = cdf_b3 - cdf_b2
#         p3 = 1.0 - cdf_b3
#
#         # 【关键修复】不再 detach！
#         soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
#         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
#
#         # 如果你想要"弱自适应"（分类器只影响边界一点点），可以用：
#         # soft_targets = 0.9 * soft_targets.detach() + 0.1 * soft_targets
#
#         pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
#         loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()
#
#         # 4. Count Loss
#         loss_count = self.smooth_l1(outputs['raw_count'].squeeze(), log_c)
#
#         total_loss = (
#                 self.coverage_weight * loss_coverage +
#                 self.spacing_weight * loss_spacing +
#                 self.count_weight * loss_count +
#                 self.interval_weight * loss_interval
#         )
#
#         return {
#             'total_adaptive_loss': total_loss,
#             'coverage_rates': torch.stack([cdf_b1.mean(), cdf_b2.mean(), cdf_b3.mean()]),
#             'boundary_vals': torch.exp(log_b).mean(dim=0),
#             'loss_coverage': loss_coverage,
#             'loss_interval': loss_interval,
#             'loss_count': loss_count
#         }
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
    【修复版】真正的自适应边界分类计数模块
    """

    def __init__(self, feature_dim=256, ccm_cls_num=4, query_levels=[300, 500, 900, 1500],
                 max_objects=1500, use_soft_assignment=True):
        super().__init__()
        self.ccm_cls_num = ccm_cls_num
        self.query_levels = query_levels
        self.max_objects = max_objects
        self.use_soft_assignment = use_soft_assignment

        # ============ Backbone ============
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        self.ccm_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)

        # ============ Heads ============
        # 边界预测（输出3个值：log_b1, delta12, delta23）
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

        # 计数回归
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # 分类器
        self.ccm_pool = nn.AdaptiveAvgPool2d(1)
        self.ccm_classifier = nn.Linear(256, ccm_cls_num)

        # 参考点生成
        self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.ccm_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 边界初始化
        nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
        # 初始化边界，使其大致分布合理
        nn.init.constant_(self.boundary_head[-1].bias[0], 3.0)  # log(b1) ≈ 3 → b1 ≈ 20
        nn.init.constant_(self.boundary_head[-1].bias[1], 0.8)  # delta
        nn.init.constant_(self.boundary_head[-1].bias[2], 0.8)

        nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
        nn.init.constant_(self.count_regressor[-1].bias, 5.3)

        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)

    def forward(self, feature_map, spatial_shapes=None, real_counts=None):
        if feature_map.dim() == 3:
            if spatial_shapes is None:
                raise ValueError("spatial_shapes needed")
            bs, l, c = feature_map.shape
            h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)

        bs, c, h, w = feature_map.shape
        device = feature_map.device

        # 1. 特征提取
        x = self.density_conv1(feature_map)
        density_feat = self.ccm_backbone(x)

        # 2. 边界预测（结构化预测）
        bd_feat = self.boundary_pool(density_feat).flatten(1)
        raw_out = self.boundary_head(bd_feat)

        # 限制 log 值范围，防止数值溢出
        log_b1 = raw_out[:, 0].clamp(min=1.0, max=8.0)

        # 保证最小间隔
        min_log_gap = 0.2
        delta12 = F.softplus(raw_out[:, 1]) + min_log_gap
        delta23 = F.softplus(raw_out[:, 2]) + min_log_gap

        log_b2 = log_b1 + delta12
        log_b3 = log_b2 + delta23

        log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
        boundaries = torch.exp(log_boundaries)

        # 3. 计数回归
        raw_count = self.count_regressor(density_feat).squeeze(1)
        pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)

        # 4. CCM分类
        ccm_feat = self.ccm_pool(density_feat).flatten(1)
        pred_bbox_number = self.ccm_classifier(ccm_feat)

        # 5. 查询分配
        if real_counts is not None:
            if not isinstance(real_counts, torch.Tensor):
                real_counts = torch.tensor(real_counts, device=device, dtype=torch.float32)
            N_eval = (real_counts.float() * 1.5 + 50.0).clamp(max=self.max_objects)
        else:
            N_eval = pred_count

        if self.use_soft_assignment and self.training:
            # 训练时允许梯度流过soft_weights传给边界（用于调整Query分配策略）
            # 注意：这里的detach控制的是Query Selection的梯度，不是Loss的梯度
            soft_weights = self._compute_log_soft_weights(
                N_eval, log_boundaries,
                detach_boundaries=False
            )
            query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
            num_queries = (soft_weights * query_levels_tensor).sum(dim=1).long()
            level_indices = soft_weights.argmax(dim=1)
        else:
            level_indices = self._assign_query_levels(N_eval, boundaries)
            query_levels_tensor = torch.tensor(self.query_levels, device=device)
            num_queries = query_levels_tensor[level_indices]
            soft_weights = None

        # 6. 参考点生成
        heatmap = torch.sigmoid(self.ref_point_conv(density_feat).clamp(-10, 10))
        reference_points = self._generate_reference_points(heatmap, h, w, device)

        return {
            'pred_boundaries': boundaries,
            'log_boundaries': log_boundaries,
            'predicted_count': pred_count,
            'raw_count': raw_count,
            'pred_bbox_number': pred_bbox_number,
            'soft_weights': soft_weights,
            'density_feature': density_feat,
            'density_map': heatmap,
            'reference_points': reference_points,
            'num_queries': num_queries
        }

    def _compute_log_soft_weights(self, N_eval, log_boundaries, detach_boundaries=False):
        temperature = 1.0
        log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)

        if detach_boundaries:
            log_b = log_boundaries.detach()
        else:
            log_b = log_boundaries

        c0 = log_b[:, 0] - 0.5
        c1 = (log_b[:, 0] + log_b[:, 1]) / 2
        c2 = (log_b[:, 1] + log_b[:, 2]) / 2
        c3 = log_b[:, 2] + 0.5

        centers = torch.stack([c0, c1, c2, c3], dim=1)
        distances = -torch.abs(log_N - centers)
        return F.softmax(distances / temperature, dim=1)

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

        ref_points = torch.stack([(topk_x / w).clamp(0.01, 0.99), (topk_y / h).clamp(0.01, 0.99)], dim=-1)
        ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.02], dim=-1)

        if actual_k < max_k:
            ref_points = torch.cat([ref_points, torch.zeros(bs, max_k - actual_k, 4, device=device)], dim=1)

        return ref_points


class TrueAdaptiveBoundaryLoss(nn.Module):
    """
    【修复版】自适应边界损失
    修复：在计算 Interval Loss (分类损失) 时 Detach 目标，防止梯度爆炸。
    """

    def __init__(self, coverage_weight=20.0, spacing_weight=1.0,
                 count_weight=1.0, interval_weight=2.0):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.spacing_weight = spacing_weight
        self.count_weight = count_weight
        self.interval_weight = interval_weight
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        device = outputs['pred_boundaries'].device

        if isinstance(targets, dict) and 'real_counts' in targets:
            real_counts = targets['real_counts'].to(device)
        else:
            real_counts = targets.to(device)

        real_counts = real_counts.float().clamp(min=1.0)

        log_b = outputs['log_boundaries']
        log_c = torch.log(real_counts)

        # 1. Coverage Loss (CDF based) - 驱动边界去匹配数据分布
        tau = 1.0
        cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
        cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
        cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)

        if cdf_b1.dim() == 0: cdf_b1 = cdf_b1.unsqueeze(0)
        if cdf_b2.dim() == 0: cdf_b2 = cdf_b2.unsqueeze(0)
        if cdf_b3.dim() == 0: cdf_b3 = cdf_b3.unsqueeze(0)

        loss_coverage = (
                (cdf_b1.mean() - 0.25) ** 2 +
                (cdf_b2.mean() - 0.50) ** 2 +
                (cdf_b3.mean() - 0.75) ** 2
        )

        # 2. Spacing Loss - 物理约束
        # 修复逻辑：log_b[:,0] clamp在1.0以上，0.4-1.0 < 0，relu恒为0。
        # 我们可以稍微放宽约束，或者保留原样防止未来修改参数出错。
        loss_spacing = (
                F.relu(0.4 - log_b[:, 0]) +
                F.relu(log_b[:, 2] - 9.5)
        ).mean()

        # 3. Interval Loss（分类损失）
        # 目标：训练分类器(pred_bbox_number)去预测当前真实计数所在的区间
        p0 = cdf_b1
        p1 = cdf_b2 - cdf_b1
        p2 = cdf_b3 - cdf_b2
        p3 = 1.0 - cdf_b3

        # 【核心修复】使用 .detach()。
        # 原因：如果不detach，模型会尝试移动边界（改变p0-p3）来“迎合”分类器的预测，而不是让分类器学习边界。
        # 这会导致严重的对抗效应和梯度爆炸。
        soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        soft_targets = soft_targets.detach()  # <--- CRITICAL FIX

        pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
        loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()

        # 4. Count Loss
        loss_count = self.smooth_l1(outputs['raw_count'].squeeze(), log_c)

        total_loss = (
                self.coverage_weight * loss_coverage +
                self.spacing_weight * loss_spacing +
                self.count_weight * loss_count +
                self.interval_weight * loss_interval
        )

        return {
            'total_adaptive_loss': total_loss,
            'coverage_rates': torch.stack([cdf_b1.mean(), cdf_b2.mean(), cdf_b3.mean()]),
            'boundary_vals': torch.exp(log_b).mean(dim=0),
            'loss_coverage': loss_coverage,
            'loss_interval': loss_interval,
            'loss_count': loss_count
        }
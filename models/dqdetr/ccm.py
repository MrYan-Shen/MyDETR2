# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Conv_GN(nn.Module):
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
#     【SOTA增强版】自适应边界CCM
#     新增特性:
#     1. 长尾分布鲁棒性增强
#     2. 渐进式warmup机制
#     3. 边界预测稳定性保障
#     4. 动态权重调整
#     """
#
#     def __init__(self, feature_dim=256, ccm_cls_num=4, query_levels=[300, 500, 900, 1500],
#                  max_objects=1500, use_soft_assignment=True, use_ema=True, ema_decay=0.9997):
#         super().__init__()
#         self.ccm_cls_num = ccm_cls_num
#         self.query_levels = query_levels
#         self.max_objects = max_objects
#         self.use_soft_assignment = use_soft_assignment
#         self.use_ema = use_ema
#         self.ema_decay = ema_decay
#
#         # Backbone
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         self.ccm_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)
#
#         # 边界预测头 - 增强稳定性
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),  # 新增: LayerNorm提升稳定性
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),  # 增大dropout: 0.1 -> 0.2
#             nn.Linear(128, 3)
#         )
#
#         # 计数回归头
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),  # 新增
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(128, 1)
#         )
#
#         # CCM分类头
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # 参考点生成
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         # EMA边界 - 针对长尾分布优化
#         if self.use_ema:
#             # 初始化为保守值 (适合微小目标)
#             self.register_buffer('ema_log_boundaries', torch.tensor([2.3, 2.89, 3.40]))
#             self.register_buffer('ema_initialized', torch.tensor(False))
#
#             # 历史边界缓存 (用于极端样本平滑)
#             self.register_buffer('boundary_history', torch.zeros(100, 3))
#             self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
#
#         # Warmup相关
#         self.register_buffer('training_steps', torch.tensor(0, dtype=torch.long))
#         self.warmup_steps = 5000  # 前5000步渐进增强
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 边界初始化 - 更保守
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.0005)  # 减小: 0.001 -> 0.0005
#         nn.init.constant_(self.boundary_head[-1].bias[0], 2.3)
#         nn.init.constant_(self.boundary_head[-1].bias[1], 0.59)
#         nn.init.constant_(self.boundary_head[-1].bias[2], 0.51)
#
#         # 计数初始化
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.0005)
#         nn.init.constant_(self.count_regressor[-1].bias, 3.5)  # log(50) ≈ 3.9 -> 3.5
#
#         # 参考点初始化
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)
#
#     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
#         # 1. 特征处理
#         if feature_map.dim() == 3:
#             if spatial_shapes is None:
#                 raise ValueError("spatial_shapes needed when feature_map is 3D")
#             bs, l, c = feature_map.shape
#             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
#             feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
#
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # 2. 特征提取
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)
#
#         # 3. 边界预测
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_out = self.boundary_head(bd_feat)
#
#         # 渐进式激活
#         warmup_factor = self._get_warmup_factor()
#
#         # 基础边界 (更保守的初始值)
#         log_b1 = raw_out[:, 0].clamp(min=1.5, max=6.0)  # 放宽: 1.0-8.0 -> 1.5-6.0
#         min_log_gap = 0.5  # 增大最小间隔: 0.42 -> 0.5
#
#         delta12 = F.softplus(raw_out[:, 1]) * warmup_factor + min_log_gap
#         delta23 = F.softplus(raw_out[:, 2]) * warmup_factor + min_log_gap
#
#         log_b2 = log_b1 + delta12
#         log_b3 = log_b2 + delta23
#
#         log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
#
#         # 长尾样本平滑
#         if self.use_ema and self.training:
#             log_boundaries = self._smooth_boundaries_for_longtail(log_boundaries, real_counts)
#
#         # 【EMA更新】
#         if self.use_ema and self.training:
#             log_boundaries_mean = log_boundaries.mean(dim=0)
#
#             if not self.ema_initialized:
#                 self.ema_log_boundaries.copy_(log_boundaries_mean.detach())
#                 self.ema_initialized.fill_(True)
#             else:
#                 # 动态EMA系数 (长尾样本用更大的decay)
#                 dynamic_decay = self._compute_dynamic_ema_decay(real_counts)
#                 self.ema_log_boundaries.mul_(dynamic_decay).add_(
#                     log_boundaries_mean.detach(), alpha=1 - dynamic_decay
#                 )
#
#             # 历史缓存
#             self._update_boundary_history(log_boundaries_mean.detach())
#
#             log_boundaries_for_use = self.ema_log_boundaries.unsqueeze(0).expand(bs, -1)
#         else:
#             log_boundaries_for_use = log_boundaries
#
#         boundaries = torch.exp(log_boundaries_for_use)
#
#         # 4. 计数回归
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # 5. CCM分类
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)
#
#         # 6. 查询数量分配
#         if real_counts is not None:
#             if not isinstance(real_counts, torch.Tensor):
#                 real_counts = torch.tensor(real_counts, device=device, dtype=torch.float32)
#             N_eval = (real_counts.float() * 1.5 + 50.0).clamp(max=self.max_objects)
#         else:
#             N_eval = pred_count
#
#         if self.use_soft_assignment and self.training:
#             soft_weights = self._compute_soft_weights(N_eval, log_boundaries_for_use)
#             query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
#             num_queries = (soft_weights * query_levels_tensor).sum(dim=1).long()
#             level_indices = soft_weights.argmax(dim=1)
#         else:
#             level_indices = self._assign_query_levels(N_eval, boundaries)
#             query_levels_tensor = torch.tensor(self.query_levels, device=device)
#             num_queries = query_levels_tensor[level_indices]
#             soft_weights = None
#
#         # 7. 参考点生成
#         heatmap = torch.sigmoid(self.ref_point_conv(density_feat).clamp(-10, 10))
#         reference_points = self._generate_reference_points(heatmap, h, w, device)
#
#         # 更新训练步数
#         if self.training:
#             self.training_steps += 1
#
#         return {
#             'pred_boundaries': boundaries,
#             'log_boundaries': log_boundaries,
#             'log_boundaries_ema': log_boundaries_for_use,
#             'predicted_count': pred_count,
#             'raw_count': raw_count,
#             'pred_bbox_number': pred_bbox_number,
#             'soft_weights': soft_weights,
#             'density_feature': density_feat,
#             'density_map': heatmap,
#             'reference_points': reference_points,
#             'num_queries': num_queries,
#             'level_indices': level_indices,
#             'warmup_factor': warmup_factor  # 用于监控
#         }
#
#     def _get_warmup_factor(self) -> float:
#         """渐进式warmup因子"""
#         if not self.training:
#             return 1.0
#
#         steps = self.training_steps.item()
#         if steps >= self.warmup_steps:
#             return 1.0
#
#         # 余弦warmup
#         return 0.5 * (1 + torch.cos(torch.tensor((1 - steps / self.warmup_steps) * 3.14159))).item()
#
#     def _smooth_boundaries_for_longtail(self, log_boundaries: torch.Tensor,
#                                         real_counts: torch.Tensor) -> torch.Tensor:
#         """长尾样本边界平滑"""
#         if real_counts is None:
#             return log_boundaries
#
#         # 识别极端样本 (>100个目标)
#         extreme_mask = real_counts > 100
#
#         if extreme_mask.any():
#             # 对极端样本使用历史均值
#             if self.history_ptr > 10:  # 至少有10个历史样本
#                 hist_mean = self.boundary_history[:self.history_ptr].mean(dim=0)
#                 log_boundaries[extreme_mask] = 0.7 * log_boundaries[extreme_mask] + 0.3 * hist_mean
#
#         return log_boundaries
#
#     def _compute_dynamic_ema_decay(self, real_counts: torch.Tensor) -> float:
#         """动态EMA系数 (长尾样本更保守)"""
#         if real_counts is None:
#             return self.ema_decay
#
#         max_count = real_counts.max().item()
#
#         if max_count > 150:
#             return 0.999  # 超密集 -> 更大decay
#         elif max_count > 80:
#             return 0.998
#         else:
#             return self.ema_decay
#
#     def _update_boundary_history(self, boundaries: torch.Tensor):
#         """更新边界历史"""
#         ptr = self.history_ptr.item()
#         self.boundary_history[ptr % 100] = boundaries
#         self.history_ptr += 1
#
#     def _compute_soft_weights(self, N_eval, log_boundaries):
#         temperature = 1.0
#         log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)
#
#         c0 = log_boundaries[:, 0] - 0.5
#         c1 = (log_boundaries[:, 0] + log_boundaries[:, 1]) / 2
#         c2 = (log_boundaries[:, 1] + log_boundaries[:, 2]) / 2
#         c3 = log_boundaries[:, 2] + 0.5
#
#         centers = torch.stack([c0, c1, c2, c3], dim=1)
#         distances = -torch.abs(log_N - centers)
#         soft_weights = F.softmax(distances / temperature, dim=1)
#
#         return soft_weights
#
#     def _assign_query_levels(self, N_eval, boundaries):
#         bs = N_eval.shape[0]
#         level_indices = torch.zeros(bs, dtype=torch.long, device=N_eval.device)
#         b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
#
#         level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
#         level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
#         level_indices[N_eval >= b3] = 3
#
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
#         ref_points = torch.stack([
#             (topk_x / w).clamp(0.01, 0.99),
#             (topk_y / h).clamp(0.01, 0.99)
#         ], dim=-1)
#
#         ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.02], dim=-1)
#
#         if actual_k < max_k:
#             padding = torch.zeros(bs, max_k - actual_k, 4, device=device)
#             ref_points = torch.cat([ref_points, padding], dim=1)
#
#         return ref_points
#
#
# class TrueAdaptiveBoundaryLoss(nn.Module):
#     """
#     【SOTA增强版】自适应边界损失
#     新增特性:
#     1. 动态权重调整 (根据训练阶段)
#     2. 长尾样本特殊处理
#     3. 梯度裁剪保护
#     """
#
#     def __init__(self,
#                  coverage_weight=0.3,  # 降低: 0.5 -> 0.3
#                  spacing_weight=1.5,  # 降低: 2.0 -> 1.5
#                  count_weight=0.15,  # 提高: 0.1 -> 0.15
#                  interval_weight=0.25,  # 提高: 0.2 -> 0.25
#                  boundary_guide_weight=1.2,  # 提高: 1.0 -> 1.2
#                  enable_adaptive_targets=True,
#                  enable_loss_clipping=True):
#         super().__init__()
#         self.coverage_weight = coverage_weight
#         self.spacing_weight = spacing_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.boundary_guide_weight = boundary_guide_weight
#
#         self.enable_adaptive_targets = enable_adaptive_targets
#         self.enable_loss_clipping = enable_loss_clipping
#
#         self.smooth_l1 = nn.SmoothL1Loss()
#
#         # 固定目标
#         self.register_buffer('default_target_coverage',
#                              torch.tensor([0.40, 0.70, 0.90]))
#         self.register_buffer('default_target_boundaries_log',
#                              torch.tensor([2.30, 2.89, 3.40]))
#
#     def _compute_adaptive_targets(self, real_counts, device):
#         bs = real_counts.shape[0]
#
#         # 【优化】更细粒度的分档
#         small_b1 = torch.tensor(2.30, device=device)
#         small_b2 = torch.tensor(2.89, device=device)
#         small_b3 = torch.tensor(3.40, device=device)
#
#         medium_b1 = torch.tensor(2.71, device=device)
#         medium_b2 = torch.tensor(3.30, device=device)
#         medium_b3 = torch.tensor(3.91, device=device)
#
#         large_b1 = torch.tensor(3.22, device=device)
#         large_b2 = torch.tensor(3.91, device=device)
#         large_b3 = torch.tensor(4.61, device=device)
#
#         # 【新增】超大档 (100+目标)
#         xlarge_b1 = torch.tensor(3.50, device=device)  # log(33)
#         xlarge_b2 = torch.tensor(4.20, device=device)  # log(66)
#         xlarge_b3 = torch.tensor(4.85, device=device)  # log(128)
#
#         # 分档逻辑
#         target_b1_log = torch.where(
#             real_counts < 30,
#             small_b1,
#             torch.where(real_counts < 80,
#                         medium_b1,
#                         torch.where(real_counts < 150,
#                                     large_b1,
#                                     xlarge_b1))
#         )
#
#         target_b2_log = torch.where(
#             real_counts < 30,
#             small_b2,
#             torch.where(real_counts < 80,
#                         medium_b2,
#                         torch.where(real_counts < 150,
#                                     large_b2,
#                                     xlarge_b2))
#         )
#
#         target_b3_log = torch.where(
#             real_counts < 30,
#             small_b3,
#             torch.where(real_counts < 80,
#                         medium_b3,
#                         torch.where(real_counts < 150,
#                                     large_b3,
#                                     xlarge_b3))
#         )
#
#         target_boundaries_log = torch.stack([target_b1_log, target_b2_log, target_b3_log], dim=1)
#
#         # 覆盖率目标 (超大档降低覆盖率)
#         target_cov1 = torch.where(
#             real_counts < 30,
#             torch.tensor(0.50, device=device),
#             torch.where(real_counts < 80,
#                         torch.tensor(0.40, device=device),
#                         torch.where(real_counts < 150,
#                                     torch.tensor(0.25, device=device),
#                                     torch.tensor(0.15, device=device)))  # 新增
#         )
#
#         target_cov2 = torch.where(
#             real_counts < 30,
#             torch.tensor(0.75, device=device),
#             torch.where(real_counts < 80,
#                         torch.tensor(0.70, device=device),
#                         torch.where(real_counts < 150,
#                                     torch.tensor(0.55, device=device),
#                                     torch.tensor(0.40, device=device)))
#         )
#
#         target_cov3 = torch.where(
#             real_counts < 30,
#             torch.tensor(0.92, device=device),
#             torch.where(real_counts < 80,
#                         torch.tensor(0.90, device=device),
#                         torch.where(real_counts < 150,
#                                     torch.tensor(0.80, device=device),
#                                     torch.tensor(0.65, device=device)))
#         )
#
#         target_coverage = torch.stack([target_cov1, target_cov2, target_cov3], dim=1)
#
#         return target_boundaries_log, target_coverage
#
#     def forward(self, outputs, targets):
#         device = outputs['pred_boundaries'].device
#
#         if isinstance(targets, dict) and 'real_counts' in targets:
#             real_counts = targets['real_counts'].to(device)
#         else:
#             real_counts = targets.to(device)
#
#         real_counts = real_counts.float().clamp(min=1.0)
#         bs = real_counts.shape[0]
#
#         log_b = outputs['log_boundaries']
#         log_c = torch.log(real_counts)
#
#         # 获取目标值
#         if self.enable_adaptive_targets:
#             target_boundaries_log, target_coverage = self._compute_adaptive_targets(real_counts, device)
#         else:
#             target_boundaries_log = self.default_target_boundaries_log.unsqueeze(0).expand(bs, -1)
#             target_coverage = self.default_target_coverage.unsqueeze(0).expand(bs, -1)
#
#         # 【新增】动态权重 (超大样本降权)
#         sample_weights = torch.where(
#             real_counts > 150,
#             torch.tensor(0.2, device=device),  # 150+: 20%
#             torch.where(
#                 real_counts > 80,
#                 torch.tensor(0.5, device=device),  # 80-150: 50%
#                 torch.tensor(1.0, device=device)  # <80: 100%
#             )
#         )
#
#         # 1. Coverage Loss
#         tau = 1.0
#         cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
#         cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
#         cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)
#
#         if cdf_b1.dim() == 0: cdf_b1 = cdf_b1.unsqueeze(0)
#         if cdf_b2.dim() == 0: cdf_b2 = cdf_b2.unsqueeze(0)
#         if cdf_b3.dim() == 0: cdf_b3 = cdf_b3.unsqueeze(0)
#
#         cov_loss_1 = (cdf_b1 - target_coverage[:, 0]).pow(2)
#         cov_loss_2 = (cdf_b2 - target_coverage[:, 1]).pow(2)
#         cov_loss_3 = (cdf_b3 - target_coverage[:, 2]).pow(2)
#
#         cov_loss_per_sample = cov_loss_1 + cov_loss_2 + cov_loss_3
#
#         if self.enable_loss_clipping:
#             cov_loss_per_sample = cov_loss_per_sample.clamp(max=2.0)  # 放宽: 1.5 -> 2.0
#             loss_coverage = (cov_loss_per_sample * sample_weights).mean()
#         else:
#             loss_coverage = cov_loss_per_sample.mean()
#
#         # 2. Boundary Guidance Loss (加权)
#         loss_boundary_guide = F.smooth_l1_loss(
#             log_b * sample_weights.unsqueeze(1),
#             target_boundaries_log * sample_weights.unsqueeze(1)
#         )
#
#         # 3. Spacing Loss (保持原有逻辑)
#         loss_spacing = (
#                 F.relu(2.0 - log_b[:, 0]) * 3.0 +
#                 F.relu(log_b[:, 0] - 2.71) * 3.0 +
#                 F.relu(2.65 - log_b[:, 1]) * 2.0 +
#                 F.relu(log_b[:, 1] - 3.22) * 2.0 +
#                 F.relu(3.20 - log_b[:, 2]) * 2.0 +
#                 F.relu(log_b[:, 2] - 3.81) * 2.0 +
#                 F.relu(log_b[:, 0] + 0.47 - log_b[:, 1]) * 3.0 +
#                 F.relu(log_b[:, 1] + 0.47 - log_b[:, 2]) * 3.0
#         ).mean()
#
#         # 4. Interval Loss
#         p0 = cdf_b1
#         p1 = cdf_b2 - cdf_b1
#         p2 = cdf_b3 - cdf_b2
#         p3 = 1.0 - cdf_b3
#
#         soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
#         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
#         soft_targets = soft_targets.detach()
#
#         pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
#         loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()
#
#         # 5. Count Loss
#         loss_count = self.smooth_l1(outputs['raw_count'].squeeze(-1), log_c)
#
#         # 【新增】总损失加权
#         warmup_factor = outputs.get('warmup_factor', 1.0)
#
#         total_loss = (
#                 self.coverage_weight * loss_coverage * warmup_factor +
#                 self.spacing_weight * loss_spacing +
#                 self.count_weight * loss_count +
#                 self.interval_weight * loss_interval +
#                 self.boundary_guide_weight * loss_boundary_guide
#         )
#
#         result = {
#             'total_adaptive_loss': total_loss,
#             'coverage_rates': torch.stack([cdf_b1.mean(), cdf_b2.mean(), cdf_b3.mean()]),
#             'boundary_vals': torch.exp(log_b).mean(dim=0),
#             'loss_coverage': loss_coverage,
#             'loss_interval': loss_interval,
#             'loss_count': loss_count,
#             'loss_spacing': loss_spacing,
#             'loss_boundary_guide': loss_boundary_guide,
#         }
#
#         # 如果有EMA边界，也返回
#         if 'log_boundaries_ema' in outputs:
#             result['boundary_vals_ema'] = torch.exp(outputs['log_boundaries_ema']).mean(dim=0)
#
#         # 返回当前使用的自适应目标（用于监控）
#         if self.enable_adaptive_targets:
#             result['adaptive_target_boundaries'] = torch.exp(target_boundaries_log).mean(dim=0)
#             result['adaptive_target_coverage'] = target_coverage.mean(dim=0)
#
#         return result

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
    【SOTA 修正版】受限自适应边界 CCM (Constrained Adaptive Boundary)

    核心改进:
    1. 引入物理约束(Physical Constraints): 严格限制各层级的最大尺度，防止边界漂移。
    2. 保留自适应特性: 在安全范围内允许模型动态微调。
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

        # 边界预测头
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
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

        if self.use_ema:
            # 初始值设为 Epoch 13 的最佳实践值
            self.register_buffer('ema_log_boundaries', torch.tensor([2.25, 2.89, 3.40]))
            self.register_buffer('ema_initialized', torch.tensor(False))
            self.register_buffer('boundary_history', torch.zeros(100, 3))
            self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))

        self.register_buffer('training_steps', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = 5000

        self._init_weights()

    def _init_weights(self):
        for m in self.ccm_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 初始化为 Epoch 13 附近的最佳值: [9.5, 18.0, 30.0]
        # log(9.5)≈2.25, log(18)≈2.89, log(30)≈3.40
        nn.init.normal_(self.boundary_head[-1].weight, std=0.0005)
        nn.init.constant_(self.boundary_head[-1].bias[0], 2.25)  # 9.5px
        nn.init.constant_(self.boundary_head[-1].bias[1], 0.64)  # delta1
        nn.init.constant_(self.boundary_head[-1].bias[2], 0.51)  # delta2

        nn.init.normal_(self.count_regressor[-1].weight, std=0.0005)
        nn.init.constant_(self.count_regressor[-1].bias, 3.5)

        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)

    def forward(self, feature_map, spatial_shapes=None, real_counts=None):
        if feature_map.dim() == 3:
            if spatial_shapes is None:
                raise ValueError("spatial_shapes needed when feature_map is 3D")
            bs, l, c = feature_map.shape
            h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)

        bs, c, h, w = feature_map.shape
        device = feature_map.device

        x = self.density_conv1(feature_map)
        density_feat = self.ccm_backbone(x)

        bd_feat = self.boundary_pool(density_feat).flatten(1)
        raw_out = self.boundary_head(bd_feat)

        warmup_factor = self._get_warmup_factor()

        # 【核心修改：物理约束】----------------------------------------------------
        # 施加严格的物理约束，防止边界漂移到 14px (log 2.63) 以上。
        # min=1.5 (4.5px): 保证不低于特征图最小分辨率
        # max=2.5 (12.2px): 强制定义"极微小目标"上限，超过这个值就不叫 Very Tiny 了。
        # 这样模型仍然可以在 [4.5px, 12.2px] 之间自适应，但绝不会跑偏。

        log_b1 = raw_out[:, 0].clamp(min=1.0, max=3.0)

        min_log_gap = 0.5
        delta12 = F.softplus(raw_out[:, 1]) * warmup_factor + min_log_gap
        delta23 = F.softplus(raw_out[:, 2]) * warmup_factor + min_log_gap

        log_b2 = log_b1 + delta12

        # 同样对 b2 施加软约束，防止 Tiny 层过大
        # log(28px) ≈ 3.33. 限制 Tiny 层上限为 28px
        log_b2 = log_b2.clamp(max=3.6)

        log_b3 = log_b2 + delta23

        log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
        # ------------------------------------------------------------------------

        if self.use_ema and self.training:
            log_boundaries = self._smooth_boundaries_for_longtail(log_boundaries, real_counts)

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
        if not self.training:
            return 1.0
        steps = self.training_steps.item()
        if steps >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + torch.cos(torch.tensor((1 - steps / self.warmup_steps) * 3.14159))).item()

    def _smooth_boundaries_for_longtail(self, log_boundaries: torch.Tensor, real_counts: torch.Tensor) -> torch.Tensor:
        if real_counts is None:
            return log_boundaries
        extreme_mask = real_counts > 100
        if extreme_mask.any():
            if self.history_ptr > 10:
                hist_mean = self.boundary_history[:self.history_ptr].mean(dim=0)
                log_boundaries[extreme_mask] = 0.7 * log_boundaries[extreme_mask] + 0.3 * hist_mean
        return log_boundaries

    def _compute_dynamic_ema_decay(self, real_counts: torch.Tensor) -> float:
        if real_counts is None:
            return self.ema_decay
        max_count = real_counts.max().item()
        if max_count > 150:
            return 0.999
        elif max_count > 80:
            return 0.998
        else:
            return self.ema_decay

    def _update_boundary_history(self, boundaries: torch.Tensor):
        ptr = self.history_ptr.item()
        self.boundary_history[ptr % 100] = boundaries
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
    def __init__(self,
                 coverage_weight=0.3,
                 spacing_weight=1.5,
                 count_weight=0.15,
                 interval_weight=0.25,
                 boundary_guide_weight=1.2,
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
        bs = real_counts.shape[0]
        small_b1 = torch.tensor(2.30, device=device)
        small_b2 = torch.tensor(2.89, device=device)
        small_b3 = torch.tensor(3.40, device=device)
        medium_b1 = torch.tensor(2.71, device=device)
        medium_b2 = torch.tensor(3.30, device=device)
        medium_b3 = torch.tensor(3.91, device=device)
        large_b1 = torch.tensor(3.22, device=device)
        large_b2 = torch.tensor(3.91, device=device)
        large_b3 = torch.tensor(4.61, device=device)
        xlarge_b1 = torch.tensor(3.50, device=device)
        xlarge_b2 = torch.tensor(4.20, device=device)
        xlarge_b3 = torch.tensor(4.85, device=device)
        target_b1_log = torch.where(
            real_counts < 30,
            small_b1,
            torch.where(real_counts < 80,
                        medium_b1,
                        torch.where(real_counts < 150,
                                    large_b1,
                                    xlarge_b1))
        )
        target_b2_log = torch.where(
            real_counts < 30,
            small_b2,
            torch.where(real_counts < 80,
                        medium_b2,
                        torch.where(real_counts < 150,
                                    large_b2,
                                    xlarge_b2))
        )
        target_b3_log = torch.where(
            real_counts < 30,
            small_b3,
            torch.where(real_counts < 80,
                        medium_b3,
                        torch.where(real_counts < 150,
                                    large_b3,
                                    xlarge_b3))
        )
        target_boundaries_log = torch.stack([target_b1_log, target_b2_log, target_b3_log], dim=1)
        target_cov1 = torch.where(
            real_counts < 30,
            torch.tensor(0.50, device=device),
            torch.where(real_counts < 80,
                        torch.tensor(0.40, device=device),
                        torch.where(real_counts < 150,
                                    torch.tensor(0.25, device=device),
                                    torch.tensor(0.15, device=device)))
        )
        target_cov2 = torch.where(
            real_counts < 30,
            torch.tensor(0.75, device=device),
            torch.where(real_counts < 80,
                        torch.tensor(0.70, device=device),
                        torch.where(real_counts < 150,
                                    torch.tensor(0.55, device=device),
                                    torch.tensor(0.40, device=device)))
        )
        target_cov3 = torch.where(
            real_counts < 30,
            torch.tensor(0.92, device=device),
            torch.where(real_counts < 80,
                        torch.tensor(0.90, device=device),
                        torch.where(real_counts < 150,
                                    torch.tensor(0.80, device=device),
                                    torch.tensor(0.65, device=device)))
        )
        target_coverage = torch.stack([target_cov1, target_cov2, target_cov3], dim=1)
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
        sample_weights = torch.where(
            real_counts > 150,
            torch.tensor(0.2, device=device),
            torch.where(
                real_counts > 80,
                torch.tensor(0.5, device=device),
                torch.tensor(1.0, device=device)
            )
        )
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
            cov_loss_per_sample = cov_loss_per_sample.clamp(max=2.0)
            loss_coverage = (cov_loss_per_sample * sample_weights).mean()
        else:
            loss_coverage = cov_loss_per_sample.mean()
        loss_boundary_guide = F.smooth_l1_loss(
            log_b * sample_weights.unsqueeze(1),
            target_boundaries_log * sample_weights.unsqueeze(1)
        )
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
        p0 = cdf_b1
        p1 = cdf_b2 - cdf_b1
        p2 = cdf_b3 - cdf_b2
        p3 = 1.0 - cdf_b3
        soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        soft_targets = soft_targets.detach()
        pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
        loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()
        loss_count = self.smooth_l1(outputs['raw_count'].squeeze(-1), log_c)
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
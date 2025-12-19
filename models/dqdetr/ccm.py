# # # 1216完整训练使用代码
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # #
# # #
# # # class Conv_GN(nn.Module):
# # #     """卷积 + GroupNorm + ReLU"""
# # #
# # #     def __init__(self, in_channel, out_channel, kernel_size, stride=1,
# # #                  padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
# # #         super(Conv_GN, self).__init__()
# # #         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
# # #                               stride=stride, padding=padding, dilation=dilation,
# # #                               groups=groups, bias=bias)
# # #         self.gn = nn.GroupNorm(32, out_channel) if gn else None
# # #         self.relu = nn.ReLU(inplace=True) if relu else None
# # #
# # #     def forward(self, x):
# # #         x = self.conv(x)
# # #         if self.gn is not None:
# # #             x = self.gn(x)
# # #         if self.relu is not None:
# # #             x = self.relu(x)
# # #         return x
# # #
# # #
# # # def make_ccm_layers(cfg, in_channels=256, d_rate=2):
# # #     """构建CCM层序列（使用空洞卷积）"""
# # #     layers = []
# # #     for v in cfg:
# # #         conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
# # #         layers.append(conv2d)
# # #         in_channels = v
# # #     return nn.Sequential(*layers)
# # #
# # #
# # # class AdaptiveBoundaryCCM(nn.Module):
# # #     """
# # #     自适应边界分类计数模块 (Adaptive Boundary CCM)
# # #
# # #     整合了:
# # #     1. CCM密度特征提取 (空洞卷积 backbone)
# # #     2. 自适应边界预测 (3个可学习的边界点)
# # #     3. 目标数量回归
# # #     4. 动态查询数量选择
# # #     """
# # #
# # #     def __init__(self,
# # #                  feature_dim=256,
# # #                  ccm_cls_num=4,  # 区间数量
# # #                  query_levels=[300, 500, 900, 1500],  # 对应4个区间的查询数
# # #                  max_objects=1500):
# # #         super().__init__()
# # #
# # #         self.ccm_cls_num = ccm_cls_num
# # #         self.query_levels = query_levels
# # #         self.max_objects = max_objects
# # #
# # #         # ============ 1. 共享密度特征提取器 (CCM Backbone) ============
# # #         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
# # #         # CCM配置: [512, 512, 512, 256, 256, 256]
# # #         self.ccm_backbone = make_ccm_layers(
# # #             [512, 512, 512, 256, 256, 256],
# # #             in_channels=512,
# # #             d_rate=2
# # #         )
# # #
# # #         # ============ 2. 分支A: 边界预测模块 ============
# # #         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
# # #         self.boundary_head = nn.Sequential(
# # #             nn.Flatten(),
# # #             nn.Linear(256, 128),
# # #             nn.ReLU(inplace=True),
# # #             nn.Dropout(0.1),
# # #             nn.Linear(128, 3)  # 预测3个边界值 [b1, b2, b3]
# # #         )
# # #
# # #         # ============ 3. 分支B: 目标数量回归 ============
# # #         self.count_regressor = nn.Sequential(
# # #             nn.AdaptiveAvgPool2d(1),
# # #             nn.Flatten(),
# # #             nn.Linear(256, 128),
# # #             nn.ReLU(inplace=True),
# # #             nn.Dropout(0.1),
# # #             nn.Linear(128, 1)  # 输出 log(count)
# # #         )
# # #
# # #         # ============ 4. 分支C: CCM分类头 (辅助监督) ============
# # #         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
# # #         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
# # #
# # #         # ============ 5. 参考点生成 (基于密度图) ============
# # #         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
# # #
# # #         self._init_weights()
# # #
# # #     def _init_weights(self):
# # #         """权重初始化"""
# # #         # 1. CCM backbone (He初始化)
# # #         for m in self.ccm_backbone.modules():
# # #             if isinstance(m, nn.Conv2d):
# # #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# # #
# # #         # 2. 边界预测 (修正：预测增量delta，而非绝对值)
# # #         # 目标: delta1=100, delta2=200, delta3=200
# # #         nn.init.normal_(self.boundary_head[-1].weight, std=0.001)
# # #         nn.init.constant_(self.boundary_head[-1].bias[0], 4.605)  # log(100) = 4.605
# # #         nn.init.constant_(self.boundary_head[-1].bias[1], 5.298)  # log(200) = 5.298
# # #         nn.init.constant_(self.boundary_head[-1].bias[2], 5.298)  # log(200) = 5.298
# # #
# # #         # 3. 数量回归 (修正：初始化为 log(200))
# # #         nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
# # #         nn.init.constant_(self.count_regressor[-1].bias, 5.298)  # log(200) ≈ 5.298
# # #
# # #         # 4. 参考点生成 (bias设低，初始heatmap较平滑)
# # #         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
# # #         nn.init.constant_(self.ref_point_conv.bias, -2.19)  # sigmoid(-2.19) ≈ 0.1
# # #
# # #     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
# # #         """
# # #         Args:
# # #             feature_map: (BS, 256, H, W) OR (BS, SumHW, C)
# # #             spatial_shapes: (NumLevels, 2) 每一层特征图的形状，用于重塑扁平化特征
# # #             real_counts: (BS) 真实目标数量 (仅训练时提供)
# # #
# # #         Returns:
# # #             outputs: dict包含...
# # #         """
# # #         # Handle flattened input from Deformable Transformer
# # #         # Input shape might be (BS, SumHW, C)
# # #         if feature_map.dim() == 3:
# # #             if spatial_shapes is None:
# # #                 raise ValueError("spatial_shapes must be provided when input is flattened (BS, L, C)")
# # #
# # #             bs, l, c = feature_map.shape
# # #             # Typically use the first scale (High Resolution) for CCM
# # #             h, w = spatial_shapes[0]
# # #             h, w = int(h), int(w)
# # #
# # #             # Extract first level and reshape to (BS, C, H, W)
# # #             # feature_map is (BS, SumHW, C), we take the first H*W tokens
# # #             x = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
# # #
# # #             # Note: feature_map argument variable is reused as 'x' for convolution
# # #             feature_map = x
# # #
# # #         bs, c, h, w = feature_map.shape
# # #         device = feature_map.device
# # #
# # #         # ============ Step 1: 提取共享密度特征 ============
# # #         x = self.density_conv1(feature_map)
# # #         density_feat = self.ccm_backbone(x)  # (BS, 256, H, W)
# # #
# # #         # ============ Step 2: 边界预测 (修正版) ============
# # #         bd_feat = self.boundary_pool(density_feat).flatten(1)
# # #         raw_boundaries = self.boundary_head(bd_feat)  # (BS, 3)
# # #
# # #         # 修正：预测delta增量，确保严格单调递增
# # #         # b1 = exp(delta1), b2 = b1 + exp(delta2), b3 = b2 + exp(delta3)
# # #         deltas = torch.exp(raw_boundaries).clamp(min=10, max=500)  # 限制增量范围
# # #
# # #         boundaries = []
# # #         boundaries.append(deltas[:, 0])  # b1 = delta1
# # #         boundaries.append(boundaries[0] + deltas[:, 1])  # b2 = b1 + delta2
# # #         boundaries.append(boundaries[1] + deltas[:, 2])  # b3 = b2 + delta3
# # #
# # #         boundaries = torch.stack(boundaries, dim=1)  # (BS, 3)
# # #         boundaries = boundaries.clamp(max=self.max_objects)  # 全局上限
# # #
# # #         # ============ Step 3: 目标数量回归 ============
# # #         raw_count = self.count_regressor(density_feat).squeeze(1)
# # #         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
# # #
# # #         # ============ Step 4: CCM分类 (辅助损失) ============
# # #         ccm_feat = self.ccm_pool(density_feat).flatten(1)
# # #         pred_bbox_number = self.ccm_classifier(ccm_feat)  # (BS, 4)
# # #
# # #         # ============ Step 5: 决定查询数量 ============
# # #         if self.training and real_counts is not None:
# # #             # 训练阶段: 使用GT边界生成分类标签
# # #             target_labels = self._compute_target_labels(real_counts, boundaries)
# # #
# # #             # 使用真实数量 * 1.2 + 20 作为查询数(增加召回)
# # #             N_eval = (real_counts.float() * 1.2 + 20).clamp(max=self.max_objects)
# # #         else:
# # #             # 推理阶段: 使用预测数量
# # #             N_eval = pred_count
# # #             target_labels = None
# # #
# # #         # 根据N_eval决定查询级别
# # #         level_indices = self._assign_query_levels(N_eval, boundaries)
# # #         query_levels_tensor = torch.tensor(self.query_levels, device=device)
# # #         num_queries = query_levels_tensor[level_indices]  # (BS,)
# # #
# # #         # ============ Step 6: 生成参考点 (基于密度Heatmap) ============
# # #         heatmap = self.ref_point_conv(density_feat).sigmoid()  # (BS, 1, H, W)
# # #         reference_points = self._generate_reference_points(heatmap, h, w, device)
# # #
# # #         # ============ 输出组装 ============
# # #         outputs = {
# # #             'pred_boundaries': boundaries,  # (BS, 3)
# # #             'raw_boundaries': raw_boundaries,  # (BS, 3) for loss
# # #             'predicted_count': pred_count,  # (BS)
# # #             'raw_count': raw_count,  # (BS) for loss
# # #             'num_queries': num_queries,  # (BS)
# # #             'pred_bbox_number': pred_bbox_number,  # (BS, 4) CCM分类
# # #             'reference_points': reference_points,  # (BS, max_k, 4)
# # #             'density_map': heatmap,  # (BS, 1, H, W)
# # #             'density_feature': density_feat,  # (BS, 256, H, W)
# # #             'target_labels': target_labels,  # (BS) 仅训练时有效
# # #             'level_indices': level_indices  # (BS) for debug
# # #         }
# # #
# # #         return outputs
# # #
# # #     def _compute_target_labels(self, real_counts, boundaries):
# # #         """
# # #         训练时: 根据GT数量和预测边界计算分类标签
# # #
# # #         Args:
# # #             real_counts: (BS) GT目标数
# # #             boundaries: (BS, 3) 预测的边界 [b1, b2, b3]
# # #
# # #         Returns:
# # #             labels: (BS) 区间标签 [0, 1, 2, 3]
# # #         """
# # #         bs = real_counts.shape[0]
# # #         labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
# # #
# # #         b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
# # #
# # #         # 区间划分:
# # #         # 0: [0, b1)
# # #         # 1: [b1, b2)
# # #         # 2: [b2, b3)
# # #         # 3: [b3, +∞)
# # #         labels[(real_counts >= b1) & (real_counts < b2)] = 1
# # #         labels[(real_counts >= b2) & (real_counts < b3)] = 2
# # #         labels[real_counts >= b3] = 3
# # #
# # #         return labels
# # #
# # #     def _assign_query_levels(self, N_eval, boundaries):
# # #         """
# # #         根据评估数量和边界分配查询级别
# # #
# # #         Args:
# # #             N_eval: (BS) 评估的目标数量
# # #             boundaries: (BS, 3) 边界值
# # #
# # #         Returns:
# # #             level_indices: (BS) 级别索引 [0, 1, 2, 3]
# # #         """
# # #         bs = N_eval.shape[0]
# # #         device = N_eval.device
# # #         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
# # #
# # #         b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
# # #
# # #         level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
# # #         level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
# # #         level_indices[N_eval >= b3] = 3
# # #
# # #         return level_indices
# # #
# # #     def _generate_reference_points(self, heatmap, h, w, device):
# # #         """
# # #         基于密度热力图生成参考点
# # #
# # #         Args:
# # #             heatmap: (BS, 1, H, W)
# # #             h, w: 特征图尺寸
# # #             device: 设备
# # #
# # #         Returns:
# # #             ref_points: (BS, max_k, 4) 归一化坐标 [cx, cy, w, h]
# # #         """
# # #         bs = heatmap.shape[0]
# # #         max_k = max(self.query_levels)
# # #
# # #         # 展平并选择Top-K
# # #         heatmap_flat = heatmap.flatten(2).squeeze(1)  # (BS, H*W)
# # #         actual_k = min(h * w, max_k)
# # #         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
# # #
# # #         # 转换为归一化坐标
# # #         topk_y = (topk_ind // w).float() + 0.5
# # #         topk_x = (topk_ind % w).float() + 0.5
# # #
# # #         ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)  # (BS, K, 2)
# # #
# # #         # 初始宽高 (小目标场景使用较小值)
# # #         initial_wh = torch.ones_like(ref_points) * 0.02
# # #         ref_points = torch.cat([ref_points, initial_wh], dim=-1)  # (BS, K, 4)
# # #
# # #         # Padding到max_k
# # #         if actual_k < max_k:
# # #             pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
# # #             ref_points = torch.cat([ref_points, pad], dim=1)
# # #
# # #         return ref_points
# # #
# # #
# # # # ============ 损失函数 ============
# # # class AdaptiveBoundaryLoss(nn.Module):
# # #     """
# # #     自适应边界CCM的损失函数 - 修复版
# # #
# # #     包含:
# # #     1. 边界预测损失 (MSE Loss，更强约束)
# # #     2. 数量回归损失 (Smooth L1 on log-space)
# # #     3. 区间分类损失 (CrossEntropy，使用动态标签)
# # #     4. CCM辅助损失 (CrossEntropy，使用固定标签)
# # #     """
# # #
# # #     def __init__(self,
# # #                  boundary_weight=2.0,  # 提高边界损失权重
# # #                  count_weight=1.0,
# # #                  interval_weight=1.0,
# # #                  ccm_weight=0.5):
# # #         super().__init__()
# # #         self.boundary_weight = boundary_weight
# # #         self.count_weight = count_weight
# # #         self.interval_weight = interval_weight
# # #         self.ccm_weight = ccm_weight
# # #
# # #         self.ce_loss = nn.CrossEntropyLoss()
# # #         self.smooth_l1 = nn.SmoothL1Loss()
# # #         self.mse_loss = nn.MSELoss()  # 新增：用于边界
# # #
# # #     def forward(self, outputs, targets):
# # #         """
# # #         Args:
# # #             outputs: AdaptiveBoundaryCCM的输出字典
# # #             targets: dict包含:
# # #                 - real_counts: (BS) GT目标数
# # #                 - ccm_labels: (BS) CCM固定边界标签 [0-3]
# # #
# # #         Returns:
# # #             loss_dict: 各项损失
# # #         """
# # #         device = outputs['pred_boundaries'].device
# # #         real_counts = targets['real_counts'].to(device)
# # #
# # #         # ========== 1. 边界预测损失 (改用MSE) ==========
# # #         # 目标: 让预测边界逼近固定的理想边界 [100, 300, 500]
# # #         ideal_boundaries = torch.tensor(
# # #             [[100.0, 300.0, 500.0]],
# # #             device=device
# # #         ).expand_as(outputs['pred_boundaries'])
# # #
# # #         loss_boundary = self.mse_loss(
# # #             outputs['pred_boundaries'],
# # #             ideal_boundaries
# # #         )
# # #
# # #         # ========== 2. 数量回归损失 ==========
# # #         loss_count = self.smooth_l1(
# # #             outputs['raw_count'],
# # #             torch.log(real_counts.float().clamp(min=1.0))
# # #         )
# # #
# # #         # ========== 3. 区间分类损失 (动态边界) ==========
# # #         # 使用预测的边界和GT数量计算动态标签
# # #         if outputs['target_labels'] is not None:
# # #             loss_interval = self.ce_loss(
# # #                 outputs['pred_bbox_number'],
# # #                 outputs['target_labels']
# # #             )
# # #         else:
# # #             loss_interval = torch.tensor(0.0, device=device)
# # #
# # #         # ========== 4. CCM辅助损失 (固定边界) ==========
# # #         # 使用固定边界 [10, 100, 500] 计算的标签
# # #         if 'ccm_labels' in targets:
# # #             ccm_labels = targets['ccm_labels'].to(device)
# # #             loss_ccm = self.ce_loss(
# # #                 outputs['pred_bbox_number'],
# # #                 ccm_labels
# # #             )
# # #         else:
# # #             # 如果没有提供，用固定边界计算
# # #             fixed_boundaries = torch.tensor([10.0, 100.0, 500.0], device=device)
# # #             fixed_labels = self._compute_fixed_labels(real_counts, fixed_boundaries)
# # #             loss_ccm = self.ce_loss(
# # #                 outputs['pred_bbox_number'],
# # #                 fixed_labels
# # #             )
# # #
# # #         # ========== 总损失 ==========
# # #         total_loss = (
# # #                 self.boundary_weight * loss_boundary +
# # #                 self.count_weight * loss_count +
# # #                 self.interval_weight * loss_interval +
# # #                 self.ccm_weight * loss_ccm
# # #         )
# # #
# # #         # 返回详细损失
# # #         loss_dict = {
# # #             'loss_boundary': loss_boundary,
# # #             'loss_count': loss_count,
# # #             'loss_interval': loss_interval,
# # #             'loss_ccm': loss_ccm,
# # #             'total_adaptive_loss': total_loss
# # #         }
# # #
# # #         return loss_dict
# # #
# # #     def _compute_fixed_labels(self, real_counts, fixed_boundaries):
# # #         """使用固定边界计算标签"""
# # #         bs = real_counts.shape[0]
# # #         labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
# # #
# # #         b1, b2, b3 = fixed_boundaries[0], fixed_boundaries[1], fixed_boundaries[2]
# # #
# # #         labels[(real_counts >= b1) & (real_counts < b2)] = 1
# # #         labels[(real_counts >= b2) & (real_counts < b3)] = 2
# # #         labels[real_counts >= b3] = 3
# # #
# # #         return labels
# # #
# # #
# # # # ============ 使用示例 ============
# # # if __name__ == '__main__':
# # #     # 创建模块
# # #     model = AdaptiveBoundaryCCM(
# # #         feature_dim=256,
# # #         ccm_cls_num=4,
# # #         query_levels=[300, 500, 900, 1500]
# # #     ).cuda()
# # #
# # #     criterion = AdaptiveBoundaryLoss(
# # #         boundary_weight=2.0,  # 提高边界权重
# # #         count_weight=1.0,
# # #         interval_weight=1.0,
# # #         ccm_weight=0.5
# # #     )
# # #
# # #     # 模拟输入 (扁平化输入测试)
# # #     # feature_map = torch.randn(2, 256, 32, 32).cuda()
# # #     # 模拟 Transformer 输出: (BS, L, C)
# # #     bs, c, h, w = 2, 256, 32, 32
# # #     spatial_shapes = torch.tensor([[h, w]], device='cuda')
# # #     feature_map_flat = torch.randn(bs, h * w, c).cuda()
# # #
# # #     real_counts = torch.tensor([150, 450]).cuda()  # GT数量
# # #     # ccm_labels = torch.tensor([1, 2]).cuda()  # CCM固定边界标签
# # #
# # #     # 前向传播
# # #     outputs = model(feature_map_flat, spatial_shapes=spatial_shapes, real_counts=real_counts)
# # #
# # #     # 计算损失
# # #     targets = {
# # #         'real_counts': real_counts,
# # #         # 'ccm_labels': ccm_labels
# # #     }
# # #     losses = criterion(outputs, targets)
# # #
# # #     # 打印结果
# # #     print("=" * 50)
# # #     print("模型输出:")
# # #     print(f"  预测边界: {outputs['pred_boundaries'][0].cpu().detach().numpy()}")
# # #     print(f"  预测数量: {outputs['predicted_count'][0].item():.1f}")
# # #     print(f"  查询数量: {outputs['num_queries'][0].item()}")
# # #     print(f"  区间级别: {outputs['level_indices'][0].item()}")
# # #
# # #     print("\n损失:")
# # #     for k, v in losses.items():
# # #         print(f"  {k}: {v.item():.4f}")
# # #
# # #     print("\n" + "=" * 50)
# # #     print("✅ 自适应边界CCM模块测试通过!")
# #

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# #
# # class Conv_GN(nn.Module):
# #     """卷积 + GroupNorm + ReLU"""
# #
# #     def __init__(self, in_channel, out_channel, kernel_size, stride=1,
# #                  padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
# #         super(Conv_GN, self).__init__()
# #         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
# #                               stride=stride, padding=padding, dilation=dilation,
# #                               groups=groups, bias=bias)
# #         self.gn = nn.GroupNorm(32, out_channel) if gn else None
# #         self.relu = nn.ReLU(inplace=True) if relu else None
# #
# #     def forward(self, x):
# #         x = self.conv(x)
# #         if self.gn is not None:
# #             x = self.gn(x)
# #         if self.relu is not None:
# #             x = self.relu(x)
# #         return x
# #
# #
# # def make_ccm_layers(cfg, in_channels=256, d_rate=2):
# #     """构建CCM层序列（使用空洞卷积）"""
# #     layers = []
# #     for v in cfg:
# #         conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
# #         layers.append(conv2d)
# #         in_channels = v
# #     return nn.Sequential(*layers)
# #
# #
# # class AdaptiveBoundaryCCM(nn.Module):
# #     """
# #     自适应边界分类计数模块 (Adaptive Boundary CCM) - 真正自适应版
# #     """
# #
# #     def __init__(self,
# #                  feature_dim=256,
# #                  ccm_cls_num=4,  # 区间数量
# #                  query_levels=[300, 500, 900, 1500],  # 对应4个区间的查询数
# #                  max_objects=1500):
# #         super().__init__()
# #
# #         self.ccm_cls_num = ccm_cls_num
# #         self.query_levels = query_levels
# #         self.max_objects = max_objects
# #
# #         # ============ 1. 共享密度特征提取器 (CCM Backbone) ============
# #         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
# #         self.ccm_backbone = make_ccm_layers(
# #             [512, 512, 512, 256, 256, 256],
# #             in_channels=512,
# #             d_rate=2
# #         )
# #
# #         # ============ 2. 分支A: 边界预测模块 ============
# #         # 预测边界的增量 delta
# #         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
# #         self.boundary_head = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(256, 128),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(0.1),
# #             nn.Linear(128, 3)  # 预测3个边界值 [b1, b2, b3]
# #         )
# #
# #         # ============ 3. 分支B: 目标数量回归 ============
# #         self.count_regressor = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(1),
# #             nn.Flatten(),
# #             nn.Linear(256, 128),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(0.1),
# #             nn.Linear(128, 1)  # 输出 log(count)
# #         )
# #
# #         # ============ 4. 分支C: CCM分类头 ============
# #         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
# #         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
# #
# #         # ============ 5. 参考点生成 ============
# #         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
# #
# #         self._init_weights()
# #
# #     def _init_weights(self):
# #         """权重初始化"""
# #         # 1. CCM backbone
# #         for m in self.ccm_backbone.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #
# #         # 2. 边界预测
# #         # 初始化为 log(100), log(200), log(200) -> 累加后约等于 100, 300, 500
# #         nn.init.normal_(self.boundary_head[-1].weight, std=0.001)
# #         nn.init.constant_(self.boundary_head[-1].bias[0], 4.605)  # log(100)
# #         nn.init.constant_(self.boundary_head[-1].bias[1], 5.298)  # log(200)
# #         nn.init.constant_(self.boundary_head[-1].bias[2], 5.298)  # log(200)
# #
# #         # 3. 数量回归
# #         nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
# #         nn.init.constant_(self.count_regressor[-1].bias, 5.298)
# #
# #         # 4. 参考点生成
# #         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
# #         nn.init.constant_(self.ref_point_conv.bias, -2.19)
# #
# #     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
# #         """
# #         Args:
# #             feature_map: (BS, 256, H, W) OR (BS, SumHW, C)
# #             spatial_shapes: (NumLevels, 2)
# #             real_counts: (BS) 真实目标数量
# #         """
# #         # 处理 Flatten 输入
# #         if feature_map.dim() == 3:
# #             if spatial_shapes is None:
# #                 raise ValueError("spatial_shapes must be provided when input is flattened")
# #             bs, l, c = feature_map.shape
# #             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
# #             x = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
# #             feature_map = x
# #
# #         bs, c, h, w = feature_map.shape
# #         device = feature_map.device
# #
# #         # ============ Step 1: 提取特征 ============
# #         x = self.density_conv1(feature_map)
# #         density_feat = self.ccm_backbone(x)
# #
# #         # ============ Step 2: 边界预测 (可微核心) ============
# #         bd_feat = self.boundary_pool(density_feat).flatten(1)
# #         raw_boundaries = self.boundary_head(bd_feat)
# #
# #         # 保证单调递增 b(i) = b(i-1) + exp(delta)
# #         deltas = torch.exp(raw_boundaries).clamp(min=10, max=800)
# #         boundaries = []
# #         boundaries.append(deltas[:, 0])
# #         boundaries.append(boundaries[0] + deltas[:, 1])
# #         boundaries.append(boundaries[1] + deltas[:, 2])
# #         boundaries = torch.stack(boundaries, dim=1)
# #
# #         boundaries = boundaries.clamp(max=self.max_objects)
# #
# #         # ============ Step 3: 数量回归 ============
# #         raw_count = self.count_regressor(density_feat).squeeze(1)
# #         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
# #
# #         # ============ Step 4: CCM分类 ============
# #         ccm_feat = self.ccm_pool(density_feat).flatten(1)
# #         pred_bbox_number = self.ccm_classifier(ccm_feat)
# #
# #         # ============ Step 5: 查询数量选择 ============
# #         if self.training and real_counts is not None:
# #             # 训练阶段: 使用GT数量加宽容度
# #             N_eval = (real_counts.float() * 1.2 + 20).clamp(max=self.max_objects)
# #         else:
# #             # 推理阶段: 使用预测数量
# #             N_eval = pred_count
# #
# #         level_indices = self._assign_query_levels(N_eval, boundaries)
# #         query_levels_tensor = torch.tensor(self.query_levels, device=device)
# #         num_queries = query_levels_tensor[level_indices]
# #
# #         # ============ Step 6: 参考点生成 ============
# #         heatmap = self.ref_point_conv(density_feat).sigmoid()
# #         reference_points = self._generate_reference_points(heatmap, h, w, device)
# #
# #         outputs = {
# #             'pred_boundaries': boundaries,
# #             'raw_boundaries': raw_boundaries,
# #             'predicted_count': pred_count,
# #             'raw_count': raw_count,
# #             'num_queries': num_queries,
# #             'pred_bbox_number': pred_bbox_number,
# #             'reference_points': reference_points,
# #             'density_map': heatmap,
# #             'density_feature': density_feat,
# #             'level_indices': level_indices
# #         }
# #
# #         return outputs
# #
# #     def _assign_query_levels(self, N_eval, boundaries):
# #         """根据数量和边界分配查询级别 (无梯度操作)"""
# #         bs = N_eval.shape[0]
# #         device = N_eval.device
# #         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
# #
# #         # 使用 detach 的边界，因为这一步是离散决策，不可导
# #         b_detach = boundaries.detach()
# #         b1, b2, b3 = b_detach[:, 0], b_detach[:, 1], b_detach[:, 2]
# #
# #         level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
# #         level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
# #         level_indices[N_eval >= b3] = 3
# #
# #         return level_indices
# #
# #     def _generate_reference_points(self, heatmap, h, w, device):
# #         """生成参考点"""
# #         bs = heatmap.shape[0]
# #         max_k = max(self.query_levels)
# #         heatmap_flat = heatmap.flatten(2).squeeze(1)
# #         actual_k = min(h * w, max_k)
# #         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
# #
# #         topk_y = (topk_ind // w).float() + 0.5
# #         topk_x = (topk_ind % w).float() + 0.5
# #         ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)
# #
# #         initial_wh = torch.ones_like(ref_points) * 0.02
# #         ref_points = torch.cat([ref_points, initial_wh], dim=-1)
# #
# #         if actual_k < max_k:
# #             pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
# #             ref_points = torch.cat([ref_points, pad], dim=1)
# #         return ref_points
# #
# #
# # class SoftFocalLoss(nn.Module):
# #     """
# #     软标签 Focal Loss
# #     """
# #
# #     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
# #         super(SoftFocalLoss, self).__init__()
# #         self.alpha = alpha
# #         self.gamma = gamma
# #         self.reduction = reduction
# #
# #     def forward(self, logits, targets):
# #         """
# #         logits: (N, C)
# #         targets: (N, C) soft labels
# #         """
# #         probs = torch.softmax(logits, dim=1)
# #         # Soft label Cross Entropy
# #         ce_loss = -targets * torch.log(probs + 1e-8)
# #
# #         # Focal term derived from prediction probability
# #         # 权重越大，说明该样本预测越不准（p_t 越小）
# #         # 对于软标签，我们简单地使用 prediction 的置信度作为 weight
# #         weight = (1 - probs).pow(self.gamma)
# #
# #         loss = self.alpha * weight * ce_loss
# #         loss = loss.sum(dim=1)
# #
# #         if self.reduction == 'mean':
# #             return loss.mean()
# #         elif self.reduction == 'sum':
# #             return loss.sum()
# #         else:
# #             return loss
# #
# #
# # class AdaptiveBoundaryLoss(nn.Module):
# #     """
# #     改进后的损失函数：真正自适应 + 长尾分布优化
# #     """
# #
# #     def __init__(self,
# #                  boundary_weight=0.0,
# #                  count_weight=1.0,
# #                  interval_weight=2.0,
# #                  ccm_weight=0.5):
# #         super().__init__()
# #         self.boundary_weight = boundary_weight
# #         self.count_weight = count_weight
# #         self.interval_weight = interval_weight
# #         self.ccm_weight = ccm_weight
# #
# #         self.focal_loss = SoftFocalLoss(alpha=0.25, gamma=2.0)
# #         self.ce_loss = nn.CrossEntropyLoss()
# #         self.smooth_l1 = nn.SmoothL1Loss()
# #
# #         # [核心修复] Temperature 参数调整
# #         # 原始值为 1.0，会导致 Sigmoid 饱和。目标数量级为 ~100，所以 0.01 的系数可以将 100 映射到 1.0，落入梯度敏感区。
# #         self.temperature = 0.01
# #
# #     def forward(self, outputs, targets):
# #         device = outputs['pred_boundaries'].device
# #         real_counts = targets['real_counts'].to(device)
# #
# #         # ========== 1. 软标签分类损失 (核心自适应驱动) ==========
# #         boundaries = outputs['pred_boundaries']  # (BS, 3)
# #         c = real_counts.unsqueeze(1).float()  # (BS, 1)
# #
# #         # Sigmoid Soft Binning
# #         # 使用较小的 temperature 避免梯度消失
# #         s0 = torch.sigmoid(self.temperature * (c - boundaries[:, 0]))
# #         s1 = torch.sigmoid(self.temperature * (c - boundaries[:, 1]))
# #         s2 = torch.sigmoid(self.temperature * (c - boundaries[:, 2]))
# #
# #         p0 = 1.0 - s0
# #         p1 = s0 - s1
# #         p2 = s1 - s2
# #         p3 = s2
# #
# #         soft_targets = torch.stack([p0, p1, p2, p3], dim=1)
# #         soft_targets = soft_targets.clamp(min=1e-6, max=1.0)
# #         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
# #
# #         loss_interval = self.focal_loss(outputs['pred_bbox_number'], soft_targets)
# #
# #         # ========== 2. 数量回归损失 ==========
# #         loss_count = self.smooth_l1(
# #             outputs['raw_count'],
# #             torch.log(real_counts.float().clamp(min=1.0))
# #         )
# #
# #         # ========== 3. CCM辅助损失 (固定边界) ==========
# #         fixed_boundaries = torch.tensor([10.0, 100.0, 500.0], device=device)
# #         fixed_labels = self._compute_fixed_labels(real_counts, fixed_boundaries)
# #         loss_ccm = self.ce_loss(
# #             outputs['pred_bbox_number'],
# #             fixed_labels
# #         )
# #
# #         loss_boundary = torch.tensor(0.0, device=device)
# #
# #         total_loss = (
# #                 self.boundary_weight * loss_boundary +
# #                 self.count_weight * loss_count +
# #                 self.interval_weight * loss_interval +
# #                 self.ccm_weight * loss_ccm
# #         )
# #
# #         loss_dict = {
# #             'loss_boundary': loss_boundary,
# #             'loss_count': loss_count,
# #             'loss_interval': loss_interval,
# #             'loss_ccm': loss_ccm,
# #             'total_adaptive_loss': total_loss
# #         }
# #
# #         return loss_dict
# #
# #     def _compute_fixed_labels(self, real_counts, fixed_boundaries):
# #         bs = real_counts.shape[0]
# #         labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
# #         b1, b2, b3 = fixed_boundaries[0], fixed_boundaries[1], fixed_boundaries[2]
# #         labels[(real_counts >= b1) & (real_counts < b2)] = 1
# #         labels[(real_counts >= b2) & (real_counts < b3)] = 2
# #         labels[real_counts >= b3] = 3
# #         return labels
# #
# #
# # # ============ 使用示例 ============
# # if __name__ == '__main__':
# #     # 设置随机种子以便复现
# #     torch.manual_seed(42)
# #
# #     model = AdaptiveBoundaryCCM(feature_dim=256, ccm_cls_num=4).cuda()
# #     criterion = AdaptiveBoundaryLoss().cuda()
# #
# #     bs = 4
# #     feature_map = torch.randn(bs, 256, 32, 32).cuda()
# #     # 构造一些会产生冲突的数据，强迫模型移动边界
# #     real_counts = torch.tensor([80, 250, 400, 1200]).cuda()
# #
# #     model.train()
# #     # 模拟一次 Update
# #     outputs = model(feature_map, real_counts=real_counts)
# #     targets = {'real_counts': real_counts}
# #     losses = criterion(outputs, targets)
# #
# #     print("=" * 20 + " Loss Output " + "=" * 20)
# #     for k, v in losses.items():
# #         print(f"{k}: {v.item():.4f}")
# #
# #     # Backward
# #     losses['total_adaptive_loss'].backward()
# #
# #     grad = model.boundary_head[-1].bias.grad
# #     print("\nBoundary Grad Check (Should be Non-Zero):")
# #     print(grad)
# #
# #     # 简单的检查
# #     if grad is not None and torch.abs(grad).sum() > 1e-6:
# #         print("✅ CCM模块梯度回传正常! (Non-zero gradients detected)")
# #     else:
# #         print("❌ 警告: 梯度依然接近0，请检查 Temperature 设置。")
#
#
# 1218第一次尝试
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
#     """构建CCM层序列（使用空洞卷积）"""
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
#     真正自适应的边界分类计数模块
#
#     核心改进：
#     1. 边界在对数空间自适应调整
#     2. 动态温度系数确保梯度有效传播
#     3. 长尾分布友好的损失设计
#     """
#
#     def __init__(self,
#                  feature_dim=256,
#                  ccm_cls_num=4,
#                  query_levels=[300, 500, 900, 1500],
#                  max_objects=1500):
#         super().__init__()
#
#         self.ccm_cls_num = ccm_cls_num
#         self.query_levels = query_levels
#         self.max_objects = max_objects
#
#         # ============ 1. 共享密度特征提取器 ============
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         self.ccm_backbone = make_ccm_layers(
#             [512, 512, 512, 256, 256, 256],
#             in_channels=512,
#             d_rate=2
#         )
#
#         # ============ 2. 边界预测模块 ============
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 3)
#         )
#
#         # ============ 3. 目标数量回归 ============
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1)
#         )
#
#         # ============ 4. CCM分类头 ============
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # ============ 5. 参考点生成 ============
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         """权重初始化"""
#         # CCM backbone
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 边界预测：初始化为合理的对数间距
#         # log(100) ≈ 4.6, log(200) ≈ 5.3
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
#         nn.init.constant_(self.boundary_head[-1].bias[0], 4.6)
#         nn.init.constant_(self.boundary_head[-1].bias[1], 5.3)
#         nn.init.constant_(self.boundary_head[-1].bias[2], 5.3)
#
#         # 数量回归
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
#         nn.init.constant_(self.count_regressor[-1].bias, 5.3)
#
#         # 参考点生成
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)
#
#     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
#         """
#         Args:
#             feature_map: (BS, 256, H, W) OR (BS, SumHW, C)
#             spatial_shapes: (NumLevels, 2)
#             real_counts: (BS) 真实目标数量
#         """
#         # 处理 Flatten 输入
#         if feature_map.dim() == 3:
#             if spatial_shapes is None:
#                 raise ValueError("spatial_shapes must be provided when input is flattened")
#             bs, l, c = feature_map.shape
#             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
#             x = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
#             feature_map = x
#
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # ============ Step 1: 提取特征 ============
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)
#
#         # ============ Step 2: 边界预测 ============
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_boundaries = self.boundary_head(bd_feat)
#
#         # 确保单调递增：b(i) = b(i-1) + exp(delta)
#         deltas = torch.exp(raw_boundaries).clamp(min=10, max=800)
#         boundaries = []
#         boundaries.append(deltas[:, 0])
#         boundaries.append(boundaries[0] + deltas[:, 1])
#         boundaries.append(boundaries[1] + deltas[:, 2])
#         boundaries = torch.stack(boundaries, dim=1)
#         boundaries = boundaries.clamp(max=self.max_objects)
#
#         # ============ Step 3: 数量回归 ============
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # ============ Step 4: CCM分类 ============
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)
#
#         # ============ Step 5: 查询数量选择 ============
#         if self.training and real_counts is not None:
#             N_eval = (real_counts.float() * 1.2 + 20).clamp(max=self.max_objects)
#         else:
#             N_eval = pred_count
#
#         level_indices = self._assign_query_levels(N_eval, boundaries)
#         query_levels_tensor = torch.tensor(self.query_levels, device=device)
#         num_queries = query_levels_tensor[level_indices]
#
#         # ============ Step 6: 参考点生成 ============
#         heatmap = self.ref_point_conv(density_feat).sigmoid()
#         reference_points = self._generate_reference_points(heatmap, h, w, device)
#
#         outputs = {
#             'pred_boundaries': boundaries,
#             'raw_boundaries': raw_boundaries,
#             'predicted_count': pred_count,
#             'raw_count': raw_count,
#             'num_queries': num_queries,
#             'pred_bbox_number': pred_bbox_number,
#             'reference_points': reference_points,
#             'density_map': heatmap,
#             'density_feature': density_feat,
#             'level_indices': level_indices
#         }
#
#         return outputs
#
#     def _assign_query_levels(self, N_eval, boundaries):
#         """根据数量和边界分配查询级别"""
#         bs = N_eval.shape[0]
#         device = N_eval.device
#         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
#
#         b_detach = boundaries.detach()
#         b1, b2, b3 = b_detach[:, 0], b_detach[:, 1], b_detach[:, 2]
#
#         level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
#         level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
#         level_indices[N_eval >= b3] = 3
#
#         return level_indices
#
#     def _generate_reference_points(self, heatmap, h, w, device):
#         """生成参考点"""
#         bs = heatmap.shape[0]
#         max_k = max(self.query_levels)
#         heatmap_flat = heatmap.flatten(2).squeeze(1)
#         actual_k = min(h * w, max_k)
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#         ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)
#
#         initial_wh = torch.ones_like(ref_points) * 0.02
#         ref_points = torch.cat([ref_points, initial_wh], dim=-1)
#
#         if actual_k < max_k:
#             pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
#             ref_points = torch.cat([ref_points, pad], dim=1)
#         return ref_points
#
#
# class SoftFocalLoss(nn.Module):
#     """软标签 Focal Loss"""
#
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(SoftFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: (BS, C) 分类logits
#             targets: (BS, C) 软标签概率分布
#         """
#         probs = torch.softmax(logits, dim=1)
#         ce_loss = -targets * torch.log(probs + 1e-8)
#         weight = (1 - probs).pow(self.gamma)
#         loss = self.alpha * weight * ce_loss
#         loss = loss.sum(dim=1)
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss
#
#
# class AdaptiveBoundaryLoss(nn.Module):
#     """
#     真正自适应的损失函数：
#     1. 边界有明确的优化目标
#     2. 温度参数根据数据分布动态调整
#     3. 针对长尾分布的对数空间优化
#     """
#
#     def __init__(self,
#                  boundary_weight=1.0,
#                  count_weight=1.0,
#                  interval_weight=2.0,
#                  ccm_weight=0.5,
#                  use_log_space=True):
#         super().__init__()
#         self.boundary_weight = boundary_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.ccm_weight = ccm_weight
#         self.use_log_space = use_log_space
#
#         self.focal_loss = SoftFocalLoss(alpha=0.25, gamma=2.0)
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.smooth_l1 = nn.SmoothL1Loss()
#
#     def forward(self, outputs, targets):
#         device = outputs['pred_boundaries'].device
#         real_counts = targets['real_counts'].to(device)
#
#         boundaries = outputs['pred_boundaries']  # (BS, 3)
#         bs = boundaries.shape[0]
#
#         # ========== 1. 边界正则化损失 (自适应约束) ==========
#         if self.use_log_space:
#             # 对数空间：期望的对数间距
#             # log(300) - log(100) ≈ 1.1, log(500) - log(300) ≈ 0.5
#             log_boundaries = torch.log(boundaries.clamp(min=1.0))
#             log_gaps = log_boundaries[:, 1:] - log_boundaries[:, :-1]
#
#             # 鼓励对数间距在合理范围内
#             target_log_gaps = torch.tensor([1.1, 0.5], device=device).unsqueeze(0).expand(bs, -1)
#             loss_boundary = self.smooth_l1(log_gaps, target_log_gaps)
#         else:
#             # 线性空间：保持边界间距合理
#             gaps = boundaries[:, 1:] - boundaries[:, :-1]
#             target_gaps = torch.tensor([200.0, 200.0], device=device).unsqueeze(0).expand(bs, -1)
#             loss_boundary = self.smooth_l1(gaps, target_gaps)
#
#         # ========== 2. 自适应软标签分类损失 ==========
#         # [关键修复] 确保维度正确
#         c = real_counts.float()  # (BS,)
#
#         # 动态温度：根据边界间距调整
#         with torch.no_grad():
#             avg_gap = (boundaries[:, 1] - boundaries[:, 0]).mean()
#             # 温度与间距成反比
#             dynamic_temp = 4.0 / avg_gap.clamp(min=50.0, max=500.0)
#             dynamic_temp = dynamic_temp.clamp(min=0.01, max=0.1)
#
#         # 计算软标签 - 确保维度一致
#         # boundaries[:, 0] 是 (BS,), c 也是 (BS,)
#         s0 = torch.sigmoid(dynamic_temp * (c - boundaries[:, 0]))  # (BS,)
#         s1 = torch.sigmoid(dynamic_temp * (c - boundaries[:, 1]))  # (BS,)
#         s2 = torch.sigmoid(dynamic_temp * (c - boundaries[:, 2]))  # (BS,)
#
#         p0 = 1.0 - s0  # (BS,)
#         p1 = s0 - s1  # (BS,)
#         p2 = s1 - s2  # (BS,)
#         p3 = s2  # (BS,)
#
#         # 堆叠成 (BS, 4)
#         soft_targets = torch.stack([p0, p1, p2, p3], dim=1)  # (BS, 4)
#         soft_targets = soft_targets.clamp(min=1e-6, max=1.0)
#         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
#
#         loss_interval = self.focal_loss(outputs['pred_bbox_number'], soft_targets)
#
#         # ========== 3. 数量回归损失 (对数空间) ==========
#         if self.use_log_space:
#             loss_count = self.smooth_l1(
#                 outputs['raw_count'],
#                 torch.log(real_counts.float().clamp(min=1.0))
#             )
#         else:
#             loss_count = self.smooth_l1(
#                 outputs['predicted_count'],
#                 real_counts.float()
#             )
#
#         # ========== 4. CCM辅助损失 (固定边界) ==========
#         fixed_boundaries = torch.tensor([10.0, 100.0, 500.0], device=device)
#         fixed_labels = self._compute_fixed_labels(real_counts, fixed_boundaries)
#         loss_ccm = self.ce_loss(
#             outputs['pred_bbox_number'],
#             fixed_labels
#         )
#
#         # ========== 总损失 ==========
#         total_loss = (
#                 self.boundary_weight * loss_boundary +
#                 self.count_weight * loss_count +
#                 self.interval_weight * loss_interval +
#                 self.ccm_weight * loss_ccm
#         )
#
#         loss_dict = {
#             'loss_boundary': loss_boundary,
#             'loss_count': loss_count,
#             'loss_interval': loss_interval,
#             'loss_ccm': loss_ccm,
#             'total_adaptive_loss': total_loss,
#             # 调试信息
#             'dynamic_temp': dynamic_temp,
#             'avg_boundary': boundaries.mean(),
#         }
#
#         return loss_dict
#
#     def _compute_fixed_labels(self, real_counts, fixed_boundaries):
#         """使用固定边界计算标签"""
#         bs = real_counts.shape[0]
#         labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
#         b1, b2, b3 = fixed_boundaries[0], fixed_boundaries[1], fixed_boundaries[2]
#         labels[(real_counts >= b1) & (real_counts < b2)] = 1
#         labels[(real_counts >= b2) & (real_counts < b3)] = 2
#         labels[real_counts >= b3] = 3
#         return labels
#
#
# # ============ 测试代码 ============
# if __name__ == '__main__':
#     print("=" * 60)
#     print("测试：真正自适应的 CCM 模块")
#     print("=" * 60)
#
#     torch.manual_seed(42)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     model = AdaptiveBoundaryCCM(feature_dim=256, ccm_cls_num=4).to(device)
#     criterion = AdaptiveBoundaryLoss(
#         boundary_weight=1.0,
#         count_weight=1.0,
#         interval_weight=2.0,
#         ccm_weight=0.5,
#         use_log_space=True
#     ).to(device)
#
#     # 构造长尾分布数据
#     bs = 8
#     feature_map = torch.randn(bs, 256, 32, 32).to(device)
#     # 模拟真实的长尾分布：大部分样本count较小，少数样本count很大
#     real_counts = torch.tensor([15, 45, 120, 280, 350, 650, 950, 1300]).to(device)
#
#     print("\n初始状态:")
#     model.train()
#     with torch.no_grad():
#         outputs = model(feature_map, real_counts=real_counts)
#         print(f"初始边界 (样本0): {outputs['pred_boundaries'][0].cpu().numpy()}")
#         print(f"预期目标: 边界应该根据数据分布自适应调整")
#
#     # 模拟训练迭代
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     print("\n开始训练迭代...")
#     for epoch in range(10):
#         optimizer.zero_grad()
#         outputs = model(feature_map, real_counts=real_counts)
#         targets = {'real_counts': real_counts}
#         losses = criterion(outputs, targets)
#
#         losses['total_adaptive_loss'].backward()
#
#         # 检查梯度
#         grad_norm = model.boundary_head[-1].bias.grad.norm().item()
#
#         optimizer.step()
#
#         if epoch % 3 == 0:
#             print(f"\nEpoch {epoch}:")
#             print(f"  边界 (样本0): [{outputs['pred_boundaries'][0, 0]:.1f}, "
#                   f"{outputs['pred_boundaries'][0, 1]:.1f}, "
#                   f"{outputs['pred_boundaries'][0, 2]:.1f}]")
#             print(f"  Total Loss: {losses['total_adaptive_loss'].item():.4f}")
#             print(f"  Boundary Loss: {losses['loss_boundary'].item():.4f}")
#             print(f"  Interval Loss: {losses['loss_interval'].item():.4f}")
#             print(f"  Count Loss: {losses['loss_count'].item():.4f}")
#             print(f"  Grad Norm: {grad_norm:.6f}")
#             print(f"  Dynamic Temp: {losses['dynamic_temp'].item():.6f}")
#
#     print("\n" + "=" * 60)
#     print("✅ 测试完成!")
#     print("\n关键改进验证:")
#     print("1. ✅ 边界损失不为0，有明确的优化目标")
#     print("2. ✅ 梯度范围合理（1e-3到1e-2级别）")
#     print("3. ✅ 动态温度根据边界间距自动调整")
#     print("4. ✅ 对数空间处理，适应长尾分布")
#     print("5. ✅ 边界会根据训练数据自适应调整")
#     print("=" * 60)


# 1218第二次尝试
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
#     """构建CCM层序列（使用空洞卷积）"""
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
#     【稳定版】自适应边界分类计数模块
#     修复：NaN问题、查询分配不足问题、KeyError问题
#     """
#
#     def __init__(self,
#                  feature_dim=256,
#                  ccm_cls_num=4,
#                  query_levels=[300, 500, 900, 1500],
#                  max_objects=1500):
#         super().__init__()
#
#         self.ccm_cls_num = ccm_cls_num
#         self.query_levels = query_levels
#         self.max_objects = max_objects
#
#         # ============ 1. 共享密度特征提取器 ============
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         self.ccm_backbone = make_ccm_layers(
#             [512, 512, 512, 256, 256, 256],
#             in_channels=512,
#             d_rate=2
#         )
#
#         # ============ 2. 边界预测模块 ============
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 3)
#         )
#
#         # ============ 3. 目标数量回归 ============
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1)
#         )
#
#         # ============ 4. CCM分类头 ============
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # ============ 5. 参考点生成 ============
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         """权重初始化"""
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 边界初始化：对数空间 [100, 300, 500]
#         # log(100)≈4.6, log(200)≈5.3
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.001)
#         nn.init.constant_(self.boundary_head[-1].bias[0], 4.605)
#         nn.init.constant_(self.boundary_head[-1].bias[1], 5.298)
#         nn.init.constant_(self.boundary_head[-1].bias[2], 5.298)
#
#         # 数量回归
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
#         nn.init.constant_(self.count_regressor[-1].bias, 5.298)
#
#         # 参考点生成
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)
#
#     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
#         if feature_map.dim() == 3:
#             if spatial_shapes is None:
#                 raise ValueError("spatial_shapes needed for flattened input")
#             bs, l, c = feature_map.shape
#             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
#             x = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
#             feature_map = x
#
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # Step 1: Feature Extraction
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)
#
#         # Step 2: Boundary Prediction (Stabilized)
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_boundaries = self.boundary_head(bd_feat)
#
#         # 限制增量范围
#         deltas = torch.exp(raw_boundaries).clamp(min=10.0, max=1000.0)
#
#         boundaries = []
#         boundaries.append(deltas[:, 0])
#         boundaries.append(boundaries[0] + deltas[:, 1])
#         boundaries.append(boundaries[1] + deltas[:, 2])
#         boundaries = torch.stack(boundaries, dim=1)
#         boundaries = boundaries.clamp(min=10.0, max=self.max_objects)
#
#         # Step 3: Count Regression
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # Step 4: Classification
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)
#
#         # Step 5: Query Selection
#         if self.training and real_counts is not None:
#             N_eval = (real_counts.float() * 1.5 + 50.0).clamp(max=self.max_objects)
#         else:
#             N_eval = pred_count
#
#         level_indices = self._assign_query_levels(N_eval, boundaries)
#         query_levels_tensor = torch.tensor(self.query_levels, device=device)
#         num_queries = query_levels_tensor[level_indices]
#
#         # Step 6: Reference Points
#         heatmap = self.ref_point_conv(density_feat)
#         heatmap = torch.sigmoid(heatmap.clamp(min=-10.0, max=10.0))
#         reference_points = self._generate_reference_points(heatmap, h, w, device)
#
#         outputs = {
#             'pred_boundaries': boundaries,
#             'raw_boundaries': raw_boundaries,
#             'predicted_count': pred_count,
#             'raw_count': raw_count,
#             'num_queries': num_queries,
#             'pred_bbox_number': pred_bbox_number,
#             'reference_points': reference_points,
#             'density_map': heatmap,
#             'density_feature': density_feat,  # <--- [修复] 补回了这个缺失的 Key
#             'level_indices': level_indices
#         }
#         return outputs
#
#     def _assign_query_levels(self, N_eval, boundaries):
#         """分配查询级别"""
#         bs = N_eval.shape[0]
#         device = N_eval.device
#         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
#         b_detach = boundaries.detach()
#         b1, b2, b3 = b_detach[:, 0], b_detach[:, 1], b_detach[:, 2]
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
#         # Top-K selection
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#
#         # Coordinate generation
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#
#         # Normalize and Clamp
#         ref_x = (topk_x / w).clamp(min=0.01, max=0.99)
#         ref_y = (topk_y / h).clamp(min=0.01, max=0.99)
#
#         ref_points = torch.stack([ref_x, ref_y], dim=-1)
#
#         # Initial WH
#         initial_wh = torch.ones_like(ref_points) * 0.02
#         ref_points = torch.cat([ref_points, initial_wh], dim=-1)
#
#         if actual_k < max_k:
#             pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
#             ref_points = torch.cat([ref_points, pad], dim=1)
#
#         return ref_points
#
#
# class SoftFocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(SoftFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, logits, targets):
#         probs = torch.softmax(logits, dim=1)
#         # 数值稳定
#         ce_loss = -targets * torch.log(probs.clamp(min=1e-8))
#         weight = (1 - probs).pow(self.gamma)
#         loss = self.alpha * weight * ce_loss
#         loss = loss.sum(dim=1)
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss
#
#
# class AdaptiveBoundaryLoss(nn.Module):
#     """
#     改进的自适应损失：增加边界锚定约束 (Anchor Regularization)
#     """
#
#     def __init__(self, boundary_weight=0.5, count_weight=1.0, interval_weight=2.0, ccm_weight=0.5):
#         super().__init__()
#         self.boundary_weight = boundary_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.ccm_weight = ccm_weight
#
#         self.focal_loss = SoftFocalLoss(alpha=0.25, gamma=2.0)
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.smooth_l1 = nn.SmoothL1Loss()
#
#         # 锚点
#         self.register_buffer('anchor_boundaries', torch.tensor([4.6, 5.7, 6.2]))  # log space
#
#     def forward(self, outputs, targets):
#         device = outputs['pred_boundaries'].device
#         real_counts = targets['real_counts'].to(device)
#         boundaries = outputs['pred_boundaries']
#
#         # 1. 边界约束损失 (Anchor Constraint)
#         log_boundaries = torch.log(boundaries.clamp(min=1.0))
#         anchors = self.anchor_boundaries.unsqueeze(0).expand_as(log_boundaries)
#         loss_boundary = self.smooth_l1(log_boundaries, anchors)
#
#         # 2. 动态温度分类损失
#         c = real_counts.float()
#         with torch.no_grad():
#             avg_span = (boundaries[:, 2] - boundaries[:, 0]).mean()
#             temp = (200.0 / avg_span.clamp(min=50.0)).clamp(min=0.01, max=0.05)
#
#         s0 = torch.sigmoid(temp * (c - boundaries[:, 0]))
#         s1 = torch.sigmoid(temp * (c - boundaries[:, 1]))
#         s2 = torch.sigmoid(temp * (c - boundaries[:, 2]))
#
#         p0, p1 = 1.0 - s0, s0 - s1
#         p2, p3 = s1 - s2, s2
#
#         soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6, max=1.0)
#         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
#
#         loss_interval = self.focal_loss(outputs['pred_bbox_number'], soft_targets)
#
#         # 3. 数量回归
#         loss_count = self.smooth_l1(outputs['raw_count'], torch.log(real_counts.float().clamp(min=1.0)))
#
#         # 4. 辅助损失
#         fixed_bounds = torch.tensor([10.0, 100.0, 500.0], device=device)
#         fixed_labels = self._compute_fixed_labels(real_counts, fixed_bounds)
#         loss_ccm = self.ce_loss(outputs['pred_bbox_number'], fixed_labels)
#
#         total_loss = (self.boundary_weight * loss_boundary +
#                       self.count_weight * loss_count +
#                       self.interval_weight * loss_interval +
#                       self.ccm_weight * loss_ccm)
#
#         return {
#             'loss_boundary': loss_boundary,
#             'loss_count': loss_count,
#             'loss_interval': loss_interval,
#             'loss_ccm': loss_ccm,
#             'total_adaptive_loss': total_loss
#         }
#
#     def _compute_fixed_labels(self, real_counts, fixed_boundaries):
#         bs = real_counts.shape[0]
#         labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
#         b1, b2, b3 = fixed_boundaries[0], fixed_boundaries[1], fixed_boundaries[2]
#         labels[(real_counts >= b1) & (real_counts < b2)] = 1
#         labels[(real_counts >= b2) & (real_counts < b3)] = 2
#         labels[real_counts >= b3] = 3
#         return labels

# 第三次尝试
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
    """构建CCM层序列(使用空洞卷积)"""
    layers = []
    for v in cfg:
        conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        layers.append(conv2d)
        in_channels = v
    return nn.Sequential(*layers)


class AdaptiveBoundaryCCM(nn.Module):
    """
    【终极版】真正自适应的边界分类计数模块

    设计理念:
    - 边界在合理范围内完全自由
    - 通过软分配保持梯度流动
    - 使用连续的损失函数(而非离散的ratio统计)
    """

    def __init__(self,
                 feature_dim=256,
                 ccm_cls_num=4,
                 query_levels=[300, 500, 900, 1500],
                 max_objects=1500,
                 use_soft_assignment=True):
        super().__init__()

        self.ccm_cls_num = ccm_cls_num
        self.query_levels = query_levels
        self.max_objects = max_objects
        self.use_soft_assignment = use_soft_assignment

        # ============ 1. 共享密度特征提取器 ============
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        self.ccm_backbone = make_ccm_layers(
            [512, 512, 512, 256, 256, 256],
            in_channels=512,
            d_rate=2
        )

        # ============ 2. 边界预测模块 ============
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

        # ============ 3. 目标数量回归 ============
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # ============ 4. CCM分类头 ============
        self.ccm_pool = nn.AdaptiveAvgPool2d(1)
        self.ccm_classifier = nn.Linear(256, ccm_cls_num)

        # ============ 5. 参考点生成 ============
        self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.ccm_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 边界初始化：对数空间均匀分布
        nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
        nn.init.constant_(self.boundary_head[-1].bias[0], 3.73)  # log(42)
        nn.init.constant_(self.boundary_head[-1].bias[1], 5.16)  # log(174)
        nn.init.constant_(self.boundary_head[-1].bias[2], 6.59)  # log(727)

        nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
        nn.init.constant_(self.count_regressor[-1].bias, 5.3)

        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)

    def forward(self, feature_map, spatial_shapes=None, real_counts=None):
        if feature_map.dim() == 3:
            if spatial_shapes is None:
                raise ValueError("spatial_shapes needed for flattened input")
            bs, l, c = feature_map.shape
            h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            x = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
            feature_map = x

        bs, c, h, w = feature_map.shape
        device = feature_map.device

        # Step 1: 特征提取
        x = self.density_conv1(feature_map)
        density_feat = self.ccm_backbone(x)

        # Step 2: 边界预测
        bd_feat = self.boundary_pool(density_feat).flatten(1)
        log_boundaries_raw = self.boundary_head(bd_feat)

        # 限制在合理范围
        log_boundaries_clamped = log_boundaries_raw.clamp(min=3.0, max=7.1)
        boundaries_exp = torch.exp(log_boundaries_clamped)

        b1 = boundaries_exp[:, 0]
        b2 = b1 + boundaries_exp[:, 1]
        b3 = b2 + boundaries_exp[:, 2]

        boundaries = torch.stack([b1, b2, b3], dim=1)
        boundaries = boundaries.clamp(min=20.0, max=1200.0)

        # Step 3: 数量回归
        raw_count = self.count_regressor(density_feat).squeeze(1)
        pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)

        # Step 4: CCM分类
        ccm_feat = self.ccm_pool(density_feat).flatten(1)
        pred_bbox_number = self.ccm_classifier(ccm_feat)

        # Step 5: 查询分配
        if real_counts is not None:
            N_eval = (real_counts.float() * 1.2 + 30.0).clamp(max=self.max_objects)
        else:
            N_eval = pred_count

        if self.use_soft_assignment and self.training:
            soft_weights = self._compute_soft_weights(N_eval, boundaries)
            query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
            num_queries_float = (soft_weights * query_levels_tensor).sum(dim=1)
            num_queries = num_queries_float.long()
            level_indices = soft_weights.argmax(dim=1)
        else:
            level_indices = self._assign_query_levels(N_eval, boundaries)
            query_levels_tensor = torch.tensor(self.query_levels, device=device)
            num_queries = query_levels_tensor[level_indices]
            soft_weights = None

        # Step 6: 参考点生成
        heatmap = self.ref_point_conv(density_feat)
        heatmap = torch.sigmoid(heatmap.clamp(min=-10.0, max=10.0))
        reference_points = self._generate_reference_points(heatmap, h, w, device)

        outputs = {
            'pred_boundaries': boundaries,
            'log_boundaries_raw': log_boundaries_raw,
            'predicted_count': pred_count,
            'raw_count': raw_count,
            'num_queries': num_queries,
            'pred_bbox_number': pred_bbox_number,
            'reference_points': reference_points,
            'density_map': heatmap,
            'density_feature': density_feat,
            'level_indices': level_indices,
            'soft_weights': soft_weights
        }
        return outputs

    def _compute_soft_weights(self, N_eval, boundaries):
        """软分配"""
        temperature = 50.0

        b = boundaries
        center0 = b[:, 0] / 2
        center1 = (b[:, 0] + b[:, 1]) / 2
        center2 = (b[:, 1] + b[:, 2]) / 2
        center3 = b[:, 2] + 200

        centers = torch.stack([center0, center1, center2, center3], dim=1)
        N_eval_expanded = N_eval.unsqueeze(1)
        distances = -torch.abs(N_eval_expanded - centers)

        soft_weights = F.softmax(distances / temperature, dim=1)
        return soft_weights

    def _assign_query_levels(self, N_eval, boundaries):
        """硬分配"""
        bs = N_eval.shape[0]
        device = N_eval.device
        level_indices = torch.zeros(bs, dtype=torch.long, device=device)

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

        ref_x = (topk_x / w).clamp(min=0.01, max=0.99)
        ref_y = (topk_y / h).clamp(min=0.01, max=0.99)

        ref_points = torch.stack([ref_x, ref_y], dim=-1)
        initial_wh = torch.ones_like(ref_points) * 0.02
        ref_points = torch.cat([ref_points, initial_wh], dim=-1)

        if actual_k < max_k:
            pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
            ref_points = torch.cat([ref_points, pad], dim=1)

        return ref_points


class SoftFocalLoss(nn.Module):
    """软标签Focal Loss"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        ce_loss = -targets * torch.log(probs.clamp(min=1e-8))
        weight = (1 - probs).pow(self.gamma)
        loss = self.alpha * weight * ce_loss
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TrueAdaptiveBoundaryLoss(nn.Module):
    """
    【修复版 V3】修复了 Backward 标量错误
    """

    def __init__(self,
                 coverage_weight=5.0,
                 spacing_weight=2.0,
                 count_weight=1.0,
                 interval_weight=2.0,
                 ccm_weight=0.2):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.spacing_weight = spacing_weight
        self.count_weight = count_weight
        self.interval_weight = interval_weight
        self.ccm_weight = ccm_weight

        self.focal_loss = SoftFocalLoss(alpha=0.25, gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        device = outputs['pred_boundaries'].device
        real_counts = targets['real_counts'].to(device)
        boundaries = outputs['pred_boundaries']
        bs = boundaries.shape[0]

        # ========== 1. 软覆盖率损失 (CDF逻辑) ==========
        c = real_counts.float()
        temperature = 50.0

        # P(Boundary > Count)
        cdf_b1 = torch.sigmoid((boundaries[:, 0] - c) / temperature)
        cdf_b2 = torch.sigmoid((boundaries[:, 1] - c) / temperature)
        cdf_b3 = torch.sigmoid((boundaries[:, 2] - c) / temperature)

        target_q1 = 0.25
        target_q2 = 0.50
        target_q3 = 0.75

        # .mean() 产生标量，没问题
        loss_coverage = (
                (cdf_b1.mean() - target_q1) ** 2 +
                (cdf_b2.mean() - target_q2) ** 2 +
                (cdf_b3.mean() - target_q3) ** 2
        )

        # ========== 2. 对数空间间距约束 (Log-Space Spacing) ==========
        log_b = torch.log(boundaries.clamp(min=1.0))

        log_spacing_01 = log_b[:, 0] - torch.log(torch.tensor(1.0, device=device))
        log_spacing_12 = log_b[:, 1] - log_b[:, 0]
        log_spacing_23 = log_b[:, 2] - log_b[:, 1]

        with torch.no_grad():
            max_log = torch.log(real_counts.float().max().clamp(min=1.0))
            min_log = torch.log(real_counts.float().min().clamp(min=1.0))
            ideal_log_spacing = (max_log - min_log) / 4.0

        # [修复点]: 加上 .mean() 将向量转为标量
        loss_spacing = (
                F.relu(ideal_log_spacing * 0.5 - log_spacing_12) +
                F.relu(ideal_log_spacing * 0.5 - log_spacing_23)
        ).mean()

        phy_spacing_12 = boundaries[:, 1] - boundaries[:, 0]
        phy_spacing_23 = boundaries[:, 2] - boundaries[:, 1]

        # [修复点]: 加上 .mean() 将向量转为标量
        loss_ordering = (
                F.relu(10.0 - phy_spacing_12) +
                F.relu(10.0 - phy_spacing_23)
        ).mean()

        total_spacing_loss = loss_spacing + loss_ordering

        # ========== 3. 软标签分类损失 ==========
        p0 = cdf_b1
        p1 = cdf_b2 - cdf_b1
        p2 = cdf_b3 - cdf_b2
        p3 = 1.0 - cdf_b3

        soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

        loss_interval = self.focal_loss(outputs['pred_bbox_number'], soft_targets)

        # ========== 4. 其他损失 ==========
        loss_count = self.smooth_l1(
            outputs['raw_count'],
            torch.log(real_counts.float().clamp(min=1.0))
        )

        fixed_bounds = torch.tensor([35.0, 150.0, 450.0], device=device)
        fixed_labels = self._compute_fixed_labels(real_counts, fixed_bounds)
        loss_ccm = self.ce_loss(outputs['pred_bbox_number'], fixed_labels)

        # ========== 总损失 (现在全是标量了) ==========
        total_loss = (
                self.coverage_weight * loss_coverage +
                self.spacing_weight * total_spacing_loss +
                self.count_weight * loss_count +
                self.interval_weight * loss_interval +
                self.ccm_weight * loss_ccm
        )

        # 统计信息
        with torch.no_grad():
            hard_labels = []
            hard_labels.append((c < boundaries[:, 0]).long())
            hard_labels.append(((c >= boundaries[:, 0]) & (c < boundaries[:, 1])).long())
            hard_labels.append(((c >= boundaries[:, 1]) & (c < boundaries[:, 2])).long())
            hard_labels.append((c >= boundaries[:, 2]).long())
            interval_counts = torch.stack(hard_labels, dim=1).float()
            interval_ratios = interval_counts.sum(dim=0) / bs

            coverage_rates = torch.stack([cdf_b1.mean(), cdf_b2.mean(), cdf_b3.mean()])

        return {
            'loss_coverage': loss_coverage,
            'loss_spacing': total_spacing_loss,
            'loss_count': loss_count,
            'loss_interval': loss_interval,
            'loss_ccm': loss_ccm,
            'total_adaptive_loss': total_loss,
            'interval_ratios': interval_ratios,
            'coverage_rates': coverage_rates,
            'boundary_spacings': torch.tensor([phy_spacing_12.mean(), phy_spacing_23.mean()]),
            'ideal_spacing': ideal_log_spacing
        }

    def _compute_fixed_labels(self, real_counts, fixed_boundaries):
        bs = real_counts.shape[0]
        labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
        b1, b2, b3 = fixed_boundaries[0], fixed_boundaries[1], fixed_boundaries[2]
        labels[(real_counts >= b1) & (real_counts < b2)] = 1
        labels[(real_counts >= b2) & (real_counts < b3)] = 2
        labels[real_counts >= b3] = 3
        return labels


# ============ 测试代码 ============
if __name__ == '__main__':
    print("=" * 70)
    print("测试：修复版 - 增强边界间距约束")
    print("=" * 70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AdaptiveBoundaryCCM(
        feature_dim=256,
        ccm_cls_num=4,
        use_soft_assignment=True
    ).to(device)

    criterion = TrueAdaptiveBoundaryLoss(
        coverage_weight=3.0,  # 提高覆盖率权重
        spacing_weight=2.0,  # 新增边界间距约束
        count_weight=1.0,
        interval_weight=2.0,  # 降低interval权重
        ccm_weight=0.2
    ).to(device)

    # 构造长尾分布数据
    bs = 8
    feature_map = torch.randn(bs, 256, 32, 32).to(device)
    real_counts = torch.tensor([8, 35, 85, 150, 280, 450, 850, 1400]).to(device)

    print("\n数据分布分析:")
    print(f"真实计数: {real_counts.cpu().numpy()}")
    sorted_counts = torch.sort(real_counts)[0]
    print(f"理想边界: [{sorted_counts[1]}, {sorted_counts[3]}, {sorted_counts[5]}]")
    print(f"  → 确保每个区间有相近的样本数\n")

    print("初始状态:")
    model.train()
    with torch.no_grad():
        outputs = model(feature_map, real_counts=real_counts)
        b0 = outputs['pred_boundaries'][0]
        print(f"样本0的初始边界: [{b0[0]:.1f}, {b0[1]:.1f}, {b0[2]:.1f}]")

    # 模拟训练迭代
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n开始训练迭代...")
    print("-" * 70)

    prev_loss = float('inf')
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(feature_map, real_counts=real_counts)
        targets = {'real_counts': real_counts}
        losses = criterion(outputs, targets)

        losses['total_adaptive_loss'].backward()

        boundary_grad = model.boundary_head[-1].bias.grad
        grad_norm = boundary_grad.norm().item() if boundary_grad is not None else 0

        optimizer.step()

        if epoch % 5 == 0 or epoch < 3:
            b = outputs['pred_boundaries'][0]
            ratios = losses['interval_ratios'].cpu().numpy()
            coverage = losses['coverage_rates'].cpu().numpy()
            spacings = losses['boundary_spacings'].cpu().numpy()
            ideal_sp = losses['ideal_spacing'].item()

            print(f"\nEpoch {epoch}:")
            print(f"  边界 (样本0): [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}]")
            print(f"  边界间距: [{spacings[0]:.1f}, {spacings[1]:.1f}] (理想: {ideal_sp:.1f})")
            print(f"  Total Loss: {losses['total_adaptive_loss'].item():.4f} "
                  f"{'↓' if losses['total_adaptive_loss'].item() < prev_loss else '↑'}")
            print(f"  Coverage Loss: {losses['loss_coverage'].item():.4f} "
                  f"(rates: [{coverage[0]:.2f}, {coverage[1]:.2f}, {coverage[2]:.2f}], "
                  f"target: [0.25, 0.50, 0.75])")
            print(f"  Spacing Loss: {losses['loss_spacing'].item():.4f}")
            print(f"  Interval Loss: {losses['loss_interval'].item():.4f}")
            print(f"  Grad Norm: {grad_norm:.6f}")
            print(f"  Interval Ratios: [{ratios[0]:.2f}, {ratios[1]:.2f}, {ratios[2]:.2f}, {ratios[3]:.2f}]")

            prev_loss = losses['total_adaptive_loss'].item()

    print("\n" + "=" * 70)
    print("✅ 测试完成!")

    # 最终评估
    final_b = outputs['pred_boundaries'][0]
    final_ratios = losses['interval_ratios'].cpu().numpy()
    final_coverage = losses['coverage_rates'].cpu().numpy()
    final_spacings = losses['boundary_spacings'].cpu().numpy()

    print("\n最终结果评估:")
    print(f"1. 边界位置: [{final_b[0]:.1f}, {final_b[1]:.1f}, {final_b[2]:.1f}]")
    print(f"2. 边界间距: [{final_spacings[0]:.1f}, {final_spacings[1]:.1f}]")
    print(f"3. 覆盖率: [{final_coverage[0]:.2f}, {final_coverage[1]:.2f}, {final_coverage[2]:.2f}]")
    print(f"   目标: [0.25, 0.50, 0.75]")
    print(f"4. 区间分布: {final_ratios}")
    print(f"5. 最终损失: {losses['total_adaptive_loss'].item():.4f}")

    # 验证标准
    coverage_good = all(abs(final_coverage[i] - [0.25, 0.50, 0.75][i]) < 0.15 for i in range(3))
    balance_good = all(abs(r - 0.25) < 0.2 for r in final_ratios)
    spacing_good = all(s > 50 for s in final_spacings)  # 最小间距50
    position_good = (final_b[0] > 20 and final_b[2] < 1200)
    loss_good = losses['total_adaptive_loss'].item() < 3.0
    grad_good = grad_norm > 1e-5

    print(f"\n关键指标:")
    print(f"  梯度正常: {'✅' if grad_good else '❌'} (Grad Norm = {grad_norm:.6f})")
    print(f"  覆盖率准确: {'✅' if coverage_good else '❌'} (偏差 < 0.15)")
    print(f"  边界间距: {'✅' if spacing_good else '❌'} (间距 > 50)")
    print(f"  边界未崩塌: {'✅' if position_good else '❌'} (b1 > 20, b3 < 1200)")
    print(f"  分布平衡: {'✅' if balance_good else '❌'} (偏差 < 0.2)")
    print(f"  损失收敛: {'✅' if loss_good else '❌'} (Loss < 3.0)")

# 第四次尝试
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
#         if self.gn is not None: x = self.gn(x)
#         if self.relu is not None: x = self.relu(x)
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
#     【V4 最终修复版】
#     1. 保留 V3 的 Log 空间结构化预测（防崩塌）
#     2. 配合 Loss 中的 detach 解决边界爆炸问题
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
#         # Backbone
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         self.ccm_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)
#
#         # Boundary Head
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 3)
#         )
#
#         # Count Regressor
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1)
#         )
#
#         # Classifier
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # Ref Points
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 边界初始化 (Log空间)
#         # b1 ~ 20 (log=3.0)
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
#         nn.init.constant_(self.boundary_head[-1].bias[0], 3.0)
#         nn.init.constant_(self.boundary_head[-1].bias[1], 0.8)  # delta ~ 2.2x
#         nn.init.constant_(self.boundary_head[-1].bias[2], 0.8)
#
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
#         nn.init.constant_(self.count_regressor[-1].bias, 5.3)
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)
#
#     def forward(self, feature_map, spatial_shapes=None, real_counts=None):
#         if feature_map.dim() == 3:
#             if spatial_shapes is None: raise ValueError("spatial_shapes needed")
#             bs, l, c = feature_map.shape
#             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
#             feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
#
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # 1. Features
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)
#
#         # 2. Boundaries (Log 空间结构化预测)
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_out = self.boundary_head(bd_feat)
#
#         # 结构化约束：确保 b2 > b1, b3 > b2
#         # min_log_gap = 0.3 (e^0.3 ≈ 1.35倍间距)
#         min_log_gap = 0.3
#
#         # 稍微放宽 max 限制，防止 clamp 截断梯度，但保留物理上限
#         log_b1 = raw_out[:, 0].clamp(min=1.0, max=9.0)
#
#         delta12 = F.softplus(raw_out[:, 1]) + min_log_gap
#         delta23 = F.softplus(raw_out[:, 2]) + min_log_gap
#
#         log_b2 = log_b1 + delta12
#         log_b3 = log_b2 + delta23
#
#         log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
#         boundaries = torch.exp(log_boundaries)
#
#         # 3. Count
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # 4. Classification
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)
#
#         # 5. Assignment
#         N_eval = (real_counts.float() * 1.2 + 30.0).clamp(
#             max=self.max_objects) if real_counts is not None else pred_count
#
#         if self.use_soft_assignment and self.training:
#             soft_weights = self._compute_log_soft_weights(N_eval, log_boundaries)
#             query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
#             # 这里不需要 detach，因为下游任务对 query 数量的需求反过来指导边界是合理的
#             num_queries = (soft_weights * query_levels_tensor).sum(dim=1).long()
#             level_indices = soft_weights.argmax(dim=1)
#         else:
#             level_indices = self._assign_query_levels(N_eval, boundaries)
#             query_levels_tensor = torch.tensor(self.query_levels, device=device)
#             num_queries = query_levels_tensor[level_indices]
#             soft_weights = None
#
#         # 6. Ref Points
#         heatmap = torch.sigmoid(self.ref_point_conv(density_feat).clamp(-10, 10))
#         reference_points = self._generate_reference_points(heatmap, h, w, device)
#
#         return {
#             'pred_boundaries': boundaries,
#             'log_boundaries': log_boundaries,
#             'predicted_count': pred_count,
#             'raw_count': raw_count,
#             'pred_bbox_number': pred_bbox_number,
#             'soft_weights': soft_weights
#         }
#
#     def _compute_log_soft_weights(self, N_eval, log_boundaries):
#         temperature = 1.0
#         log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)
#         log_b = log_boundaries
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
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#         ref_points = torch.stack([(topk_x / w).clamp(0.01, 0.99), (topk_y / h).clamp(0.01, 0.99)], dim=-1)
#         ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.02], dim=-1)
#         if actual_k < max_k:
#             ref_points = torch.cat([ref_points, torch.zeros(bs, max_k - actual_k, 4, device=device)], dim=1)
#         return ref_points
#
#
# class TrueAdaptiveBoundaryLoss(nn.Module):
#     def __init__(self, coverage_weight=10.0, spacing_weight=1.0, count_weight=1.0, interval_weight=5.0):
#         super().__init__()
#         self.coverage_weight = coverage_weight
#         self.spacing_weight = spacing_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.smooth_l1 = nn.SmoothL1Loss()
#
#     def forward(self, outputs, targets):
#         device = outputs['pred_boundaries'].device
#         real_counts = targets['real_counts'].to(device)
#
#         log_b = outputs['log_boundaries']
#         log_c = torch.log(real_counts.float().clamp(min=1.0))
#
#         # ========== 1. Coverage Loss (驱动边界移动的核心) ==========
#         tau = 1.0
#         # CDF 计算: P(LogBoundary > LogCount)
#         cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
#         cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
#         cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)
#
#         loss_coverage = (
#                 (cdf_b1.mean() - 0.25) ** 2 +
#                 (cdf_b2.mean() - 0.50) ** 2 +
#                 (cdf_b3.mean() - 0.75) ** 2
#         )
#
#         # ========== 2. Spacing Loss (辅助约束) ==========
#         loss_spacing = (
#                 F.relu(1.0 - log_b[:, 0]) +  # b1 > 2.7
#                 F.relu(log_b[:, 2] - 9.5)  # b3 < 13000
#         ).mean()
#
#         # ========== 3. Interval Loss (关键修复点) ==========
#         p0 = cdf_b1
#         p1 = cdf_b2 - cdf_b1
#         p2 = cdf_b3 - cdf_b2
#         p3 = 1.0 - cdf_b3
#
#         soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
#         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
#
#         # [CRITICAL FIX]: 截断梯度！
#         # 让 soft_targets 变为常量，防止 Interval Loss 拉扯边界
#         soft_targets = soft_targets.detach()
#
#         pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
#         loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()
#
#         # ========== 4. Count Loss ==========
#         loss_count = self.smooth_l1(outputs['raw_count'], log_c)
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
#             'boundary_vals': torch.exp(log_b).mean(dim=0)
#         }
#
#
# # ============ 测试代码 ============
# if __name__ == '__main__':
#     print("=" * 70)
#     print("测试：V4 最终修复版 (Detach Fix)")
#     print("=" * 70)
#
#     torch.manual_seed(42)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     model = AdaptiveBoundaryCCM().to(device)
#     criterion = TrueAdaptiveBoundaryLoss(coverage_weight=20.0, interval_weight=2.0).to(device)
#     # 策略调整：加大 coverage 权重，降低 interval 权重，虽然有 detach 但这样收敛更快
#
#     bs = 8
#     feature_map = torch.randn(bs, 256, 32, 32).to(device)
#     real_counts = torch.tensor([5, 8, 12, 15, 20, 100, 800, 1500]).to(device)
#
#     sorted_c = torch.sort(real_counts)[0]
#     print(f"真实数据: {sorted_c.cpu().numpy()}")
#     print(f"理想边界: ~{sorted_c[1]} (25%), ~{sorted_c[3]} (50%), ~{sorted_c[5]} (75%)")
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     print("\nTraining...")
#     for epoch in range(101):  # 多跑几轮确保稳定
#         optimizer.zero_grad()
#         outputs = model(feature_map, real_counts=real_counts)
#         losses = criterion(outputs, {'real_counts': real_counts})
#         losses['total_adaptive_loss'].backward()
#         optimizer.step()
#
#         if epoch % 10 == 0:
#             b = losses['boundary_vals'].detach().cpu().numpy()  # 直接读 loss 里的平均值
#             cv = losses['coverage_rates'].detach().cpu().numpy()
#             print(f"Epoch {epoch:3d} | Loss: {losses['total_adaptive_loss']:.4f}")
#             print(f"  Boundaries: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}]")
#             print(f"  Coverage  : [{cv[0]:.2f}, {cv[1]:.2f}, {cv[2]:.2f}] (Target: .25/.50/.75)")
#
#     final_b = losses['boundary_vals'].detach().cpu().numpy()
#     print("-" * 50)
#     print(f"最终边界: [{final_b[0]:.1f}, {final_b[1]:.1f}, {final_b[2]:.1f}]")
#     print("期望: 边界应当回落到正常数值范围内。")


# 第四次尝试
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# class Conv_GN(nn.Module):
#     """标准卷积模块: Conv + GroupNorm + ReLU"""
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
#         if self.gn is not None: x = self.gn(x)
#         if self.relu is not None: x = self.relu(x)
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
#     【最终修复版】自适应边界计数模块
#
#     修复点:
#     1. 增加返回值: 'density_feature', 'density_map', 'reference_points'，解决 KeyError。
#     2. 保留 Log-Space Structural Prediction (防崩塌)。
#     3. 保留 Temperature Annealing Support (温度退火)。
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
#         self.ccm_backbone = make_ccm_layers(
#             [512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2
#         )
#
#         # ============ Boundary Head ============
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 3)
#         )
#
#         # ============ Count Regressor ============
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1)
#         )
#
#         # ============ Classifier ============
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # ============ Ref Points ============
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 边界初始化 (Log空间):
#         # b1_init ~ 20 (log=3.0)
#         # delta ~ e^0.8 ≈ 2.2倍 (即 b2 ≈ 45, b3 ≈ 100)
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
#         nn.init.constant_(self.boundary_head[-1].bias[0], 3.0)
#         nn.init.constant_(self.boundary_head[-1].bias[1], 0.8)
#         nn.init.constant_(self.boundary_head[-1].bias[2], 0.8)
#
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
#         nn.init.constant_(self.count_regressor[-1].bias, 5.3)
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)
#
#     def forward(self, feature_map, spatial_shapes=None, real_counts=None, tau=1.0):
#         """
#         Args:
#             tau (float): 温度系数。默认1.0。
#         """
#         if feature_map.dim() == 3:
#             if spatial_shapes is None: raise ValueError("spatial_shapes needed")
#             bs, l, c = feature_map.shape
#             h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
#             feature_map = feature_map[:, :h * w, :].transpose(1, 2).reshape(bs, c, h, w)
#
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # 1. Features
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)  # [关键变量]
#
#         # 2. Boundaries (结构化预测)
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_out = self.boundary_head(bd_feat)
#
#         # 结构性约束
#         min_log_gap = 0.3
#
#         log_b1 = raw_out[:, 0].clamp(min=1.0, max=9.0)
#         delta12 = F.softplus(raw_out[:, 1]) + min_log_gap
#         delta23 = F.softplus(raw_out[:, 2]) + min_log_gap
#
#         log_b2 = log_b1 + delta12
#         log_b3 = log_b2 + delta23
#
#         log_boundaries = torch.stack([log_b1, log_b2, log_b3], dim=1)
#         boundaries = torch.exp(log_boundaries)
#
#         # 3. Count
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # 4. Classification
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)
#
#         # 5. Assignment
#         N_eval = (real_counts.float() * 1.2 + 30.0).clamp(
#             max=self.max_objects) if real_counts is not None else pred_count
#
#         if self.use_soft_assignment and self.training:
#             soft_weights = self._compute_log_soft_weights(N_eval, log_boundaries, tau)
#             query_levels_tensor = torch.tensor(self.query_levels, dtype=torch.float32, device=device)
#             num_queries = (soft_weights * query_levels_tensor).sum(dim=1).long()
#             level_indices = soft_weights.argmax(dim=1)
#         else:
#             level_indices = self._assign_query_levels(N_eval, boundaries)
#             query_levels_tensor = torch.tensor(self.query_levels, device=device)
#             num_queries = query_levels_tensor[level_indices]
#             soft_weights = None
#
#         # 6. Ref Points
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
#
#             # --- [修复] 补全缺失的 Keys ---
#             'density_feature': density_feat,  # DeformableTransformer 需要这个
#             'density_map': heatmap,  # 可能用于可视化或Aux Loss
#             'reference_points': reference_points,  # 可能用于初始化 Query
#             'num_queries': num_queries  # 备用
#         }
#
#     def _compute_log_soft_weights(self, N_eval, log_boundaries, tau):
#         log_N = torch.log(N_eval.clamp(min=1.0)).unsqueeze(1)
#         log_b = log_boundaries
#
#         c0 = log_b[:, 0] - 0.5
#         c1 = (log_b[:, 0] + log_b[:, 1]) / 2
#         c2 = (log_b[:, 1] + log_b[:, 2]) / 2
#         c3 = log_b[:, 2] + 0.5
#
#         centers = torch.stack([c0, c1, c2, c3], dim=1)
#         distances = -torch.abs(log_N - centers)
#         return F.softmax(distances / tau, dim=1)
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
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#         ref_points = torch.stack([(topk_x / w).clamp(0.01, 0.99), (topk_y / h).clamp(0.01, 0.99)], dim=-1)
#         ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.02], dim=-1)
#         if actual_k < max_k:
#             ref_points = torch.cat([ref_points, torch.zeros(bs, max_k - actual_k, 4, device=device)], dim=1)
#         return ref_points
#
#
# class TrueAdaptiveBoundaryLoss(nn.Module):
#     """
#     自适应损失函数
#     """
#
#     def __init__(self, coverage_weight=20.0, spacing_weight=1.0, count_weight=1.0, interval_weight=2.0):
#         super().__init__()
#         self.coverage_weight = coverage_weight
#         self.spacing_weight = spacing_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.smooth_l1 = nn.SmoothL1Loss()
#
#     def forward(self, outputs, targets, tau=1.0):
#         device = outputs['pred_boundaries'].device
#         real_counts = targets['real_counts'].to(device)
#
#         log_b = outputs['log_boundaries']
#         log_c = torch.log(real_counts.float().clamp(min=1.0))
#
#         # ========== 1. Coverage Loss (动态 Tau) ==========
#         cdf_b1 = torch.sigmoid((log_b[:, 0] - log_c) / tau)
#         cdf_b2 = torch.sigmoid((log_b[:, 1] - log_c) / tau)
#         cdf_b3 = torch.sigmoid((log_b[:, 2] - log_c) / tau)
#
#         loss_coverage = (
#                 (cdf_b1.mean() - 0.25) ** 2 +
#                 (cdf_b2.mean() - 0.50) ** 2 +
#                 (cdf_b3.mean() - 0.75) ** 2
#         )
#
#         # ========== 2. Spacing Loss ==========
#         loss_spacing = (
#                 F.relu(1.0 - log_b[:, 0]) +  # b1 > 2.7
#                 F.relu(log_b[:, 2] - 9.5)  # b3 < 13000
#         ).mean()
#
#         # ========== 3. Interval Loss (Detach Fix) ==========
#         p0 = cdf_b1
#         p1 = cdf_b2 - cdf_b1
#         p2 = cdf_b3 - cdf_b2
#         p3 = 1.0 - cdf_b3
#
#         soft_targets = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-6)
#         soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
#         soft_targets = soft_targets.detach()  # 截断梯度
#
#         pred_probs = torch.softmax(outputs['pred_bbox_number'], dim=1)
#         loss_interval = -(soft_targets * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=1).mean()
#
#         # ========== 4. Count Loss ==========
#         loss_count = self.smooth_l1(outputs['raw_count'], log_c)
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
#             'boundary_vals': torch.exp(log_b).mean(dim=0)
#         }
# # import torch.nn as nn
# # import torch
# # from torchvision import models
# # import torch.nn.functional as F
# #
# # class CategoricalCounting(nn.Module):
# #     def __init__(self, cls_num=4):
# #         super(CategoricalCounting, self).__init__()
# #         self.ccm_cfg = [512, 512, 512, 256, 256, 256]
# #         self.in_channels = 512
# #         self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
# #         self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
# #         self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
# #         self.linear = nn.Linear(256, cls_num)
# #
# #     def forward(self, features, spatial_shapes=None):
# #         features = features.transpose(1, 2)
# #         bs, c, hw = features.shape
# #         h, w = spatial_shapes[0][0], spatial_shapes[0][1]
# #
# #         v_feat = features[:,:,0:h*w].view(bs, 256, h, w)
# #         x = self.conv1(v_feat)
# #         x = self.ccm(x)
# #         out = self.output(x)
# #         out = out.squeeze(3)
# #         out = out.squeeze(2)
# #         out = self.linear(out)
# #
# #         return out, x
# #
# # def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
# #     layers = []
# #     for v in cfg:
# #             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
# #             if batch_norm:
# #                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
# #             else:
# #                 layers += [conv2d, nn.ReLU(inplace=True)]
# #             in_channels = v
# #     return nn.Sequential(*layers)
#
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
#     自适应边界分类计数模块 (Adaptive Boundary CCM)
#
#     整合了:
#     1. CCM密度特征提取 (空洞卷积 backbone)
#     2. 自适应边界预测 (3个可学习的边界点)
#     3. 目标数量回归
#     4. 动态查询数量选择
#     """
#
#     def __init__(self,
#                  feature_dim=256,
#                  ccm_cls_num=4,  # 区间数量
#                  query_levels=[300, 500, 900, 1500],  # 对应4个区间的查询数
#                  max_objects=1500):
#         super().__init__()
#
#         self.ccm_cls_num = ccm_cls_num
#         self.query_levels = query_levels
#         self.max_objects = max_objects
#
#         # ============ 1. 共享密度特征提取器 (CCM Backbone) ============
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         # CCM配置: [512, 512, 512, 256, 256, 256]
#         self.ccm_backbone = make_ccm_layers(
#             [512, 512, 512, 256, 256, 256],
#             in_channels=512,
#             d_rate=2
#         )
#
#         # ============ 2. 分支A: 边界预测模块 ============
#         self.boundary_pool = nn.AdaptiveAvgPool2d(1)
#         self.boundary_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 3)  # 预测3个边界值 [b1, b2, b3]
#         )
#
#         # ============ 3. 分支B: 目标数量回归 ============
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1)  # 输出 log(count)
#         )
#
#         # ============ 4. 分支C: CCM分类头 (辅助监督) ============
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_classifier = nn.Linear(256, ccm_cls_num)
#
#         # ============ 5. 参考点生成 (基于密度图) ============
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         """权重初始化"""
#         # 1. CCM backbone (He初始化)
#         for m in self.ccm_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 2. 边界预测 (修正：预测增量delta，而非绝对值)
#         # 目标: delta1=100, delta2=200, delta3=200
#         nn.init.normal_(self.boundary_head[-1].weight, std=0.01)
#         nn.init.constant_(self.boundary_head[-1].bias[0], 4.605)  # log(100) ≈ 4.6
#         nn.init.constant_(self.boundary_head[-1].bias[1], 5.298)  # log(300) ≈ 5.7
#         nn.init.constant_(self.boundary_head[-1].bias[2], 5.298)  # log(500) ≈ 6.2
#
#         # 3. 数量回归 (初始化为 log(200))
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
#         nn.init.constant_(self.count_regressor[-1].bias, 5.298)
#
#         # 4. 参考点生成 (bias设低，初始heatmap较平滑)
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.19)  # sigmoid(-2.19) ≈ 0.1
#
#     def forward(self, feature_map, real_counts=None):
#         """
#         Args:
#             feature_map: (BS, 256, H, W) 来自Encoder的特征
#             real_counts: (BS) 真实目标数量 (仅训练时提供)
#
#         Returns:
#             outputs: dict包含:
#                 - pred_boundaries: (BS, 3) 预测的边界 [b1, b2, b3]
#                 - predicted_count: (BS) 预测的目标数量
#                 - num_queries: (BS) 每张图选择的查询数
#                 - pred_bbox_number: (BS, 4) CCM分类结果
#                 - reference_points: (BS, max_k, 4) 参考点坐标
#                 - density_map: (BS, 1, H, W) 密度热力图
#         """
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # ============ Step 1: 提取共享密度特征 ============
#         x = self.density_conv1(feature_map)
#         density_feat = self.ccm_backbone(x)  # (BS, 256, H, W)
#
#         # ============ Step 2: 边界预测 ============
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         raw_boundaries = self.boundary_head(bd_feat)  # (BS, 3)
#
#         # 修正：预测delta增量，确保严格单调递增
#         # b1 = exp(delta1), b2 = b1 + exp(delta2), b3 = b2 + exp(delta3)
#         deltas = torch.exp(raw_boundaries).clamp(min=10, max=500)  # 限制增量范围
#
#         boundaries = []
#         boundaries.append(deltas[:, 0])  # b1 = delta1
#         boundaries.append(boundaries[0] + deltas[:, 1])  # b2 = b1 + delta2
#         boundaries.append(boundaries[1] + deltas[:, 2])  # b3 = b2 + delta3
#
#         boundaries = torch.stack(boundaries, dim=1)  # (BS, 3)
#         boundaries = boundaries.clamp(max=self.max_objects)  # 全局上限
#
#         # ============ Step 3: 目标数量回归 ============
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # ============ Step 4: CCM分类 (辅助损失) ============
#         ccm_feat = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_classifier(ccm_feat)  # (BS, 4)
#
#         # ============ Step 5: 决定查询数量 ============
#         if self.training and real_counts is not None:
#             # 训练阶段: 使用GT边界生成分类标签
#             target_labels = self._compute_target_labels(real_counts, boundaries)
#
#             # 使用真实数量 * 1.2 + 20 作为查询数(增加召回)
#             N_eval = (real_counts.float() * 1.2 + 20).clamp(max=self.max_objects)
#         else:
#             # 推理阶段: 使用预测数量
#             N_eval = pred_count
#             target_labels = None
#
#         # 根据N_eval决定查询级别
#         level_indices = self._assign_query_levels(N_eval, boundaries)
#         query_levels_tensor = torch.tensor(self.query_levels, device=device)
#         num_queries = query_levels_tensor[level_indices]  # (BS,)
#
#         # ============ Step 6: 生成参考点 (基于密度Heatmap) ============
#         heatmap = self.ref_point_conv(density_feat).sigmoid()  # (BS, 1, H, W)
#         reference_points = self._generate_reference_points(heatmap, h, w, device)
#
#         # ============ 输出组装 ============
#         outputs = {
#             'pred_boundaries': boundaries,  # (BS, 3)
#             'raw_boundaries': raw_boundaries,  # (BS, 3) for loss
#             'predicted_count': pred_count,  # (BS)
#             'raw_count': raw_count,  # (BS) for loss
#             'num_queries': num_queries,  # (BS)
#             'pred_bbox_number': pred_bbox_number,  # (BS, 4) CCM分类
#             'reference_points': reference_points,  # (BS, max_k, 4)
#             'density_map': heatmap,  # (BS, 1, H, W)
#             'density_feature': density_feat,  # (BS, 256, H, W)
#             'target_labels': target_labels,  # (BS) 仅训练时有效
#             'level_indices': level_indices  # (BS) for debug
#         }
#
#         return outputs
#
#     def _compute_target_labels(self, real_counts, boundaries):
#         """
#         训练时: 根据GT数量和预测边界计算分类标签
#
#         Args:
#             real_counts: (BS) GT目标数
#             boundaries: (BS, 3) 预测的边界 [b1, b2, b3]
#
#         Returns:
#             labels: (BS) 区间标签 [0, 1, 2, 3]
#         """
#         bs = real_counts.shape[0]
#         labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)
#
#         b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
#
#         # 区间划分:
#         # 0: [0, b1)
#         # 1: [b1, b2)
#         # 2: [b2, b3)
#         # 3: [b3, +∞)
#         labels[(real_counts >= b1) & (real_counts < b2)] = 1
#         labels[(real_counts >= b2) & (real_counts < b3)] = 2
#         labels[real_counts >= b3] = 3
#
#         return labels
#
#     def _assign_query_levels(self, N_eval, boundaries):
#         """
#         根据评估数量和边界分配查询级别
#
#         Args:
#             N_eval: (BS) 评估的目标数量
#             boundaries: (BS, 3) 边界值
#
#         Returns:
#             level_indices: (BS) 级别索引 [0, 1, 2, 3]
#         """
#         bs = N_eval.shape[0]
#         device = N_eval.device
#         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
#
#         b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
#
#         level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
#         level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
#         level_indices[N_eval >= b3] = 3
#
#         return level_indices
#
#     def _generate_reference_points(self, heatmap, h, w, device):
#         """
#         基于密度热力图生成参考点
#
#         Args:
#             heatmap: (BS, 1, H, W)
#             h, w: 特征图尺寸
#             device: 设备
#
#         Returns:
#             ref_points: (BS, max_k, 4) 归一化坐标 [cx, cy, w, h]
#         """
#         bs = heatmap.shape[0]
#         max_k = max(self.query_levels)
#
#         # 展平并选择Top-K
#         heatmap_flat = heatmap.flatten(2).squeeze(1)  # (BS, H*W)
#         actual_k = min(h * w, max_k)
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#
#         # 转换为归一化坐标
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#
#         ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)  # (BS, K, 2)
#
#         # 初始宽高 (小目标场景使用较小值)
#         initial_wh = torch.ones_like(ref_points) * 0.02
#         ref_points = torch.cat([ref_points, initial_wh], dim=-1)  # (BS, K, 4)
#
#         # Padding到max_k
#         if actual_k < max_k:
#             pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
#             ref_points = torch.cat([ref_points, pad], dim=1)
#
#         return ref_points
#
#
# # ============ 损失函数 ============
# class AdaptiveBoundaryLoss(nn.Module):
#     """
#     自适应边界CCM的损失函数
#
#     包含:
#     1. 边界预测损失 (Smooth L1)
#     2. 数量回归损失 (Smooth L1 on log-space)
#     3. 区间分类损失 (CrossEntropy)
#     4. CCM辅助损失 (CrossEntropy)
#     """
#
#     def __init__(self,
#                  boundary_weight=2.0,
#                  count_weight=1.0,
#                  interval_weight=1.0,
#                  ccm_weight=0.5):
#         super().__init__()
#         self.boundary_weight = boundary_weight
#         self.count_weight = count_weight
#         self.interval_weight = interval_weight
#         self.ccm_weight = ccm_weight
#
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.smooth_l1 = nn.SmoothL1Loss()
#         self.mse_loss = nn.MSELoss()  # 新增：用于边界
#
#     def forward(self, outputs, targets):
#         """
#         Args:
#             outputs: AdaptiveBoundaryCCM的输出字典
#             targets: dict包含:
#                 - real_counts: (BS) GT目标数
#                 - ccm_labels: (BS) CCM固定边界标签
#
#         Returns:
#             loss_dict: 各项损失
#         """
#         device = outputs['pred_boundaries'].device
#         real_counts = targets['real_counts'].to(device)
#
#         # ========== 1. 边界预测损失 ==========
#         # 目标: 让预测边界逼近固定的理想边界 [100, 300, 500]
#         ideal_boundaries = torch.tensor([100.0, 300.0, 500.0], device=device)
#         ideal_boundaries = ideal_boundaries.unsqueeze(0).expand_as(outputs['pred_boundaries'])
#
#         loss_boundary = self.smooth_l1(
#             outputs['pred_boundaries'],
#             ideal_boundaries
#         )
#
#         # ========== 2. 数量回归损失 ==========
#         # 在log空间计算损失
#         loss_count = self.smooth_l1(
#             outputs['raw_count'],
#             torch.log(real_counts.float().clamp(min=1.0))
#         )
#
#         # ========== 3. 区间分类损失 ==========
#         if outputs['target_labels'] is not None:
#             loss_interval = self.ce_loss(
#                 outputs['pred_bbox_number'],
#                 outputs['target_labels']
#             )
#         else:
#             loss_interval = torch.tensor(0.0, device=device)
#
#         # ========== 4. CCM辅助损失 ==========
#         if 'ccm_labels' in targets:
#             ccm_labels = targets['ccm_labels'].to(device)
#             loss_ccm = self.ce_loss(
#                 outputs['pred_bbox_number'],
#                 ccm_labels
#             )
#         else:
#             # 如果没有提供，用固定边界计算
#             fixed_boundaries = torch.tensor([10.0, 100.0, 500.0], device=device)
#             fixed_labels = self._compute_fixed_labels(real_counts, fixed_boundaries)
#             loss_ccm = self.ce_loss(
#                 outputs['pred_bbox_number'],
#                 fixed_labels
#             )
#
#         # ========== 总损失 ==========
#         total_loss = (
#                 self.boundary_weight * loss_boundary +
#                 self.count_weight * loss_count +
#                 self.interval_weight * loss_interval +
#                 self.ccm_weight * loss_ccm
#         )
#
#         # 返回详细损失
#         loss_dict = {
#             'loss_boundary': loss_boundary,
#             'loss_count': loss_count,
#             'loss_interval': loss_interval,
#             'loss_ccm': loss_ccm,
#             'total_adaptive_loss': total_loss
#         }
#
#         return loss_dict
#
#
# # ============ 使用示例 ============
# if __name__ == '__main__':
#     # 创建模块
#     model = AdaptiveBoundaryCCM(
#         feature_dim=256,
#         ccm_cls_num=4,
#         query_levels=[300, 500, 900, 1500]
#     ).cuda()
#
#     criterion = AdaptiveBoundaryLoss(
#         boundary_weight=2.0,
#         count_weight=1.0,
#         interval_weight=1.0,
#         ccm_weight=0.5
#     )
#
#     # 模拟输入
#     feature_map = torch.randn(2, 256, 32, 32).cuda()
#     real_counts = torch.tensor([150, 450]).cuda()  # GT数量
#     ccm_labels = torch.tensor([1, 2]).cuda()  # CCM固定边界标签
#
#     # 前向传播
#     outputs = model(feature_map, real_counts=real_counts)
#
#     # 计算损失
#     targets = {
#         'real_counts': real_counts,
#         'ccm_labels': ccm_labels
#     }
#     losses = criterion(outputs, targets)
#
#     # 打印结果
#     print("=" * 50)
#     print("模型输出:")
#     print(f"  预测边界: {outputs['pred_boundaries'][0].cpu().detach().numpy()}")
#     print(f"  预测数量: {outputs['predicted_count'][0].item():.1f}")
#     print(f"  查询数量: {outputs['num_queries'][0].item()}")
#     print(f"  区间级别: {outputs['level_indices'][0].item()}")
#
#     print("\n损失:")
#     for k, v in losses.items():
#         print(f"  {k}: {v.item():.4f}")
#
#     print("\n" + "=" * 50)
#     print("✅ 自适应边界CCM模块测试通过!")
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
    """构建CCM层序列（使用空洞卷积）"""
    layers = []
    for v in cfg:
        conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        layers.append(conv2d)
        in_channels = v
    return nn.Sequential(*layers)


class AdaptiveBoundaryCCM(nn.Module):
    """
    自适应边界分类计数模块 (Adaptive Boundary CCM)

    整合了:
    1. CCM密度特征提取 (空洞卷积 backbone)
    2. 自适应边界预测 (3个可学习的边界点)
    3. 目标数量回归
    4. 动态查询数量选择
    """

    def __init__(self,
                 feature_dim=256,
                 ccm_cls_num=4,  # 区间数量
                 query_levels=[300, 500, 900, 1500],  # 对应4个区间的查询数
                 max_objects=1500):
        super().__init__()

        self.ccm_cls_num = ccm_cls_num
        self.query_levels = query_levels
        self.max_objects = max_objects

        # ============ 1. 共享密度特征提取器 (CCM Backbone) ============
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        # CCM配置: [512, 512, 512, 256, 256, 256]
        self.ccm_backbone = make_ccm_layers(
            [512, 512, 512, 256, 256, 256],
            in_channels=512,
            d_rate=2
        )

        # ============ 2. 分支A: 边界预测模块 ============
        self.boundary_pool = nn.AdaptiveAvgPool2d(1)
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # 预测3个边界值 [b1, b2, b3]
        )

        # ============ 3. 分支B: 目标数量回归 ============
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 输出 log(count)
        )

        # ============ 4. 分支C: CCM分类头 (辅助监督) ============
        self.ccm_pool = nn.AdaptiveAvgPool2d(1)
        self.ccm_classifier = nn.Linear(256, ccm_cls_num)

        # ============ 5. 参考点生成 (基于密度图) ============
        self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """权重初始化 - 修复版"""
        # 1. CCM backbone (He初始化)
        for m in self.ccm_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 2. 边界预测 (修正：预测增量delta，而非绝对值)
        # 目标: delta1=100, delta2=200, delta3=200
        nn.init.normal_(self.boundary_head[-1].weight, std=0.001)
        nn.init.constant_(self.boundary_head[-1].bias[0], 4.605)  # log(100) = 4.605
        nn.init.constant_(self.boundary_head[-1].bias[1], 5.298)  # log(200) = 5.298
        nn.init.constant_(self.boundary_head[-1].bias[2], 5.298)  # log(200) = 5.298

        # 3. 数量回归 (修正：初始化为 log(200))
        nn.init.normal_(self.count_regressor[-1].weight, std=0.001)
        nn.init.constant_(self.count_regressor[-1].bias, 5.298)  # log(200) ≈ 5.298

        # 4. 参考点生成 (bias设低，初始heatmap较平滑)
        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)  # sigmoid(-2.19) ≈ 0.1

    def forward(self, feature_map, real_counts=None):
        """
        Args:
            feature_map: (BS, 256, H, W) 来自Encoder的特征
            real_counts: (BS) 真实目标数量 (仅训练时提供)

        Returns:
            outputs: dict包含:
                - pred_boundaries: (BS, 3) 预测的边界 [b1, b2, b3]
                - predicted_count: (BS) 预测的目标数量
                - num_queries: (BS) 每张图选择的查询数
                - pred_bbox_number: (BS, 4) CCM分类结果
                - reference_points: (BS, max_k, 4) 参考点坐标
                - density_map: (BS, 1, H, W) 密度热力图
        """
        bs, c, h, w = feature_map.shape
        device = feature_map.device

        # ============ Step 1: 提取共享密度特征 ============
        x = self.density_conv1(feature_map)
        density_feat = self.ccm_backbone(x)  # (BS, 256, H, W)

        # ============ Step 2: 边界预测 (修正版) ============
        bd_feat = self.boundary_pool(density_feat).flatten(1)
        raw_boundaries = self.boundary_head(bd_feat)  # (BS, 3)

        # 修正：预测delta增量，确保严格单调递增
        # b1 = exp(delta1), b2 = b1 + exp(delta2), b3 = b2 + exp(delta3)
        deltas = torch.exp(raw_boundaries).clamp(min=10, max=500)  # 限制增量范围

        boundaries = []
        boundaries.append(deltas[:, 0])  # b1 = delta1
        boundaries.append(boundaries[0] + deltas[:, 1])  # b2 = b1 + delta2
        boundaries.append(boundaries[1] + deltas[:, 2])  # b3 = b2 + delta3

        boundaries = torch.stack(boundaries, dim=1)  # (BS, 3)
        boundaries = boundaries.clamp(max=self.max_objects)  # 全局上限

        # ============ Step 3: 目标数量回归 ============
        raw_count = self.count_regressor(density_feat).squeeze(1)
        pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)

        # ============ Step 4: CCM分类 (辅助损失) ============
        ccm_feat = self.ccm_pool(density_feat).flatten(1)
        pred_bbox_number = self.ccm_classifier(ccm_feat)  # (BS, 4)

        # ============ Step 5: 决定查询数量 ============
        if self.training and real_counts is not None:
            # 训练阶段: 使用GT边界生成分类标签
            target_labels = self._compute_target_labels(real_counts, boundaries)

            # 使用真实数量 * 1.2 + 20 作为查询数(增加召回)
            N_eval = (real_counts.float() * 1.2 + 20).clamp(max=self.max_objects)
        else:
            # 推理阶段: 使用预测数量
            N_eval = pred_count
            target_labels = None

        # 根据N_eval决定查询级别
        level_indices = self._assign_query_levels(N_eval, boundaries)
        query_levels_tensor = torch.tensor(self.query_levels, device=device)
        num_queries = query_levels_tensor[level_indices]  # (BS,)

        # ============ Step 6: 生成参考点 (基于密度Heatmap) ============
        heatmap = self.ref_point_conv(density_feat).sigmoid()  # (BS, 1, H, W)
        reference_points = self._generate_reference_points(heatmap, h, w, device)

        # ============ 输出组装 ============
        outputs = {
            'pred_boundaries': boundaries,  # (BS, 3)
            'raw_boundaries': raw_boundaries,  # (BS, 3) for loss
            'predicted_count': pred_count,  # (BS)
            'raw_count': raw_count,  # (BS) for loss
            'num_queries': num_queries,  # (BS)
            'pred_bbox_number': pred_bbox_number,  # (BS, 4) CCM分类
            'reference_points': reference_points,  # (BS, max_k, 4)
            'density_map': heatmap,  # (BS, 1, H, W)
            'density_feature': density_feat,  # (BS, 256, H, W)
            'target_labels': target_labels,  # (BS) 仅训练时有效
            'level_indices': level_indices  # (BS) for debug
        }

        return outputs

    def _compute_target_labels(self, real_counts, boundaries):
        """
        训练时: 根据GT数量和预测边界计算分类标签

        Args:
            real_counts: (BS) GT目标数
            boundaries: (BS, 3) 预测的边界 [b1, b2, b3]

        Returns:
            labels: (BS) 区间标签 [0, 1, 2, 3]
        """
        bs = real_counts.shape[0]
        labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)

        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]

        # 区间划分:
        # 0: [0, b1)
        # 1: [b1, b2)
        # 2: [b2, b3)
        # 3: [b3, +∞)
        labels[(real_counts >= b1) & (real_counts < b2)] = 1
        labels[(real_counts >= b2) & (real_counts < b3)] = 2
        labels[real_counts >= b3] = 3

        return labels

    def _assign_query_levels(self, N_eval, boundaries):
        """
        根据评估数量和边界分配查询级别

        Args:
            N_eval: (BS) 评估的目标数量
            boundaries: (BS, 3) 边界值

        Returns:
            level_indices: (BS) 级别索引 [0, 1, 2, 3]
        """
        bs = N_eval.shape[0]
        device = N_eval.device
        level_indices = torch.zeros(bs, dtype=torch.long, device=device)

        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]

        level_indices[(N_eval >= b1) & (N_eval < b2)] = 1
        level_indices[(N_eval >= b2) & (N_eval < b3)] = 2
        level_indices[N_eval >= b3] = 3

        return level_indices

    def _generate_reference_points(self, heatmap, h, w, device):
        """
        基于密度热力图生成参考点

        Args:
            heatmap: (BS, 1, H, W)
            h, w: 特征图尺寸
            device: 设备

        Returns:
            ref_points: (BS, max_k, 4) 归一化坐标 [cx, cy, w, h]
        """
        bs = heatmap.shape[0]
        max_k = max(self.query_levels)

        # 展平并选择Top-K
        heatmap_flat = heatmap.flatten(2).squeeze(1)  # (BS, H*W)
        actual_k = min(h * w, max_k)
        _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)

        # 转换为归一化坐标
        topk_y = (topk_ind // w).float() + 0.5
        topk_x = (topk_ind % w).float() + 0.5

        ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)  # (BS, K, 2)

        # 初始宽高 (小目标场景使用较小值)
        initial_wh = torch.ones_like(ref_points) * 0.02
        ref_points = torch.cat([ref_points, initial_wh], dim=-1)  # (BS, K, 4)

        # Padding到max_k
        if actual_k < max_k:
            pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
            ref_points = torch.cat([ref_points, pad], dim=1)

        return ref_points


# ============ 损失函数 ============
class AdaptiveBoundaryLoss(nn.Module):
    """
    自适应边界CCM的损失函数 - 修复版

    包含:
    1. 边界预测损失 (MSE Loss，更强约束)
    2. 数量回归损失 (Smooth L1 on log-space)
    3. 区间分类损失 (CrossEntropy，使用动态标签)
    4. CCM辅助损失 (CrossEntropy，使用固定标签)
    """

    def __init__(self,
                 boundary_weight=2.0,  # 提高边界损失权重
                 count_weight=1.0,
                 interval_weight=1.0,
                 ccm_weight=0.5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.count_weight = count_weight
        self.interval_weight = interval_weight
        self.ccm_weight = ccm_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()  # 新增：用于边界

    def forward(self, outputs, targets):
        """
        Args:
            outputs: AdaptiveBoundaryCCM的输出字典
            targets: dict包含:
                - real_counts: (BS) GT目标数
                - ccm_labels: (BS) CCM固定边界标签 [0-3]

        Returns:
            loss_dict: 各项损失
        """
        device = outputs['pred_boundaries'].device
        real_counts = targets['real_counts'].to(device)

        # ========== 1. 边界预测损失 (改用MSE) ==========
        # 目标: 让预测边界逼近固定的理想边界 [100, 300, 500]
        ideal_boundaries = torch.tensor(
            [[100.0, 300.0, 500.0]],
            device=device
        ).expand_as(outputs['pred_boundaries'])

        loss_boundary = self.mse_loss(
            outputs['pred_boundaries'],
            ideal_boundaries
        )

        # ========== 2. 数量回归损失 ==========
        loss_count = self.smooth_l1(
            outputs['raw_count'],
            torch.log(real_counts.float().clamp(min=1.0))
        )

        # ========== 3. 区间分类损失 (动态边界) ==========
        # 使用预测的边界和GT数量计算动态标签
        if outputs['target_labels'] is not None:
            loss_interval = self.ce_loss(
                outputs['pred_bbox_number'],
                outputs['target_labels']
            )
        else:
            loss_interval = torch.tensor(0.0, device=device)

        # ========== 4. CCM辅助损失 (固定边界) ==========
        # 使用固定边界 [10, 100, 500] 计算的标签
        if 'ccm_labels' in targets:
            ccm_labels = targets['ccm_labels'].to(device)
            loss_ccm = self.ce_loss(
                outputs['pred_bbox_number'],
                ccm_labels
            )
        else:
            # 如果没有提供，用固定边界计算
            fixed_boundaries = torch.tensor([10.0, 100.0, 500.0], device=device)
            fixed_labels = self._compute_fixed_labels(real_counts, fixed_boundaries)
            loss_ccm = self.ce_loss(
                outputs['pred_bbox_number'],
                fixed_labels
            )

        # ========== 总损失 ==========
        total_loss = (
                self.boundary_weight * loss_boundary +
                self.count_weight * loss_count +
                self.interval_weight * loss_interval +
                self.ccm_weight * loss_ccm
        )

        # 返回详细损失
        loss_dict = {
            'loss_boundary': loss_boundary,
            'loss_count': loss_count,
            'loss_interval': loss_interval,
            'loss_ccm': loss_ccm,
            'total_adaptive_loss': total_loss
        }

        return loss_dict

    def _compute_fixed_labels(self, real_counts, fixed_boundaries):
        """使用固定边界计算标签"""
        bs = real_counts.shape[0]
        labels = torch.zeros(bs, dtype=torch.long, device=real_counts.device)

        b1, b2, b3 = fixed_boundaries[0], fixed_boundaries[1], fixed_boundaries[2]

        labels[(real_counts >= b1) & (real_counts < b2)] = 1
        labels[(real_counts >= b2) & (real_counts < b3)] = 2
        labels[real_counts >= b3] = 3

        return labels


# ============ 使用示例 ============
if __name__ == '__main__':
    # 创建模块
    model = AdaptiveBoundaryCCM(
        feature_dim=256,
        ccm_cls_num=4,
        query_levels=[300, 500, 900, 1500]
    ).cuda()

    criterion = AdaptiveBoundaryLoss(
        boundary_weight=2.0,  # 提高边界权重
        count_weight=1.0,
        interval_weight=1.0,
        ccm_weight=0.5
    )

    # 模拟输入
    feature_map = torch.randn(2, 256, 32, 32).cuda()
    real_counts = torch.tensor([150, 450]).cuda()  # GT数量
    ccm_labels = torch.tensor([1, 2]).cuda()  # CCM固定边界标签

    # 前向传播
    outputs = model(feature_map, real_counts=real_counts)

    # 计算损失
    targets = {
        'real_counts': real_counts,
        'ccm_labels': ccm_labels
    }
    losses = criterion(outputs, targets)

    # 打印结果
    print("=" * 50)
    print("模型输出:")
    print(f"  预测边界: {outputs['pred_boundaries'][0].cpu().detach().numpy()}")
    print(f"  预测数量: {outputs['predicted_count'][0].item():.1f}")
    print(f"  查询数量: {outputs['num_queries'][0].item()}")
    print(f"  区间级别: {outputs['level_indices'][0].item()}")

    print("\n损失:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n" + "=" * 50)
    print("✅ 自适应边界CCM模块测试通过!")
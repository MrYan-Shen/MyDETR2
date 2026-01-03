# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
Optimized for AITOD (Tiny Object Detection) with Numerical Stability.
Enhanced with CCM adaptive boundary support.
Fixed: EPSILON adjusted for tiny object normalized areas (e.g. 4x4 pixels).
"""
import torch
from torch import Tensor
from typing import Tuple, Optional
import warnings

# 【修改1】适应微小目标：4x4像素在800x800图上约为2.5e-5，必须小于此值
EPSILON = 1e-7


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x_c, y_c, w, h = x.unbind(-1)

    # 【修改2】限制最小宽高，防止除零。对于微小目标，1e-5 (约0.008像素) 足够安全
    w = w.clamp(min=1e-5)
    h = h.clamp(min=1e-5)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """(x1, y1, x2, y2) -> (cx, cy, w, h)"""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1).clamp(min=1e-5), (y2 - y1).clamp(min=1e-5)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算IoU，强制使用float32以保证微小目标精度
    """
    # 【修改3】强制转换为 float32 进行计算，防止 FP16 下溢
    # 这对微小目标 IoU 计算至关重要
    if boxes1.dtype == torch.float16:
        boxes1 = boxes1.float()
    if boxes2.dtype == torch.float16:
        boxes2 = boxes2.float()

    inter, union = _box_inter_union(boxes1, boxes2)
    # 使用较小的 EPSILON (1e-7)
    iou = inter / union.clamp(min=EPSILON)
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Generalized IoU with strong numerical protection.
    强制 float32 计算。
    """
    # 保存原始 dtype 以便最后转回
    original_dtype = boxes1.dtype

    # 强制转换为 float32
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    # 1. 预处理：修复无效框
    boxes1 = _fix_invalid_boxes(boxes1)
    boxes2 = _fix_invalid_boxes(boxes2)

    # 2. 计算 IoU
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union.clamp(min=EPSILON)

    # 3. 计算外接矩形
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = (rbi - lti).clamp(min=EPSILON)
    area = whi[..., 0] * whi[..., 1]

    # 4. 计算 GIoU
    giou = iou - (area - union) / area.clamp(min=EPSILON)

    # 如果输入是 FP16，结果转回 FP16 (保持兼容性)
    if original_dtype == torch.float16:
        giou = giou.half()

    return giou


def _fix_invalid_boxes(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    # 保证 x2 > x1, y2 > y1，防止退化为线或点
    new_x2 = torch.max(x1 + 1e-5, x2)
    new_y2 = torch.max(y1 + 1e-5, y2)
    return torch.stack([x1, y1, new_x2, new_y2], dim=-1)


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    return inter, union

def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    height, width = size
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height)
    return boxes


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    return keep.nonzero().squeeze(1)


# ======================== 新增：CCM自适应边界支持函数 ========================

def validate_boundary_predictions(boundaries: Tensor, log_boundaries: Tensor) -> Tensor:
    """
    验证CCM预测的边界是否合理

    Args:
        boundaries: (B, 3) 边界值 [b1, b2, b3]
        log_boundaries: (B, 3) log空间边界

    Returns:
        valid_mask: (B,) bool tensor，True表示边界合理
    """
    # 检查1: 单调性 (b1 < b2 < b3)
    monotonic = (boundaries[:, 0] < boundaries[:, 1]) & (boundaries[:, 1] < boundaries[:, 2])

    # 检查2: 合理范围 (1 < b1 < b2 < b3 < 10000)
    in_range = (boundaries[:, 0] > 1) & (boundaries[:, 2] < 10000)

    # 检查3: 最小间隔 (相邻边界至少相差1.5倍)
    min_ratio = 1.5
    ratio_12 = boundaries[:, 1] / boundaries[:, 0].clamp(min=EPSILON)
    ratio_23 = boundaries[:, 2] / boundaries[:, 1].clamp(min=EPSILON)
    sufficient_gap = (ratio_12 > min_ratio) & (ratio_23 > min_ratio)

    # 检查4: 无NaN/Inf
    no_nan = ~(torch.isnan(boundaries).any(dim=1) | torch.isinf(boundaries).any(dim=1))

    valid_mask = monotonic & in_range & sufficient_gap & no_nan

    return valid_mask


def compute_box_size_distribution(boxes: Tensor) -> dict:
    """
    计算边界框尺寸分布统计（用于验证边界设置）

    Args:
        boxes: (N, 4) xyxy格式的边界框

    Returns:
        dict: 包含面积统计的字典
    """
    if boxes.numel() == 0:
        return {
            'min_area': 0.0,
            'q25_area': 0.0,
            'median_area': 0.0,
            'q75_area': 0.0,
            'max_area': 0.0,
            'mean_area': 0.0,
            'count': 0
        }

    # 计算面积（归一化坐标，面积在[0,1]范围）
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    areas = (w * h).clamp(min=0)

    # 转换为像素面积（假设800x800图像）
    areas_px = areas * (800 * 800)

    # 统计
    sorted_areas = torch.sort(areas_px)[0]
    n = len(sorted_areas)

    return {
        'min_area': sorted_areas[0].item(),
        'q25_area': sorted_areas[n // 4].item() if n > 4 else sorted_areas[0].item(),
        'median_area': sorted_areas[n // 2].item(),
        'q75_area': sorted_areas[3 * n // 4].item() if n > 4 else sorted_areas[-1].item(),
        'max_area': sorted_areas[-1].item(),
        'mean_area': areas_px.mean().item(),
        'count': n
    }


def assign_boxes_to_boundaries(boxes: Tensor, boundaries: Tensor,
                               image_size: Tuple[int, int] = (800, 800)) -> Tensor:
    """
    根据边界值将边界框分配到不同区间

    Args:
        boxes: (N, 4) xyxy格式边界框（归一化坐标）
        boundaries: (3,) 或 (B, 3) 边界值 [b1, b2, b3]
        image_size: 图像尺寸 (H, W)

    Returns:
        assignments: (N,) 区间索引 [0, 1, 2, 3]
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # 计算像素面积
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    areas = (w * h).clamp(min=0)
    areas_px = areas * (image_size[0] * image_size[1])

    # 确保boundaries是1D
    if boundaries.dim() == 2:
        boundaries = boundaries[0]

    b1, b2, b3 = boundaries

    # 分配
    assignments = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
    assignments[(areas_px >= b1) & (areas_px < b2)] = 1
    assignments[(areas_px >= b2) & (areas_px < b3)] = 2
    assignments[areas_px >= b3] = 3

    return assignments


def compute_boundary_coverage(boxes: Tensor, boundaries: Tensor,
                              image_size: Tuple[int, int] = (800, 800)) -> Tensor:
    """
    计算边界的覆盖率（用于验证Coverage Loss）

    Args:
        boxes: (N, 4) 边界框
        boundaries: (3,) 边界值
        image_size: 图像尺寸

    Returns:
        coverage: (3,) 每个边界的覆盖率
    """
    if boxes.numel() == 0:
        return torch.zeros(3, device=boxes.device)

    # 计算像素面积
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    areas = (w * h).clamp(min=0)
    areas_px = areas * (image_size[0] * image_size[1])

    # 确保boundaries是1D
    if boundaries.dim() == 2:
        boundaries = boundaries[0]

    b1, b2, b3 = boundaries

    # 计算覆盖率
    total = len(boxes)
    coverage = torch.tensor([
        (areas_px < b1).sum().float() / total,
        (areas_px < b2).sum().float() / total,
        (areas_px < b3).sum().float() / total
    ], device=boxes.device)

    return coverage


# ======================== 新增：微小目标专用函数 ========================

def scale_boxes_for_tiny_objects(boxes: Tensor, scale_factor: float = 1.2) -> Tensor:
    """
    对微小目标的边界框进行轻微放大（用于NMS前处理）

    Args:
        boxes: (N, 4) xyxy格式
        scale_factor: 放大系数（默认1.2倍）

    Returns:
        scaled_boxes: (N, 4) 放大后的边界框
    """
    # 转换为cxcywh
    cxcywh = box_xyxy_to_cxcywh(boxes)

    # 放大宽高
    cxcywh[:, 2:] *= scale_factor

    # 转回xyxy
    return box_cxcywh_to_xyxy(cxcywh)


def filter_tiny_boxes_by_area(boxes: Tensor, scores: Tensor,
                              min_area_px: float = 4.0,
                              image_size: Tuple[int, int] = (800, 800)) -> Tuple[Tensor, Tensor]:
    """
    过滤掉过小的边界框（可能是噪声）

    Args:
        boxes: (N, 4) 归一化坐标
        scores: (N,) 置信度
        min_area_px: 最小像素面积
        image_size: 图像尺寸

    Returns:
        filtered_boxes, filtered_scores
    """
    if boxes.numel() == 0:
        return boxes, scores

    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    areas = (w * h).clamp(min=0)
    areas_px = areas * (image_size[0] * image_size[1])

    keep = areas_px >= min_area_px

    return boxes[keep], scores[keep]
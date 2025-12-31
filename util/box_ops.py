# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# Utilities for bounding box manipulation and GIoU.
# """
# import torch, os
# from torchvision.ops.boxes import box_area
#
#
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)
#
#
# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)
#
#
# # modified from torchvision to also return the union
# def box_iou(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)
#
#
#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
#
#     union = area1[:, None] + area2 - inter
#
#     iou = inter / (union + 1e-6)
#     return iou, union
#
#
# def generalized_box_iou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#
#     The boxes should be in [x0, y0, x1, y1] format
#
#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#
#     iou, union = box_iou(boxes1, boxes2)
#
#     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     area = wh[:, :, 0] * wh[:, :, 1]
#
#     return iou - (area - union) / (area + 1e-6)
#
#
#
# # modified from torchvision to also return the union
# def box_iou_pairwise(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)
#
#     lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
#     rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]
#
#     wh = (rb - lt).clamp(min=0)  # [N,2]
#     inter = wh[:, 0] * wh[:, 1]  # [N]
#
#     union = area1 + area2 - inter
#
#     iou = inter / union
#     return iou, union
#
#
# def generalized_box_iou_pairwise(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#
#     Input:
#         - boxes1, boxes2: N,4
#     Output:
#         - giou: N, 4
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#     assert boxes1.shape == boxes2.shape
#     iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4
#
#     lt = torch.min(boxes1[:, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
#
#     wh = (rb - lt).clamp(min=0)  # [N,2]
#     area = wh[:, 0] * wh[:, 1]
#
#     return iou - (area - union) / area
#
# def masks_to_boxes(masks):
#     """Compute the bounding boxes around the provided masks
#
#     The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
#
#     Returns a [N, 4] tensors, with the boxes in xyxy format
#     """
#     if masks.numel() == 0:
#         return torch.zeros((0, 4), device=masks.device)
#
#     h, w = masks.shape[-2:]
#
#     y = torch.arange(0, h, dtype=torch.float)
#     x = torch.arange(0, w, dtype=torch.float)
#     # y, x = torch.meshgrid(y, x,indexing='ij')
#     y, x = torch.meshgrid(y, x)
#
#     x_mask = (masks * x.unsqueeze(0))
#     x_max = x_mask.flatten(1).max(-1)[0]
#     x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
#
#     y_mask = (masks * y.unsqueeze(0))
#     y_max = y_mask.flatten(1).max(-1)[0]
#     y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
#
#     return torch.stack([x_min, y_min, x_max, y_max], 1)
#
# if __name__ == '__main__':
#     x = torch.rand(5, 4)
#     y = torch.rand(3, 4)
#     iou, union = box_iou(x, y)
#     import ipdb; ipdb.set_trace()


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
Optimized for AITOD (Tiny Object Detection) with Numerical Stability.
"""
import torch
from torch import Tensor
from typing import Tuple
import warnings

# 【关键修改】提升数值保护阈值，防止FP16下溢
# 1e-4 对应 800像素图像中的 0.08 像素，对性能无影响但能防止NaN
EPSILON = 1e-4


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """
    (cx, cy, w, h) -> (x1, y1, x2, y2)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # 1. 数据清洗：将NaN替换为0，Inf替换为1
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    x_c, y_c, w, h = x.unbind(-1)

    # 2. 【关键】强制限制最小宽高，防止除零或生成无效框
    w = w.clamp(min=EPSILON)
    h = h.clamp(min=EPSILON)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """
    (x1, y1, x2, y2) -> (cx, cy, w, h)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1).clamp(min=EPSILON), (y2 - y1).clamp(min=EPSILON)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    inter, union = _box_inter_union(boxes1, boxes2)
    # 使用较大的 EPSILON
    iou = inter / union.clamp(min=EPSILON)
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Generalized IoU with strong numerical protection.
    """
    # 1. 预处理：修复无效框
    boxes1 = _fix_invalid_boxes(boxes1)
    boxes2 = _fix_invalid_boxes(boxes2)

    # 2. 计算基础 IoU
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union.clamp(min=EPSILON)

    # 3. 计算外接矩形
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = (rbi - lti).clamp(min=EPSILON)  # 宽高至少为 EPSILON
    area = whi[..., 0] * whi[..., 1]

    # 4. 计算 GIoU
    return iou - (area - union) / area.clamp(min=EPSILON)


def _fix_invalid_boxes(boxes: Tensor) -> Tensor:
    """内部函数：修复非法坐标"""
    if boxes.numel() == 0:
        return boxes

    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)

    # 强制 x2 > x1 + EPS, y2 > y1 + EPS
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    # 修正坐标顺序，并保证最小尺寸
    new_x2 = torch.max(x1 + EPSILON, x2)
    new_y2 = torch.max(y1 + EPSILON, y2)

    return torch.stack([x1, y1, new_x2, new_y2], dim=-1)


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0.0)  # 交集宽高非负即可
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

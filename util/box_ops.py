# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def box_nwd(boxes1, boxes2, constant=12.0):
    """
    Normalized Wasserstein Distance (NWD) with Sqrt for linear gradient.
    Args:
        boxes1: [N, 4] (cx, cy, w, h)
        boxes2: [M, 4] (cx, cy, w, h)
        constant: Scaling factor.
                  For normalized coords [0,1], typical tiny obj is 0.02-0.04.
                  Set constant ~0.03.
    """
    b1_cx, b1_cy, b1_w, b1_h = boxes1.unbind(-1)
    b2_cx, b2_cy, b2_w, b2_h = boxes2.unbind(-1)

    cx1, cy1, w1, h1 = b1_cx[:, None], b1_cy[:, None], b1_w[:, None], b1_h[:, None]
    cx2, cy2, w2, h2 = b2_cx[None, :], b2_cy[None, :], b2_w[None, :], b2_h[None, :]

    pow_delta_c = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    pow_delta_w = (w1 / 2 - w2 / 2) ** 2
    pow_delta_h = (h1 / 2 - h2 / 2) ** 2

    wasserstein_2_sq = pow_delta_c + pow_delta_w + pow_delta_h
    # 开根号，使得距离与尺度线性相关
    wasserstein_dist = torch.sqrt(wasserstein_2_sq.clamp(min=1e-7))

    nwd = torch.exp(-wasserstein_dist / constant)
    return nwd


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    return torch.stack([x_min, y_min, x_max, y_max], 1)


def validate_boundary_predictions(pred_boxes, image_size=None):
    if torch.isnan(pred_boxes).any():
        pred_boxes = torch.nan_to_num(pred_boxes, nan=0.0)
    pred_boxes = pred_boxes.clamp(min=0.0)
    if pred_boxes.shape[-1] == 4:
        pred_boxes[..., 2:] = pred_boxes[..., 2:].clamp(min=1e-6)
    return pred_boxes
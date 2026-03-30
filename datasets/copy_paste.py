# datasets/copy_paste.py
"""
Copy-Paste augmentation for AI-TOD small object detection.
Called AFTER _transforms, so:
  - img: FloatTensor [3, H, W], normalized (ImageNet mean/std)
  - target['boxes']: FloatTensor [N, 4], cxcywh normalized [0,1]
  - target['labels']: LongTensor [N], raw category_id
"""
import random
import torch


class CopyPasteSmallObjects:
    """
    专为微小目标设计的 Copy-Paste 增强。

    在每次调用时：
    1. 从当前图像中提取面积小于 area_threshold 的目标 patch，存入缓存
    2. 从缓存中随机取若干 patch，粘贴到当前图像的随机位置
    3. 粘贴前检查与现有框的重叠，避免覆盖已有目标

    参数：
        p               触发概率
        max_paste       每张图最多粘贴数量
        area_threshold  面积阈值（归一化面积，w*h），只粘贴小于此值的目标
                        32*32 在 800px 图中约为 0.0016
        min_area        最小面积，过滤噪声 patch（归一化）
        iou_threshold   粘贴框与现有框的最大允许 IoU
        cache_size      缓存最大容量
    """

    def __init__(self,
                 p=0.5,
                 max_paste=6,
                 area_threshold=0.0016,   # (32/800)^2 ≈ 0.0016，适合 800px 训练图
                 min_area=0.000025,       # (4/800)^2，过小的 patch 无意义
                 iou_threshold=0.15,
                 cache_size=300):
        self.p = p
        self.max_paste = max_paste
        self.area_threshold = area_threshold
        self.min_area = min_area
        self.iou_threshold = iou_threshold
        self.cache_size = cache_size
        # 缓存：list of {'patch': Tensor[3,h,w], 'label': int,
        #                'norm_w': float, 'norm_h': float}
        self._cache = []

    # ────────────────────────────────────────────
    # 内部工具函数
    # ────────────────────────────────────────────

    def _cxcywh_to_xyxy_pixel(self, boxes, H, W):
        """归一化 cxcywh → 像素 xyxy"""
        cx = boxes[:, 0] * W
        cy = boxes[:, 1] * H
        bw = boxes[:, 2] * W
        bh = boxes[:, 3] * H
        return torch.stack([cx - bw/2, cy - bh/2,
                            cx + bw/2, cy + bh/2], dim=1)

    def _iou_with_existing(self, new_xyxy, existing_xyxy):
        """
        new_xyxy: [4]  (单个框)
        existing_xyxy: [N, 4]
        返回最大 IoU（float）
        """
        if existing_xyxy.shape[0] == 0:
            return 0.0
        nb = new_xyxy.unsqueeze(0)          # [1,4]
        eb = existing_xyxy                   # [N,4]
        ix1 = torch.max(nb[:, 0], eb[:, 0])
        iy1 = torch.max(nb[:, 1], eb[:, 1])
        ix2 = torch.min(nb[:, 2], eb[:, 2])
        iy2 = torch.min(nb[:, 3], eb[:, 3])
        iw  = (ix2 - ix1).clamp(min=0)
        ih  = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        area_nb = (nb[:, 2]-nb[:, 0]) * (nb[:, 3]-nb[:, 1])
        area_eb = (eb[:, 2]-eb[:, 0]) * (eb[:, 3]-eb[:, 1])
        union = area_nb + area_eb - inter
        return (inter / (union + 1e-6)).max().item()

    # ────────────────────────────────────────────
    # 缓存更新
    # ────────────────────────────────────────────

    def _update_cache(self, img, boxes, labels, H, W):
        """从当前图像中提取小目标 patch 加入缓存"""
        for box, label in zip(boxes, labels):
            norm_w, norm_h = box[2].item(), box[3].item()
            area = norm_w * norm_h
            if area < self.min_area or area > self.area_threshold:
                continue

            # 像素坐标
            cx_px = box[0].item() * W
            cy_px = box[1].item() * H
            w_px  = norm_w * W
            h_px  = norm_h * H

            x1 = max(0, int(cx_px - w_px/2))
            y1 = max(0, int(cy_px - h_px/2))
            x2 = min(W, int(cx_px + w_px/2))
            y2 = min(H, int(cy_px + h_px/2))

            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

            patch = img[:, y1:y2, x1:x2].clone()
            self._cache.append({
                'patch': patch,
                'label': int(label.item()),
                'norm_w': norm_w,
                'norm_h': norm_h,
            })

        # FIFO 控制缓存大小
        if len(self._cache) > self.cache_size:
            self._cache = self._cache[-self.cache_size:]

    # ────────────────────────────────────────────
    # 主调用接口
    # ────────────────────────────────────────────

    def __call__(self, img, target):
        """
        img:    FloatTensor [3, H, W]，已归一化
        target: dict，包含 'boxes'(cxcywh norm) 和 'labels'
        """
        boxes  = target.get('boxes',  torch.zeros(0, 4))
        labels = target.get('labels', torch.zeros(0, dtype=torch.long))

        _, H, W = img.shape

        # 先更新缓存（无论是否触发粘贴）
        self._update_cache(img, boxes, labels, H, W)

        # 概率触发 + 缓存不足时跳过
        if random.random() > self.p or len(self._cache) < 5:
            return img, target

        # 现有框的像素 xyxy（用于重叠检测）
        if len(boxes) > 0:
            existing_xyxy = self._cxcywh_to_xyxy_pixel(boxes, H, W)
        else:
            existing_xyxy = torch.zeros(0, 4)

        img_out     = img.clone()
        new_boxes   = list(boxes)
        new_labels  = list(labels)

        # 随机采样候选（多取几个，过滤掉放不下或重叠的）
        n_candidates = min(self.max_paste * 4, len(self._cache))
        candidates   = random.sample(self._cache, n_candidates)
        n_pasted     = 0

        for item in candidates:
            if n_pasted >= self.max_paste:
                break

            patch  = item['patch']
            label  = item['label']
            ph     = patch.shape[1]
            pw     = patch.shape[2]

            # 尺寸检查
            if pw >= W - 4 or ph >= H - 4:
                continue

            # 随机选粘贴位置
            px = random.randint(0, W - pw - 1)
            py = random.randint(0, H - ph - 1)

            new_box_xyxy = torch.tensor(
                [float(px), float(py),
                 float(px + pw), float(py + ph)])

            # 重叠检测
            if self._iou_with_existing(new_box_xyxy, existing_xyxy) > self.iou_threshold:
                continue

            # 执行粘贴
            img_out[:, py:py+ph, px:px+pw] = patch

            # 添加标注（归一化 cxcywh）
            cx_n = (px + pw/2) / W
            cy_n = (py + ph/2) / H
            w_n  = pw / W
            h_n  = ph / H
            new_boxes.append(
                torch.tensor([cx_n, cy_n, w_n, h_n], dtype=torch.float32))
            new_labels.append(
                torch.tensor(label, dtype=torch.long))

            # 更新现有框列表（避免后续 patch 与新加入的框重叠）
            existing_xyxy = torch.cat(
                [existing_xyxy, new_box_xyxy.unsqueeze(0)], dim=0)
            n_pasted += 1

        if n_pasted > 0:
            target = dict(target)  # 浅拷贝，不修改原始
            target['boxes']  = torch.stack(new_boxes)
            target['labels'] = torch.stack(new_labels)

            # 同步更新 area 字段（取归一化面积 × 图像像素面积的近似）
            if 'area' in target:
                new_areas = [(b[2] * W) * (b[3] * H)
                             for b in target['boxes']]
                target['area'] = torch.tensor(new_areas, dtype=torch.float32)

            # iscrowd 字段补充（新增目标都不是 crowd）
            if 'iscrowd' in target:
                orig_len   = len(target['iscrowd'])
                extra_len  = len(target['boxes']) - orig_len
                if extra_len > 0:
                    target['iscrowd'] = torch.cat([
                        target['iscrowd'],
                        torch.zeros(extra_len, dtype=torch.long)
                    ])

        return img_out, target
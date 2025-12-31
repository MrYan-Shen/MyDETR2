import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import torch.nn as nn
import torch.nn.functional as F
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

# 【新增】导入自适应损失函数
from models.dqdetr.ccm import TrueAdaptiveBoundaryLoss

print_freq = 5000


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    # 【新增】初始化自适应边界损失函数
    # 这里的权重建议与 ccm.py 中保持一致或在此调整
    adaptive_criterion = TrueAdaptiveBoundaryLoss(
        coverage_weight=20.0,
        spacing_weight=1.0,
        count_weight=1.0,
        interval_weight=2.0
    ).to(device)
    adaptive_criterion.train()

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)

        # 【新增】准备自适应 Loss 需要的真实计数
        # 统计每张图的 GT 框数量
        real_counts = torch.stack([torch.tensor(t['labels'].shape[0], device=device) for t in targets])

        # 原始 targets 处理
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=args.amp):
            if need_tgt_for_training:
                # 注意：确保 model forward 接受 real_counts 参数（如果 ccm 需要它来做 Query 分配）
                # 通常 main_aitod.py 里的 model 是 DETR 包装器，它会把 extra args 传进去
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            # 【关键修复】确保 outputs 包含 ccm_outputs
            # 从 transformer 的输出中提取 ccm_outputs
            if isinstance(outputs, (tuple, list)):
                # 根据 deformable_transformer.py 的返回顺序
                # hs, references, hs_enc, ref_enc, init_box_proposal, dn_meta, ccm_outputs, num_select
                if len(outputs) >= 7:
                    ccm_outputs = outputs[6]
                else:
                    # 如果没有 ccm_outputs，创建一个空的
                    ccm_outputs = {}
            else:
                # 如果 outputs 是字典，直接获取
                ccm_outputs = outputs.get('ccm_outputs', {})

            # 1. 计算原始 DETR Loss (bbox, giou, label)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # 2. 【核心修改】计算自适应边界 Loss
            # 这会计算 coverage, spacing, interval 等损失，并反向传播给 boundary_head
            adaptive_targets = {'real_counts': real_counts}

            # 【修复】确保 ccm_outputs 包含必要的键
            if ccm_outputs and 'pred_boundaries' in ccm_outputs:
                adaptive_loss_out = adaptive_criterion(ccm_outputs, adaptive_targets)
                total_adaptive_loss = adaptive_loss_out['total_adaptive_loss']
                losses += total_adaptive_loss

                # 记录详细 Loss 以便观察
                loss_dict['loss_coverage'] = adaptive_loss_out.get('loss_coverage', torch.tensor(0.))
                loss_dict['loss_interval'] = adaptive_loss_out.get('loss_interval', torch.tensor(0.))
                loss_dict['loss_count'] = adaptive_loss_out.get('loss_count', torch.tensor(0.))
                loss_dict['ccm_loss'] = total_adaptive_loss  # 添加到 loss_dict 以便日志记录
            else:
                # 如果没有 ccm_outputs，使用 0 损失
                total_adaptive_loss = torch.tensor(0., device=device)
                loss_dict['loss_coverage'] = torch.tensor(0., device=device)
                loss_dict['loss_interval'] = torch.tensor(0., device=device)
                loss_dict['loss_count'] = torch.tensor(0., device=device)
                loss_dict['ccm_loss'] = torch.tensor(0., device=device)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        # 添加 adaptive loss 到日志（如果存在）
        if 'ccm_loss' in loss_dict_reduced_scaled:
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        else:
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values()) + total_adaptive_loss

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()

            # 【调试建议】如果你想再次验证梯度，可以在这里取消注释
            # if _cnt % 100 == 0:
            #     # 假设 CCM 在 model.module.transformer.CCM
            #     # print(model.module.transformer.CCM.boundary_head[-1].bias.grad)
            #     pass

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        # 更新日志
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
             args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # 【修复】确保 outputs 包含 num_select
        if isinstance(outputs, (tuple, list)):
            if len(outputs) >= 8:
                num_select = outputs[7]
            else:
                num_select = None
        else:
            num_select = outputs.get('num_select', None)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, num_select)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator
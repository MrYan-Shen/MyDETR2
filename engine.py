# import math
# import os
# import sys
# from typing import Iterable
#
# from util.utils import slprint, to_device
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import util.misc as utils
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator
#
# # 导入自适应损失函数
# from models.dqdetr.ccm import TrueAdaptiveBoundaryLoss
#
# print_freq = 5000
#
#
# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0,
#                     wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
#     scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
#
#     try:
#         need_tgt_for_training = args.use_dn
#     except:
#         need_tgt_for_training = False
#
#     # 初始化自适应边界损失函数
#     adaptive_criterion = TrueAdaptiveBoundaryLoss(
#         coverage_weight=0.5,
#         spacing_weight=2.0,
#         count_weight=0.1,
#         interval_weight=0.2,
#         boundary_guide_weight=1.0
#     ).to(device)
#     adaptive_criterion.train()
#
#     model.train()
#     criterion.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     if not wo_class_error:
#         metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#
#     _cnt = 0
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
#
#         samples = samples.to(device)
#
#         # 准备真实计数
#         real_counts = torch.tensor(
#             [len(t['labels']) for t in targets],
#             device=device,
#             dtype=torch.float32
#         )
#
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         with torch.amp.autocast('cuda', enabled=args.amp):
#             if need_tgt_for_training:
#                 outputs = model(samples, targets)
#             else:
#                 outputs = model(samples)
#
#             # 1. 计算原始DETR损失
#             loss_dict = criterion(outputs, targets)
#             weight_dict = criterion.weight_dict
#             losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#
#             # 2. 强制计算自适应边界损失
#             # 先初始化为0（防止后面忘记赋值）
#             total_adaptive_loss = torch.tensor(0., device=device, requires_grad=True)
#             loss_dict['loss_coverage'] = torch.tensor(0., device=device)
#             loss_dict['loss_interval'] = torch.tensor(0., device=device)
#             loss_dict['loss_count'] = torch.tensor(0., device=device)
#             loss_dict['loss_spacing'] = torch.tensor(0., device=device)
#             loss_dict['loss_boundary_guide'] = torch.tensor(0., device=device)
#             loss_dict['ccm_loss'] = torch.tensor(0., device=device)
#
#             # 检查CCM输出
#             has_ccm_output = False
#             if 'pred_boundaries' in outputs and 'log_boundaries' in outputs:
#                 has_ccm_output = True
#
#                 try:
#                     # 构造adaptive_targets
#                     adaptive_targets = {'real_counts': real_counts}
#                     adaptive_loss_out = adaptive_criterion(outputs, adaptive_targets)
#                     total_adaptive_loss = adaptive_loss_out['total_adaptive_loss']
#
#                     # 确保损失有梯度
#                     if not total_adaptive_loss.requires_grad:
#                         print(f"[ERROR] CCM loss没有梯度！")
#                         total_adaptive_loss = torch.tensor(0., device=device, requires_grad=True)
#
#                     # 累加到总损失
#                     losses = losses + total_adaptive_loss
#
#                     # 记录详细损失
#                     loss_dict['loss_coverage'] = adaptive_loss_out.get('loss_coverage', torch.tensor(0., device=device))
#                     loss_dict['loss_interval'] = adaptive_loss_out.get('loss_interval', torch.tensor(0., device=device))
#                     loss_dict['loss_count'] = adaptive_loss_out.get('loss_count', torch.tensor(0., device=device))
#                     loss_dict['loss_spacing'] = adaptive_loss_out.get('loss_spacing', torch.tensor(0., device=device))
#                     loss_dict['loss_boundary_guide'] = adaptive_loss_out.get('loss_boundary_guide', torch.tensor(0., device=device))
#                     loss_dict['ccm_loss'] = total_adaptive_loss
#
#                     # 每100个iter记录边界值
#                     if _cnt % 500 == 0 and args.rank == 0:
#                         boundary_vals = adaptive_loss_out['boundary_vals'].cpu().tolist()
#                         coverage_rates = adaptive_loss_out['coverage_rates'].cpu().tolist()
#
#                         loss_values = {
#                             'coverage': loss_dict['loss_coverage'].item(),
#                             'interval': loss_dict['loss_interval'].item(),
#                             'count': loss_dict['loss_count'].item(),
#                             'spacing': loss_dict['loss_spacing'].item(),
#                             'boundary_guide': loss_dict['loss_boundary_guide'].item(),
#                             'total_ccm': loss_dict['ccm_loss'].item()
#                         }
#
#                         # 计算边界相对误差
#                         target_boundaries = [10, 18, 30]
#                         boundary_errors = [
#                             abs(b - t) / t * 100 for b, t in zip(boundary_vals, target_boundaries)
#                         ]
#
#                         # 计算Coverage误差
#                         target_coverage = [0.40, 0.70, 0.90]
#                         coverage_errors = [
#                             abs(c - t) * 100 for c, t in zip(coverage_rates, target_coverage)
#                         ]
#
#                         monitor_msg = (
#                             f"\n{'=' * 70}\n"
#                             f"[CCM Monitor] Epoch {epoch}, Iter {_cnt}\n"
#                             f"  Boundaries: [{boundary_vals[0]:.1f}, {boundary_vals[1]:.1f}, {boundary_vals[2]:.1f}] px\n"
#                             f"  Errors:     [{boundary_errors[0]:.1f}%, {boundary_errors[1]:.1f}%, {boundary_errors[2]:.1f}%]\n"
#                             f"  Coverage:   [{coverage_rates[0]:.3f}, {coverage_rates[1]:.3f}, {coverage_rates[2]:.3f}]\n"
#                             f"  Errors:     [{coverage_errors[0]:.1f}%, {coverage_errors[1]:.1f}%, {coverage_errors[2]:.1f}%]\n"
#                             f"  Losses: CCM={loss_values['total_ccm']:.3f} "
#                             f"(Cov={loss_values['coverage']:.3f}, "
#                             f"Sp={loss_values['spacing']:.3f},"
#                             f"Guide={loss_values['boundary_guide']:.3f})\n"
#                             f"  Total DETR Loss: {losses.item():.4f}\n"
#                             f"  CCM/DETR Ratio: {loss_values['total_ccm'] / losses.item() * 100:.1f}%\n"
#                             f"{'=' * 70}"
#                         )
#                         print(monitor_msg)
#                         if logger:
#                             logger.info(monitor_msg)
#
#                 except Exception as e:
#                     if args.rank == 0:
#                         error_msg = f"[ERROR] 计算自适应损失失败 (Iter {_cnt}): {e}"
#                         print(error_msg)
#                         if logger:
#                             logger.error(error_msg)
#                         import traceback
#                         traceback.print_exc()
#
#                     # 失败时使用0损失
#                     total_adaptive_loss = torch.tensor(0., device=device, requires_grad=True)
#
#             # 如果没有CCM输出，警告
#             if not has_ccm_output and _cnt % 100 == 0 and args.rank == 0:
#                 print(f"[WARNING] Iter {_cnt}: 没有检测到CCM输出 (pred_boundaries)！")
#
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#
#         # 确保CCM损失在weight_dict中
#         # 临时添加CCM损失的权重（如果不存在）
#         if 'ccm_loss' not in weight_dict:
#             weight_dict['ccm_loss'] = 1.0
#         if 'loss_coverage' not in weight_dict:
#             weight_dict['loss_coverage'] = 1.0
#         if 'loss_interval' not in weight_dict:
#             weight_dict['loss_interval'] = 1.0
#         if 'loss_count' not in weight_dict:
#             weight_dict['loss_count'] = 1.0
#         if 'loss_spacing' not in weight_dict:
#             weight_dict['loss_spacing'] = 1.0
#         if 'loss_boundary_guide' not in weight_dict:
#             weight_dict['loss_boundary_guide'] = 1.0
#
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items()
#                                     if k in weight_dict}
#
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
#         loss_value = losses_reduced_scaled.item()
#
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)
#
#         # amp backward function
#         if args.amp:
#             optimizer.zero_grad()
#             scaler.scale(losses).backward()
#
#             if max_norm > 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             # original backward function
#             optimizer.zero_grad()
#             losses.backward()
#             if max_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()
#
#         if args.onecyclelr:
#             lr_scheduler.step()
#         if args.use_ema:
#             if epoch >= args.ema_epoch:
#                 ema_m.update(model)
#
#         # 更新日志
#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         if 'class_error' in loss_dict_reduced:
#             metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#
#         _cnt += 1
#
#     if getattr(criterion, 'loss_weight_decay', False):
#         criterion.loss_weight_decay(epoch=epoch)
#     if getattr(criterion, 'tuning_matching', False):
#         criterion.tuning_matching(epoch)
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
#     if getattr(criterion, 'loss_weight_decay', False):
#         resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
#     return resstat
#
#
# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
#              args=None, logger=None):
#     try:
#         need_tgt_for_training = args.use_dn
#     except:
#         need_tgt_for_training = False
#
#     model.eval()
#     criterion.eval()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     if not wo_class_error:
#         metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'
#
#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     useCats = True
#     try:
#         useCats = args.useCats
#     except:
#         useCats = True
#     if not useCats:
#         print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
#     coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
#
#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )
#
#     _cnt = 0
#     output_state_dict = {}
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
#
#         samples = samples.to(device)
#         targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
#
#         with torch.amp.autocast('cuda', enabled=args.amp):
#             if need_tgt_for_training:
#                 outputs = model(samples, targets)
#             else:
#                 outputs = model(samples)
#
#             loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict
#
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         if 'class_error' in loss_dict_reduced:
#             metric_logger.update(class_error=loss_dict_reduced['class_error'])
#
#         # 提取num_select（处理不同输出格式）
#         num_select = None
#         if 'num_select' in outputs:
#             ns = outputs['num_select']
#             if isinstance(ns, torch.Tensor):
#                 if ns.numel() > 1:
#                     num_select = int(ns.max().item())
#                 else:
#                     num_select = int(ns.item())
#             elif isinstance(ns, (int, float)):
#                 num_select = int(ns)
#
#         # 如果没有num_select，使用默认值
#         if num_select is None:
#             num_select = 300
#
#         # 限制范围
#         num_select = max(100, min(num_select, 1500))
#
#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes, num_select)
#
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)
#
#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name
#
#             panoptic_evaluator.update(res_pano)
#
#         if args.save_results:
#             for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
#                 gt_bbox = tgt['boxes']
#                 gt_label = tgt['labels']
#                 gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
#
#                 _res_bbox = outbbox
#                 _res_prob = res['scores']
#                 _res_label = res['labels']
#                 res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
#
#                 if 'gt_info' not in output_state_dict:
#                     output_state_dict['gt_info'] = []
#                 output_state_dict['gt_info'].append(gt_info.cpu())
#
#                 if 'res_info' not in output_state_dict:
#                     output_state_dict['res_info'] = []
#                 output_state_dict['res_info'].append(res_info.cpu())
#
#         _cnt += 1
#         if args.debug:
#             if _cnt % 15 == 0:
#                 print("BREAK!" * 5)
#                 break
#
#     if args.save_results:
#         import os.path as osp
#         savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
#         print("Saving res to {}".format(savepath))
#         torch.save(output_state_dict, savepath)
#
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()
#
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_res = panoptic_evaluator.summarize()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#
#     return stats, coco_evaluator


# 0109 A+B方案
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

# 导入自适应损失函数
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

    # ============ 初始化自适应边界损失函数（方案A+B）============
    adaptive_criterion = TrueAdaptiveBoundaryLoss(
        coverage_weight=0.5,  # 保持原值，或调整为 0.3
        spacing_weight=2.0,  # 保持原值，或调整为 1.0
        count_weight=0.1,  # 保持原值，或调整为 0.2
        interval_weight=0.2,  # 保持原值，或调整为 0.3
        boundary_guide_weight=1.0,  # 保持原值，或调整为 1.5
        enable_adaptive_targets=True,  # 启用自适应目标
        enable_loss_clipping=True  # 启用损失裁剪
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

        # 准备真实计数
        real_counts = torch.tensor(
            [len(t['labels']) for t in targets],
            device=device,
            dtype=torch.float32
        )

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            # 1. 计算原始DETR损失
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # 2. 强制计算自适应边界损失
            total_adaptive_loss = torch.tensor(0., device=device, requires_grad=True)
            loss_dict['loss_coverage'] = torch.tensor(0., device=device)
            loss_dict['loss_interval'] = torch.tensor(0., device=device)
            loss_dict['loss_count'] = torch.tensor(0., device=device)
            loss_dict['loss_spacing'] = torch.tensor(0., device=device)
            loss_dict['loss_boundary_guide'] = torch.tensor(0., device=device)
            loss_dict['ccm_loss'] = torch.tensor(0., device=device)

            # 检查CCM输出
            has_ccm_output = False
            if 'pred_boundaries' in outputs and 'log_boundaries' in outputs:
                has_ccm_output = True

                try:
                    # 构造 adaptive_targets
                    adaptive_targets = {'real_counts': real_counts}
                    adaptive_loss_out = adaptive_criterion(outputs, adaptive_targets)
                    total_adaptive_loss = adaptive_loss_out['total_adaptive_loss']

                    # 确保损失有梯度
                    if not total_adaptive_loss.requires_grad:
                        print(f"[ERROR] CCM loss没有梯度！")
                        total_adaptive_loss = torch.tensor(0., device=device, requires_grad=True)

                    # 累加到总损失
                    losses = losses + total_adaptive_loss

                    # 记录详细损失
                    loss_dict['loss_coverage'] = adaptive_loss_out.get('loss_coverage', torch.tensor(0., device=device))
                    loss_dict['loss_interval'] = adaptive_loss_out.get('loss_interval', torch.tensor(0., device=device))
                    loss_dict['loss_count'] = adaptive_loss_out.get('loss_count', torch.tensor(0., device=device))
                    loss_dict['loss_spacing'] = adaptive_loss_out.get('loss_spacing', torch.tensor(0., device=device))
                    loss_dict['loss_boundary_guide'] = adaptive_loss_out.get('loss_boundary_guide',
                                                                             torch.tensor(0., device=device))
                    loss_dict['ccm_loss'] = total_adaptive_loss

                    # ============ 每100个iter记录边界值（增强版监控）============
                    if _cnt % 100 == 0 and args.rank == 0:
                        boundary_vals = adaptive_loss_out['boundary_vals'].cpu().tolist()
                        coverage_rates = adaptive_loss_out['coverage_rates'].cpu().tolist()

                        loss_values = {
                            'coverage': loss_dict['loss_coverage'].item(),
                            'interval': loss_dict['loss_interval'].item(),
                            'count': loss_dict['loss_count'].item(),
                            'spacing': loss_dict['loss_spacing'].item(),
                            'boundary_guide': loss_dict['loss_boundary_guide'].item(),
                            'total_ccm': loss_dict['ccm_loss'].item()
                        }

                        # 获取自适应目标（如果有）
                        if 'adaptive_target_boundaries' in adaptive_loss_out:
                            adaptive_target_boundaries = adaptive_loss_out['adaptive_target_boundaries'].cpu().tolist()
                            adaptive_target_coverage = adaptive_loss_out['adaptive_target_coverage'].cpu().tolist()

                            # 计算与自适应目标的误差
                            boundary_errors = [
                                abs(b - t) / t * 100 for b, t in zip(boundary_vals, adaptive_target_boundaries)
                            ]
                            coverage_errors = [
                                abs(c - t) * 100 for c, t in zip(coverage_rates, adaptive_target_coverage)
                            ]

                            # 统计当前batch的目标数量分布
                            real_counts_cpu = real_counts.cpu().numpy()
                            count_stats = {
                                'min': real_counts_cpu.min(),
                                'max': real_counts_cpu.max(),
                                'mean': real_counts_cpu.mean(),
                            }

                            monitor_msg = (
                                f"\n{'=' * 80}\n"
                                f"[CCM Monitor - Adaptive] Epoch {epoch}, Iter {_cnt}\n"
                                f"  Batch Stats: Min={count_stats['min']:.0f}, Max={count_stats['max']:.0f}, Mean={count_stats['mean']:.1f}\n"
                                f"  Current Boundaries:  [{boundary_vals[0]:.1f}, {boundary_vals[1]:.1f}, {boundary_vals[2]:.1f}] px\n"
                                f"  Adaptive Targets:    [{adaptive_target_boundaries[0]:.1f}, {adaptive_target_boundaries[1]:.1f}, {adaptive_target_boundaries[2]:.1f}] px\n"
                                f"  Boundary Errors:     [{boundary_errors[0]:.1f}%, {boundary_errors[1]:.1f}%, {boundary_errors[2]:.1f}%]\n"
                                f"  Current Coverage:    [{coverage_rates[0]:.3f}, {coverage_rates[1]:.3f}, {coverage_rates[2]:.3f}]\n"
                                f"  Adaptive Cov Target: [{adaptive_target_coverage[0]:.3f}, {adaptive_target_coverage[1]:.3f}, {adaptive_target_coverage[2]:.3f}]\n"
                                f"  Coverage Errors:     [{coverage_errors[0]:.1f}%, {coverage_errors[1]:.1f}%, {coverage_errors[2]:.1f}%]\n"
                                f"  Losses: CCM={loss_values['total_ccm']:.3f} "
                                f"(Cov={loss_values['coverage']:.3f}, "
                                f"Guide={loss_values['boundary_guide']:.3f}, "
                                f"Sp={loss_values['spacing']:.3f})\n"
                                f"  Total DETR Loss: {losses.item():.4f}\n"
                                f"  CCM/DETR Ratio: {loss_values['total_ccm'] / losses.item() * 100:.1f}%\n"
                                f"{'=' * 80}"
                            )
                        else:
                            # 固定目标模式（兼容原版）
                            target_boundaries = [10, 18, 30]
                            target_coverage = [0.40, 0.70, 0.90]

                            boundary_errors = [
                                abs(b - t) / t * 100 for b, t in zip(boundary_vals, target_boundaries)
                            ]
                            coverage_errors = [
                                abs(c - t) * 100 for c, t in zip(coverage_rates, target_coverage)
                            ]

                            monitor_msg = (
                                f"\n{'=' * 70}\n"
                                f"[CCM Monitor] Epoch {epoch}, Iter {_cnt}\n"
                                f"  Boundaries: [{boundary_vals[0]:.1f}, {boundary_vals[1]:.1f}, {boundary_vals[2]:.1f}] px\n"
                                f"  Errors:     [{boundary_errors[0]:.1f}%, {boundary_errors[1]:.1f}%, {boundary_errors[2]:.1f}%]\n"
                                f"  Coverage:   [{coverage_rates[0]:.3f}, {coverage_rates[1]:.3f}, {coverage_rates[2]:.3f}]\n"
                                f"  Errors:     [{coverage_errors[0]:.1f}%, {coverage_errors[1]:.1f}%, {coverage_errors[2]:.1f}%]\n"
                                f"  Losses: CCM={loss_values['total_ccm']:.3f} "
                                f"(Cov={loss_values['coverage']:.3f}, "
                                f"Sp={loss_values['spacing']:.3f})\n"
                                f"Guide={loss_values['boundary_guide']:.3f})\n"
                                f"  Total DETR Loss: {losses.item():.4f}\n"
                                f"  CCM/DETR Ratio: {loss_values['total_ccm'] / losses.item() * 100:.1f}%\n"
                                f"{'=' * 70}"
                            )

                        print(monitor_msg)
                        if logger:
                            logger.info(monitor_msg)

                except Exception as e:
                    if args.rank == 0:
                        error_msg = f"[ERROR] 计算自适应损失失败 (Iter {_cnt}): {e}"
                        print(error_msg)
                        if logger:
                            logger.error(error_msg)
                        import traceback
                        traceback.print_exc()

                    # 失败时使用0损失
                    total_adaptive_loss = torch.tensor(0., device=device, requires_grad=True)

            # 如果没有CCM输出，警告
            if not has_ccm_output and _cnt % 100 == 0 and args.rank == 0:
                print(f"[WARNING] Iter {_cnt}: 没有检测到CCM输出 (pred_boundaries)！")

        # ============ 后续代码保持不变 ============
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        # 确保CCM损失在weight_dict中
        if 'ccm_loss' not in weight_dict:
            weight_dict['ccm_loss'] = 1.0
        if 'loss_coverage' not in weight_dict:
            weight_dict['loss_coverage'] = 1.0
        if 'loss_interval' not in weight_dict:
            weight_dict['loss_interval'] = 1.0
        if 'loss_count' not in weight_dict:
            weight_dict['loss_count'] = 1.0
        if 'loss_spacing' not in weight_dict:
            weight_dict['loss_spacing'] = 1.0
        if 'loss_boundary_guide' not in weight_dict:
            weight_dict['loss_boundary_guide'] = 1.0

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items()
                                    if k in weight_dict}

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()

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

    # ============ 后续代码完全保持不变 ============
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
    output_state_dict = {}
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

        # 提取num_select（处理不同输出格式）
        num_select = None
        if 'num_select' in outputs:
            ns = outputs['num_select']
            if isinstance(ns, torch.Tensor):
                if ns.numel() > 1:
                    num_select = int(ns.max().item())
                else:
                    num_select = int(ns.item())
            elif isinstance(ns, (int, float)):
                num_select = int(ns)

        # 如果没有num_select，使用默认值
        if num_select is None:
            num_select = 300

        # 限制范围
        num_select = max(100, min(num_select, 1500))

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



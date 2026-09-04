#!/usr/bin/env python
"""
Train TSViT Baseline for T3S Crop Mapping
==========================================

Trains the Temporal-Spatial ViT (TSViT, Tarasiou et al. 2023) on the Swiss
crop mapping dataset. Uses the same SatelliteDataset and train/val/test splits
as train_utae.py for a fair comparison.

Because TSViT operates on fixed 16×16 spatial patches, each tile from
SatelliteDataset is cropped on-the-fly. During training a random 16×16 crop
is taken per tile; during validation / test a centre crop is used.

Usage (2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=13347 train_tsvit.py \\
      --res_dir ./storage/results_tsvit --satellite sentinel --ablation_split SA1

Author: Ozgur (Agroscope) — T3S paper baseline integration
"""

import argparse
import json
import os
import pickle as pkl
import pprint
import calendar
import math
import random
import time
import sys
import re
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score
import fsspec
import zarr

import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler


class SkipDistributedSampler(DistributedSampler):
    """DistributedSampler that skips the first `skip_first` indices on the first iteration.
    Enables fast resume: the DataLoader workers never load skipped batches."""
    def __init__(self, *args, skip_first=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_first = skip_first

    def __iter__(self):
        indices = list(super().__iter__())
        skip = self._skip_first
        self._skip_first = 0  # only apply once; subsequent epochs iterate fully
        return iter(indices[skip:])

    def __len__(self):
        return max(0, super().__len__() - self._skip_first)
import torch.utils.data as data
import torchnet as tnt
import torch.distributed as dist
from torch.amp import autocast

from src import utils
from src.utils import zarr_cache
from src.dataset import SatelliteDataset
from src.tsvit.TSViTdense import TSViT
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.utils import calculate_f1_score, compute_ece
from src.learning.losses import create_loss_function, MultiTaskLoss


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### Argument parsing
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
parser = argparse.ArgumentParser(description="TSViT baseline for T3S crop mapping")

# TSViT model config
parser.add_argument("--img_res", default=16, type=int,
                    help="Spatial patch size for TSViT (default: 16)")
parser.add_argument("--patch_size", default=2, type=int,
                    help="TSViT internal patch size (default: 2)")
parser.add_argument("--max_seq_len", default=24, type=int,
                    help="Temporal sequence length (default: 24)")
parser.add_argument("--dim", default=128, type=int)
parser.add_argument("--temporal_depth", default=4, type=int)
parser.add_argument("--spatial_depth", default=4, type=int)
parser.add_argument("--heads", default=4, type=int)
parser.add_argument("--dim_head", default=32, type=int)
parser.add_argument("--scale_dim", default=4, type=int)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--emb_dropout", default=0.0, type=float)
parser.add_argument("--pool", default="cls", type=str, choices=["cls", "mean"])
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--bias_initialization", action="store_true",
                    help="Initialize final bias using log(class frequency).")

# Data parameters
parser.add_argument("--dataset_folder", type=str, default=None, help="Root of the SwissCrop25 dataset (sentinel2/, labels/, temperature/ subdirs). If not set, falls back to hardcoded HPC paths.")
parser.add_argument("--label_sheet_file", type=str, default='label_sheet.csv')
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--num_bands", default=9, type=int)
parser.add_argument("--train_dataset_portion", default=1.0, type=float)
parser.add_argument("--val_dataset_portion", default=1.0, type=float)
parser.add_argument("--test_dataset_portion", default=1.0, type=float)
parser.add_argument("--temporal_length", default=24, type=int)
parser.add_argument("--truncate_month", default=None, type=int, help="Truncate test data to first N months (1-12); implies eval-only mode")
parser.add_argument("--eval_only", action="store_true", help="Skip training; load best checkpoint from res_dir and run test only")
parser.add_argument("--eval_all_months", action="store_true", help="Load full-year test data once; evaluate at each month cutoff in one pass (12x faster than --truncate_month loop)")
parser.add_argument("--variable_t", action="store_true", help="In --eval_all_months mode, compact each sample to its real observations only (variable T per month) instead of zero-padding future slots")
parser.add_argument("--use_temperature_calendar", action="store_true")
parser.add_argument("--use_temperature_subsampling", action="store_true")
parser.add_argument("--no_sliding_subsample", action="store_true")
parser.add_argument("--use_temperature_calendar_no_sliding_subsample", action="store_true")
parser.add_argument("--use_fixed_temperature_subsampling", action="store_true")
parser.add_argument("--use_upsampling", action="store_true", default=False)
parser.add_argument("--target_ratio", default=0.05, type=float)
parser.add_argument("--minority_threshold", default=0.0005, type=float)
parser.add_argument("--min_class_pixels", default=None, type=int)
parser.add_argument("--simulate_landsat", action="store_true")
parser.add_argument("--revisit_time", default=8, type=int)
parser.add_argument("--oli_degrade_prob", default=0.0, type=float)
parser.add_argument("--use_sensor_flag", action="store_true")
parser.add_argument("--no_normalize_timestamps", action="store_true", help="Return raw GDD and DOY (1-365) instead of z-scored; needed for sinusoidal PE")
parser.add_argument("--use_gdd_pe", action="store_true", help="Use z-scored CGDD as temporal PE instead of DOY (bench condition 3)")
parser.add_argument("--temp_mean_var", default="CGDD_mean", type=str)
parser.add_argument("--temp_spatial_var", default="None", type=str)
parser.add_argument("--pad_value", default=0, type=float)

# Experiment parameters
parser.add_argument("--res_dir", type=str, default='./storage/results_tsvit')
parser.add_argument("--checkpoint_dir", default=None, type=str)
parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoints and start fresh")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--pct_start", default=0.05, type=float)
parser.add_argument("--checkpoint_steps", default=500, type=int, help="Save model_latest.pth every N optimizer steps (0 = epoch-only)")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size per GPU (patches, not tiles)")
parser.add_argument("--accumulate_steps", default=1, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lr_start", default=1e-4, type=float)
parser.add_argument("--lr_end", default=1e-7, type=float)
parser.add_argument("--fold", default=None, type=int)
parser.add_argument("--num_classes", default=71, type=int)
parser.add_argument("--ignore_index", default=0, type=int)
parser.add_argument("--val_every", default=10, type=int)

# Misc parameters
parser.add_argument("--rdm_seed", default=6666, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--display_step", default=50, type=int)
parser.add_argument("--cache", dest="cache", action="store_true")
parser.set_defaults(cache=False)
parser.add_argument("--satellite", default="sentinel", choices=["landsat", "sentinel"])
parser.add_argument("--ablation_split", default=None,
                    choices=["SA1", "SA2", "SA3", "S1", "S2", "S3", "S4", "S5", "ALL", "ALL_wo_19_20"])

# Loss parameters
parser.add_argument("--use_class_balance_loss", action="store_true")
parser.add_argument("--beta_class_balance", default=0.99999, type=float)
parser.add_argument("--w_ce", default=1.0, type=float)
parser.add_argument("--w_focal", default=0, type=float)
parser.add_argument("--w_dice", default=0, type=float)
parser.add_argument("--w_logcosh_dice", default=0, type=float)
parser.add_argument("--w_kl", default=0, type=float)
parser.add_argument("--focal_gamma", default=2, type=float)
parser.add_argument("--w_tversky", default=0, type=float)
parser.add_argument("--tversky_alpha", default=0.3, type=float)
parser.add_argument("--tversky_beta", default=0.7, type=float)


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### TSViT Patch Dataset Wrapper
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

class TSViTPatchDataset(data.Dataset):
    """Wraps SatelliteDataset to return 16×16 patches formatted for TSViT.

    Each item is a tuple ``(patch, label)`` where:
    - ``patch``: float32 tensor of shape ``[T, img_res, img_res, C+1]``
      — the last channel carries the scalar GDD timestamp broadcast to every
      spatial location, matching TSViT's expected format.
    - ``label``: int64 tensor of shape ``[img_res, img_res]``

    Training mode: one random crop per tile per __getitem__ call.
    Eval mode: centre crop for deterministic evaluation.
    """

    def __init__(self, base_dataset, img_res=16, training=True, use_gdd_pe=False):
        self.base = base_dataset
        self.img_res = img_res
        self.training = training
        self.use_gdd_pe = use_gdd_pe

        # Always tile the full tile for all splits (train/val/test consistent)
        (imgs, _), gt, *_ = base_dataset[0]
        H, W = imgs.shape[-2], imgs.shape[-1]
        self._n_h = math.ceil(H / img_res)
        self._n_w = math.ceil(W / img_res)
        self._n_patches = self._n_h * self._n_w

    def __len__(self):
        return len(self.base)  # one item = one tile

    @property
    def mapping_dict(self):
        return self.base.mapping_dict

    def __getitem__(self, idx):
        r = self.img_res
        (images, dates), ground_truth, *_ = self.base[idx]  # single zarr read

        ground_truth = ground_truth[0] if isinstance(ground_truth, list) else ground_truth

        if isinstance(dates, tuple):
            if self.use_gdd_pe:
                time_stamps = dates[0]  # z-scored CGDD
            else:
                time_stamps = dates[1] if len(dates) >= 2 else dates[0]  # DOY
        else:
            time_stamps = dates

        T, C, H, W = images.shape

        pad_h = max(0, self._n_h * r - H)
        pad_w = max(0, self._n_w * r - W)

        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            ground_truth = F.pad(ground_truth.unsqueeze(0),
                                 (0, pad_w, 0, pad_h)).squeeze(0)

        ts = time_stamps.float()
        ts_spatial = ts.view(T, 1, 1, 1).expand(T, 1, r, r).contiguous()

        patches = []
        gt_patches = []
        for ph in range(self._n_h):
            for pw in range(self._n_w):
                h0 = ph * r
                w0 = pw * r
                patch    = images[:, :, h0:h0+r, w0:w0+r]   # [T, C, r, r]
                gt_patch = ground_truth[h0:h0+r, w0:w0+r]   # [r, r]
                patch = torch.cat([patch, ts_spatial], dim=1)  # [T, C+1, r, r]
                patch = patch.permute(0, 2, 3, 1)             # [T, r, r, C+1]
                patches.append(patch)
                gt_patches.append(gt_patch)

        return torch.stack(patches), torch.stack(gt_patches).long()
        # shapes: [n_patches, T, r, r, C+1] and [n_patches, r, r]


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### Helper functions
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def tsvit_collate_fn(batch):
    # batch: list of B tile items, each ([n_patches, T, r, r, C+1], [n_patches, r, r])
    # flatten tiles → flat patch batch for iterate()
    patches = torch.cat([item[0] for item in batch], dim=0)  # [B*n_patches, T, r, r, C+1]
    labels  = torch.cat([item[1] for item in batch], dim=0)  # [B*n_patches, r, r]
    return patches, labels

def list_to_tuple(batch):
    """Convert batch structure from lists back to tuples after DataLoader pickling."""
    if not isinstance(batch, (list, tuple)) or len(batch) not in (2, 3):
        return batch
    if isinstance(batch, list):
        batch = tuple(batch)
    x_dates = batch[0]
    if isinstance(x_dates, list):
        x_dates = tuple(x_dates)
    if len(x_dates) >= 2 and isinstance(x_dates[1], list):
        x_dates = (x_dates[0], tuple(x_dates[1]))
    return (x_dates,) + batch[1:]

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return tuple(recursive_todevice(item, device) for item in x)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def convert_to_native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    if config.checkpoint_dir is None:
        config.checkpoint_dir = config.res_dir

def checkpoint(fold, trainlog, config):
    log_path = os.path.join(config.res_dir, "trainlog.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as infile:
            try:
                log = json.load(infile)
            except json.JSONDecodeError:
                log = {}
    else:
        log = {}
    latest_epoch = max(int(k) for k in trainlog.keys())
    epoch_data = trainlog.get(latest_epoch, trainlog.get(str(latest_epoch)))
    log[str(latest_epoch)] = epoch_data
    with open(log_path, "w") as outfile:
        json.dump(log, outfile, indent=4, default=convert_to_native)

def test_all_months(model, base_loader, device, config, amp_dtype=torch.float32):
    """
    One pass over the full-year *tile-level* test loader (SatelliteDataset, not TSViTPatchDataset).
    Distributed: each rank handles months where (m - 1) % world_size == rank.
    Computes IoU/Acc + ECE/NLL/Brier per month over the full test set.
    Saves test_metrics_month{M}.json + conf_mat_month{M}.pkl for each assigned month.
    """
    DOY_MEAN, DOY_STD = 183.0, 105.1
    r = config.img_res  # 16
    test_year = config.test_years[0]

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    assigned_months = [m for m in range(1, 13) if (m - 1) % world_size == rank]
    if not assigned_months:
        if dist.is_initialized():
            dist.barrier()
        return

    month_cutoffs = {
        m: sum(calendar.monthrange(test_year, mm)[1] for mm in range(1, m + 1))
        for m in assigned_months
    }

    iou_meters = {
        m: IoU(num_classes=config.num_classes, ignore_index=config.ignore_index, cm_device=config.device)
        for m in assigned_months
    }
    conf_lists = {m: [] for m in assigned_months}
    corr_lists = {m: [] for m in assigned_months}
    gt_lists = {m: [] for m in assigned_months}
    nll_sums = {m: 0.0 for m in assigned_months}
    brier_sums = {m: 0.0 for m in assigned_months}
    valid_counts = {m: 0 for m in assigned_months}

    print(f"[rank {rank}] eval_all_months months: {assigned_months}", flush=True)

    model.eval()
    n_batches = len(base_loader)

    for i, batch in enumerate(base_loader):
        if (i + 1) % 200 == 0 and rank == 0:
            print(f"  [eval_all_months] batch {i+1}/{n_batches}", flush=True)

        # Batch from SatelliteDataset: ((images, dates_tuple), gt, alpine)
        batch = list_to_tuple(batch)
        batch = recursive_todevice(batch, device)
        (images, dates), gt, *_ = batch
        gt = gt[0] if isinstance(gt, list) else gt

        if isinstance(dates, tuple):
            cgdd_d = dates[0]
            doy_d  = dates[1] if len(dates) >= 2 else None
        else:
            cgdd_d, doy_d = dates, None

        if doy_d is not None:
            doy_raw = doy_d if getattr(config, 'no_normalize_timestamps', False) \
                      else doy_d * DOY_STD + DOY_MEAN
        else:
            doy_raw = cgdd_d

        # PE timestamps: z-scored CGDD when --use_gdd_pe, else DOY
        pe_raw = cgdd_d if getattr(config, 'use_gdd_pe', False) else doy_raw

        B, T, C, H, W = images.shape
        is_real = images.abs().sum(dim=(2, 3, 4)) > 0  # (B, T)

        n_h = math.ceil(H / r)
        n_w = math.ceil(W / r)
        pad_h = max(0, n_h * r - H)
        pad_w = max(0, n_w * r - W)
        if pad_h > 0 or pad_w > 0:
            images_pad = F.pad(images, (0, pad_w, 0, pad_h))
            gt_pad = F.pad(gt.unsqueeze(1).float(), (0, pad_w, 0, pad_h),
                           value=config.ignore_index).squeeze(1).long()
        else:
            images_pad = images
            gt_pad = gt

        for month in assigned_months:
            cutoff_doy = month_cutoffs[month]
            keep = is_real & (doy_raw <= cutoff_doy)  # (B, T)

            H_pad = images_pad.shape[-2]
            W_pad = images_pad.shape[-1]

            if getattr(config, 'variable_t', False):
                # Compact: gather real timesteps per sample, pad batch to max_T
                T_counts = keep.sum(dim=1)  # (B,)
                seq_T = int(T_counts.max().item())
                if seq_T == 0:
                    continue
                images_seq = torch.zeros(B, seq_T, C, H_pad, W_pad,
                                         device=images_pad.device, dtype=images_pad.dtype)
                pe_seq     = torch.zeros(B, seq_T, device=pe_raw.device, dtype=pe_raw.dtype)
                for b_idx in range(B):
                    t_idx = keep[b_idx].nonzero(as_tuple=False).view(-1)
                    n = t_idx.numel()
                    images_seq[b_idx, :n] = images_pad[b_idx, t_idx]
                    pe_seq[b_idx, :n]     = pe_raw[b_idx, t_idx]
            else:
                if not keep.any():
                    continue
                seq_T = T
                k = keep.to(images_pad.dtype)
                images_seq = images_pad * k.view(B, T, 1, 1, 1)
                pe_seq     = pe_raw * k

            # Build patch tensor: (B*n_patches, seq_T, r, r, C+1)
            patch_rows, gt_rows = [], []
            for ph in range(n_h):
                for pw in range(n_w):
                    h0, w0 = ph * r, pw * r
                    p = images_seq[:, :, :, h0:h0+r, w0:w0+r]         # (B, seq_T, C, r, r)
                    ts = pe_seq.view(B, seq_T, 1, 1, 1).expand(B, seq_T, 1, r, r)
                    p = torch.cat([p, ts], dim=2)                       # (B, seq_T, C+1, r, r)
                    patch_rows.append(p.permute(0, 1, 3, 4, 2))        # (B, seq_T, r, r, C+1)
                    gt_rows.append(gt_pad[:, h0:h0+r, w0:w0+r])        # (B, r, r)

            all_patches = torch.stack(patch_rows, dim=1).view(B * n_h * n_w, seq_T, r, r, C + 1)
            all_gt      = torch.stack(gt_rows,    dim=1).view(B * n_h * n_w, r, r).long()

            with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
                out = model(all_patches)  # (B*n_patches, num_classes, r, r)

            prob = torch.softmax(out, dim=1)
            confidence, pred_eval = torch.max(prob, dim=1)
            iou_meters[month].add(pred_eval, all_gt)

            conf_lists[month].append(confidence.cpu().float().numpy())
            corr_lists[month].append((pred_eval == all_gt).cpu().numpy())
            gt_lists[month].append(all_gt.cpu().numpy())

            eps = 1e-10
            log_prob = torch.log(prob + eps)
            nll_batch = F.nll_loss(log_prob, all_gt, reduction='none', ignore_index=config.ignore_index)
            one_hot = F.one_hot(all_gt, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
            brier_batch = torch.sum((prob - one_hot) ** 2, dim=1)
            valid_mask = (all_gt != config.ignore_index)
            nll_sums[month] += nll_batch[valid_mask].sum().item()
            brier_sums[month] += brier_batch[valid_mask].sum().item()
            valid_counts[month] += valid_mask.sum().item()

    for month in assigned_months:
        cutoff_doy = month_cutoffs[month]
        miou, acc = iou_meters[month].get_miou_acc(sync=False)
        global_iou = iou_meters[month].get_global_iou(sync=False)
        cm_val = iou_meters[month].conf_metric.value()
        ece = compute_ece(conf_lists[month], corr_lists[month], gt_lists[month],
                          config.ignore_index, num_bins=15)
        nll = nll_sums[month] / valid_counts[month] if valid_counts[month] > 0 else float('nan')
        brier = brier_sums[month] / valid_counts[month] if valid_counts[month] > 0 else float('nan')
        class_df = iou_meters[month].get_per_class_metrics(
            class_mapping=base_loader.dataset.mapping_dict, sync=False)
        print(f"\n--- Month {month} (DOY ≤ {cutoff_doy}) Acc: {acc:.2f}%  mIoU: {miou:.4f}%  ECE: {ece:.4f}  NLL: {nll:.4f}  Brier: {brier:.4f} ---", flush=True)
        print(class_df.to_string(index=False))
        metrics = {
            "test_accuracy": acc, "test_IoU": miou, "test_global_IoU": global_iou,
            "test_ECE": ece, "test_NLL": nll, "test_Brier": brier,
        }
        save_results(1, metrics, cm_val.cpu().float().numpy(), config,
                     filename=f"test_metrics_month{month}.json")

    print(f"\n[rank {rank}] [eval_all_months] Saved test_metrics for months {assigned_months}", flush=True)
    if dist.is_initialized():
        dist.barrier()


def save_results(fold, metrics, conf_mat, config, filename="test_metrics.json"):
    with open(os.path.join(config.res_dir, filename), "w") as outfile:
        json.dump(metrics, outfile, indent=4, default=convert_to_native)
    pkl_name = filename.replace(".json", ".pkl")
    pkl.dump(conf_mat, open(os.path.join(config.res_dir, pkl_name), "wb"))

def overall_performance(config):
    cm = np.zeros((config.num_classes, config.num_classes))
    for fold in [1]:
        cm += pkl.load(open(os.path.join(config.res_dir, "test_metrics.pkl"), "rb"))

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:", flush=True)
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]), flush=True)

    with open(os.path.join(config.res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4, default=convert_to_native))


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### Training functions
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def sync_across_gpus(tensor_or_float):
    """Synchronize a scalar value across all GPUs in DDP."""
    if not dist.is_initialized():
        return tensor_or_float

    if not isinstance(tensor_or_float, torch.Tensor):
        tensor = torch.tensor(tensor_or_float, device='cuda')
    else:
        tensor = tensor_or_float.clone().detach()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()

    return tensor.item() if not isinstance(tensor_or_float, torch.Tensor) else tensor


def iterate(model, data_loader, criterion, config, optimizer=None, scheduler=None, mode="train", device=None, local_rank=0, force_no_sync=False, epoch=0, checkpoint_steps=0, resume_step=0, skip_handled=False):
    """
    Single epoch iteration function with support for training, validation, and testing modes.

    Args:
        force_no_sync: If True, disable all distributed synchronization (for single-GPU evaluation)
    """

    ### Setup of meters and timers
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    f1_scores = []
    t_start = time.time()
    accumulation_counter = 0
    opt_step = resume_step
    loss_components = {}
    # Windowed meters for display (last display_step batches only)
    window_losses = []
    window_iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    if mode in ["test", "val"]:
        eval_confidences_list = []
        eval_correctness_list = []
        eval_ground_truths = []
        total_nll_sum = 0.0
        total_brier_sum = 0.0
        total_valid_pixels = 0

    ### Start of main loop
    if mode == "train" and optimizer is not None:
        optimizer.zero_grad()

    skip_batches = 0 if skip_handled else (resume_step * config.accumulate_steps if mode == "train" else 0)
    if skip_batches > 0 and local_rank == 0:
        print(f"[Resume] Skipping {skip_batches} batches (resume from opt_step={resume_step})", flush=True)
    elif skip_handled and resume_step > 0 and local_rank == 0:
        print(f"[Resume] Fast-resume from opt_step={resume_step} (sampler-based skip)", flush=True)

    t_prev_batch_end = time.time()
    n_batches = len(data_loader)
    _step_offset = resume_step * config.accumulate_steps if (skip_handled and mode == "train") else 0
    for i, (x, y) in enumerate(data_loader):
        if i < skip_batches:
            continue
        t0 = time.time()
        x = x.to(device)           # [B*n_patches, T, r, r, C+1]
        y_eval = y.long().to(device)  # [B*n_patches, r, r]

        # Strip empty (zero-padded) timesteps so TSViT only sees real observations.
        # Detect via spectral channels (last channel is DOY timestamp, may be non-zero for empty slots).
        if getattr(config, 'variable_t', False):
            is_real = x[:, :, :, :, :-1].abs().sum(dim=(2, 3, 4)) > 0  # (B_flat, T)
            # Drop samples where every timestep is empty (degenerate all-zero batches)
            n_real = is_real.sum(dim=1)
            if (n_real == 0).any():
                valid = n_real > 0
                if not valid.any():
                    continue
                x, y_eval, is_real = x[valid], y_eval[valid], is_real[valid]
            if not is_real.all():
                max_T = int(is_real.sum(dim=1).max().item())
                x_c = torch.zeros(x.shape[0], max_T, *x.shape[2:], device=x.device, dtype=x.dtype)
                for b in range(x.shape[0]):
                    t_idx = is_real[b].nonzero(as_tuple=False).view(-1)
                    n = t_idx.numel()
                    x_c[b, :n] = x[b, t_idx]
                x = x_c

        # Forward pass
        amp_dtype = getattr(config, 'amp_dtype', torch.float32)
        if mode == "train":
            with autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)     # [B, num_classes, r, r]
        else:
            with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)

        # Compute loss
        if mode == "train":
            with autocast(device_type="cuda", dtype=amp_dtype):
                if isinstance(criterion, MultiTaskLoss):
                    loss, batch_loss_dict = criterion(out, y_eval, return_components=True)
                    for key, val in batch_loss_dict.items():
                        if key not in loss_components:
                            loss_components[key] = []
                        loss_components[key].append(val.item() if torch.is_tensor(val) else val)
                else:
                    loss = criterion(out, y_eval)
                    batch_loss_dict = {'ce_loss': loss.item()}
                    for key, val in batch_loss_dict.items():
                        if key not in loss_components:
                            loss_components[key] = []
                        loss_components[key].append(val)
        else:
            with autocast(device_type="cuda", dtype=amp_dtype):
                if isinstance(criterion, MultiTaskLoss):
                    loss = criterion(out, y_eval, return_components=False)
                else:
                    loss = criterion(out, y_eval)

        # Backward pass and weight updates
        if mode == "train":
            loss = loss / config.accumulate_steps
            accumulation_counter += 1
            loss.backward()

            if accumulation_counter == config.accumulate_steps:
                _bad_grad = False
                for _p in model.parameters():
                    if _p.grad is not None and not _p.grad.isfinite().all():
                        _p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        _bad_grad = True
                if _bad_grad and local_rank == 0:
                    print(f"[GRAD-WARN] NaN/Inf gradients zeroed at opt_step={opt_step}", flush=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                accumulation_counter = 0
                opt_step += 1
                if checkpoint_steps > 0 and local_rank == 0 and opt_step % checkpoint_steps == 0:
                    _m = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'epoch': epoch - 1,
                        'step': opt_step,
                        'model_state_dict': _m.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    }, os.path.join(config.checkpoint_dir, 'model_latest.pth'))
                    print(f"[Step ckpt] epoch={epoch} step={opt_step}", flush=True)

        # Compute evaluation metrics
        if mode in ["test", "val"]:
            prob = torch.softmax(out, dim=1)
            confidence, pred_eval = torch.max(prob, dim=1)
            eval_confidences_list.append(confidence.cpu().float().numpy())
            eval_correctness_list.append((pred_eval == y_eval).cpu().numpy())
            eval_ground_truths.append(y_eval.cpu().numpy())

            eps = 1e-10
            log_prob = torch.log(prob + eps)
            nll_batch = F.nll_loss(log_prob, y_eval, reduction='none', ignore_index=config.ignore_index)
            one_hot = F.one_hot(y_eval, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
            brier_batch = torch.sum((prob - one_hot)**2, dim=1)
            valid_mask = (y_eval != config.ignore_index)
            total_nll_sum += nll_batch[valid_mask].sum().item()
            total_brier_sum += brier_batch[valid_mask].sum().item()
            total_valid_pixels += valid_mask.sum().item()
        else:
            with torch.no_grad():
                pred_eval = out.argmax(dim=1)

        iou_meter.add(pred_eval, y_eval)
        window_iou_meter.add(pred_eval, y_eval)
        _loss_val = loss.item() * (config.accumulate_steps if mode == "train" else 1)
        loss_meter.add(_loss_val)
        window_losses.append(_loss_val)
        if mode != "train" or (i + 1) % config.display_step == 0:
            batch_f1 = calculate_f1_score(pred_eval, y_eval, num_classes=config.num_classes, ignore_index=config.ignore_index)
            f1_scores.append(batch_f1)

        # Display training information
        if (i + 1) % config.display_step == 0 and local_rank == 0:
            miou, acc = window_iou_meter.get_miou_acc(sync=False)
            torch.cuda.synchronize()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tend = time.time() - t0

            loss_str = f"Loss: {np.mean(window_losses):.4f}"
            if mode == "train" and loss_components:
                component_strs = [f"{k}: {np.mean(v):.4f}" for k, v in loss_components.items() if not k.endswith('_weight')]
                if component_strs:
                    loss_str += f" ({', '.join(component_strs)})"

            print(
                "[{}] Step [{}/{}], Time: {:.4f}, Acc: {:.2f}%, mIoU: {:.2f}%, F1: {:.2f}%, {}".format(
                    current_time, i + 1 + _step_offset, n_batches + _step_offset, tend, acc, miou, batch_f1 * 100, loss_str
                ), flush=True
            )
            # Reset windowed meters
            window_losses = []
            window_iou_meter = IoU(
                num_classes=config.num_classes,
                ignore_index=config.ignore_index,
                cm_device=config.device,
            )

    # Final optimizer step if needed
    if mode == "train" and accumulation_counter > 0:
        for _p in model.parameters():
            if _p.grad is not None and not _p.grad.isfinite().all():
                _p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    # End of epoch metrics computation
    t_end = time.time()
    total_time = t_end - t_start
    if local_rank == 0:
        print("Epoch time: {:.1f}s".format(total_time), flush=True)

    should_sync = (mode in ["test", "val"]) and not force_no_sync
    miou, acc = iou_meter.get_miou_acc(sync=should_sync)
    global_iou = iou_meter.get_global_iou(sync=should_sync)

    if should_sync:
        avg_f1 = sync_across_gpus(np.mean(f1_scores))
        avg_loss = sync_across_gpus(loss_meter.value()[0])
    else:
        avg_f1 = np.mean(f1_scores)
        avg_loss = loss_meter.value()[0]

    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): avg_loss,
        "{}_IoU".format(mode): miou,
        "{}_global_IoU".format(mode): global_iou,
        "{}_F1".format(mode): avg_f1,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "train" and loss_components:
        for key, values in loss_components.items():
            metrics[f"{mode}_{key}"] = np.mean(values)

    lr_str = None
    if mode == "train" and scheduler is not None:
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]['lr']
        metrics[f"{mode}_lr"] = current_lr
        lr_str = f"LR: {current_lr:.6f} "

    if mode == "train":
        if local_rank == 0:
            print(f"\n--- Epoch summary ---")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Averages, Acc: {acc:.2f}%, mIoU: {miou:.2f}%, global_IoU: {global_iou:.2f}%, F1: {avg_f1 * 100:.2f}%")
            if lr_str:
                print(f"{lr_str}")
            class_metrics_df = iou_meter.get_per_class_metrics(class_mapping=data_loader.dataset.mapping_dict, sync=False)
            print(f"\n--- Class statistics ---")
            print(class_metrics_df.to_string(index=False))

    if mode in ["test", "val"]:
        if should_sync:
            total_nll_sum = sync_across_gpus(total_nll_sum)
            total_brier_sum = sync_across_gpus(total_brier_sum)
            total_valid_pixels = sync_across_gpus(float(total_valid_pixels))

        ece_value = compute_ece(eval_confidences_list, eval_correctness_list, eval_ground_truths, config.ignore_index, num_bins=15)

        if should_sync:
            ece_value = sync_across_gpus(ece_value)

        avg_nll = total_nll_sum / total_valid_pixels if total_valid_pixels > 0 else float('nan')
        avg_brier = total_brier_sum / total_valid_pixels if total_valid_pixels > 0 else float('nan')
        metrics[f"{mode}_ECE"] = ece_value
        metrics[f"{mode}_NLL"] = avg_nll
        metrics[f"{mode}_Brier"] = avg_brier

        class_metrics_df = iou_meter.get_per_class_metrics(class_mapping=data_loader.dataset.mapping_dict, sync=should_sync)
        if local_rank == 0:
            print(f"\n--- {mode.capitalize()} Class statistics ---")
            print(class_metrics_df.to_string(index=False))

        if mode == "test":
            return metrics, iou_meter.conf_metric.value()
        else:
            return metrics
    else:
        return metrics


def run_training_phase(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    config, device, global_rank, start_epoch, end_epoch, trainlog, best_mIoU, best_epoch, phase_name="", resume_step=0
):
    """Run a training phase."""
    for epoch in range(start_epoch, end_epoch + 1):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if global_rank == 0:
            prefix = f"[{phase_name}] " if phase_name else ""
            print(f"{prefix}EPOCH {epoch}/{end_epoch}", flush=True)

        # Training step
        model.train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            mode="train",
            device=device,
            local_rank=global_rank,
            epoch=epoch,
            checkpoint_steps=getattr(config, 'checkpoint_steps', 0),
            resume_step=resume_step if epoch == start_epoch else 0,
            skip_handled=isinstance(train_loader.sampler, SkipDistributedSampler) and epoch == start_epoch,
        )

        # Save model weights
        if global_rank == 0:
            model_path_latest = os.path.join(config.res_dir, f"model_latest.pth")
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint_dict, model_path_latest)

        # Evaluate every 'val_every' epochs on validation set
        if epoch % config.val_every == 0:
            if global_rank == 0:
                print("Validating . . .", flush=True)
            model.eval()
            val_metrics = iterate(
                model,
                data_loader=val_loader,
                criterion=criterion,
                config=config,
                optimizer=None,
                scheduler=None,
                mode="val",
                device=device,
                local_rank=global_rank
            )
            if global_rank == 0:
                print("Val - Loss: {:.4f}, Acc: {:.2f}%, IoU: {:.4f}%, Global IoU: {:.4f}%, F1: {:.2f}%, ECE: {:.4f}, NLL: {:.4f}, Brier: {:.4f}".format(
                    val_metrics["val_loss"],
                    val_metrics["val_accuracy"],
                    val_metrics["val_IoU"],
                    val_metrics["val_global_IoU"],
                    val_metrics["val_F1"] * 100,
                    val_metrics.get("val_ECE", 0.0),
                    val_metrics.get("val_NLL", 0.0),
                    val_metrics.get("val_Brier", 0.0)
                ), flush=True)

                model_path_epoch = os.path.join(config.res_dir, f"model_epoch_{epoch}.pth")
                torch.save(checkpoint_dict, model_path_epoch)

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(1, trainlog, config)

                if val_metrics["val_IoU"] > best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    torch.save(model.module.state_dict(), os.path.join(config.res_dir, "model.pth"))
                    best_epoch = epoch
            else:
                trainlog[epoch] = {**train_metrics, **val_metrics}
        else:
            trainlog[epoch] = {**train_metrics}
            if global_rank == 0:
                checkpoint(1, trainlog, config)

    return trainlog, best_mIoU, best_epoch


def main(config):
    """Main training function with DDP support."""

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Setup of DDP, devices and seeds
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    if config.device == "cuda":
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    config.device = device

    config.amp_dtype = torch.bfloat16

    # Scale learning rate based on world size (linear scaling rule)
    config.base_lr = config.lr
    effective_batch_size = config.batch_size * world_size * config.accumulate_steps
    reference_batch_size = 4 * 4 * 4   # Reference: 4 tiles/GPU * 4 GPUs * 4 accumulate_steps = 64 tiles
    lr_scale = effective_batch_size / reference_batch_size
    config.lr = config.base_lr * lr_scale

    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"DDP Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {config.batch_size}")
        print(f"  Gradient accumulation steps: {config.accumulate_steps}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Reference batch size: {reference_batch_size}")
        print(f"  LR scaling factor: {lr_scale:.2f}x")
        print(f"  Base LR: {config.base_lr:.6f}")
        print(f"  Scaled LR: {config.lr:.6f}")
        print(f"{'='*60}\n")

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    torch.cuda.manual_seed_all(config.rdm_seed)

    prepare_output(config)

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Dataset setup
    bands = config.bands[:config.num_bands]

    dt_train_base = SatelliteDataset(
        config.satellite_train,
        config.gt_train,
        config.temp_train,
        label_sheet_file=config.label_sheet_file,
        label_columns=config.label_columns,
        ignore_index=config.ignore_index,
        use_temperature_calendar=config.use_temperature_calendar,
        temporal_length=config.temporal_length,
        use_temperature_subsampling=config.use_temperature_subsampling,
        bands=bands,
        cloud_band=config.cloud_band,
        sample_percentage=config.train_dataset_portion,
        no_sliding_subsample=config.no_sliding_subsample,
        use_temperature_calendar_no_sliding_subsample=config.use_temperature_calendar_no_sliding_subsample,
        use_fixed_temperature_subsampling=config.use_fixed_temperature_subsampling,
        augmentation=True,
        simulate_landsat=config.simulate_landsat,
        revisit_time=config.revisit_time,
        temp_mean_var=config.temp_mean_var,
        temp_spatial_var=config.temp_spatial_var,
        seed=config.rdm_seed,
        cgdd_bounds_gpkg=config.cgdd_bounds,
        oli_degrade_prob=getattr(config, 'oli_degrade_prob', 0.0),
        use_sensor_flag=config.use_sensor_flag,
        normalize_timestamps=not config.no_normalize_timestamps,
    )
    if config.use_upsampling:
        dt_train_base.data_files = dt_train_base.upsample_tiles_with_minority_classes(
            pixel_count_files=None,
            minority_threshold=config.minority_threshold,
            target_ratio=config.target_ratio,
            max_replicas=8,
            min_class_pixels=config.min_class_pixels,
            top_k_tiles_per_class=100
        )

    dt_val_base = SatelliteDataset(
        config.satellite_val,
        config.gt_val,
        config.temp_val,
        label_sheet_file=config.label_sheet_file,
        label_columns=config.label_columns,
        ignore_index=config.ignore_index,
        use_temperature_calendar=config.use_temperature_calendar,
        temporal_length=config.temporal_length,
        use_temperature_subsampling=config.use_temperature_subsampling,
        bands=bands,
        cloud_band=config.cloud_band,
        band_stats=dt_train_base.channel_stats,
        temp_stats=dt_train_base.gdd_stats,
        sample_percentage=config.val_dataset_portion,
        no_sliding_subsample=config.no_sliding_subsample,
        use_temperature_calendar_no_sliding_subsample=config.use_temperature_calendar_no_sliding_subsample,
        use_fixed_temperature_subsampling=config.use_fixed_temperature_subsampling,
        augmentation=False,
        simulate_landsat=config.simulate_landsat,
        revisit_time=config.revisit_time,
        temp_mean_var=config.temp_mean_var,
        temp_spatial_var=config.temp_spatial_var,
        seed=config.rdm_seed,
        cgdd_bounds_gpkg=config.cgdd_bounds,
        oli_degrade_prob=0.0,
        use_sensor_flag=config.use_sensor_flag,
        normalize_timestamps=not config.no_normalize_timestamps,
    )

    # Temporal truncation: compute exact end-of-month DOY for the actual test year (handles leap years).
    # Always divide by 365 so that int(365 * truncate_portion) recovers month_end_doy exactly.
    # December of a leap year gives truncate_portion > 1 → no truncation (correct).
    if getattr(config, 'eval_all_months', False):
        _test_truncate = 1.0
    elif config.truncate_month:
        _test_year = config.test_years[0]
        _month_end_doy = sum(calendar.monthrange(_test_year, m)[1]
                             for m in range(1, config.truncate_month + 1))
        _test_truncate = _month_end_doy / 365.0
    else:
        _test_truncate = 1.0

    dt_test_base = SatelliteDataset(
        config.satellite_test,
        config.gt_test,
        config.temp_test,
        label_sheet_file=config.label_sheet_file,
        label_columns=config.label_columns,
        ignore_index=config.ignore_index,
        use_temperature_calendar=config.use_temperature_calendar,
        temporal_length=config.temporal_length,
        use_temperature_subsampling=config.use_temperature_subsampling,
        bands=bands,
        cloud_band=config.cloud_band,
        band_stats=dt_train_base.channel_stats,
        temp_stats=dt_train_base.gdd_stats,
        sample_percentage=config.test_dataset_portion,
        no_sliding_subsample=config.no_sliding_subsample,
        use_temperature_calendar_no_sliding_subsample=config.use_temperature_calendar_no_sliding_subsample,
        use_fixed_temperature_subsampling=config.use_fixed_temperature_subsampling,
        augmentation=False,
        simulate_landsat=config.simulate_landsat,
        revisit_time=config.revisit_time,
        temp_mean_var=config.temp_mean_var,
        temp_spatial_var=config.temp_spatial_var,
        seed=config.rdm_seed,
        cgdd_bounds_gpkg=config.cgdd_bounds,
        oli_degrade_prob=0.0,
        truncate_portion=_test_truncate,
        use_sensor_flag=config.use_sensor_flag,
        normalize_timestamps=not config.no_normalize_timestamps,
    )

    # Wrap in TSViT patch datasets
    _use_gdd_pe = getattr(config, 'use_gdd_pe', False)
    dt_train = TSViTPatchDataset(dt_train_base, img_res=config.img_res, training=True,  use_gdd_pe=_use_gdd_pe)
    dt_val   = TSViTPatchDataset(dt_val_base,   img_res=config.img_res, training=False, use_gdd_pe=_use_gdd_pe)
    dt_test  = TSViTPatchDataset(dt_test_base,  img_res=config.img_res, training=False, use_gdd_pe=_use_gdd_pe)

    # TSViTPatchDataset is not a SatelliteDataset, so we use manual DistributedSampler
    val_sampler = DistributedSampler(dt_val, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_loader = DataLoader(
        dt_val, sampler=val_sampler,
        batch_size=config.batch_size, drop_last=False,
        num_workers=config.num_workers, persistent_workers=True, pin_memory=True,
        collate_fn=tsvit_collate_fn
    )

    # Test loader is plain (no DDP) — final eval runs on rank 0 only
    test_loader = DataLoader(
        dt_test,
        batch_size=config.batch_size, shuffle=False, drop_last=False,
        num_workers=config.num_workers, persistent_workers=True, pin_memory=True,
        collate_fn=tsvit_collate_fn
    )

    # Tile-level base loader for eval_all_months (SatelliteDataset, not TSViTPatchDataset)
    base_test_loader = DataLoader(
        dt_test_base,
        batch_size=config.batch_size, shuffle=False, drop_last=False,
        num_workers=config.num_workers, persistent_workers=True, pin_memory=True,
        collate_fn=lambda x: utils.pad_collate(x, pad_value=0),
    )

    # Save dataset normalization statistics to config for reproducibility
    config.channel_stats = dt_train_base.channel_stats
    config.gdd_stats = dt_train_base.gdd_stats
    config.gdd_mean_var = dt_train_base.gdd_mean_var
    config.gdd_spatial_var = dt_train_base.gdd_spatial_var
    config.use_spatial_gdd = dt_train_base.use_spatial_gdd

    if global_rank == 0:
        print("Train tiles: {}, Val: {}, Test: {}".format(len(dt_train_base), len(dt_val_base), len(dt_test_base)), flush=True)

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Model setup
    model_config = {
        'img_res':        config.img_res,
        'patch_size':     config.patch_size,
        'max_seq_len':    config.max_seq_len,
        'num_channels':   config.num_bands + 1,   # +1 for timestamp channel
        'num_classes':    config.num_classes,
        'dim':            config.dim,
        'temporal_depth': config.temporal_depth,
        'spatial_depth':  config.spatial_depth,
        'heads':          config.heads,
        'dim_head':       config.dim_head,
        'dropout':        config.dropout,
        'emb_dropout':    config.emb_dropout,
        'pool':           config.pool,
        'scale_dim':      config.scale_dim,
    }
    model = TSViT(model_config)

    config.N_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if global_rank == 0:
        if not (config.restart and config.restart_epoch):
            with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
                file.write(json.dumps(vars(config), default=str, indent=4))
            print("TOTAL TRAINABLE PARAMETERS:", config.N_params, flush=True)
    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False)

    # Initialize final bias using class frequencies
    if config.bias_initialization:
        weights_np = dt_train_base.get_class_weights(beta=1.0)
        weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
        inv_weights = 1.0 / (weights + 1e-6)
        freq = inv_weights / inv_weights.sum()
        bias_init = torch.log(freq + 1e-6)
        with torch.no_grad():
            conv_layers = [m for m in model.module.modules() if isinstance(m, nn.Conv2d)]
            if conv_layers:
                final_conv = conv_layers[-1]
                if final_conv.bias is not None and final_conv.bias.shape[0] == len(bias_init):
                    final_conv.bias.data.copy_(bias_init)
                    if global_rank == 0:
                        print(f"Initialized bias for {final_conv.bias.shape[0]} classes", flush=True)

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Get class weights
    class_weights = None
    if config.use_class_balance_loss:
        class_weights_np = dt_train_base.get_class_weights(beta=config.beta_class_balance)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
        class_weights[config.ignore_index] = 0

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Initialize tracking variables
    trainlog = {}
    best_mIoU = 0
    best_epoch = None
    start_epoch = 0

    # Load from checkpoint if restarting
    if config.restart and config.restart_epoch is not None and config.restart_epoch >= 0:
        with open(os.path.join(config.res_dir, f"conf_restart{config.restart_epoch}.json"), "w") as file:
            file.write(json.dumps(vars(config), default=str, indent=4))

        if config.use_latest_checkpoint:
            snapshot = os.path.join(config.checkpoint_dir, "model_latest.pth")
        else:
            snapshot = os.path.join(config.checkpoint_dir, f"model_epoch_{config.restart_epoch}.pth")

        checkpoint_dict = torch.load(snapshot, map_location={'cuda:0': f'cuda:{local_rank}'})
        model.module.load_state_dict(checkpoint_dict['model_state_dict'])
        start_epoch = checkpoint_dict.get('epoch', config.restart_epoch)
        assert start_epoch == config.restart_epoch, "Restart epoch mismatch!"

        trainlog_path = os.path.join(config.checkpoint_dir, "trainlog.json")
        if os.path.exists(trainlog_path):
            with open(trainlog_path, "r") as f:
                trainlog = json.load(f)
                # Find best epoch from training log (val_IoU is the per-epoch key)
                for epoch_str, metrics in trainlog.items():
                    epoch_num = int(epoch_str)
                    if "val_IoU" in metrics:
                        if metrics["val_IoU"] > best_mIoU:
                            best_mIoU = metrics["val_IoU"]
                            best_epoch = epoch_num

        if global_rank == 0:
            print(f"Restarting from epoch {start_epoch}", flush=True)
            if best_epoch is not None:
                print(f"Previous best: Epoch {best_epoch}, IoU {best_mIoU:.4f}%", flush=True)

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### TRAINING PHASE
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    if global_rank == 0:
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60 + "\n")

    # Create training dataloader (manual DistributedSampler for patch dataset)
    _skip_samples = getattr(config, 'resume_step', 0) * config.accumulate_steps * config.batch_size
    if _skip_samples > 0:
        train_sampler = SkipDistributedSampler(dt_train, num_replicas=world_size, rank=global_rank,
                                               shuffle=True, skip_first=_skip_samples)
    else:
        train_sampler = DistributedSampler(dt_train, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(
        dt_train, sampler=train_sampler,
        batch_size=config.batch_size, drop_last=True,
        num_workers=config.num_workers, persistent_workers=True, pin_memory=True,
        collate_fn=tsvit_collate_fn
    )

    # Create criterion (multi-task loss)
    main_criterion = create_loss_function(
        config=config,
        class_weights=class_weights,
        warmup_batches=0,
        restart=config.restart
    )
    if hasattr(main_criterion, 'to'):
        main_criterion = main_criterion.to(device)

    # Create optimizer and scheduler
    main_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_loader) // config.accumulate_steps
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = max(1, int(config.pct_start * total_steps))
    main_scheduler = torch.optim.lr_scheduler.SequentialLR(
        main_optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                main_optimizer,
                start_factor=config.lr_start / config.lr,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                main_optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=config.lr_end,
            ),
        ],
        milestones=[warmup_steps],
    )

    # Load optimizer/scheduler state if restarting
    if config.restart and config.restart_epoch > 0:
        main_optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        main_scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])

    # Recovery: if we resumed from an epoch checkpoint (resume_step==0 means training completed
    # but validation hadn't started yet), check trainlog for what is still missing and run it.
    if config.restart and start_epoch > 0 and config.resume_step == 0:
        existing_entry = trainlog.get(str(start_epoch), trainlog.get(start_epoch, {}))
        if "val_IoU" not in existing_entry:
            if global_rank == 0:
                print(f"\n[RECOVERY] Epoch {start_epoch} crashed during validation — re-running now...", flush=True)
            model.eval()
            recovery_val_metrics = iterate(
                model,
                data_loader=val_loader,
                criterion=main_criterion,
                config=config,
                optimizer=None,
                scheduler=None,
                mode="val",
                device=device,
                local_rank=global_rank
            )
            if global_rank == 0:
                print("Recovery Val - Loss: {:.4f}, Acc: {:.2f}%, IoU: {:.4f}%".format(
                    recovery_val_metrics["val_loss"],
                    recovery_val_metrics["val_accuracy"],
                    recovery_val_metrics["val_IoU"]
                ), flush=True)
                prior_train = {k: v for k, v in existing_entry.items() if k.startswith("train_")}
                trainlog[start_epoch] = {**prior_train, **recovery_val_metrics}
                checkpoint(1, trainlog, config)
                model_path_epoch = os.path.join(config.res_dir, f"model_epoch_{start_epoch}.pth")
                torch.save(checkpoint_dict, model_path_epoch)
                if recovery_val_metrics["val_IoU"] > best_mIoU:
                    best_mIoU = recovery_val_metrics["val_IoU"]
                    torch.save(model.module.state_dict(), os.path.join(config.res_dir, "model.pth"))
                    best_epoch = start_epoch
                    print(f"New best model at epoch {best_epoch}!", flush=True)

    if global_rank == 0:
        print(f"\nTraining configuration:")
        print(f"  Epochs: 1 to {config.epochs}")
        print(f"  Dataset size: {len(dt_train_base)} tiles × {dt_train._n_patches} patches = {len(dt_train_base) * dt_train._n_patches} patches/epoch")
        print("  Loss Configuration:")
        loss_config = []
        if hasattr(config, 'w_ce') and config.w_ce > 0:
            loss_config.append(f"    CE: {config.w_ce}")
        if hasattr(config, 'w_focal') and config.w_focal > 0:
            loss_config.append(f"    Focal: {config.w_focal}")
        if hasattr(config, 'w_dice') and config.w_dice > 0:
            loss_config.append(f"    Dice: {config.w_dice}")
        if hasattr(config, 'w_logcosh_dice') and config.w_logcosh_dice > 0:
            loss_config.append(f"    LogCosh Dice: {config.w_logcosh_dice}")
        if hasattr(config, 'w_kl') and config.w_kl > 0:
            loss_config.append(f"    KL Area: {config.w_kl}")
        if hasattr(config, 'w_tversky') and config.w_tversky > 0:
            loss_config.append(f"    Tversky: {config.w_tversky} (alpha={config.tversky_alpha}, beta={config.tversky_beta})")
        for lc in loss_config:
            print(lc)
        if config.use_class_balance_loss:
            print(f"    Class-balanced weighting: ON (beta={config.beta_class_balance})")
        else:
            print(f"    Class-balanced weighting: OFF")
        print(f"  LR schedule: LinearWarmup+Cosine, max_lr={config.lr:.2e}, lr_start={config.lr_start:.2e}, lr_end={config.lr_end:.2e}, warmup={config.pct_start:.0%}")
        print()

    # Run training (skip when eval_only or truncate_month is set)
    _eval_only = (getattr(config, 'eval_only', False) or
                  bool(getattr(config, 'truncate_month', None)) or
                  getattr(config, 'eval_all_months', False))
    if not _eval_only:
        main_start = max(1, start_epoch + 1)
        main_end = config.epochs
        trainlog, best_mIoU, best_epoch = run_training_phase(
            model, train_loader, val_loader, main_criterion,
            main_optimizer, main_scheduler,
            config, device, global_rank,
            main_start, main_end, trainlog, best_mIoU, best_epoch,
            phase_name="MAIN TRAINING",
            resume_step=getattr(config, 'resume_step', 0),
        )

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Final evaluation on best model
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    # Recover best_epoch from trainlog if not set (eval_only / truncate_month path)
    if best_epoch is None:
        trainlog_path = os.path.join(config.res_dir, "trainlog.json")
        if os.path.exists(trainlog_path):
            with open(trainlog_path, "r") as f:
                _tl = json.load(f)
            _best_iou = 0.0
            for _ep_str, _m in _tl.items():
                if "val_IoU" in _m and _m["val_IoU"] > _best_iou:
                    _best_iou = _m["val_IoU"]
                    best_epoch = int(_ep_str)

    if global_rank == 0:
        print("\n" + "="*60)
        print("EVALUATING BEST MODEL")
        print("="*60 + "\n")
        print(f"Testing best model from epoch {best_epoch}...", flush=True)

    if getattr(config, 'eval_all_months', False):
        # Distributed eval: all ranks participate, each handles a subset of months
        checkpoint_path = os.path.join(config.res_dir, "model.pth")
        model.module.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        test_all_months(
            model, base_test_loader, device, config,
            amp_dtype=getattr(config, 'amp_dtype', torch.float32),
        )
    elif global_rank == 0:
        checkpoint_path = os.path.join(config.res_dir, "model.pth")
        model.module.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=main_criterion,
            config=config,
            optimizer=None,
            scheduler=None,
            mode="test",
            device=device,
            local_rank=global_rank,
            force_no_sync=True
        )

        test_metrics["best_epoch"] = best_epoch
        print("Best Model (Epoch {}) - Loss: {:.4f}, Acc: {:.2f}%, IoU: {:.4f}%, Global IoU: {:.4f}%, F1: {:.2f}%, ECE: {:.4f}, NLL: {:.4f}, Brier: {:.4f}".format(
            best_epoch,
            test_metrics["test_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_IoU"],
            test_metrics["test_global_IoU"],
            test_metrics["test_F1"] * 100,
            test_metrics.get("test_ECE", 0.0),
            test_metrics.get("test_NLL", 0.0),
            test_metrics.get("test_Brier", 0.0)
        ), flush=True)

        _metrics_file = (f"test_metrics_month{config.truncate_month}.json"
                         if config.truncate_month else "test_metrics.json")
        save_results(1, test_metrics, conf_mat.cpu().float().numpy(), config, filename=_metrics_file)
        if not config.truncate_month:
            overall_performance(config)

    # Clean shutdown
    torch.distributed.destroy_process_group()


if __name__ == "__main__":

    config = parser.parse_args()
    if config.min_class_pixels is None:
        config.min_class_pixels = 100 if config.satellite == "sentinel" else 11

    # ~~~ YEAR SPLITS ~~~
    _ablation_splits = {
        "SA1": {"train": [2019, 2020, 2021, 2022], "val": [2023], "test": [2024]},
        "SA2": {"train": [2019, 2020, 2021, 2024], "val": [2022], "test": [2023]},
        "SA3": {"train": [2019, 2020, 2023, 2024], "val": [2021], "test": [2022]},
        "S5":  {"train": [2019, 2020, 2021, 2022, 2023], "val": [2024], "test": [2025]},
        "S4":  {"train": [2019, 2020, 2021, 2022, 2025], "val": [2023], "test": [2024]},
        "S3":  {"train": [2019, 2020, 2021, 2024, 2025], "val": [2022], "test": [2023]},
        "S2":  {"train": [2019, 2020, 2023, 2024, 2025], "val": [2021], "test": [2022]},
        "S1":  {"train": [2019, 2022, 2023, 2024, 2025], "val": [2020], "test": [2021]},
        "ALL": {"train": [2019, 2020, 2021, 2022, 2023, 2024, 2025], "test": [2025]},
        "ALL_wo_19_20": {"train": [2021, 2022, 2023, 2024, 2025], "test": [2025]}
    }
    if config.ablation_split is not None:
        _split = _ablation_splits[config.ablation_split]
    else:
        _split = {"train": [2019, 2020, 2021, 2022], "val": [2023], "test": [2024]}
    _train_years = _split["train"]
    _val_years   = _split.get("val", [])
    _test_years  = _split["test"]
    config.test_years = _test_years  # needed for calendar-aware truncate_month DOY computation

    # ~~~ DATA PATHS ~~~
    if config.satellite == "landsat":
        config.bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
        config.num_bands = 6
        config.cloud_band = "QA_PIXEL"
        _sat_roots = [
            '/capstor/store/cscs/2go/go57/data/satellite/landsat/raw/CH/45',
            '/capstor/store/cscs/2go/go57/data/satellite/landsat/raw/CH/7',
            '/capstor/store/cscs/2go/go57/data/satellite/landsat/raw/CH/89',
        ]
        _gt_root   = '/capstor/scratch/cscs/tlauber/020_crop1990/data/GTs_Landsat'
        _temp_root = '/capstor/store/cscs/2go/go57/data/climate/cgdd/landsat'
        config.cgdd_bounds = '/users/tlauber/scratch/020_crop1990/data/CGDD_bounds/cgdd_bounds_LS_1991_2020.gpkg'
        _sat_ext  = '.zarr.zip.json'
        _gt_ext   = '.tar.json'
        _temp_ext = '.tar.json'
    else:  # sentinel
        config.bands = ['s2_B02', 's2_B03', 's2_B04', 's2_B08', 's2_B05', 's2_B06', 's2_B07', 's2_B8A', 's2_B11', 's2_B12']
        config.num_bands = 10
        config.cloud_band = "s2_mask"
        if config.dataset_folder is not None:
            _sat_root  = os.path.join(config.dataset_folder, 'sentinel2')
            _gt_root   = os.path.join(config.dataset_folder, 'labels')
            _temp_root = os.path.join(config.dataset_folder, 'temperature')
            config.cgdd_bounds = None
        else:
            _sat_root  = '/capstor/store/cscs/2go/go57/data/satellite/sentinel2/raw/CH'
            _gt_root   = '/capstor/scratch/cscs/tlauber/020_crop1990/data/GTs_Sentinel'
            _temp_root = '/capstor/scratch/cscs/tlauber/020_crop1990/data/CGDD'
            config.cgdd_bounds = '/users/tlauber/scratch/020_crop1990/data/CGDD_bounds/cgdd_bounds_S2_1991_2020.gpkg'
        _sat_ext  = '.json'
        _gt_ext   = '.tar.json'
        _temp_ext = '.tar.json'

    if config.satellite == "landsat":
        config.satellite_train = [[f'{r}/{y}{_sat_ext}' for r in _sat_roots] for y in _train_years]
        config.satellite_val   = [[f'{r}/{y}{_sat_ext}' for r in _sat_roots] for y in _val_years]
        config.satellite_test  = [[f'{r}/{y}{_sat_ext}' for r in _sat_roots] for y in _test_years]
    else:
        config.satellite_train = [f'{_sat_root}/{y}{_sat_ext}' for y in _train_years]
        config.satellite_val   = [f'{_sat_root}/{y}{_sat_ext}' for y in _val_years]
        config.satellite_test  = [f'{_sat_root}/{y}{_sat_ext}' for y in _test_years]
    config.gt_train        = [f'{_gt_root}/{y}{_gt_ext}'     for y in _train_years]
    config.gt_val          = [f'{_gt_root}/{y}{_gt_ext}'     for y in _val_years]
    config.gt_test         = [f'{_gt_root}/{y}{_gt_ext}'     for y in _test_years]
    config.temp_train      = [f'{_temp_root}/{y}{_temp_ext}' for y in _train_years]
    config.temp_val        = [f'{_temp_root}/{y}{_temp_ext}' for y in _val_years]
    config.temp_test       = [f'{_temp_root}/{y}{_temp_ext}' for y in _test_years]

    config.label_sheet_file = "./SwissCrop25.xlsx"
    config.label_columns = ["Crop_Label", "Crop_Label_lv2", "Crop_Label_lv3"]
    config.num_classes = 71

    prepare_output(config)  # sets checkpoint_dir = res_dir if not given

    # Using latest checkpoint if it exists
    if not config.overwrite and config.checkpoint_dir is not None and os.path.exists(config.checkpoint_dir):
        config.restart = False
        config.restart_epoch = None
        config.use_latest_checkpoint = False
        config.resume_step = 0

        latest_checkpoint_path = os.path.join(config.checkpoint_dir, "model_latest.pth")
        print(f"Checking for checkpoint at: {latest_checkpoint_path}", flush=True)
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint_dict = torch.load(latest_checkpoint_path, map_location='cpu')
                if 'epoch' in checkpoint_dict:
                    config.restart = True
                    config.restart_epoch = checkpoint_dict['epoch']
                    config.use_latest_checkpoint = True
                    config.resume_step = checkpoint_dict.get('step', 0)
                    _display_epoch = config.restart_epoch + 1 if config.resume_step > 0 else config.restart_epoch
                    print(f"✓ Using latest checkpoint from epoch: {_display_epoch}, step: {config.resume_step}", flush=True)
                else:
                    print(f"Warning: model_latest.pth exists but has no 'epoch' key", flush=True)
            except Exception as e:
                print(f"Warning: Could not load model_latest.pth: {e}", flush=True)
        else:
            print(f"model_latest.pth not found", flush=True)

        if not config.restart:
            checkpoint_fold_dir = config.checkpoint_dir
            if os.path.exists(checkpoint_fold_dir):
                epoch_numbers = []
                for filename in os.listdir(checkpoint_fold_dir):
                    match = re.search(r"_epoch_(\d+)", filename)
                    if match:
                        epoch_numbers.append(int(match.group(1)))
                if epoch_numbers:
                    config.restart = True
                    config.restart_epoch = max(epoch_numbers)
                    config.use_latest_checkpoint = False
                    print(f"✓ Using epoch checkpoint from epoch: {config.restart_epoch}", flush=True)
                else:
                    print(f"No epoch checkpoints found in {checkpoint_fold_dir}", flush=True)
            else:
                print(f"Checkpoint fold directory not found: {checkpoint_fold_dir}", flush=True)
    else:
        config.restart = False
        config.restart_epoch = None
        config.use_latest_checkpoint = False
        config.resume_step = 0
        if config.checkpoint_dir is not None:
            print(f"Checkpoint directory does not exist: {config.checkpoint_dir}", flush=True)

    # Override rdm_seed with seed if provided
    if config.seed is not None:
        config.rdm_seed = config.seed

    start_time = time.time()
    print("Training started at:", time.ctime(start_time), flush=True)
    pprint.pprint(config)
    sys.stdout.flush()
    main(config)
    print("Training ended at:", time.ctime(time.time()), flush=True)
    print(f"Training duration: {(time.time() - start_time) / 60:.2f} minutes", flush=True)

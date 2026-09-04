"""
Multi-task loss module.
Includes: KL Area Loss, Focal Loss, CE Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    """
    def __init__(self, gamma=2, weight=None, ignore_index=None):
        """
        :param gamma: Focusing parameter gamma.
        :param weight: A tensor of shape (num_classes,) for per-class weighting.
        :param ignore_index: Class index to ignore.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input: Tensor of shape (N, C, H, W) (raw logits).
        :param target: Tensor of shape (N, H, W) with class indices.
        :return: scalar focal loss.
        """
        # Compute log softmax over the class dimension
        logpt = F.log_softmax(input, dim=1)  # shape: (N, C, H, W)
        pt = torch.exp(logpt)  # shape: (N, C, H, W)

        # Reshape tensors: flatten spatial dimensions to simplify indexing.
        N, C, H, W = input.shape
        logpt = logpt.view(N, C, -1)      # shape: (N, C, H*W)
        pt = pt.view(N, C, -1)            # shape: (N, C, H*W)
        target = target.view(N, -1)       # shape: (N, H*W)

        # Gather the log-probabilities and probabilities corresponding to target labels.
        # target.unsqueeze(1) reshapes target to (N, 1, H*W) so we can gather along dim=1.
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)  # shape: (N, H*W)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)          # shape: (N, H*W)

        # Create a mask to ignore specified pixels.
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=input.device)
        logpt = logpt[valid_mask]
        pt = pt[valid_mask]
        target_valid = target[valid_mask]

        # Compute the focal loss term.
        loss = - (1 - pt) ** self.gamma * logpt

        # Optionally apply per-class weighting.
        if self.weight is not None:
            # Gather the weights for the valid target labels.
            weight = self.weight[target_valid]
            loss = loss * weight

        return loss.mean()

class TverskyLoss(nn.Module):
    """
    Multiclass Tversky loss.
    alpha controls FP penalty, beta controls FN penalty (alpha + beta = 1 typical).
    When alpha=beta=0.5 this reduces to Dice loss.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        N, C, H, W = probs.shape

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        targets_masked = targets.clone()
        targets_masked[~valid_mask] = 0
        one_hot = F.one_hot(targets_masked, num_classes=C).permute(0, 3, 1, 2).float()

        mask = valid_mask.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask

        probs_flat = probs.reshape(N, C, -1)
        oh_flat = one_hot.reshape(N, C, -1)

        TP = (probs_flat * oh_flat).sum(dim=(0, 2))
        FP = (probs_flat * (1 - oh_flat)).sum(dim=(0, 2))
        FN = ((1 - probs_flat) * oh_flat).sum(dim=(0, 2))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky).mean()


class KLAreaLoss(nn.Module):
    """
    KL divergence between predicted and target class distributions,
    using running (global) estimates with optional warm-up or restart behavior.
    """

    def __init__(self, ignore_index=None, num_classes=None, eps=1e-8, momentum=0.9,
                 warmup_batches=100, restart=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.momentum = momentum
        self.num_classes = num_classes
        self.warmup_batches = warmup_batches
        self.restart = restart  # if True, skip warm-up entirely

        # Running estimates and counter
        self.register_buffer('running_pred_dist', None)
        self.register_buffer('running_target_dist', None)
        self.register_buffer('num_updates', torch.tensor(0.0))

    def _update_running(self, name, current):
        """EMA update for a buffer"""
        buf = getattr(self, name)
        if buf is None:
            setattr(self, name, current.detach())
        else:
            buf.mul_(self.momentum).add_((1 - self.momentum) * current.detach())

    def forward(self, logits, targets):
        device, dtype = logits.device, logits.dtype
        B, C, H, W = logits.shape
        if self.num_classes is None:
            self.num_classes = C

        probs = F.softmax(logits, dim=1)

        # mask ignore index
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).to(device=device, dtype=dtype)
        else:
            valid_mask = torch.ones((B, H, W), device=device, dtype=dtype)

        probs_flat = probs.reshape(B, C, -1)
        mask_flat = valid_mask.reshape(B, -1)
        targets_flat = targets.reshape(B * H * W)

        valid_pixels = mask_flat.sum().clamp(min=1.0)

        # per-batch predicted distribution
        pred_hist = (probs_flat * mask_flat.unsqueeze(1)).sum(dim=(0, 2))

        # per-batch target distribution
        mask_flat_1d = mask_flat.reshape(-1)
        valid_idx = torch.nonzero(mask_flat_1d, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        vals = targets_flat[valid_idx]
        target_hist = torch.bincount(vals, minlength=C).to(device=device, dtype=dtype)

        # normalize to probability distributions
        pred_dist = pred_hist / valid_pixels
        target_dist = target_hist / valid_pixels

        # update global estimates
        self._update_running('running_pred_dist', pred_dist)
        self._update_running('running_target_dist', target_dist)
        self.num_updates += 1

        # --- warm-up logic ---
        if not self.restart and self.num_updates < self.warmup_batches:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # use running estimates
        global_pred = self.running_pred_dist.clamp(min=self.eps)
        global_target = self.running_target_dist.clamp(min=self.eps)

        kl = (global_target * (torch.log(global_target) - torch.log(global_pred))).sum()

        return kl


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss manager.
    """
    def __init__(
        self,
        loss_weights,
        num_classes,
        ignore_index=None,
        restart=False,
        warmup_batches=100,
        class_weights=None,
        gamma=2,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    ):
        """
        Args:
            loss_weights: Dict of loss names to weights, e.g. {'ce': 1.0, 'dice': 0.5}
            num_classes: Number of classes
            ignore_index: Index to ignore in loss computation
            restart: Whether training is a restart (KL loss skips warmup)
            warmup_batches: Number of warm-up batches for KL loss
            class_weights: Class weights for CE or Focal loss
            gamma: Focusing parameter for Focal loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.restart = restart

        self.task_names = list(loss_weights.keys())
        self.losses = nn.ModuleDict()
        
        # Cross-Entropy Loss — wrapped to return 0 on all-ignored batches
        # (nn.CrossEntropyLoss returns NaN when every target pixel is ignore_index)
        if 'ce' in loss_weights:
            _ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
            _ign = ignore_index

            class _SafeCE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ce = _ce

                def forward(self, input, target):
                    if _ign is not None and (target != _ign).sum() == 0:
                        return (input * 0).sum()  # 0 with grad_fn, safe for DDP
                    return self.ce(input, target)

            self.losses['ce'] = _SafeCE()

        # Focal Loss
        if 'focal' in loss_weights:
            self.losses['focal'] = FocalLoss(
                gamma=gamma,
                weight=class_weights,
                ignore_index=ignore_index
            )
        
        # Tversky Loss
        if 'tversky' in loss_weights:
            self.losses['tversky'] = TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                ignore_index=ignore_index,
            )

        # KL Area Loss
        if 'kl' in loss_weights or 'kl_area' in loss_weights:
            loss_key = 'kl' if 'kl' in loss_weights else 'kl_area'
            self.losses[loss_key] = KLAreaLoss(
                ignore_index=ignore_index,
                num_classes=num_classes,
                warmup_batches=warmup_batches,
                restart=self.restart
            )

        self.task_weights = [
            torch.tensor(loss_weights[name], dtype=torch.float32)
            for name in self.task_names
        ]

    def forward(self, logits, targets, return_components=False):
        """
        Args:
            logits: [B, C, H, W] - model outputs
            targets: [B, H, W] - ground truth labels
            return_components: If True, also return dict of individual loss values

        Returns:
            total_loss or (total_loss, loss_dict)
        """
        loss_dict = {}
        for name in self.task_names:
            if name not in self.losses:
                continue
            loss_dict[name] = self.losses[name](logits, targets)

        # Compute weighted total loss
        total_loss = 0.0
        weight_idx = 0
        for name in self.task_names:
            if name not in loss_dict:
                continue
            weight = self.task_weights[weight_idx].to(logits.device)
            total_loss += weight * loss_dict[name]
            weight_idx += 1

        if return_components:
            weight_dict = {}
            weight_idx = 0
            for name in self.task_names:
                if name not in loss_dict:
                    continue
                weight_dict[f'{name}_weight'] = self.task_weights[weight_idx].item()
                weight_idx += 1
            return total_loss, {**loss_dict, **weight_dict}

        return total_loss


def create_loss_function(config, class_weights=None, warmup_batches=100, restart=False,
                         num_classes_override=None):
    """
    Factory function to create a MultiTaskLoss based on config.

    Args:
        config: Training configuration with w_ce, w_focal, w_dice, w_logcosh_dice, w_kl
        class_weights: Optional class weights for CE/Focal loss
        warmup_batches: Number of warm-up batches for KL loss
        restart: Whether training is a restart (KL loss skips warmup)
        num_classes_override: Override config.num_classes

    Returns:
        MultiTaskLoss instance
    """
    loss_weights = {}

    if hasattr(config, 'w_ce') and config.w_ce > 0:
        loss_weights['ce'] = config.w_ce

    if hasattr(config, 'w_focal') and config.w_focal > 0:
        loss_weights['focal'] = config.w_focal

    if hasattr(config, 'w_kl') and config.w_kl > 0:
        loss_weights['kl'] = config.w_kl

    if hasattr(config, 'w_tversky') and config.w_tversky > 0:
        loss_weights['tversky'] = config.w_tversky

    num_classes = num_classes_override if num_classes_override is not None else config.num_classes

    return MultiTaskLoss(
        loss_weights=loss_weights,
        num_classes=num_classes,
        ignore_index=config.ignore_index,
        restart=restart,
        warmup_batches=warmup_batches,
        class_weights=class_weights,
        gamma=getattr(config, 'focal_gamma', 2),
        tversky_alpha=getattr(config, 'tversky_alpha', 0.3),
        tversky_beta=getattr(config, 'tversky_beta', 0.7),
    )
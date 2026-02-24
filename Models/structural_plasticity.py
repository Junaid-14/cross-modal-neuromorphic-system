"""
Structural plasticity: kill low-firing neurons and breed from top (alpha) neurons.
Compatible with snnTorch LIF; modifies Conv2d and Linear weights in-place under torch.no_grad().
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def apply_structural_plasticity(
    model: nn.Module,
    firing_rates: Dict[str, torch.Tensor],
    epoch: int,
    *,
    kill_start_epoch: int = 10,
    kill_threshold: float = 0.01,
    alpha_frac: float = 0.05,
    breed_sigma: float = 0.01,
) -> None:
    """
    In-place: kill neurons with mean firing rate < kill_threshold, then breed from
    top alpha_frac neurons into dead slots (clone + Gaussian noise). Only runs
    kill when epoch >= kill_start_epoch.

    firing_rates: dict mapping layer name (as in model.named_modules()) to 1D
        tensor of mean firing rate per output unit (out_channels for Conv2d, out_features for Linear).
    """
    if not firing_rates:
        return

    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in firing_rates:
                continue
            rate = firing_rates[name]
            if rate is None or rate.numel() == 0:
                continue
            rate = rate.to(module.weight.device).float()

            if isinstance(module, nn.Conv2d):
                _apply_plasticity_conv2d(module, rate, epoch, kill_start_epoch, kill_threshold, alpha_frac, breed_sigma)
            elif isinstance(module, nn.Linear):
                _apply_plasticity_linear(module, rate, epoch, kill_start_epoch, kill_threshold, alpha_frac, breed_sigma)


def _apply_plasticity_conv2d(
    layer: nn.Conv2d,
    rate: torch.Tensor,
    epoch: int,
    kill_start_epoch: int,
    kill_threshold: float,
    alpha_frac: float,
    breed_sigma: float,
) -> None:
    # rate: (out_channels,)
    dead = rate < kill_threshold
    n_alpha = max(1, int(rate.numel() * alpha_frac))
    alpha_inds = rate.argsort(descending=True)[:n_alpha]

    if epoch >= kill_start_epoch and dead.any():
        layer.weight[dead, :, :, :] = 0.0
        if layer.bias is not None:
            layer.bias[dead] = 0.0

        dead_inds = dead.nonzero(as_tuple=True)[0]
        if dead_inds.numel() > 0 and alpha_inds.numel() > 0:
            for i, d in enumerate(dead_inds):
                a = alpha_inds[i % alpha_inds.numel()]
                layer.weight[d : d + 1].copy_(
                    layer.weight[a : a + 1]
                    + breed_sigma * torch.randn_like(layer.weight[a : a + 1], device=layer.weight.device)
                )
            if layer.bias is not None:
                for i, d in enumerate(dead_inds):
                    a = alpha_inds[i % alpha_inds.numel()]
                    layer.bias[d] = (layer.bias[a] + breed_sigma * torch.randn(1, device=layer.bias.device)).item()


def _apply_plasticity_linear(
    layer: nn.Linear,
    rate: torch.Tensor,
    epoch: int,
    kill_start_epoch: int,
    kill_threshold: float,
    alpha_frac: float,
    breed_sigma: float,
) -> None:
    # rate: (out_features,)
    dead = rate < kill_threshold
    n_alpha = max(1, int(rate.numel() * alpha_frac))
    alpha_inds = rate.argsort(descending=True)[:n_alpha]

    if epoch >= kill_start_epoch and dead.any():
        layer.weight[dead, :] = 0.0
        if layer.bias is not None:
            layer.bias[dead] = 0.0

        dead_inds = dead.nonzero(as_tuple=True)[0]
        if dead_inds.numel() > 0 and alpha_inds.numel() > 0:
            for i, d in enumerate(dead_inds):
                a = alpha_inds[i % alpha_inds.numel()]
                layer.weight[d : d + 1].copy_(
                    layer.weight[a : a + 1]
                    + breed_sigma * torch.randn_like(layer.weight[a : a + 1], device=layer.weight.device)
                )
            if layer.bias is not None:
                for i, d in enumerate(dead_inds):
                    a = alpha_inds[i % alpha_inds.numel()]
                    layer.bias[d] = (layer.bias[a] + breed_sigma * torch.randn(1, device=layer.bias.device)).item()

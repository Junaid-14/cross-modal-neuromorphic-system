"""
Training pipeline utilities for SNN experiments.
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.structural_plasticity import apply_structural_plasticity
from .losses import SupervisedContrastiveLoss
from .settings import TrainingConfig


# Adjustable energy constants (in picojoules)
ANN_ENERGY_PER_FLOP_PJ = 100.0
SNN_ENERGY_PER_SYNOP_PJ = 0.1


def _is_lif(module: nn.Module) -> bool:
    """True if module is snnTorch Leaky (LIF)."""
    c = type(module)
    return c.__name__ == "Leaky" and "snntorch" in (c.__module__ or "")


def _build_lif_to_layer(model: nn.Module) -> Dict[str, str]:
    """Map each LIF module full name to the preceding Conv2d/Linear layer full name."""
    # Preserve module registration order (do NOT sort by name).
    # Sorting breaks the true layer->LIF pairing for names like:
    # fc_input, fc1, fc_hidden, ... which are not lexicographically ordered
    # the same way they are executed/declared.
    ordered = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear)) or _is_lif(mod)
    ]
    lif_to_layer: Dict[str, str] = {}
    last_layer: Optional[str] = None
    for name, mod in ordered:
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            last_layer = name
        elif _is_lif(mod) and last_layer is not None:
            lif_to_layer[name] = last_layer
    return lif_to_layer


def collect_firing_rates(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run model over batches, collect LIF spike tensors, and return mean firing rate
    per output unit for each Conv2d/Linear layer (keyed by layer name).
    """
    lif_to_layer = _build_lif_to_layer(model)
    if not lif_to_layer:
        return {}

    # Accumulate spike sum (per neuron) and total positions per layer
    sum_spikes: Dict[str, torch.Tensor] = defaultdict(lambda: None)
    total_positions: Dict[str, float] = defaultdict(float)

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            spk = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(spk, torch.Tensor):
                return
            spk = spk.detach()
            # Neuron dim is 1 (channels for Conv2d, features for Linear)
            if spk.dim() == 4:
                # (B, C, H, W)
                positions = spk.size(0) * spk.size(2) * spk.size(3)
                s = spk.sum(dim=(0, 2, 3))
            elif spk.dim() == 2:
                # (B, F)
                positions = spk.size(0)
                s = spk.sum(dim=0)
            else:
                positions = spk.numel() // max(1, spk.size(1))
                s = spk.sum(dim=tuple(d for d in range(spk.dim()) if d != 1))
            # MPS does not support float64 tensors; move to CPU before widening dtype.
            s = s.to("cpu").to(torch.float64)
            if sum_spikes[layer_name] is None:
                sum_spikes[layer_name] = s
            else:
                sum_spikes[layer_name] = sum_spikes[layer_name] + s
            total_positions[layer_name] += positions
        return hook

    hooks = []
    try:
        for lif_name, layer_name in lif_to_layer.items():
            mod = dict(model.named_modules())[lif_name]
            hooks.append(mod.register_forward_hook(make_hook(layer_name)))

        model.eval()
        n_batches = 0
        with torch.no_grad():
            for data, _ in data_loader:
                if max_batches is not None and n_batches >= max_batches:
                    break
                data = data.to(device)
                _ = model(data)
                n_batches += 1
    finally:
        for h in hooks:
            h.remove()

    firing_rates = {}
    for layer_name in sum_spikes:
        s = sum_spikes[layer_name]
        p = total_positions[layer_name]
        if p > 0 and s is not None:
            firing_rates[layer_name] = (s / p).float()
    return firing_rates


def _synapses_per_spike(layer: nn.Module) -> float:
    """Approximate SynOps contribution per emitted spike for a layer output."""
    if isinstance(layer, nn.Linear):
        return float(layer.in_features)
    if isinstance(layer, nn.Conv2d):
        kh, kw = layer.kernel_size
        fan_in = (layer.in_channels // layer.groups) * kh * kw
        return float(fan_in)
    return 1.0


class SpikeCounterHook:
    """
    Forward-hook profiler for snnTorch LIF layers.

    Counts emitted spikes per sample and estimates SynOps using:
        SynOps ~= emitted_spikes * fan_in(preceding layer)
    for each hooked LIF layer.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self._attached = False
        self._module_lookup = dict(model.named_modules())
        self._lif_names = [name for name, mod in model.named_modules() if _is_lif(mod)]
        self._lif_to_layer = _build_lif_to_layer(model)
        self._synapse_factors = self._build_synapse_factors()
        self.reset()

    def _build_synapse_factors(self) -> Dict[str, float]:
        factors: Dict[str, float] = {}
        for lif_name in self._lif_names:
            layer_name = self._lif_to_layer.get(lif_name)
            if layer_name is None:
                factors[lif_name] = 1.0
                continue
            layer = self._module_lookup.get(layer_name)
            factors[lif_name] = _synapses_per_spike(layer) if layer is not None else 1.0
        return factors

    def reset(self) -> None:
        self.num_forwards = 0
        self.layer_total_spikes: Dict[str, float] = defaultdict(float)
        self.layer_total_synops: Dict[str, float] = defaultdict(float)
        self._sample_spikes: List[torch.Tensor] = []
        self._sample_synops: List[torch.Tensor] = []
        self._active_spikes: Optional[torch.Tensor] = None
        self._active_synops: Optional[torch.Tensor] = None

    def attach(self) -> "SpikeCounterHook":
        if self._attached:
            return self

        self._handles.append(self.model.register_forward_pre_hook(self._on_model_forward_pre))
        self._handles.append(self.model.register_forward_hook(self._on_model_forward_post))

        for lif_name in self._lif_names:
            module = self._module_lookup[lif_name]
            self._handles.append(module.register_forward_hook(self._make_lif_hook(lif_name)))

        self._attached = True
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attached = False

    def _on_model_forward_pre(self, _module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        batch_size: Optional[int] = None
        if inputs:
            first = inputs[0]
            if isinstance(first, torch.Tensor) and first.dim() > 0:
                batch_size = int(first.size(0))

        if batch_size is None or batch_size <= 0:
            self._active_spikes = None
            self._active_synops = None
            return

        self._active_spikes = torch.zeros(batch_size, dtype=torch.float64)
        self._active_synops = torch.zeros(batch_size, dtype=torch.float64)

    def _on_model_forward_post(self, _module: nn.Module, _inputs: Tuple[Any, ...], _output: Any) -> None:
        if self._active_spikes is not None and self._active_synops is not None:
            self._sample_spikes.append(self._active_spikes)
            self._sample_synops.append(self._active_synops)
        self._active_spikes = None
        self._active_synops = None
        self.num_forwards += 1

    def _make_lif_hook(self, lif_name: str):
        layer_key = self._lif_to_layer.get(lif_name, lif_name)
        syn_factor = self._synapse_factors.get(lif_name, 1.0)

        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            spk = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(spk, torch.Tensor) or spk.dim() == 0:
                return

            spk_detached = spk.detach()
            if spk_detached.size(0) <= 0:
                return

            per_sample_spikes = spk_detached.reshape(spk_detached.size(0), -1).sum(dim=1)
            # MPS does not support float64 tensors; move to CPU first, then cast.
            per_sample_spikes = per_sample_spikes.to("cpu").to(torch.float64)
            per_sample_synops = per_sample_spikes * syn_factor

            self.layer_total_spikes[layer_key] += float(per_sample_spikes.sum().item())
            self.layer_total_synops[layer_key] += float(per_sample_synops.sum().item())

            if (
                self._active_spikes is not None
                and self._active_synops is not None
                and self._active_spikes.numel() == per_sample_spikes.numel()
            ):
                self._active_spikes += per_sample_spikes
                self._active_synops += per_sample_synops

        return hook

    def summary(self) -> Dict[str, Any]:
        spikes_per_sample = (
            torch.cat(self._sample_spikes, dim=0) if self._sample_spikes else torch.empty(0, dtype=torch.float64)
        )
        synops_per_sample = (
            torch.cat(self._sample_synops, dim=0) if self._sample_synops else torch.empty(0, dtype=torch.float64)
        )
        num_samples = int(spikes_per_sample.numel())

        layer_totals = {}
        for lif_name in self._lif_names:
            layer_key = self._lif_to_layer.get(lif_name, lif_name)
            layer_totals[layer_key] = {
                "spikes": float(self.layer_total_spikes.get(layer_key, 0.0)),
                "synops": float(self.layer_total_synops.get(layer_key, 0.0)),
                "synapses_per_spike": float(self._synapse_factors.get(lif_name, 1.0)),
                "lif_module": lif_name,
            }

        total_spikes = float(spikes_per_sample.sum().item()) if num_samples else 0.0
        total_synops = float(synops_per_sample.sum().item()) if num_samples else 0.0

        return {
            "num_lif_layers": len(self._lif_names),
            "num_forwards": int(self.num_forwards),
            "num_samples": num_samples,
            "spikes_per_sample": spikes_per_sample,
            "synops_per_sample": synops_per_sample,
            "total_spikes": total_spikes,
            "total_synops": total_synops,
            "mean_spikes_per_sample": (total_spikes / num_samples) if num_samples else 0.0,
            "mean_synops_per_sample": (total_synops / num_samples) if num_samples else 0.0,
            "layer_totals": layer_totals,
            "synops_estimate": "emitted_spikes * fan_in(preceding Conv/Linear)",
        }


def attach_spike_counter_hook(model: nn.Module) -> SpikeCounterHook:
    """Attach spike-count hooks to all LIF layers and return the active hook object."""
    return SpikeCounterHook(model).attach()


def profile_network_activity(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run inference and profile total network spikes + SynOps across all LIF layers.

    Returns a dict containing per-sample spike counts and SynOps, plus totals.
    """
    counter = attach_spike_counter_hook(model)
    n_batches = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if max_batches is not None and n_batches >= max_batches:
                    break
                data = batch[0] if isinstance(batch, (tuple, list)) else batch
                data = data.to(device)
                _ = model(data)
                n_batches += 1
    finally:
        counter.remove()

    summary = counter.summary()
    summary["num_batches"] = n_batches
    return summary


TEMPORAL_SNAPSHOT_FRACTIONS: Tuple[Tuple[str, float], ...] = (
    ("t25", 0.25),
    ("t50", 0.50),
    ("t75", 0.75),
    ("t100", 1.00),
)


def _infer_num_timesteps(batch_data: torch.Tensor) -> int:
    """
    Infer temporal length from batched event tensors.

    Expected shapes are typically [B, T, ...] for this repository.
    """
    if not isinstance(batch_data, torch.Tensor) or batch_data.dim() < 2:
        return 1
    t_steps = int(batch_data.size(1))
    return max(1, t_steps)


def _resolve_snapshot_steps(
    total_steps: int,
    snapshots: Sequence[Tuple[str, float]] = TEMPORAL_SNAPSHOT_FRACTIONS,
) -> Dict[str, int]:
    """Return 1-indexed timestep indices for snapshot capture."""
    import math

    total_steps = max(1, int(total_steps))
    resolved: Dict[str, int] = {}
    for label, frac in snapshots:
        if frac >= 1.0:
            step = total_steps
        else:
            step = int(math.ceil(float(frac) * total_steps))
            step = max(1, min(total_steps, step))
        resolved[label] = step
    return resolved


def _select_temporal_latent_lif_name(model: nn.Module) -> str:
    """
    Pick a latent LIF layer for temporal engram snapshots.

    Preference:
    1) deepest hidden LIF inside backbone (excluding output layer)
    2) deepest hidden LIF in model (excluding output layer)
    3) final LIF in model
    """
    lif_names = [name for name, mod in model.named_modules() if _is_lif(mod)]
    if not lif_names:
        raise ValueError("No snnTorch Leaky (LIF) layers found in model.")

    backbone_hidden = [n for n in lif_names if n.startswith("backbone.") and not n.endswith("lif_out")]
    if backbone_hidden:
        return backbone_hidden[-1]

    hidden = [n for n in lif_names if not n.endswith("lif_out")]
    if hidden:
        return hidden[-1]

    return lif_names[-1]


class TemporalEngramSnapshotHook:
    """
    Capture cumulative latent spike embeddings at fixed temporal snapshots.

    The hook attaches to one latent LIF layer (typically the deepest hidden layer)
    and stores running spike-count embeddings at {25%, 50%, 75%, 100%} of T_max.
    """

    def __init__(
        self,
        model: nn.Module,
        target_lif_name: Optional[str] = None,
        snapshots: Sequence[Tuple[str, float]] = TEMPORAL_SNAPSHOT_FRACTIONS,
    ):
        self.model = model
        self.snapshots = tuple(snapshots)
        self.module_lookup = dict(model.named_modules())
        self.target_lif_name = target_lif_name or _select_temporal_latent_lif_name(model)
        target_module = self.module_lookup.get(self.target_lif_name)
        if target_module is None or not _is_lif(target_module):
            raise ValueError(f"Temporal snapshot hook target is not a LIF layer: {self.target_lif_name}")

        self._target_module = target_module
        self._handle: Optional[Any] = None

        self._step = 0
        self._snapshot_steps: Dict[str, int] = {}
        self._labels_by_step: Dict[int, List[str]] = defaultdict(list)
        self._running_embedding: Optional[torch.Tensor] = None
        self._batch_snapshots: Dict[str, torch.Tensor] = {}
        self._batch_size: Optional[int] = None

    def attach(self) -> "TemporalEngramSnapshotHook":
        if self._handle is None:
            self._handle = self._target_module.register_forward_hook(self._on_lif_forward)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def start_batch(self, batch_data: torch.Tensor) -> None:
        total_steps = _infer_num_timesteps(batch_data)
        self._snapshot_steps = _resolve_snapshot_steps(total_steps, self.snapshots)
        self._labels_by_step = defaultdict(list)
        for label, step in self._snapshot_steps.items():
            self._labels_by_step[int(step)].append(label)

        self._step = 0
        self._running_embedding = None
        self._batch_snapshots = {}
        self._batch_size = int(batch_data.size(0)) if batch_data.dim() > 0 else None

    def _on_lif_forward(self, _module: nn.Module, _input: Any, output: Any) -> None:
        spk = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(spk, torch.Tensor) or spk.dim() == 0:
            return

        spk_detached = spk.detach()
        batch_size = int(spk_detached.size(0))
        flat = spk_detached.reshape(batch_size, -1).to(device="cpu", dtype=torch.float32)

        if self._running_embedding is None:
            self._running_embedding = torch.zeros_like(flat, dtype=torch.float32)
        self._running_embedding += flat

        self._step += 1
        for label in self._labels_by_step.get(self._step, []):
            self._batch_snapshots[label] = self._running_embedding.clone()

    def finish_batch(self) -> Dict[str, torch.Tensor]:
        if self._running_embedding is None:
            return {}

        # Backfill any missing snapshot labels (possible if duplicate snapshot indices collapse).
        for label in self._snapshot_steps:
            if label not in self._batch_snapshots:
                self._batch_snapshots[label] = self._running_embedding.clone()

        return {k: v.clone() for k, v in self._batch_snapshots.items()}

    @property
    def snapshot_steps(self) -> Dict[str, int]:
        return dict(self._snapshot_steps)


def make_temporal_silhouette_callback(
    val_loader: DataLoader,
    device: torch.device,
    max_samples: int = 512,
    target_lif_name: Optional[str] = None,
    snapshots: Sequence[Tuple[str, float]] = TEMPORAL_SNAPSHOT_FRACTIONS,
) -> Callable[[nn.Module, Dict[str, Any], int], Dict[str, Any]]:
    """
    Build a silhouette callback that measures temporal engram formation.

    For each validation sample, this captures cumulative latent spike embeddings
    at 25%, 50%, 75%, and 100% of the sequence and computes silhouette scores
    on up to `max_samples` examples (default 512) under `torch.no_grad()`.
    """

    def temporal_silhouette_callback(
        model: nn.Module,
        _history: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        from sklearn.metrics import silhouette_score  # local import for optional dependency

        set_global_seed(seed, deterministic=True)

        collector = TemporalEngramSnapshotHook(
            model=model,
            target_lif_name=target_lif_name,
            snapshots=snapshots,
        ).attach()

        embeddings_by_snapshot: Dict[str, List[torch.Tensor]] = defaultdict(list)
        labels_by_snapshot: Dict[str, List[torch.Tensor]] = defaultdict(list)
        snapshot_steps_observed: Dict[str, int] = {}
        samples_collected = 0

        try:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if samples_collected >= max_samples:
                        break

                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        continue

                    data, labels = batch[0], batch[1]
                    remaining = max(0, int(max_samples - samples_collected))
                    if remaining <= 0:
                        break

                    if isinstance(data, torch.Tensor) and data.size(0) > remaining:
                        data = data[:remaining]
                        labels = labels[:remaining]

                    if not isinstance(data, torch.Tensor) or not isinstance(labels, torch.Tensor):
                        continue

                    collector.start_batch(data)
                    data = data.to(device)
                    _ = model(data)
                    batch_snapshots = collector.finish_batch()
                    snapshot_steps_observed.update(collector.snapshot_steps)

                    labels_cpu = labels.detach().cpu()
                    batch_count = int(labels_cpu.size(0))
                    if batch_count <= 0:
                        continue

                    for snap_label, embedding in batch_snapshots.items():
                        if embedding.numel() == 0:
                            continue
                        embeddings_by_snapshot[snap_label].append(embedding[:batch_count].cpu())
                        labels_by_snapshot[snap_label].append(labels_cpu.clone())

                    samples_collected += batch_count
        finally:
            collector.remove()

        temporal_scores: Dict[str, float] = {}
        for snap_label, _ in snapshots:
            emb_chunks = embeddings_by_snapshot.get(snap_label, [])
            lab_chunks = labels_by_snapshot.get(snap_label, [])
            if not emb_chunks or not lab_chunks:
                temporal_scores[snap_label] = float("nan")
                continue

            emb = torch.cat(emb_chunks, dim=0).cpu().numpy()
            y = torch.cat(lab_chunks, dim=0).cpu().numpy()

            # Require at least two classes and two samples for silhouette.
            unique_classes = set(y.tolist()) if y.size else set()
            if emb.shape[0] < 2 or len(unique_classes) < 2:
                temporal_scores[snap_label] = float("nan")
                continue

            try:
                temporal_scores[snap_label] = float(silhouette_score(emb, y))
            except Exception:
                temporal_scores[snap_label] = float("nan")

        return {
            "temporal_silhouette": temporal_scores,
            "snapshot_steps": snapshot_steps_observed,
            "num_samples_temporal_eval": int(samples_collected),
            "temporal_target_lif": collector.target_lif_name,
        }

    return temporal_silhouette_callback


def _count_parameters(model: nn.Module) -> int:
    """Count total parameters (trainable + frozen) for fairness checks."""
    return sum(p.numel() for p in model.parameters())


def _collapse_temporal_input_for_ann(x: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Collapse SNN temporal input [B, T, ...] into a single ANN forward input.

    Uses sum over time to preserve event-count magnitude.
    """
    if not isinstance(x, torch.Tensor):
        return x

    if kind == "visual":
        if x.dim() == 5:
            # [B, T, C, H, W] -> [B, C, H, W]
            return x.sum(dim=1)
        return x

    if kind == "audio":
        if x.dim() == 5:
            # Typical SHD path sometimes carries singleton spatial dims.
            if x.size(2) == 1 and x.size(3) == 1:
                x = x.squeeze(2).squeeze(2)
        if x.dim() == 3:
            # [B, T, F] -> [B, F]
            return x.sum(dim=1)
        return x

    return x


class _ANNBackboneShadow(nn.Module):
    """
    ANN shadow for SNN backbones (same weights, no temporal loop, ReLU activations).
    """

    def __init__(self, snn_backbone: nn.Module):
        super().__init__()
        self.relu = nn.ReLU()

        if all(hasattr(snn_backbone, name) for name in ("conv1", "conv2", "fc1", "fc_out")):
            self.kind = "visual"
            self.conv1 = copy.deepcopy(snn_backbone.conv1)
            self.conv2 = copy.deepcopy(snn_backbone.conv2)
            self.fc1 = copy.deepcopy(snn_backbone.fc1)
            self.fc_out = copy.deepcopy(snn_backbone.fc_out)
        elif all(hasattr(snn_backbone, name) for name in ("fc_input", "fc1", "fc_hidden", "fc_out")):
            self.kind = "audio"
            self.fc_input = copy.deepcopy(snn_backbone.fc_input)
            self.fc1 = copy.deepcopy(snn_backbone.fc1)
            self.fc_hidden = copy.deepcopy(snn_backbone.fc_hidden)
            self.fc_out = copy.deepcopy(snn_backbone.fc_out)
        else:
            raise ValueError(f"Unsupported backbone for ANN shadow: {type(snn_backbone).__name__}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _collapse_temporal_input_for_ann(x, self.kind)
        if self.kind == "visual":
            h1 = self.relu(self.conv1(x))
            h2 = self.relu(self.conv2(h1))
            h3 = self.relu(self.fc1(h2.flatten(1)))
            out = self.relu(self.fc_out(h3))
            return out, h3

        h_in = self.relu(self.fc_input(x))
        h1 = self.relu(self.fc1(h_in))
        h_hidden = self.relu(self.fc_hidden(h1))
        out = self.relu(self.fc_out(h_hidden))
        return out, h_hidden


class ANNShadowModel(nn.Module):
    """
    Non-spiking ANN baseline shadow cloned from an SNN model.

    - Same parameterized layers (deep-copied weights/parameters)
    - LIF activations replaced with ReLU
    - Temporal loop removed by summing inputs over time before a single forward pass
    """

    def __init__(self, snn_model: nn.Module):
        super().__init__()
        self.source_model_name = getattr(snn_model, "name", type(snn_model).__name__)
        self.input_type = getattr(snn_model, "input_type", None)
        self.relu = nn.ReLU()

        if not hasattr(snn_model, "backbone"):
            raise ValueError("ANN shadow currently expects SNN model to expose `.backbone`.")

        self.backbone = _ANNBackboneShadow(snn_model.backbone)

        self.has_hopfield = hasattr(snn_model, "hopfield")
        self.has_hgrn = hasattr(snn_model, "hgrn")
        self.has_head = hasattr(snn_model, "fc_out")

        if self.has_hopfield:
            self.hopfield = copy.deepcopy(snn_model.hopfield)
        if self.has_hgrn:
            self.hgrn = copy.deepcopy(snn_model.hgrn)
        if self.has_head:
            self.fc_out = copy.deepcopy(snn_model.fc_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spk_like, features = self.backbone(x)

        # Baseline/SCL variants end at the backbone head.
        if not self.has_head:
            return spk_like, features

        h = features
        if self.has_hopfield:
            h = self.hopfield(h)
        if self.has_hgrn:
            h_prev = torch.zeros(h.size(0), h.size(1), device=h.device, dtype=h.dtype)
            h = self.hgrn(h, h_prev)

        out = self.relu(self.fc_out(h))
        return out, h


def create_ann_baseline(snn_model: nn.Module) -> nn.Module:
    """
    Create a non-spiking ANN shadow baseline from an SNN model.

    The ANN preserves the same parameterized layers/weights (deep copy), swaps
    LIF behavior for ReLU activations, and removes explicit time-step loops.
    """
    ann_model = ANNShadowModel(snn_model)

    snn_params = _count_parameters(snn_model)
    ann_params = _count_parameters(ann_model)
    if snn_params != ann_params:
        raise ValueError(
            "ANN baseline parameter-count mismatch: "
            f"SNN={snn_params}, ANN={ann_params}. Comparison would be unfair."
        )

    return ann_model


def calculate_model_flops(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Manual FLOP counter for Conv2d/Linear layers for one forward pass.

    Counts multiply-add as 2 FLOPs and includes bias adds (if present).
    """
    model = model.to(device)
    model.eval()

    total_flops = 0.0
    flops_by_module: Dict[str, float] = defaultdict(float)
    module_to_name = {module: name for name, module in model.named_modules()}
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs: Tuple[Any, ...], output: Any) -> None:
        nonlocal total_flops
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(out, torch.Tensor) or out.dim() != 4:
            return
        batch, out_channels, out_h, out_w = out.shape
        kh, kw = module.kernel_size
        in_per_group = module.in_channels // module.groups
        macs_per_out = in_per_group * kh * kw
        flops_per_out = (2.0 * macs_per_out) + (1.0 if module.bias is not None else 0.0)
        flops = float(batch * out_channels * out_h * out_w) * flops_per_out
        name = module_to_name.get(module, module.__class__.__name__)
        flops_by_module[name] += flops
        total_flops += flops

    def linear_hook(module: nn.Linear, _inputs: Tuple[Any, ...], output: Any) -> None:
        nonlocal total_flops
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(out, torch.Tensor) or out.dim() < 1:
            return
        out_features = int(out.shape[-1])
        outer = int(out.numel() // max(1, out_features))
        macs_per_out = module.in_features
        flops_per_out = (2.0 * macs_per_out) + (1.0 if module.bias is not None else 0.0)
        flops = float(outer * out_features) * flops_per_out
        name = module_to_name.get(module, module.__class__.__name__)
        flops_by_module[name] += flops
        total_flops += flops

    def hopfield_hook(module: nn.Module, inputs: Tuple[Any, ...], _output: Any) -> None:
        """
        Approximate ModernHopfieldLayer compute dominated by two dense matmuls:
          (B, I) x (I, M) and (B, M) x (M, I)
        """
        nonlocal total_flops
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            return
        if not hasattr(module, "memory"):
            return

        memory = getattr(module, "memory")
        if not isinstance(memory, torch.Tensor) or memory.dim() != 2:
            return

        batch = int(x.size(0))
        input_size = int(x.size(1))
        memory_size = int(memory.size(0))

        flops_similarity = float(batch * memory_size) * (2.0 * input_size)
        flops_retrieval = float(batch * input_size) * (2.0 * memory_size)
        flops = flops_similarity + flops_retrieval

        name = module_to_name.get(module, module.__class__.__name__)
        flops_by_module[name] += flops
        total_flops += flops

    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif type(module).__name__ == "ModernHopfieldLayer":
                handles.append(module.register_forward_hook(hopfield_hook))

        with torch.no_grad():
            _ = model(sample_input.to(device))
    finally:
        for h in handles:
            h.remove()

    batch_size = int(sample_input.size(0)) if isinstance(sample_input, torch.Tensor) and sample_input.dim() > 0 else 1
    flops_per_sample = total_flops / max(1, batch_size)

    return {
        "total_flops": float(total_flops),
        "flops_per_sample": float(flops_per_sample),
        "batch_size": batch_size,
        "flops_by_module": dict(flops_by_module),
        "counting_rule": (
            "Conv2d/Linear (+Hopfield matmul approximation); multiply-add=2 FLOPs; bias add included"
        ),
    }


def compute_energy_improvement_ratio(
    ann_flops_per_sample: float,
    snn_mean_synops_per_sample: float,
    ann_energy_per_flop_pj: float = ANN_ENERGY_PER_FLOP_PJ,
    snn_energy_per_synop_pj: float = SNN_ENERGY_PER_SYNOP_PJ,
) -> Dict[str, float]:
    """
    Compute ANN-vs-SNN energy ratio using configurable per-op constants (pJ).
    """
    energy_ann_pj = float(ann_flops_per_sample) * float(ann_energy_per_flop_pj)
    energy_snn_pj = float(snn_mean_synops_per_sample) * float(snn_energy_per_synop_pj)
    ratio = float("inf") if energy_snn_pj <= 0 else (energy_ann_pj / energy_snn_pj)
    return {
        "ann_energy_pj": energy_ann_pj,
        "snn_energy_pj": energy_snn_pj,
        "energy_improvement_ratio": ratio,
        "ann_energy_per_flop_pj": float(ann_energy_per_flop_pj),
        "snn_energy_per_synop_pj": float(snn_energy_per_synop_pj),
    }


def _get_single_sample_batch(
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    for batch in data_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 1:
            continue
        data = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        if isinstance(data, torch.Tensor) and data.size(0) > 1:
            data = data[:1]
            if isinstance(labels, torch.Tensor):
                labels = labels[:1]
        if isinstance(data, torch.Tensor):
            return data.to(device), labels
    raise ValueError("Could not retrieve a tensor batch from data_loader for ANN FLOP profiling.")


def make_energy_methodology_callback(
    eval_loader: DataLoader,
    device: torch.device,
    profile_max_batches: Optional[int] = None,
    ann_energy_per_flop_pj: float = ANN_ENERGY_PER_FLOP_PJ,
    snn_energy_per_synop_pj: float = SNN_ENERGY_PER_SYNOP_PJ,
) -> Callable[[nn.Module, Dict[str, Any], int], Dict[str, Any]]:
    """
    Build a callback that compares SNN SynOps against a shadow ANN FLOP baseline.
    """

    def energy_callback(
        model: nn.Module,
        _history: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        set_global_seed(seed, deterministic=True)

        with torch.no_grad():
            snn_profile = profile_network_activity(
                model=model,
                data_loader=eval_loader,
                device=device,
                max_batches=profile_max_batches,
            )

            ann_model = create_ann_baseline(model).to(device)
            sample_input, _ = _get_single_sample_batch(eval_loader, device=device)
            ann_flops = calculate_model_flops(
                model=ann_model,
                sample_input=sample_input,
                device=device,
            )

        snn_params = _count_parameters(model)
        ann_params = _count_parameters(ann_model)
        param_match = int(snn_params == ann_params)

        energy = compute_energy_improvement_ratio(
            ann_flops_per_sample=float(ann_flops["flops_per_sample"]),
            snn_mean_synops_per_sample=float(snn_profile["mean_synops_per_sample"]),
            ann_energy_per_flop_pj=ann_energy_per_flop_pj,
            snn_energy_per_synop_pj=snn_energy_per_synop_pj,
        )

        return {
            "energy": {
                "ann": {
                    "flops_single_forward": float(ann_flops["total_flops"]),
                    "flops_per_sample": float(ann_flops["flops_per_sample"]),
                    "params": float(ann_params),
                },
                "snn": {
                    "total_synops_profiled": float(snn_profile["total_synops"]),
                    "mean_synops_per_sample": float(snn_profile["mean_synops_per_sample"]),
                    "num_samples_profiled": float(snn_profile["num_samples"]),
                    "num_batches_profiled": float(snn_profile.get("num_batches", 0)),
                },
                "constants_pj": {
                    "ann_per_flop": float(ann_energy_per_flop_pj),
                    "snn_per_synop": float(snn_energy_per_synop_pj),
                },
                "ratio": float(energy["energy_improvement_ratio"]),
                "ann_energy_pj": float(energy["ann_energy_pj"]),
                "snn_energy_pj": float(energy["snn_energy_pj"]),
                "parameter_count_match": float(param_match),
                "snn_params": float(snn_params),
                "ann_params": float(ann_params),
            }
        }

    return energy_callback


class TrainingTracker:
    """Logs metrics and handles early stopping + checkpointing."""

    def __init__(self, model_name: str, model: nn.Module, checkpoint_dir: Path, patience: int = 7):
        self.model_name = model_name
        self.model = model
        self.model_path = checkpoint_dir / f"{model_name}_best.pth"
        self.patience = patience
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_ce_loss": [],
            "train_scl_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_ce_loss": [],
            "val_scl_loss": [],
            "best_val_acc": 0.0,
            "epochs_trained": 0,
            "lr": [],
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, epoch: int, metrics: dict) -> bool:
        self.history["train_loss"].append(metrics["train_loss"])
        self.history["train_ce_loss"].append(metrics["train_ce_loss"])
        self.history["train_scl_loss"].append(metrics["train_scl_loss"])
        self.history["train_acc"].append(metrics["train_acc"])
        self.history["val_loss"].append(metrics["val_loss"])
        self.history["val_ce_loss"].append(metrics["val_ce_loss"])
        self.history["val_scl_loss"].append(metrics["val_scl_loss"])
        self.history["val_acc"].append(metrics["val_acc"])
        self.history["lr"].append(metrics["lr"])
        self.history["epochs_trained"] = epoch + 1

        if metrics["val_acc"] > self.best_val_acc:
            self.best_val_acc = metrics["val_acc"]
            self.history["best_val_acc"] = self.best_val_acc
            self.epochs_no_improve = 0
            self.save_checkpoint(epoch)
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve < self.patience

    def save_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_acc": self.best_val_acc,
            },
            self.model_path,
        )

    def get_history(self) -> Dict[str, Any]:
        return self.history


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    con_loss_fn: Optional[nn.Module],
    contrastive_weight: float,
    device: torch.device,
    gradient_clip: float,
) -> Tuple[float, float, float, float]:
    total_loss = total_ce_loss = total_con_loss = 0.0
    total_correct = total_samples = 0
    model.train()

    pbar = tqdm(data_loader, desc="[Train]", leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        spk_sum, features = model(data)

        loss_ce = ce_loss_fn(spk_sum, labels)
        if contrastive_weight > 0 and con_loss_fn is not None:
            loss_con = con_loss_fn(features, labels)
            loss = loss_ce + contrastive_weight * loss_con
        else:
            loss = loss_ce
            loss_con = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_ce_loss += loss_ce.item() * data.size(0)
        total_con_loss += loss_con.item() * data.size(0)
        total_correct += (spk_sum.argmax(1) == labels).sum().item()
        total_samples += data.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * total_correct / total_samples:.2f}%")

    avg_loss = total_loss / (total_samples + 1e-9)
    avg_ce_loss = total_ce_loss / (total_samples + 1e-9)
    avg_con_loss = total_con_loss / (total_samples + 1e-9)
    avg_acc = (total_correct / (total_samples + 1e-9)) * 100
    return avg_loss, avg_ce_loss, avg_con_loss, avg_acc


def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    ce_loss_fn: nn.Module,
    con_loss_fn: Optional[nn.Module],
    contrastive_weight: float,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    total_loss = total_ce_loss = total_con_loss = 0.0
    total_correct = total_samples = 0
    model.eval()

    pbar = tqdm(data_loader, desc="[Val]", leave=False)
    with torch.no_grad():
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            spk_sum, features = model(data)

            loss_ce = ce_loss_fn(spk_sum, labels)
            if contrastive_weight > 0 and con_loss_fn is not None:
                loss_con = con_loss_fn(features, labels)
                loss = loss_ce + contrastive_weight * loss_con
            else:
                loss = loss_ce
                loss_con = torch.tensor(0.0, device=device)

            total_loss += loss.item() * data.size(0)
            total_ce_loss += loss_ce.item() * data.size(0)
            total_con_loss += loss_con.item() * data.size(0)
            total_correct += (spk_sum.argmax(1) == labels).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / (total_samples + 1e-9)
    avg_ce_loss = total_ce_loss / (total_samples + 1e-9)
    avg_con_loss = total_con_loss / (total_samples + 1e-9)
    avg_acc = (total_correct / (total_samples + 1e-9)) * 100
    return avg_loss, avg_ce_loss, avg_con_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    dataset_name: str,
    use_contrastive: bool,
    device: torch.device,
    config: TrainingConfig,
) -> Tuple[nn.Module, Dict[str, Any], float]:
    lr = config.learning_rate
    scl_weight = config.contrastive_weight if use_contrastive else 0.0

    ce_loss_fn = nn.CrossEntropyLoss()
    con_loss_fn = None
    if use_contrastive:
        con_loss_fn = SupervisedContrastiveLoss(
            temperature=config.contrastive_temperature
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    tracker = TrainingTracker(model_name, model, config.checkpoint_dir, config.patience)

    start_time = time.time()
    enable_structural_plasticity = bool(getattr(config, "enable_structural_plasticity", False))
    plasticity_interval = int(getattr(config, "structural_plasticity_interval", 5))
    plasticity_max_batches = int(getattr(config, "structural_plasticity_max_batches", 50))
    kill_start_epoch = int(getattr(config, "structural_plasticity_kill_start_epoch", 10))
    kill_threshold = float(getattr(config, "structural_plasticity_kill_threshold", 0.01))
    alpha_frac = float(getattr(config, "structural_plasticity_alpha_frac", 0.05))
    breed_sigma = float(getattr(config, "structural_plasticity_breed_sigma", 0.01))

    for epoch in range(config.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_ce, train_scl, train_acc = train_step(
            model,
            train_loader,
            optimizer,
            ce_loss_fn,
            con_loss_fn,
            scl_weight,
            device,
            config.gradient_clip,
        )
        val_loss, val_ce, val_scl, val_acc = test_step(
            model, test_loader, ce_loss_fn, con_loss_fn, scl_weight, device
        )
        scheduler.step()

        # Structural plasticity every plasticity_interval epochs (kill only when epoch >= kill_start_epoch)
        if enable_structural_plasticity and plasticity_interval > 0 and (epoch + 1) % plasticity_interval == 0:
            firing_rates = collect_firing_rates(model, train_loader, device, max_batches=plasticity_max_batches)
            if firing_rates:
                apply_structural_plasticity(
                    model,
                    firing_rates,
                    epoch,
                    kill_start_epoch=kill_start_epoch,
                    kill_threshold=kill_threshold,
                    alpha_frac=alpha_frac,
                    breed_sigma=breed_sigma,
                )

        if not tracker.log_epoch(
            epoch,
            {
                "train_loss": train_loss,
                "train_ce_loss": train_ce,
                "train_scl_loss": train_scl,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_ce_loss": val_ce,
                "val_scl_loss": val_scl,
                "val_acc": val_acc,
                "lr": current_lr,
            },
        ):
            break

    total_time = (time.time() - start_time) / 60
    print(
        f"Training finished: {model_name} on {dataset_name} "
        f"({tracker.best_val_acc:.2f}% best, {total_time:.2f} min)"
    )
    return model, tracker.history, tracker.best_val_acc


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python/NumPy/PyTorch seeds for repeatable runs."""
    random.seed(seed)
    try:
        import numpy as np  # local import to keep pipeline import lightweight

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _flatten_numeric_metrics(obj: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_numeric_metrics(value, child_prefix))
        return flat

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1 and prefix:
            flat[prefix] = float(obj.detach().cpu().item())
        return flat

    if isinstance(obj, Real) and prefix:
        flat[prefix] = float(obj)

    return flat


def _metric_mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "std_dev": 0.0, "n": 0, "n_valid": 0}
    tensor = torch.tensor(list(values), dtype=torch.float64)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "std_dev": nan, "n": int(tensor.numel()), "n_valid": 0}
    std = float(finite.std(unbiased=finite.numel() > 1).item()) if finite.numel() > 1 else 0.0
    return {
        "mean": float(finite.mean().item()),
        "std": std,
        "std_dev": std,
        "n": int(tensor.numel()),
        "n_valid": int(finite.numel()),
    }


def _build_temporal_silhouette_summary(flat_summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Group temporal silhouette stats for easier reporting/plotting.

    Handles both:
    - `temporal_silhouette.t25`
    - `<prefix>.temporal_silhouette.t25` (e.g., vision/audio prefixes)
    """
    grouped: Dict[str, Any] = {}
    for metric_key, stats in flat_summary.items():
        if metric_key.startswith("temporal_silhouette."):
            snap = metric_key.split("temporal_silhouette.", 1)[1]
            grouped[snap] = stats
            continue
        if ".temporal_silhouette." in metric_key:
            prefix, snap = metric_key.split(".temporal_silhouette.", 1)
            grouped.setdefault(prefix, {})[snap] = stats
    return grouped


def _extract_energy_summary(flat_summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Collect aggregated energy-methodology metrics from the flat summary map."""
    energy_summary: Dict[str, Any] = {}
    for metric_key, stats in flat_summary.items():
        if metric_key.startswith("energy."):
            energy_summary[metric_key] = stats
    return energy_summary


def _json_safe(obj: Any) -> Any:
    """Convert tensors/Paths/non-finite floats into JSON-safe values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _json_safe(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if not torch.isfinite(torch.tensor(obj)):
            return None
        return obj
    return obj


def _build_final_specs_payload(
    summary: Dict[str, Dict[str, float]],
    temporal_silhouette_summary: Dict[str, Any],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """
    Build the final_specs.json payload requested for reporting.
    """
    acc = summary.get("accuracy", {})

    # Prefer top-level temporal summary; if nested prefixes exist, export all.
    temporal_export: Dict[str, Any]
    if temporal_silhouette_summary and any(
        isinstance(v, dict) and "mean" not in v for v in temporal_silhouette_summary.values()
    ):
        temporal_export = temporal_silhouette_summary
    else:
        temporal_export = {
            snap: temporal_silhouette_summary.get(snap, {})
            for snap in ("t25", "t50", "t75", "t100")
            if snap in temporal_silhouette_summary
        }

    energy_fields = {
        "ann_flops_single_forward": summary.get("energy.ann.flops_single_forward", {}),
        "ann_flops_per_sample": summary.get("energy.ann.flops_per_sample", {}),
        "snn_total_synops_profiled": summary.get("energy.snn.total_synops_profiled", {}),
        "snn_mean_synops_per_sample": summary.get("energy.snn.mean_synops_per_sample", {}),
        "energy_improvement_ratio": summary.get("energy.ratio", {}),
        "ann_energy_pj": summary.get("energy.ann_energy_pj", {}),
        "snn_energy_pj": summary.get("energy.snn_energy_pj", {}),
        "ann_energy_per_flop_pj": summary.get("energy.constants_pj.ann_per_flop", {}),
        "snn_energy_per_synop_pj": summary.get("energy.constants_pj.snn_per_synop", {}),
        "parameter_count_match": summary.get("energy.parameter_count_match", {}),
        "snn_params": summary.get("energy.snn_params", {}),
        "ann_params": summary.get("energy.ann_params", {}),
    }

    payload = {
        "num_runs": len(seeds),
        "seeds": [int(s) for s in seeds],
        "accuracy": {
            "mean": acc.get("mean"),
            "std_dev": acc.get("std_dev", acc.get("std")),
        },
        "temporal_silhouette": temporal_export,
        "compute_comparison": {
            "total_synops_vs_total_flops": {
                "snn_total_synops_profiled": energy_fields["snn_total_synops_profiled"],
                "ann_flops_single_forward": energy_fields["ann_flops_single_forward"],
            },
            "mean_synops_vs_mean_flops_per_sample": {
                "snn_mean_synops_per_sample": energy_fields["snn_mean_synops_per_sample"],
                "ann_flops_per_sample": energy_fields["ann_flops_per_sample"],
            },
            "energy_constants_pj": {
                "ann_per_flop": energy_fields["ann_energy_per_flop_pj"],
                "snn_per_synop": energy_fields["snn_energy_per_synop_pj"],
            },
            "energies_pj": {
                "ann": energy_fields["ann_energy_pj"],
                "snn": energy_fields["snn_energy_pj"],
            },
            "energy_improvement_ratio": energy_fields["energy_improvement_ratio"],
            "fairness": {
                "parameter_count_match": energy_fields["parameter_count_match"],
                "snn_params": energy_fields["snn_params"],
                "ann_params": energy_fields["ann_params"],
            },
        },
    }
    return _json_safe(payload)


def _extract_silhouette_metrics(silhouette_result: Any) -> Dict[str, float]:
    if silhouette_result is None:
        return {}

    if isinstance(silhouette_result, torch.Tensor) and silhouette_result.numel() == 1:
        return {"silhouette": float(silhouette_result.detach().cpu().item())}

    if isinstance(silhouette_result, Real):
        return {"silhouette": float(silhouette_result)}

    flat = _flatten_numeric_metrics(silhouette_result)
    sil_metrics = {k: v for k, v in flat.items() if "silhouette" in k.lower()}
    if sil_metrics:
        return sil_metrics
    return flat


def run_seed_iterator(
    train_once_fn: Callable[[int], Tuple[nn.Module, Dict[str, Any], float]],
    num_runs: int = 5,
    base_seed: int = 42,
    seeds: Optional[Sequence[int]] = None,
    silhouette_fn: Optional[Callable[[nn.Module, Dict[str, Any], int], Any]] = None,
    energy_fn: Optional[Callable[[nn.Module, Dict[str, Any], int], Any]] = None,
    deterministic: bool = True,
    final_specs_path: Optional[Path] = Path("final_specs.json"),
) -> Dict[str, Any]:
    """
    Wrap a single-run training function and aggregate metrics over multiple seeds.

    Expected train_once_fn signature:
        train_once_fn(seed) -> (model, history, best_val_acc)

    Optional silhouette_fn signature:
        silhouette_fn(model, history, seed) -> float | dict

    Optional energy_fn signature:
        energy_fn(model, history, seed) -> dict

    Exports `final_specs.json` by default (set final_specs_path=None to disable).
    """
    run_seeds = list(seeds) if seeds is not None else [base_seed + i for i in range(num_runs)]
    if not run_seeds:
        raise ValueError("run_seed_iterator requires at least one seed.")
    final_specs_target = Path(final_specs_path) if final_specs_path is not None else None

    runs: List[Dict[str, Any]] = []
    metric_series: Dict[str, List[float]] = defaultdict(list)
    last_model: Optional[nn.Module] = None
    last_history: Optional[Dict[str, Any]] = None

    for idx, seed in enumerate(run_seeds, start=1):
        print(f"[Seed Run {idx}/{len(run_seeds)}] seed={seed}")
        set_global_seed(seed, deterministic=deterministic)

        model, history, best_val_acc = train_once_fn(seed)
        last_model, last_history = model, history

        run_metrics: Dict[str, float] = {"accuracy": float(best_val_acc)}

        if silhouette_fn is not None:
            silhouette_metrics = _extract_silhouette_metrics(silhouette_fn(model, history, seed))
            run_metrics.update(silhouette_metrics)

        if energy_fn is not None:
            energy_metrics = _flatten_numeric_metrics(energy_fn(model, history, seed))
            run_metrics.update(energy_metrics)

        for metric_name, metric_value in run_metrics.items():
            metric_series[metric_name].append(float(metric_value))

        runs.append(
            {
                "seed": int(seed),
                "metrics": run_metrics,
                "best_val_acc": float(best_val_acc),
                "history": history,
            }
        )

    summary = {metric_name: _metric_mean_std(values) for metric_name, values in metric_series.items()}
    temporal_silhouette_summary = _build_temporal_silhouette_summary(summary)
    energy_summary = _extract_energy_summary(summary)

    final_specs = _build_final_specs_payload(
        summary=summary,
        temporal_silhouette_summary=temporal_silhouette_summary,
        seeds=run_seeds,
    )
    if final_specs_target is not None:
        final_specs_target.parent.mkdir(parents=True, exist_ok=True)
        with open(final_specs_target, "w", encoding="utf-8") as f:
            json.dump(final_specs, f, indent=2)

    return {
        "seeds": [int(s) for s in run_seeds],
        "num_runs": len(run_seeds),
        "runs": runs,
        "summary": summary,
        "temporal_silhouette_summary": temporal_silhouette_summary,
        "energy_summary": energy_summary,
        "final_specs": final_specs,
        "final_specs_path": str(final_specs_target) if final_specs_target is not None else None,
        "last_model": last_model,
        "last_history": last_history,
    }

"""
Training configuration and defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    gradient_clip: float = 1.0
    num_epochs: int = 30
    patience: int = 7
    checkpoint_dir: Path = Path("checkpoints")
    enable_structural_plasticity: bool = False
    structural_plasticity_interval: int = 5
    structural_plasticity_max_batches: int = 50
    structural_plasticity_kill_start_epoch: int = 10
    structural_plasticity_kill_threshold: float = 0.01
    structural_plasticity_alpha_frac: float = 0.05
    structural_plasticity_breed_sigma: float = 0.01

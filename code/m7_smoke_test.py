# %% CELL 0
# !pip install --upgrade jinja2 -q
# !pip install matplotlib
# !pip install seaborn

# %% CELL 1
"""Cell 2: Install Missing Dependencies (Kaggle has most pre-installed)"""

print("="*80)
print("📦 INSTALLING DEPENDENCIES")
print("="*80)

# Kaggle pre-installs: torch, numpy, pandas, matplotlib, scikit-learn, seaborn, tqdm
# We only need to install neuromorphic-specific packages

print("\nInstalling neuromorphic packages...")
# !pip install -q snntorch
# !pip install -q tonic
# !pip install tonic

print("Installing NLP packages (if needed)...")
# !pip install -q transformers
# !pip install -q datasets

print("\n✅ All dependencies installed successfully!")
print("   Pre-installed by Kaggle: torch, numpy, pandas, matplotlib, scikit-learn, seaborn, tqdm")
print("   Newly installed: snntorch, tonic, transformers, datasets")
print("="*80 + "\n")

# %% CELL 2
# System imports
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch

# ── MPS/CPU compatibility helpers ────────────────────────────────────────────
def safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS has no empty_cache API; OS manages unified memory


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# snnTorch & Tonic
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import tonic
from tonic import datasets, transforms

# Sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    confusion_matrix,
    classification_report
)

# Progress bars
from tqdm.auto import tqdm

# Weights & Biases (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.backends.mps.is_available():
    print("GPU: Apple Silicon MPS")
elif torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# %% CELL 3
# ── Install ───────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(['pip', 'install', '-q', 'snntorch', 'tonic'], check=False)

# ── Imports ───────────────────────────────────────────────────────────────────
import os, json, time, copy, pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import snntorch as snn
from snntorch import surrogate, utils
import tonic
import tonic.transforms

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"✅ Device: {device}")
if torch.backends.mps.is_available():
    print("   GPU: Apple Silicon MPS")
    print("   Memory: Shared unified memory (OS managed)")
elif torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR        = Path('./ICONS_M7')
DATA_DIR        = BASE_DIR / 'datasets'
DATASETS_DIR    = BASE_DIR / 'datasets'   # alias used by SHD loader
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
RESULTS_DIR     = BASE_DIR / 'results'
FIGURES_DIR     = BASE_DIR / 'figures'
for d in [DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print(f"✅ Directories ready under {BASE_DIR}")

# ── Global CONFIG (matches Paper 2 confirmed run) ─────────────────────────────
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'patience': 5,
    'gradient_clip': 1.0,
    'beta': 0.9,
    'dropout': 0.2,
    'hidden_dim': 512,
    'num_patterns': 100,
    'num_gru_layers': 2,
    'num_workers': 0,
    'time_steps': 25,
    'pin_memory': True,
    'use_contrastive': True,
    'contrastive_temperature': 0.07,
    'contrastive_weight': 0.1,
    'seed': 42,
    'device': device,
    'save_dir': RESULTS_DIR,
}
spike_grad = surrogate.atan()
print("✅ CONFIG ready")


# %% CELL 4
"""Cell 7: N-MNIST Dataset Preparation"""

print("="*80)
print("📊 N-MNIST DATASET LOADING")
print("="*80)

def get_nmnist_loaders(batch_size=32, num_workers=2):
    print("\n🔄 Loading N-MNIST...")
    
    sensor_size = tonic.datasets.NMNIST.sensor_size
    print(f"   Sensor size: {sensor_size}")
    
    transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    
    print("   Loading train set...")
    train_dataset = tonic.datasets.NMNIST(
        save_to=str(DATA_DIR), train=True, transform=transform)
    
    print("   Loading test set...")
    test_dataset = tonic.datasets.NMNIST(
        save_to=str(DATA_DIR), train=False, transform=transform)
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    
    print("\n   📦 Creating disk cache...")
    cached_train = tonic.DiskCachedDataset(
        train_dataset, cache_path=str(DATA_DIR / 'nmnist_train_cache'))
    cached_test = tonic.DiskCachedDataset(
        test_dataset, cache_path=str(DATA_DIR / 'nmnist_test_cache'))
    
    train_loader = DataLoader(
        cached_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=nmnist_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        cached_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=nmnist_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    dataset_info = {
        'name': 'N-MNIST', 'num_classes': 10,
        'input_channels': 2, 'spatial_size': (34, 34), 'time_steps': 25,
        'train_samples': len(train_dataset), 'test_samples': len(test_dataset),
    }
    
    print("\n✅ N-MNIST loaded successfully!")
    print(f"   Batch shape: (batch, 25, 2, 34, 34)")
    return train_loader, test_loader, dataset_info


# ── Collate function defined at module scope (accessible everywhere) ───────────
def nmnist_collate_fn(batch):
    events, labels = zip(*batch)
    fixed = []
    for e in events:
        if not isinstance(e, torch.Tensor):
            e = torch.from_numpy(e).float()
        if e.shape[0] > 25:
            e = e[:25]
        elif e.shape[0] < 25:
            e = torch.cat([e, torch.zeros(25 - e.shape[0], *e.shape[1:])], dim=0)
        fixed.append(e)
    return torch.stack(fixed), torch.tensor(labels)


# ── Load N-MNIST ───────────────────────────────────────────────────────────────
train_loader_nmnist, test_loader_nmnist, nmnist_info = get_nmnist_loaders(
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers']
)

data_sample, label_sample = next(iter(train_loader_nmnist))
print(f"\n🧪 Sample batch:")
print(f"   Data shape:  {data_sample.shape}")
print(f"   Labels:      {label_sample[:8].tolist()}")
print("\n✅ N-MNIST ready for training!")
print("="*80 + "\n")

# ── Val split for CL experiments ──────────────────────────────────────────────
_sensor_size = tonic.datasets.NMNIST.sensor_size
_transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000),
    tonic.transforms.ToFrame(sensor_size=_sensor_size, time_window=1000),
])
_train_ds = tonic.datasets.NMNIST(
    save_to=str(DATA_DIR), train=True, transform=_transform)
_cached_train = tonic.DiskCachedDataset(
    _train_ds, cache_path=str(DATA_DIR / 'nmnist_train_cache'))

_n_val   = int(0.1 * len(_cached_train))
_n_train = len(_cached_train) - _n_val
_train_split, _val_split = torch.utils.data.random_split(
    _cached_train, [_n_train, _n_val],
    generator=torch.Generator().manual_seed(42)
)
val_loader_nmnist = DataLoader(
    _val_split, batch_size=CONFIG['batch_size'],
    shuffle=False, collate_fn=nmnist_collate_fn, num_workers=0
)
print(f"✅ N-MNIST val loader: {_n_val} samples")

# %% CELL 5
"""Cell 8: SHD Dataset Preparation"""

print("="*80)
print("📊 SHD DATASET LOADING")
print("="*80)

def shd_collate_fn(batch):
    events, labels = zip(*batch)
    return torch.stack(events), torch.tensor(labels)

def events_to_dense(events, label):
    time_bins, channels = 100, 700
    dense = torch.zeros(time_bins, channels)
    if len(events) > 0:
        max_time = events['t'].max() if len(events) > 0 else 1
        time_indices = (events['t'] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events['x'].astype(int)
        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0
    return dense.unsqueeze(1).unsqueeze(1), label

def get_shd_loaders(batch_size=32):
    print("\n🔄 Loading SHD...")
    
    print("   Loading train set...")
    train_dataset = tonic.datasets.SHD(save_to=str(DATASETS_DIR), train=True)
    print("   Loading test set...")
    test_dataset  = tonic.datasets.SHD(save_to=str(DATASETS_DIR), train=False)
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Test samples:  {len(test_dataset):,}")
    
    print("\n   🔄 Converting to dense format...")
    train_data = [events_to_dense(e, l) for e, l in tqdm(train_dataset, desc="   Train")]
    test_data  = [events_to_dense(e, l) for e, l in tqdm(test_dataset,  desc="   Test")]
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=shd_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        collate_fn=shd_collate_fn, pin_memory=True
    )
    
    dataset_info = {
        'name': 'SHD', 'num_classes': 20, 'input_channels': 1,
        'spatial_size': (1, 700), 'time_steps': 100,
        'train_samples': len(train_data), 'test_samples': len(test_data),
    }
    
    print("\n✅ SHD loaded successfully!")
    print(f"   Batch shape: (batch, 100, 1, 1, 700)")
    return train_loader, test_loader, dataset_info, train_data


# ── Load SHD ───────────────────────────────────────────────────────────────────
train_loader_shd, test_loader_shd, shd_info, shd_train_data = get_shd_loaders(
    batch_size=CONFIG['batch_size']
)

data_sample, label_sample = next(iter(train_loader_shd))
print(f"\n🧪 Sample batch:")
print(f"   Data shape:  {data_sample.shape}")
print(f"   Labels:      {label_sample[:8].tolist()}")
print("\n✅ SHD ready for training!")
print("="*80 + "\n")

# ── Val split for CL experiments ──────────────────────────────────────────────
_shd_n_val   = int(0.1 * len(shd_train_data))
_shd_n_train = len(shd_train_data) - _shd_n_val
_shd_train_split, _shd_val_split = torch.utils.data.random_split(
    shd_train_data, [_shd_n_train, _shd_n_val],
    generator=torch.Generator().manual_seed(42)
)
train_loader_shd_cl = DataLoader(
    _shd_train_split, batch_size=CONFIG['batch_size'],
    shuffle=True, collate_fn=shd_collate_fn, num_workers=0, pin_memory=True
)
val_loader_shd = DataLoader(
    _shd_val_split, batch_size=CONFIG['batch_size'],
    shuffle=False, collate_fn=shd_collate_fn, num_workers=0
)
print(f"✅ SHD val loader:   {_shd_n_val} samples")
print(f"✅ SHD train (CL):   {_shd_n_train} samples")
print(f"   Shape check: {next(iter(train_loader_shd_cl))[0].shape}")

# %% CELL 6
"""Cell 8.5: DVS-Gesture Dataset Preparation"""

def get_dvs_gesture_loaders(batch_size=32):
    print("\n🔄 Loading DVS-Gesture...")
    
    # DVS-Gesture: 11 classes, 128×128 spatial, ~1K samples
    # Need to pool to 34×34 to match N-MNIST backbone, or use adaptive pooling
    transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(
            sensor_size=(128, 128, 2),  # (H, W, C) — 2 channels (polarity)
            time_window=5000,
        ),
    ])
    
    train_dataset = tonic.datasets.DVSGesture(
        save_to=str(DATA_DIR), train=True, transform=transform)
    test_dataset = tonic.datasets.DVSGesture(
        save_to=str(DATA_DIR), train=False, transform=transform)
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    
    # Collate: pad/truncate to T=25, pool spatial 128→34
    def dvs_collate_fn(batch):
        events, labels = zip(*batch)
        fixed = []
        for e in events:
            if not isinstance(e, torch.Tensor):
                e = torch.from_numpy(e).float()
            # Temporal: pad/truncate to 25 timesteps
            if e.shape[0] > 25:
                e = e[:25]
            elif e.shape[0] < 25:
                e = torch.cat([
                    e, 
                    torch.zeros(25 - e.shape[0], *e.shape[1:])
                ], dim=0)
            # Spatial: adaptive pool from 128×128 → 34×34
            if e.shape[-2:] != (34, 34):
                e = F.adaptive_avg_pool2d(
                    e.view(-1, e.shape[-3], e.shape[-2], e.shape[-1]),
                    (34, 34)
                ).view(25, e.shape[-3], 34, 34)
            fixed.append(e)
        return torch.stack(fixed), torch.tensor(labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=dvs_collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=dvs_collate_fn, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader, {
        'name': 'DVS-Gesture', 'num_classes': 11,
        'input_channels': 2, 'spatial_size': (34, 34), 'time_steps': 25,
    }

# %% CELL 7
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import tonic
import numpy as np

# Note: Ensure CONFIG, DATA_DIR, and get_dvs_gesture_loaders are defined in your environment.

print("=" * 80)
print("📊 DVS-GESTURE DATASET LOADING")
print("=" * 80)

# --- Load DVS-Gesture --------------------------------------------------------
try:
    train_loader_dvs, test_loader_dvs, dvs_info = get_dvs_gesture_loaders(
        batch_size=CONFIG['batch_size']
    )
except Exception as e:
    print(f"\n🚨 DVS-Gesture download/load FAILED: {e}")
    print("   → FALL BACK TO SSC-35 (notify Blessing for loader)\n")
    raise RuntimeError("DVS-Gesture unavailable — abort. Use SSC-35 fallback.") from e

# Quick shape check
data_sample, label_sample = next(iter(train_loader_dvs))
print(f"\n🧪 DVS sample batch:")
print(f"   Data shape:  {data_sample.shape}")
print(f"   Labels:      {label_sample[:8].tolist()}")
print(f"   Classes:     {dvs_info['num_classes']}")

# --- Validation split for CL experiments -------------------------------------
# DVS-Gesture train is relatively small (~1K), so we perform a 10% val split
_dvs_train_dataset = train_loader_dvs.dataset
_dvs_n_val = int(0.1 * len(_dvs_train_dataset))
_dvs_n_train = len(_dvs_train_dataset) - _dvs_n_val

# Define transforms for deterministic splitting
_dvs_transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000),
    tonic.transforms.ToFrame(
        sensor_size=(128, 128, 2),
        time_window=5000,
    ),
])

def dvs_collate_fn(batch):
    """
    Normalizes batches:
    1. Truncates/Pads temporal dimension to 25 frames.
    2. Resizes spatial dimensions to 34x34 via adaptive pooling.
    """
    events, labels = zip(*batch)
    fixed = []
    
    for e in events:
        if not isinstance(e, torch.Tensor):
            e = torch.from_numpy(e).float()
        
        # Temporal padding/clipping to 25 frames
        if e.shape[0] > 25:
            e = e[:25]
        elif e.shape[0] < 25:
            e = torch.cat([
                e,
                torch.zeros(25 - e.shape[0], *e.shape[1:])
            ], dim=0)
            
        # Spatial resizing to 34x34
        if e.shape[-2:] != (34, 34):
            e = F.adaptive_avg_pool2d(
                e.view(-1, e.shape[-3], e.shape[-2], e.shape[-1]),
                (34, 34)
            ).view(25, e.shape[-3], 34, 34)
            
        fixed.append(e)
        
    return torch.stack(fixed), torch.tensor(labels)

# Re-instantiate raw dataset for splitting
_dvs_train_raw = tonic.datasets.DVSGesture(
    save_to=str(DATA_DIR), 
    train=True, 
    transform=_dvs_transform
)

_dvs_train_split, _dvs_val_split = random_split(
    _dvs_train_raw, 
    [_dvs_n_train, _dvs_n_val],
    generator=torch.Generator().manual_seed(42)
)

# Create Final DataLoaders
train_loader_dvs_cl = DataLoader(
    _dvs_train_split, 
    batch_size=CONFIG['batch_size'],
    shuffle=True, 
    collate_fn=dvs_collate_fn, 
    num_workers=0, 
    pin_memory=True
)

val_loader_dvs = DataLoader(
    _dvs_val_split, 
    batch_size=CONFIG['batch_size'],
    shuffle=False, 
    collate_fn=dvs_collate_fn, 
    num_workers=0, 
    pin_memory=True
)

print(f"✅ DVS val loader:   {_dvs_n_val} samples")
print(f"✅ DVS train (CL):   {_dvs_n_train} samples")
print(f"✅ DVS test:         {len(test_loader_dvs.dataset)} samples")
print(f"   Shape check:      {next(iter(train_loader_dvs_cl))[0].shape}")
print("=" * 80 + "\n")

# %% CELL 9
"""Cell 6: Supervised Contrastive Loss (Biggest Improvement!)"""

print("="*80)
print("🔥 SUPERVISED CONTRASTIVE LOSS")
print("="*80)

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss
    
    Key improvement: Pushes same-class samples together in feature space
    while pushing different-class samples apart.
    
    Expected improvement: +0.5-1.0% accuracy
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: (batch, feature_dim) - L2 normalized features
            labels: (batch,) - class labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label mask (1 for same class, 0 for different)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss

# Test the loss
print("\n🧪 Testing Contrastive Loss:")
test_features = torch.randn(8, 128).to(device)
test_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(device)

contrastive_loss = SupervisedContrastiveLoss(temperature=0.07)
loss = contrastive_loss(test_features, test_labels)

print(f"   Test loss: {loss.item():.4f}")
print(f"   ✅ Loss computed successfully!")

print("\n💡 This loss will be combined with cross-entropy:")
print("   Total Loss = CrossEntropy + 0.1 × Contrastive")

print("\n✅ Supervised Contrastive Loss ready!")
print("="*80 + "\n")

# %% CELL 10
"""Cell 10: Modern Hopfield Layer (Associative Memory)"""

print("="*80)
print("🧠 MODERN HOPFIELD LAYER")
print("="*80)

class ModernHopfieldLayer(nn.Module):
    """
    Modern Hopfield Layer with continuous states
    
    Improvements:
    ✅ Better Hopfield: Scaled dot-product attention + temperature
    ✅ Skip Connection: Residual connection for gradient flow
    ✅ LayerNorm: Stabilizes training
    
    This acts as associative memory - given a noisy/partial pattern,
    it retrieves the closest stored pattern from memory.
    """
    
    def __init__(self, input_size=512, memory_size=256, temperature=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.memory_size = memory_size
        self.temperature = temperature
        
        # Learnable memory patterns
        self.memory = nn.Parameter(torch.randn(memory_size, input_size))
        nn.init.xavier_uniform_(self.memory)
        
        # Layer normalization
        self.ln = nn.LayerNorm(input_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_size) - Input features from SNN
        Returns:
            retrieved: (batch_size, input_size) - Retrieved patterns with residual
        """
        # Normalize features and memory
        x_norm = F.normalize(x, p=2, dim=1)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        
        # ✅ Better Hopfield: Scaled dot-product attention
        similarity = torch.matmul(x_norm, mem_norm.T) / np.sqrt(self.input_size)
        
        # ✅ Better Hopfield: Temperature scaling
        attention_weights = F.softmax(similarity / self.temperature, dim=1)
        
        # Retrieve from memory (weighted sum)
        retrieved = torch.matmul(attention_weights, mem_norm)
        
        # ✅ Skip Connection: Residual + LayerNorm
        retrieved = self.ln(retrieved + x)
        
        return retrieved


# Test Hopfield layer
print("\n🧪 Testing Modern Hopfield Layer:")
test_hopfield = ModernHopfieldLayer(
    input_size=512,
    memory_size=256,
    temperature=0.1
).to(device)

dummy_features = torch.randn(4, 512).to(device)
retrieved = test_hopfield(dummy_features)

print(f"   Input shape: {dummy_features.shape}")
print(f"   Output shape: {retrieved.shape}")
print(f"   Memory patterns: {test_hopfield.memory.shape}")
print(f"   Parameters: {sum(p.numel() for p in test_hopfield.parameters()):,}")

del test_hopfield, dummy_features
safe_empty_cache()

print("\n✅ Modern Hopfield Layer ready!")
print("="*80 + "\n")

# %% CELL 11
"""Cell 11: Improved HGRN Gate (Temporal Gating)"""

print("="*80)
print("🧠 IMPROVED HGRN GATE")
print("="*80)

class ImprovedHGRNGate(nn.Module):
    """
    Hierarchical Gated Recurrent Network Gate
    
    Improvements:
    ✅ LayerNorm: Stabilizes recurrent dynamics
    ✅ Xavier Initialization: Better gradient flow
    
    This provides temporal gating (like GRU) to decide what information
    to keep from previous timesteps and what to update.
    """
    
    def __init__(self, input_size=512, hidden_size=512):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_r.weight)
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_z.weight)
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_h.weight)
        
        # ✅ LayerNorm for stability
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h_prev):
        """
        Args:
            x: (batch_size, input_size) - Current input
            h_prev: (batch_size, hidden_size) - Previous hidden state
        Returns:
            h_new: (batch_size, hidden_size) - Updated hidden state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Reset gate (what to forget)
        r = torch.sigmoid(self.ln_r(self.W_r(combined)))
        
        # Update gate (how much to update)
        z = torch.sigmoid(self.ln_z(self.W_z(combined)))
        
        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.ln_h(self.W_h(combined_reset)))
        
        # New hidden state (weighted combination)
        h_new = (1 - z) * h_prev + z * h_tilde
        
        return h_new


# Test HGRN gate
print("\n🧪 Testing Improved HGRN Gate:")
test_hgrn = ImprovedHGRNGate(
    input_size=512,
    hidden_size=512
).to(device)

dummy_input = torch.randn(4, 512).to(device)
dummy_hidden = torch.randn(4, 512).to(device)
new_hidden = test_hgrn(dummy_input, dummy_hidden)

print(f"   Input shape: {dummy_input.shape}")
print(f"   Hidden shape: {dummy_hidden.shape}")
print(f"   Output shape: {new_hidden.shape}")
print(f"   Parameters: {sum(p.numel() for p in test_hgrn.parameters()):,}")

del test_hgrn, dummy_input, dummy_hidden
safe_empty_cache()

print("\n✅ Improved HGRN Gate ready!")
print("="*80 + "\n")

# %% CELL 12
"""Cell 9: SNN Backbone Architecture (Conv2D for N-MNIST)"""

print("="*80)
print("🧠 SNN BACKBONE ARCHITECTURE")
print("="*80)

# Define surrogate gradient
spike_grad = surrogate.atan()

class SNN_Backbone(nn.Module):
    """
    Main SNN backbone with Leaky Integrate-and-Fire neurons
    
    Improvements applied:
    ✅ Larger hidden layers (1024 -> 512)
    ✅ Returns both output spikes AND features for contrastive loss
    
    Architecture:
    - Conv2d(2, 32, 5) + MaxPool + LIF
    - Conv2d(32, 64, 5) + MaxPool + LIF
    - FC(1600, 1024) + LIF
    - FC(1024, 512) + LIF  ← Feature layer
    - FC(512, num_classes) + LIF ← Output layer
    """
    
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.flatten = nn.Flatten()
        
        # Calculate flattened size
        # For N-MNIST (34x34): after conv1(5)+pool(2): 15x15
        #                      after conv2(5)+pool(2): 5x5
        # So: 64 * 5 * 5 = 1600
        flat_size = 64 * 5 * 5
        
        # ✅ Larger hidden layers
        self.fc1 = nn.Linear(flat_size, 1024)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Feature layer (512-dim for memory/assemblies)
        self.fc_hidden = nn.Linear(1024, 512)
        self.lif_hidden = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, channels, height, width)
        Returns:
            spk_sum: (batch, num_classes) - Summed spikes for classification
            features: (batch, 512) - Mean features for contrastive loss
        """
        # Reset hidden states
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.lif_hidden.reset_hidden()
        self.lif_out.reset_hidden()
        
        # [Batch, Time, C, H, W] -> [Time, Batch, C, H, W]
        x = x.permute(1, 0, 2, 3, 4)
        
        spk_out_rec = []
        features_rec = []
        
        # Process each time step
        for step in range(x.shape[0]):
            # Conv + LIF layers
            spk1 = self.lif1(self.maxpool1(self.conv1(x[step])))
            spk2 = self.lif2(self.maxpool2(self.conv2(spk1)))
            
            # Flatten
            flat_spk2 = self.flatten(spk2)
            
            # FC layers
            spk3 = self.lif3(self.fc1(flat_spk2))
            
            # Feature layer (this is what we'll use for contrastive loss)
            spk_hidden = self.lif_hidden(self.fc_hidden(spk3))
            
            # Output layer
            spk_out, _ = self.lif_out(self.fc_out(spk_hidden))
            
            spk_out_rec.append(spk_out)
            features_rec.append(spk_hidden)
        
        # Sum spikes over time for classification
        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        
        # Average features over time for contrastive loss
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        
        return spk_sum, features


# Test the backbone
print("\n🧪 Testing SNN Backbone:")
test_backbone = SNN_Backbone(input_channels=2, num_classes=10).to(device)

# Create dummy input: (batch=4, time=25, channels=2, height=34, width=34)
dummy_input = torch.randn(4, 25, 2, 34, 34).to(device)
spk_sum, features = test_backbone(dummy_input)

print(f"   Input shape: {dummy_input.shape}")
print(f"   Output spikes shape: {spk_sum.shape}")
print(f"   Features shape: {features.shape}")
print(f"   Parameters: {sum(p.numel() for p in test_backbone.parameters()):,}")

del test_backbone, dummy_input
safe_empty_cache()

print("\n✅ SNN Backbone ready!")
print("="*80 + "\n")

# %% CELL 13

print("="*80)
print("🌐 COMPREHENSIVE CROSS-MODAL ANALYSIS")
print("="*80)
print("\n📌 KEY CONTRIBUTION: Testing architecture generalization across modalities")
print("   • Visual: N-MNIST (Conv2D Backbone)")
print("   • Auditory: SHD (Linear Backbone)")
print("   • Goal: Prove that HYBRID COMPONENTS generalize across modalities")
print("   • Same components (Hopfield, HGRN, SCL), different input heads\n")

# ============================================================
# PART 1: Define Native SHD Models (1D Architecture)
# ============================================================

print("\n" + "="*70)
print("PART 1: DEFINING NATIVE 1D MODELS FOR SHD")
print("="*70)
print("\n🔨 Creating architecturally-principled 1D models...")
print("   (Linear layers for audio, NOT forced into 2D Conv)")

class SNN_Backbone_SHD(nn.Module):
    """
    Native SNN backbone for SHD (1D audio input)
    Uses Linear (FC) layers instead of Conv2D
    """
    def __init__(self, input_size=700, num_classes=20):
        super().__init__()
        
        self.fc_input = nn.Linear(input_size, 1024)
        self.lif_input = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Larger Hidden Layers: 1024 -> 512
        self.fc1 = nn.Linear(1024, 1024)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.fc_hidden = nn.Linear(1024, 512)  # 512 feature/memory size
        self.lif_hidden = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        # x: (batch, time, 1, 1, 700) -> squeeze to (batch, time, 700)
        if len(x.shape) == 5:
            x = x.squeeze(2).squeeze(2)  # Remove spatial dimensions
        
        # Reset hidden states
        self.lif_input.reset_hidden()
        self.lif3.reset_hidden()
        self.lif_hidden.reset_hidden()
        self.lif_out.reset_hidden()

        # [Batch, Time, Features] -> [Time, Batch, Features]
        x = x.permute(1, 0, 2)
        
        spk_out_rec = []
        features_rec = []

        for step in range(x.shape[0]):
            spk_in = self.lif_input(self.fc_input(x[step]))
            spk3 = self.lif3(self.fc1(spk_in))
            spk_hidden = self.lif_hidden(self.fc_hidden(spk3))
            spk_out, _ = self.lif_out(self.fc_out(spk_hidden))
            
            spk_out_rec.append(spk_out)
            features_rec.append(spk_hidden)

        # Sum spikes over time for classification
        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        # Average features over time for contrastive loss
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        
        return spk_sum, features


# Define all 5 model variants for SHD

class Model_1_SHD(SNN_Backbone_SHD):
    """TRUE Baseline - No SCL"""
    def __init__(self, input_size=700, num_classes=20):
        super().__init__(input_size, num_classes)
        self.name = "Baseline_NoSCL_SHD"


class Model_2_SHD(SNN_Backbone_SHD):
    """Baseline with SCL"""
    def __init__(self, input_size=700, num_classes=20):
        super().__init__(input_size, num_classes)
        self.name = "Baseline_SCL_SHD"


class Model_3_SHD(nn.Module):
    """SNN + Hopfield"""
    def __init__(self, input_size=700, num_classes=20):
        super().__init__()
        self.backbone = SNN_Backbone_SHD(input_size, num_classes)
        self.hopfield = ModernHopfieldLayer(input_size=512, memory_size=256, temperature=0.1)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        self.name = "SNN_Hopfield_SHD"
    
    def forward(self, x):
        _, features = self.backbone(x)
        retrieved_features = self.hopfield(features)
        spk_out, _ = self.lif_out(self.fc_out(retrieved_features))
        return spk_out, retrieved_features


class Model_4_SHD(nn.Module):
    """SNN + HGRN (Expected BEST)"""
    def __init__(self, input_size=700, num_classes=20):
        super().__init__()
        self.backbone = SNN_Backbone_SHD(input_size, num_classes)
        self.hgrn = ImprovedHGRNGate(input_size=512, hidden_size=512)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        self.name = "SNN_HGRN_SHD"

    def forward(self, x):
        _, features = self.backbone(x)
        batch_size = features.shape[0]
        h = torch.zeros(batch_size, 512).to(features.device)
        h = self.hgrn(features, h)
        spk_out, _ = self.lif_out(self.fc_out(h))
        return spk_out, h


class Model_5_SHD(nn.Module):
    """Full Hybrid (Hopfield + HGRN)"""
    def __init__(self, input_size=700, num_classes=20):
        super().__init__()
        self.backbone = SNN_Backbone_SHD(input_size, num_classes)
        self.hopfield = ModernHopfieldLayer(input_size=512, memory_size=256, temperature=0.1)
        self.hgrn = ImprovedHGRNGate(input_size=512, hidden_size=512)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        self.name = "Full_Hybrid_SHD"
        
    def forward(self, x):
        _, features = self.backbone(x)
        retrieved = self.hopfield(features)
        batch_size = retrieved.shape[0]
        h = torch.zeros(batch_size, 512).to(retrieved.device)
        h = self.hgrn(retrieved, h)
        spk_out, _ = self.lif_out(self.fc_out(h))
        return spk_out, h


print("✅ All 5 native SHD models (1D Linear backbone) defined")
print("\n📊 Model Parameter Counts:")

for ModelClass in [Model_1_SHD, Model_2_SHD, Model_3_SHD, Model_4_SHD, Model_5_SHD]:
    test_model = ModelClass(num_classes=shd_info['num_classes']).to(device)
    params = sum(p.numel() for p in test_model.parameters())
    print(f"   {test_model.name:25s}: {params:>10,} parameters")
    del test_model
    safe_empty_cache()

print("\n✅ SHD model classes ready")

# %% CELL 14
"""Cell 12: Complete Model Definitions (5 Models)"""

print("="*80)
print("🏗️  COMPLETE MODEL ARCHITECTURES")
print("="*80)

class Model_1_Baseline_SNN(nn.Module):
    """
    Model 1: TRUE BASELINE (NO Contrastive Loss)
    
    Just the SNN backbone trained with CrossEntropy only.
    This proves the baseline performance without any improvements.
    """
    
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        self.backbone = SNN_Backbone(input_channels, num_classes)
        self.name = "Baseline_SNN_NoSCL"
    
    def forward(self, x):
        return self.backbone(x)


class Model_2_Baseline_SNN_SCL(nn.Module):
    """
    Model 2: Baseline with Supervised Contrastive Loss
    
    Same SNN backbone, but trained with SCL to form engrams.
    This shows the improvement from contrastive loss alone.
    """
    
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        self.backbone = SNN_Backbone(input_channels, num_classes)
        self.name = "Baseline_SNN_SCL"
    
    def forward(self, x):
        return self.backbone(x)


class Model_3_SNN_Hopfield(nn.Module):
    """
    Model 3: SNN + Modern Hopfield (with SCL)
    
    Adds associative memory after the feature layer.
    Tests if Hopfield memory helps leverage the engrams.
    """
    
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        
        # Get backbone features
        self.backbone = SNN_Backbone(input_channels, num_classes)
        
        # Add Hopfield layer
        self.hopfield = ModernHopfieldLayer(
            input_size=512,
            memory_size=256,
            temperature=0.1
        )
        
        # New output layer after Hopfield
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.name = "SNN_Hopfield"
    
    def forward(self, x):
        # Get SNN features
        spk_sum, features = self.backbone(x)
        
        # Apply Hopfield memory
        retrieved_features = self.hopfield(features)
        
        # New classification from retrieved features
        spk_out, _ = self.lif_out(self.fc_out(retrieved_features))
        
        return spk_out, retrieved_features


class Model_4_SNN_HGRN(nn.Module):
    """
    Model 4: SNN + HGRN Gate (with SCL)
    
    Adds temporal gating after the feature layer.
    Tests if recurrent gating helps leverage the engrams.
    """
    
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        
        # Get backbone
        self.backbone = SNN_Backbone(input_channels, num_classes)
        
        # Add HGRN gate
        self.hgrn = ImprovedHGRNGate(
            input_size=512,
            hidden_size=512
        )
        
        # New output layer after HGRN
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.name = "SNN_HGRN"
    
    def forward(self, x):
        # Get SNN features
        spk_sum, features = self.backbone(x)
        
        # Initialize hidden state (batch_size, 512)
        batch_size = features.shape[0]
        h = torch.zeros(batch_size, 512).to(features.device)
        
        # Apply HGRN gate
        h = self.hgrn(features, h)
        
        # New classification from gated features
        spk_out, _ = self.lif_out(self.fc_out(h))
        
        return spk_out, h


class Model_5_Full_Hybrid(nn.Module):
    """
    Model 5: SNN + Hopfield + HGRN (with SCL)
    
    Combines both Hopfield memory AND HGRN gating.
    Tests if both components together provide maximum benefit.
    """
    
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        
        # Backbone
        self.backbone = SNN_Backbone(input_channels, num_classes)
        
        # Hopfield layer
        self.hopfield = ModernHopfieldLayer(
            input_size=512,
            memory_size=256,
            temperature=0.1
        )
        
        # HGRN gate
        self.hgrn = ImprovedHGRNGate(
            input_size=512,
            hidden_size=512
        )
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.name = "Full_Hybrid"
    
    def forward(self, x):
        # Get SNN features
        spk_sum, features = self.backbone(x)
        
        # Apply Hopfield
        retrieved = self.hopfield(features)
        
        # Apply HGRN
        batch_size = retrieved.shape[0]
        h = torch.zeros(batch_size, 512).to(retrieved.device)
        h = self.hgrn(retrieved, h)
        
        # Classification
        spk_out, _ = self.lif_out(self.fc_out(h))
        
        return spk_out, h


print("\n✅ All 5 models defined:")
print("   1. Baseline_SNN_NoSCL  - TRUE baseline (CrossEntropy only)")
print("   2. Baseline_SNN_SCL    - With contrastive loss")
print("   3. SNN_Hopfield        - + Hopfield memory")
print("   4. SNN_HGRN            - + HGRN gating (Expected BEST)")
print("   5. Full_Hybrid         - + Both components")

print("\n🧪 Testing all models:")
for ModelClass in [Model_1_Baseline_SNN, Model_2_Baseline_SNN_SCL, 
                   Model_3_SNN_Hopfield, Model_4_SNN_HGRN, Model_5_Full_Hybrid]:
    model = ModelClass(input_channels=2, num_classes=10).to(device)
    dummy = torch.randn(2, 25, 2, 34, 34).to(device)
    out, feat = model(dummy)
    params = sum(p.numel() for p in model.parameters())
    print(f"   {model.name:25s}: output {out.shape}, features {feat.shape}, params {params:,}")
    del model, dummy
    safe_empty_cache()

print("\n✅ All models ready for training!")
print("="*80 + "\n")

# %% CELL 15
"""
Cell 13: Main Training Pipeline (Corrected)
--------------------------------------------
This cell provides the COMPLETE, CORRECTED training functions.

"""
from typing import Dict, Any
# ============================================================================
# 1. TrainingTracker (Handles Early Stopping)
# ============================================================================
class TrainingTracker:
    """Logs metrics and handles model checkpointing with early stopping."""
    def __init__(self, model_name: str, model: nn.Module, checkpoint_dir: Path, patience: int = 7):
        self.model_name = model_name
        self.model = model
        # Use the provided checkpoint directory
        self.model_path = checkpoint_dir / f"{model_name}_best.pth"
        self.patience = patience
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_ce_loss': [], 'train_scl_loss': [],
            'val_loss': [], 'val_acc': [], 'val_ce_loss': [], 'val_scl_loss': [],
            'best_val_acc': 0.0, 'epochs_trained': 0, 'lr': []
        }
        print(f"Tracking training for: {model_name} (Patience: {patience})")
        print(f"   Saving best model to: {self.model_path}")

    def log_epoch(self, epoch: int, metrics: dict) -> bool:
        """
        Logs metrics. Returns True if training should continue, False if early stopping.
        """
        self.history['train_loss'].append(metrics['train_loss'])
        self.history['train_ce_loss'].append(metrics['train_ce_loss'])
        self.history['train_scl_loss'].append(metrics['train_scl_loss'])
        self.history['train_acc'].append(metrics['train_acc'])
        self.history['val_loss'].append(metrics['val_loss'])
        self.history['val_ce_loss'].append(metrics['val_ce_loss'])
        self.history['val_scl_loss'].append(metrics['val_scl_loss'])
        self.history['val_acc'].append(metrics['val_acc'])
        self.history['lr'].append(metrics['lr'])
        self.history['epochs_trained'] = epoch + 1

        if metrics['val_acc'] > self.best_val_acc:
            self.best_val_acc = metrics['val_acc']
            self.history['best_val_acc'] = self.best_val_acc
            self.epochs_no_improve = 0
            self.save_checkpoint(epoch)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"  🛑 Early stopping triggered after {self.patience} epochs with no improvement.")
            return False # Stop training
        return True # Continue training

    def save_checkpoint(self, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, self.model_path)
        print(f"  ✨ New best model saved! (Epoch {epoch}, Val Acc: {self.best_val_acc:.2f}%)")

    def get_history(self):
        return self.history

# ============================================================================
# 2. train_step Function
# ============================================================================
def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    con_loss_fn: nn.Module, # Can be None
    contrastive_weight: float,
    device: torch.device
) -> tuple[float, float, float, float]:
    
    total_loss, total_ce_loss, total_con_loss, total_correct, total_samples = 0, 0, 0, 0, 0
    model.train()
    
    pbar = tqdm(data_loader, desc="[Train]", leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        
        # ✅ CORRECT: Expects (spk_sum, features) from new models
        spk_sum, features = model(data)
        
        loss_ce = ce_loss_fn(spk_sum, labels)
        
        # ✅ FIX: Only calculate con_loss if weight > 0 and fn exists
        if contrastive_weight > 0 and con_loss_fn is not None:
            loss_con = con_loss_fn(features, labels) 
            loss = loss_ce + contrastive_weight * loss_con
        else:
            loss = loss_ce # CE-Only training
            loss_con = torch.tensor(0.0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['gradient_clip'])
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_ce_loss += loss_ce.item() * data.size(0)
        total_con_loss += loss_con.item() * data.size(0)
        total_correct += (spk_sum.argmax(1) == labels).sum().item()
        total_samples += data.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*total_correct/total_samples:.2f}%"
        })
        
    avg_loss = total_loss / (total_samples + 1e-9)
    avg_ce_loss = total_ce_loss / (total_samples + 1e-9)
    avg_con_loss = total_con_loss / (total_samples + 1e-9)
    avg_acc = (total_correct / (total_samples + 1e-9)) * 100
    
    return avg_loss, avg_ce_loss, avg_con_loss, avg_acc

# ============================================================================
# 3. test_step Function
# ============================================================================
def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    ce_loss_fn: nn.Module,
    con_loss_fn: nn.Module, # Can be None
    contrastive_weight: float,
    device: torch.device
) -> tuple[float, float, float, float]:
    
    total_loss, total_ce_loss, total_con_loss, total_correct, total_samples = 0, 0, 0, 0, 0
    model.eval()
    
    pbar = tqdm(data_loader, desc="[Val]", leave=False)
    with torch.no_grad():
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            # ✅ CORRECT: Expects (spk_sum, features)
            spk_sum, features = model(data)
            
            loss_ce = ce_loss_fn(spk_sum, labels)
            
            # ✅ FIX: Calculate all losses for validation
            if contrastive_weight > 0 and con_loss_fn is not None:
                loss_con = con_loss_fn(features, labels)
                loss = loss_ce + contrastive_weight * loss_con
            else:
                loss = loss_ce
                loss_con = torch.tensor(0.0)
                
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

# ============================================================================
# 4. Main `train_model` Function (Fixes Bug)
# ============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    dataset_name: str,
    use_contrastive: bool, # This is from your script
    num_epochs: int,
    patience: int,
    device: torch.device
) -> tuple[nn.Module, Dict[str, Any], float]:
    
    # Get LR and SCL weight from global CONFIG
    lr = CONFIG['learning_rate']
    scl_weight = CONFIG['contrastive_weight'] if use_contrastive else 0.0

    print("\n" + "="*70)
    print(f"Training: {model_name} on {dataset_name}")
    print("="*70)
    print(f"Contrastive Loss: {'✅ Enabled' if use_contrastive else '❌ Disabled (TRUE Baseline)'}")
    print(f"Max Epochs: {num_epochs} | Patience: {patience}")
    print(f"Learning Rate: {lr} | Batch Size: {train_loader.batch_size}")
    print("="*70)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    con_loss_fn = None
    if use_contrastive:
        con_loss_fn = SupervisedContrastiveLoss(temperature=CONFIG['contrastive_temperature']).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Use CHECKPOINTS_DIR from your setup cell
    tracker = TrainingTracker(
        model_name, 
        model,
        CHECKPOINTS_DIR, 
        patience
    )
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {current_lr:.1e}")
        
        # --- Run Train Step ---
        train_loss, train_ce, train_scl, train_acc = train_step(
            model, train_loader, optimizer, ce_loss_fn, con_loss_fn, scl_weight, device
        )
        
        # --- Run Validation Step ---
        val_loss, val_ce, val_scl, val_acc = test_step(
            model, test_loader, ce_loss_fn, con_loss_fn, scl_weight, device
        )
        scheduler.step()
        
        # --- Print Epoch Summary ---
        print(f"  [Train] Loss: {train_loss:.4f} (CE: {train_ce:.4f}, SCL: {train_scl:.4f}) | Acc: {train_acc:.2f}%")
        print(f"  [Val]   Loss: {val_loss:.4f} (CE: {val_ce:.4f}, SCL: {val_scl:.4f}) | Acc: {val_acc:.2f}%")

        # --- Check for Early Stopping ---
        if not tracker.log_epoch(epoch, {
            'train_loss': train_loss, 'train_ce_loss': train_ce, 'train_scl_loss': train_scl, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_ce_loss': val_ce, 'val_scl_loss': val_scl, 'val_acc': val_acc,
            'lr': current_lr
        }):
            break # Early stopping triggered
            
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    print("\n" + "="*70)
    print(f"🏁 TRAINING FINISHED: {model_name}")
    print(f"Total time: {total_time:.2f} minutes")
    print(f"Best Validation Accuracy: {tracker.best_val_acc:.2f}%")
    print("="*70 + "\n")
    
    # Return the model, its full history, and the best accuracy
    return model, tracker.history, tracker.best_val_acc

print("✅ Training Pipeline (Cell 13) is defined and fixes all bugs.")

# %% CELL 16
print("=" * 80)
print("🏗️  M7: UNIFIED CROSS-MODAL ADAPTIVE MODEL")
print("=" * 80)

import gc
if 'M7_ContinualAdaptiveModel' in dir():
    del M7_ContinualAdaptiveModel
gc.collect()
safe_empty_cache()


class M7_ContinualAdaptiveModel(nn.Module):
    """
    M7 uses proven backbones from Paper 2.
    The backbone forward() returns (spk_sum [B, num_classes], features [B, 512]).
    M7 replaces the backbone's final classification with:
      adaptive gate → weighted Hopfield/HGRN → task head → logits [B, num_classes]
    """
    FEAT = 512

    def __init__(self, tasks: dict):
        super().__init__()
        self.tasks = tasks

        # ── Proven backbones ──────────────────────────────────────────────
        self.backbone_nmnist = SNN_Backbone(
            input_channels=2,
            num_classes=tasks['nmnist']['num_classes']
        )
        self.backbone_shd = SNN_Backbone_SHD(
            input_size=700,
            num_classes=tasks['shd']['num_classes']
        )
        self.backbone_dvs = SNN_Backbone(
            input_channels=2,
            num_classes=tasks['dvs']['num_classes']
        )

        # ── Shared memory mechanisms ──────────────────────────────────────
        self.hopfield = ModernHopfieldLayer(
            input_size=self.FEAT, memory_size=256, temperature=0.1)
        self.hgrn = ImprovedHGRNGate(
            input_size=self.FEAT, hidden_size=self.FEAT)

        # ── Adaptive gate: [B, 512] → [B, 2] ─────────────────────────────
        self.adaptive_gate = nn.Sequential(
            nn.Linear(self.FEAT, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

        # ── Per-task classification heads ─────────────────────────────────
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.FEAT),
                nn.Linear(self.FEAT, cfg['num_classes'])
            )
            for name, cfg in tasks.items()
        })

        # ── Replay heads (embedding → logits, no backbone) ───────────────
        self.replay_heads = nn.ModuleDict({
            name: nn.Linear(self.FEAT, cfg['num_classes'])
            for name, cfg in tasks.items()
        })

    @torch.no_grad()
    def _get_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Returns [B, 1, 512] for EmbeddingReplayBuffer."""
        if modality == 'nmnist':
            _, feats = self.backbone_nmnist(x)
        elif modality == 'shd':
            _, feats = self.backbone_shd(x)
        elif modality == 'dvs':
            _, feats = self.backbone_dvs(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return feats.unsqueeze(1)  # [B, 1, 512]

    def forward(self, x: torch.Tensor, modality: str, task_name: str = None):
        """
        Returns (logits [B, num_classes], gate [B, 2], features [B, 512])
        """
        if task_name is None:
            task_name = modality

        # Step 1: backbone → features [B, 512]
        if modality == 'nmnist':
            _, features = self.backbone_nmnist(x)
        elif modality == 'shd':
            _, features = self.backbone_shd(x)
        elif modality == 'dvs':
            _, features = self.backbone_dvs(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Step 2: adaptive gate
        gate = self.adaptive_gate(features)         # [B, 2]
        w_h  = gate[:, 0].unsqueeze(1)              # [B, 1]
        w_g  = gate[:, 1].unsqueeze(1)              # [B, 1]

        # Step 3: memory mechanisms
        hop_out = self.hopfield(features)            # [B, 512]
        h_state = self.hgrn(features,
                            torch.zeros_like(features))  # [B, 512]

        # Step 4: weighted combination
        combined = w_h * hop_out + w_g * h_state    # [B, 512]

        # Step 5: task head → logits
        logits = self.heads[task_name](combined)     # [B, num_classes]

        return logits, gate, features

    def forward_from_embeddings(self, embs: torch.Tensor,
                                task_name: str) -> torch.Tensor:
        """Replay path: [B, 512] → logits [B, num_classes]"""
        return self.replay_heads[task_name](embs)


# ── Smoke test ────────────────────────────────────────────────────────────────
TASKS = {
    'nmnist': {'num_classes': 10, 'type': 'visual'},
    'shd':    {'num_classes': 20, 'type': 'audio'},
    'dvs':    {'num_classes': 11, 'type': 'visual_motion'},
}
_m = M7_ContinualAdaptiveModel(tasks=TASKS).to(device)

_x = torch.randn(4, 25, 2, 34, 34).to(device)
_o, _g, _f = _m(_x, 'nmnist')
assert _o.shape == (4, 10), f"Expected (4,10) got {_o.shape}"
print(f"✅ M7 N-MNIST: logits={tuple(_o.shape)}, gate={tuple(_g.shape)}, feat={tuple(_f.shape)}")

_x = torch.randn(4, 100, 1, 1, 700).to(device)
_o, _g, _f = _m(_x, 'shd')
assert _o.shape == (4, 20), f"Expected (4,20) got {_o.shape}"
print(f"✅ M7 SHD:     logits={tuple(_o.shape)}, gate={tuple(_g.shape)}, feat={tuple(_f.shape)}")

_x = torch.randn(4, 25, 2, 34, 34).to(device)
_o, _g, _f = _m(_x, 'dvs')
assert _o.shape == (4, 11), f"Expected (4,11) got {_o.shape}"
print(f"✅ M7 DVS:     logits={tuple(_o.shape)}, gate={tuple(_g.shape)}, feat={tuple(_f.shape)}")

print(f"✅ M7 params:  {sum(p.numel() for p in _m.parameters()):,}")
del _m, _x, _o, _g, _f
safe_empty_cache()
print("✅ M7_ContinualAdaptiveModel ready")

# %% CELL 17
print("=" * 80)
print("🧠 CONTINUAL LEARNING: EMBEDDING REPLAY BUFFER")
print("=" * 80)


class EmbeddingReplayBuffer:
    """
    Stores SCL embeddings per class in a fixed-size reservoir.
    Operates entirely in embedding space — no raw spikes stored.

    Biological analogy: hippocampal indexing theory — compressed
    episodic traces that can reinstate cortical representations
    without replaying full sensory input.

    Buffer size: num_classes × samples_per_class × feature_dim
    For N-MNIST (10 cls, 50 samples, 512-d) = 256 KB — negligible.
    """

    def __init__(self, feature_dim: int = 512,
                 samples_per_class: int = 50,
                 device=None):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.feature_dim       = feature_dim
        self.samples_per_class = samples_per_class
        self.device            = device
        self._store: dict      = {}
        self._task_labels: dict = {}  # class_id → task_id

    @torch.no_grad()
    def populate(self, model, loader, # Fixed: changed data_loader to loader
                 modality: str, task_id: int,
                 label_offset: int = 0) -> None:
        """
        Extract embeddings from loader using model._get_features,
        then reservoir-sample up to samples_per_class per class.

        label_offset: add to labels so T1 and T2 classes don't collide
        """
        model.eval()
        class_embs = defaultdict(list)

        for data, labels in loader:
            data   = data.to(self.device)
            labels = labels.to(self.device)
            feats  = model._get_features(data, modality)  # Expects [B, T, 512] or [B, 512]
            
            # Ensure we have a flat [B, 512] representation
            if feats.dim() == 3:
                feats = feats.mean(dim=1)  # average over time
            
            for feat, lbl in zip(feats, labels):
                cls = int(lbl.item()) + label_offset
                class_embs[cls].append(feat.cpu())

        for cls, emb_list in class_embs.items():
            all_embs = torch.stack(emb_list)
            n        = all_embs.size(0)
            idx      = torch.randperm(n)[:self.samples_per_class]
            self._store[cls]       = all_embs[idx]
            self._task_labels[cls] = task_id

        total = sum(v.size(0) for v in self._store.values())
        print(f"  ✅ Buffer populated | task={task_id} | modality={modality} | "
              f"classes={len(class_embs)} | total_samples={total}")

    def sample(self, batch_size: int, task_ids=None):
        """
        Sample a balanced batch from the buffer.
        Returns (embeddings [B, 512], labels [B]) on self.device.
        """
        if task_ids is not None:
            valid = {cls: emb for cls, emb in self._store.items()
                     if self._task_labels[cls] in task_ids}
        else:
            valid = self._store

        if not valid:
            return None, None

        per_class = max(1, batch_size // len(valid))
        emb_list, lbl_list = [], []

        for cls, embs in valid.items():
            n   = embs.size(0)
            idx = torch.randint(0, n, (per_class,))
            emb_list.append(embs[idx])
            lbl_list.append(torch.full((per_class,), cls, dtype=torch.long))

        embs   = torch.cat(emb_list, dim=0)[:batch_size].to(self.device)
        labels = torch.cat(lbl_list, dim=0)[:batch_size].to(self.device)
        return embs, labels

    def stats(self) -> dict:
        by_task = defaultdict(int)
        for cls, embs in self._store.items():
            by_task[self._task_labels[cls]] += embs.size(0)
        return {
            'total':       sum(v.size(0) for v in self._store.values()), # Fixed key name
            'num_classes': len(self._store),
            'by_task':     dict(by_task),
        }


_buf = EmbeddingReplayBuffer(feature_dim=512, samples_per_class=50)
print("✅ EmbeddingReplayBuffer defined")

# %% CELL 18
print("=" * 80)
print("🔄 CONTINUAL LEARNER")
print("=" * 80)


class ContinualLearner:
    """
    Sequential task trainer with embedding replay.

    Accuracy matrix R[step][task_name]:
      step=1: after T1 (N-MNIST) only
      step=2: after T2 (SHD) training

    CL Metrics:
      Forgetting F   = R[1][nmnist] − R[2][nmnist]   (lower is better)
      Fwd Transfer   = R[2][shd]   − R[1][shd]       (higher is better)

    Uses same AdamW + CosineAnnealingLR + TrainingTracker pattern as Paper 2,
    but with embedding replay loss added on top of main task CE.
    """

    def __init__(self, model, buffer, device=None,
                 replay_batch=32, replay_weight=0.3):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model         = model.to(device)
        self.buffer        = buffer
        self.device        = device
        self.replay_batch  = replay_batch
        self.replay_weight = replay_weight
        self.acc_matrix    = {}
        self.history       = defaultdict(list)

    @torch.no_grad()
    def evaluate(self, loader, modality, task_name) -> float:
        self.model.eval()
        correct = total = 0
        for data, labels in loader:
            data, labels = data.to(self.device), labels.to(self.device)
            spk, _, _ = self.model(data, modality, task_name)
            correct  += (spk.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        return 100. * correct / total if total > 0 else 0.0

    def train_task(self, task_name, modality,
                   train_loader, val_loader, test_loader,
                   num_epochs=50, patience=12, lr=1e-3,
                   use_replay=False, replay_task_ids=None,
                   use_scl=True, scl_weight=0.1,
                   step_id=0) -> dict:

        print(f"\n{'='*70}")
        print(f"  Task: {task_name.upper()} | Modality: {modality} "
              f"| Replay: {'✅' if use_replay else '❌'} "
              f"| SCL: {'✅' if use_scl else '❌'}")
        print(f"  lr={lr} | epochs={num_epochs} | patience={patience}")
        print(f"{'='*70}")

        optimizer  = torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=CONFIG['weight_decay'])
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6)
        ce_loss    = nn.CrossEntropyLoss()
        scl_loss_fn = SupervisedContrastiveLoss(
            temperature=CONFIG['contrastive_temperature']).to(self.device)

        best_val     = 0.0
        best_state   = None
        patience_ctr = 0
        ckpt = CHECKPOINTS_DIR / f'M7_{task_name}_step{step_id}.pth'

        task_map = {i+1: k for i, k in enumerate(self.model.tasks.keys())}

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = correct = total = 0
            cur_lr = optimizer.param_groups[0]['lr']

            pbar = tqdm(train_loader,
                        desc=f'  Ep {epoch+1:3d}/{num_epochs} lr={cur_lr:.1e}',
                        ncols=95, leave=False)

            for data, labels in pbar:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                spk, gate, feats = self.model(data, modality, task_name)
                loss = ce_loss(spk, labels)

                # SCL on backbone features
                if use_scl:
                    loss = loss + scl_weight * scl_loss_fn(feats, labels)

                # Gate entropy regularization
                gate_ent = -(gate * torch.log(gate + 1e-10)).sum(dim=1).mean()
                loss     = loss + 0.01 * gate_ent

                # Embedding replay
                if use_replay and self.buffer.stats()['total'] > 0:
                    embs, r_lbl = self.buffer.sample(
                        self.replay_batch, task_ids=replay_task_ids)
                    if embs is not None:
                        rtask    = task_map.get(
                            replay_task_ids[0] if replay_task_ids else 1,
                            list(self.model.tasks.keys())[0])
                        r_logits = self.model.forward_from_embeddings(embs, rtask)
                        n_cls    = self.model.tasks[rtask]['num_classes']
                        loss     = loss + self.replay_weight * ce_loss(
                                       r_logits, r_lbl % n_cls)

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), CONFIG['gradient_clip'])
                optimizer.step()

                correct    += (spk.argmax(1) == labels).sum().item()
                total      += labels.size(0)
                total_loss += loss.item()
                pbar.set_postfix(acc=f'{100*correct/total:.1f}%')

            scheduler.step()
            train_acc = 100. * correct / total
            val_acc   = self.evaluate(val_loader, modality, task_name)

            self.history[f'{task_name}_train'].append(train_acc)
            self.history[f'{task_name}_val'].append(val_acc)

            # Gate stats every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.model.eval()
                g_list = []
                with torch.no_grad():
                    for d, _ in val_loader:
                        _, g, _ = self.model(d.to(self.device), modality, task_name)
                        g_list.append(g.cpu())
                        if len(g_list) > 8: break
                g_mean = torch.cat(g_list).mean(0)
                print(f"  Ep {epoch+1:3d}: Train={train_acc:.2f}% | "
                      f"Val={val_acc:.2f}% | "
                      f"Gate[hop={g_mean[0]:.3f} hgrn={g_mean[1]:.3f}]")

            if val_acc > best_val:
                best_val     = val_acc
                patience_ctr = 0
                best_state   = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}
                torch.save({'model_state_dict': best_state,
                            'best_val_acc': best_val}, ckpt)
                print(f"  ✨ New best! Val={val_acc:.2f}%")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  🛑 Early stop at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        test_acc = self.evaluate(test_loader, modality, task_name)
        print(f"\n  ✅ {task_name.upper()} | Test={test_acc:.2f}% | "
              f"BestVal={best_val:.2f}%")
        return {'task': task_name, 'step': step_id,
                'test_acc': test_acc, 'best_val': best_val}

    def compute_metrics(self) -> dict:
        tasks = list(self.model.tasks.keys())
        T1, T2 = tasks[0], tasks[1]
        m = {}
        if 1 in self.acc_matrix and 2 in self.acc_matrix:
            R11 = self.acc_matrix[1].get(T1, 0)
            R21 = self.acc_matrix[2].get(T1, 0)
            R12 = self.acc_matrix[1].get(T2, 0)
            R22 = self.acc_matrix[2].get(T2, 0)
            m['forgetting']       = R11 - R21
            m['forward_transfer'] = R22 - R12
            m['R_T1_after_T1']    = R11
            m['R_T1_after_T2']    = R21
            m['R_T2_after_T2']    = R22
        return m


print("✅ ContinualLearner defined")


# %% CELL 8
"""Cell 8.5c: DVS-Gesture Single-Task Smoke Test"""

print("="*80)
print("🧪 DVS-GESTURE SINGLE-TASK SMOKE TEST")
print("="*80)

torch.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Build a temporary M7 model with DVS task
TASKS_SMOKE = {
    'nmnist': {'num_classes': 10, 'type': 'visual'},
    'shd':    {'num_classes': 20, 'type': 'audio'},
    'dvs':    {'num_classes': 11, 'type': 'visual_motion'},
}

model_smoke = M7_ContinualAdaptiveModel(tasks=TASKS_SMOKE).to(device)
buffer_smoke = EmbeddingReplayBuffer(
    feature_dim=512, samples_per_class=50, device=str(device))
trainer = ContinualLearner(
    model=model_smoke, buffer=buffer_smoke, device=str(device),
    replay_batch=32, replay_weight=0.3)

result_smoke = trainer.train_task(
    'dvs', 'dvs',
    train_loader_dvs_cl, val_loader_dvs, test_loader_dvs,
    num_epochs=30, patience=12, lr=1e-3,
    use_replay=False, use_scl=True, scl_weight=0.1,
    step_id=0)

dvs_test_acc = result_smoke['test_acc']
dvs_best_val = result_smoke['best_val']

print("\n" + "="*80)
print("📊 DVS SMOKE TEST RESULTS")
print("="*80)
print(f"   DVS Test Accuracy:  {dvs_test_acc:.2f}%")
print(f"   DVS Best Val:       {dvs_best_val:.2f}%")
print(f"   Classes:            11 (chance = {100/11:.1f}%)")

# Shape sanity checks
xb, yb = next(iter(test_loader_dvs))
xb = xb.to(device)
logits, gate, features = model_smoke(xb, 'dvs', 'dvs')
print(f"\n🧪 Tensor shape checks:")
print(f"   Input:    {xb.shape}   (expect [B,25,2,34,34])")
print(f"   Logits:   {logits.shape}   (expect [B,11])")
print(f"   Gate:     {gate.shape}     (expect [B,2])")
print(f"   Features: {features.shape} (expect [B,512])")

# Validate correctness
assert logits.shape == (xb.shape[0], 11), f"Logits shape mismatch: {logits.shape}"
assert gate.shape == (xb.shape[0], 2), f"Gate shape mismatch: {gate.shape}"
assert features.shape == (xb.shape[0], 512), f"Features shape mismatch: {features.shape}"

# Flag if below chance
CHANCE_DVS = 100.0 / 11.0  # ~9.09%
if dvs_test_acc <= CHANCE_DVS + 1.0:
    print(f"\n🚨 ALERT: DVS accuracy {dvs_test_acc:.2f}% is at or below chance ({CHANCE_DVS:.1f}%)")
    print("   → FALL BACK TO SSC-35 (notify Blessing for loader)")
    raise RuntimeError("DVS single-task accuracy below chance — aborting. Use SSC-35 fallback.")
elif dvs_test_acc < 30.0:
    print(f"\n⚠️  WARNING: DVS accuracy {dvs_test_acc:.2f}% < 30% target")
    print("   → Non-trivial but sub-optimal; proceed with caution.")
else:
    print(f"\n✅ DVS smoke test PASSED: {dvs_test_acc:.2f}% >> chance ({CHANCE_DVS:.1f}%)")

# Cleanup
del model_smoke, buffer_smoke, trainer
safe_empty_cache()
print("="*80 + "\n")



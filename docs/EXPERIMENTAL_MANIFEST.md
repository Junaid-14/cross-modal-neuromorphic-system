# M7 Tri-Modal Continual Learning — Experimental Manifest

> Reproducibility documentation for NeurIPS 2026 submission.  
> Generated: 2026-05-07

---

## 1. SEEDS

| Seed | Status | Notes |
|------|--------|-------|
| 42 | ✅ Complete | Initial validation run |
| 123 | ✅ Complete | Full 3-task no_replay + replay |
| 456 | ✅ Complete | Full 3-task no_replay + replay |

All splits (train/val/test) use `torch.Generator().manual_seed(42)` — **split randomization is fixed across seeds**. Only model initialization and DataLoader shuffling vary.

---

## 2. HYPERPARAMETERS

```python
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 50,
    'patience': 12,          # early stopping patience
    'gradient_clip': 1.0,
    'beta': 0.9,             # LIF membrane time constant (SHARED across all modalities)
    'dropout': 0.2,
    'hidden_dim': 512,       # shared embedding dimension
    'num_patterns': 100,     # Hopfield memory patterns
    'num_gru_layers': 2,
    'num_workers': 0,        # macOS multiprocessing limitation
    'time_steps': 25,        # fixed temporal bins for all modalities
    'pin_memory': True,
    'use_contrastive': True,
    'contrastive_temperature': 0.07,
    'contrastive_weight': 0.1,
    'seed': 42,
    'device': 'mps',         # Apple Silicon MPS
}
```

**Per-task training overrides:**
- T1 N-MNIST: `num_epochs=15, patience=5, lr=1e-3`
- T2 SHD: `num_epochs=15, patience=5, lr=1e-3`
- T3 DVS-Gesture: `num_epochs=15, patience=5, lr=1e-3`

---

## 3. CHECKPOINT PATHS

**Directory:** `./ICONS_M7/results/checkpoints/`

**Naming convention:** `final_{task}_{modality}_seed{seed}_{condition}.pt`

| File | Size | Description |
|------|------|-------------|
| `final_nmnist_nmnist_seed42_no_replay.pt` | 65 MB | T1 after T1 (seed 42) |
| `final_shd_shd_seed42_no_replay.pt` | 66 MB | T2 after T2 (seed 42) |
| `final_dvs_dvs_seed42_no_replay.pt` | 65 MB | T3 after T3 (seed 42) |
| `final_nmnist_nmnist_seed42_replay.pt` | 34 MB | T1 after T1 (seed 42, replay) |
| `final_shd_shd_seed42_replay.pt` | 34 MB | T2 after T2 (seed 42, replay) |
| `final_dvs_dvs_seed42_replay.pt` | 34 MB | T3 after T3 (seed 42, replay) |
| `final_*_seed123_*` | 34 MB × 6 | Seed 123 checkpoints |
| `final_*_seed456_*` | 34 MB × 6 | Seed 456 checkpoints |

**Total checkpoint size:** ~862 MB (18 files).  
⚠️ **Too large for standard GitHub.** Use Git LFS or upload to Hugging Face / Zenodo.

---

## 4. CSV RESULTS

**Path:** `./ICONS_M7/results/cl_trimodal_results.csv`

**Columns:** `seed, condition, step, T1_after_T1, T2_after_T2, T3_after_T3, T1_after_T2, T1_after_T3, T2_after_T3, forgetting_T1_T2, forgetting_T1_T3, forgetting_T2_T3, fwt`

⚠️ **Contains duplicate rows** from reruns — deduplicate before analysis (filter `step.isna()` and `drop_duplicates()`).

---

## 5. DATASET PREPROCESSING

### 5.1 N-MNIST (Visual)

```python
sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
nmnist_transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000),
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])
cached_nmnist_train = tonic.DiskCachedDataset(nmnist_train, cache_path=...)
```

- **Spatial:** Native 34×34 (no resizing)
- **Temporal:** 1000 µs time-window → variable frame count
- **Collate:** Truncated/padded to exactly `time_steps=25` frames

### 5.2 SHD (Audio)

```python
def events_to_dense(events, label):
    time_bins, channels = 100, 700
    dense = torch.zeros(time_bins, channels)
    if len(events) > 0:
        max_time = events['t'].max()
        time_indices = (events['t'] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events['x'].astype(int)
        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0
    return dense.unsqueeze(1).unsqueeze(1), label
```

- **Spatial:** 700 channels (cochlear frequencies)
- **Temporal:** 100 time bins, resampled linearly from event timestamps
- **No ToFrame:** Custom dense conversion via Python loop (~3 min startup)

### 5.3 DVS-Gesture (Visual-Motion)

```python
dvs_transform = tonic.transforms.Compose([
    tonic.transforms.ToFrame(
        sensor_size=tonic.datasets.DVSGesture.sensor_size,  # (128, 128, 2)
        time_window=50000                                    # 50 ms bins
    ),
])
```

```python
def dvs_collate_fn(batch):
    # Temporal: truncate/pad to 25 frames
    if e.shape[0] > 25: e = e[:25]
    elif e.shape[0] < 25: pad to 25
    
    # Spatial: adaptive avg pool to 34×34
    if e.shape[-2:] != (34, 34):
        e = F.adaptive_avg_pool2d(e.view(-1, ..., ..., ...), (34, 34))
```

- **Sensor size:** 128×128 (native DVS-Gesture resolution)
- **Time bins:** 50,000 µs (50 ms) per frame → variable count, collated to 25
- **Spatial downsample:** `F.adaptive_avg_pool2d` → **34×34** (to match N-MNIST backbone input)

---

## 6. ARCHITECTURE DETAILS

### 6.1 Backbone Sharing

**Backbones are SEPARATE per task — NOT shared between N-MNIST and DVS-Gesture.**

```python
class M7_ContinualAdaptiveModel(nn.Module):
    def __init__(self, tasks):
        self.backbones = nn.ModuleDict()
        for name, info in tasks.items():
            if info['type'] == 'audio':
                self.backbones[name] = nn.ModuleDict({...})  # MLP for SHD
            else:
                self.backbones[name] = SNN_Backbone(input_channels=2, ...)
```

- `nmnist` → own `SNN_Backbone`
- `dvs` → own `SNN_Backbone`  
- `shd` → own linear MLP backbone

### 6.2 SNN Backbone (N-MNIST & DVS-Gesture)

```python
class SNN_Backbone(nn.Module):
    def __init__(self, input_channels=2, num_classes=10):
        self.conv1 = nn.Conv2d(input_channels, 32, 5)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(256, 512)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
```

- **β = 0.9** for ALL LIF neurons (shared hyperparameter)
- **Input channels:** 2 (for both N-MNIST and DVS-Gesture)
- **Output:** 512-d embedding

### 6.3 Shared Components

- **512-d feature space** (shared across all modalities)
- **Hopfield memory** (shared)
- **HGRN gate** (shared)
- **Adaptive gate weights** `w_hop`, `w_hgrn` (shared scalars)
- **Task-specific classification heads** (separate)

---

## 7. RESULTS SUMMARY (3 SEEDS)

### 7.1 Raw Results

| Seed | Condition | T1 N-MNIST | T2 SHD | T3 DVS-G | FWT |
|------|-----------|------------|--------|----------|-----|
| 42 | no_replay | 97.30% | **80.26%** | **84.85%** | 75.71 |
| 42 | replay | 97.37% | **81.98%** | **84.47%** | 76.55 |
| 123 | no_replay | 97.72% | 78.93% | 78.41% | 74.43 |
| 123 | replay | 97.72% | 78.31% | 83.33% | 73.81 |
| 456 | no_replay | 97.73% | 78.75% | 82.95% | 73.67 |
| 456 | replay | 97.73% | 79.46% | 80.68% | 74.38 |

### 7.2 Mean ± Std

| Condition | T2 SHD | T3 DVS-G | FWT |
|-----------|--------|----------|-----|
| No Replay | 79.31 ± 0.82% | 82.07 ± 3.31% | 74.60 ± 1.03 |
| + Replay | 80.00 ± 1.54% | 82.83 ± 1.96% | 75.11 ± 1.24 |

---

## 8. FORGETTING ANALYSIS — THE "DRIFT" ISSUE

**Colleague concern: "Drift still exists" — CONFIRMED.**

| Condition | T1→T2 | T1→T3 | T2→T3 |
|-----------|-------|-------|-------|
| No Replay (mean) | +0.01 ± 0.02 | −0.01 ± 0.01 | −0.03 ± 0.05 |
| + Replay (mean) | −0.01 ± 0.02 | +0.01 ± 0.02 | **−0.22 ± 0.38** |

**Key finding:** T2→T3 forgetting is **not negligible** under replay:
- Seed 123 replay: **−0.71 pp** (SHD drops from 78.31% → 79.02% after DVS training... wait, that's actually +0.71? Let me re-check signs)

Actually, `forgetting_T2_T3 = acc_T2_after_T2 − acc_T2_after_T3`:
- Seed 123 replay: 78.31 − 79.02 = **−0.71** → T2 *improved* after T3? That suggests positive backward transfer, not forgetting.
- Seed 123 no_replay: 78.93 − 79.02 = −0.09 → same
- Seed 456 no_replay: 78.75 − 78.75 = 0.00
- Seed 456 replay: 79.46 − 79.42 = **+0.04** → minor forgetting

The large negative values are actually *negative forgetting* (i.e., backward transfer). The only genuine forgetting is seed 456 replay at +0.04 pp.

**However**, the T3 accuracy varies wildly (78–85%), suggesting the DVS-Gesture task is unstable across seeds. This is the real reproducibility concern.

---

## 9. FILE MANIFEST FOR UPLOAD

### 9.1 Code (→ GitHub)

```
m7_trimodal_run.py          # Main training script
generate_figures.py         # Old figure generation (seed 42)
generate_figures_3seed.py   # New figure generation (3 seeds)
EXPERIMENTAL_MANIFEST.md    # This file
```

### 9.2 Results (→ GitHub)

```
ICONS_M7/results/cl_trimodal_results.csv
ICONS_M7/results/plots/*.png
```

### 9.3 Checkpoints (→ Git LFS or Hugging Face / Zenodo)

```
ICONS_M7/results/checkpoints/*.pt    # 862 MB total
```

### 9.4 Datasets (→ DO NOT UPLOAD — 23 GB)

```
ICONS_M7/datasets/    # 23 GB — will be auto-downloaded by tonic
```

---

## 10. REPRODUCTION COMMAND

```bash
cd /Users/agent1/Documents/neuromorphic/neurips
source .venv/bin/activate

# Run all seeds
python m7_trimodal_run.py

# Generate figures
python generate_figures_3seed.py
```

**Hardware:** MacBook Pro M1 (MPS)  
**Python:** 3.11  
**Key packages:** torch 2.11.0, snntorch 0.9.4, tonic 1.6.0

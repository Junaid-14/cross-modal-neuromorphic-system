# Notebook Guide — NeurIPS2026_M7_clean (3) (1).ipynb

**55 cells** | ~3–5 hrs runtime (Cell 30) | RTX 4090 recommended

---

## Run Order Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  SETUP PHASE (~20 min)                                              │
│  Cells 0→9 → Cell 10 → Cell 11                                      │
├─────────────────────────────────────────────────────────────────────┤
│  MODEL DEFINITION (~2 min)                                          │
│  Cells 12→28                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  MAIN EXPERIMENT (~4–5 hrs)                                         │
│  Cell 30                                                            │
├─────────────────────────────────────────────────────────────────────┤
│  ANALYSIS (~10 min)                                                 │
│  Cells 31→54 (pick what you need)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Section 1 — Environment Setup (Cells 0–5)

| Cell | ID | Purpose | Runtime |
|------|----|---------|---------|
| 0 | — | Title markdown | — |
| 1 | — | Section header | — |
| 2 | `fd3182c3` | Install `jinja2`, `matplotlib`, `seaborn` | 30s |
| 3 | `2714a9bc` | Install `snntorch`, `tonic`, `transformers`, `datasets` | 1–2 min |
| 4 | `5c508ef1` | System imports + device check | <1s |
| 5 | `811ce402` | **Global CONFIG** + directories + `spike_grad` | <1s |

### Key Global Objects Defined

```python
CONFIG = {
    'batch_size': 32, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
    'max_epochs': 30, 'patience': 5, 'gradient_clip': 1.0,
    'beta': 0.9, 'dropout': 0.2, 'hidden_dim': 512,
    'num_patterns': 100, 'num_gru_layers': 2, 'num_workers': 0,
    'time_steps': 25, 'pin_memory': True,
    'use_contrastive': True, 'contrastive_temperature': 0.07,
    'contrastive_weight': 0.1, 'seed': 42,
    'device': device, 'save_dir': RESULTS_DIR,
}
spike_grad = surrogate.atan()
```

---

## Section 2 — Datasets (Cells 6–11)

| Cell | ID | Purpose | Outputs |
|------|----|---------|---------|
| 6 | — | Section header | — |
| 7 | `fa9f4dd9` | **N-MNIST loader** | `train_loader_nmnist`, `test_loader_nmnist`, `nmnist_info`, `val_loader_nmnist` |
| 8 | `297c6efb` | **SHD loader** | `train_loader_shd`, `test_loader_shd`, `shd_info`, `shd_train_data`, `train_loader_shd_cl`, `val_loader_shd` |
| 9 | `0bd0955b` | **DVS-Gesture loader definition** | `get_dvs_gesture_loaders()` function |
| 10 | `dvs-loader-exec` | **Execute DVS loader** | `train_loader_dvs`, `test_loader_dvs`, `dvs_info`, `train_loader_dvs_cl`, `val_loader_dvs` |
| 11 | `dvs-smoke-test` | **DVS smoke test** | Validates single-task accuracy > chance |

### N-MNIST (Cell 7)
- Transform: `Denoise(filter_time=10000)` → `ToFrame(sensor_size, time_window=1000)`
- Collate: `nmnist_collate_fn` — pad/truncate to T=25
- Cached via `tonic.DiskCachedDataset`
- Val split: 10% from train (6,000 / 54,000)

### SHD (Cell 8)
- Transform: `events_to_dense` — 100 time bins × 700 channels
- Collate: `shd_collate_fn` — simple stack
- Val split: 10% from train (815 / 7,341)

### DVS-Gesture (Cells 9–10)
- Transform: `Denoise(filter_time=10000)` → `ToFrame((128,128,2), time_window=5000)`
- Collate: `dvs_collate_fn` — pad/truncate to T=25 + adaptive pool 128→34
- Val split: 10% from train
- **Try/except wrapper** for download failure → SSC-35 fallback

### Smoke Test (Cell 11)
- Seeds: 42
- Temporary `M7_ContinualAdaptiveModel(tasks={'dvs':...})`
- Runs `ContinualLearner.train_task('dvs', 'dvs', ...)` for 30 epochs
- **Assertions**: logits=(B,11), gate=(B,2), features=(B,512)
- **Guards**:
  - `≤ 10.1%` → RuntimeError (below chance)
  - `< 30%` → warning
  - `≥ 30%` → PASS

---

## Section 3 — Building Blocks (Cells 12–15)

| Cell | ID | Class | Purpose |
|------|----|-------|---------|
| 12 | — | Section header | — |
| 13 | `b73d8000` | `SupervisedContrastiveLoss` | SCL engram formation (temperature=0.07) |
| 14 | `e6dda3be` | `ModernHopfieldLayer` | Associative memory (256 patterns × 512-d) |
| 15 | `4d01e114` | `ImprovedHGRNGate` | Temporal gating (GRU-like, LayerNorm) |

### SupervisedContrastiveLoss
- Input: `(features [B, 512], labels [B])`
- Normalizes features, computes similarity matrix
- Returns mean negative log-likelihood over positive pairs

### ModernHopfieldLayer
- Memory: `nn.Parameter(torch.randn(256, 512))`
- Forward: scaled dot-product attention + temperature + residual + LayerNorm

### ImprovedHGRNGate
- Gates: reset `r` + update `z`
- Combined: `(1-z) * h_prev + z * h_tilde`
- LayerNorm on all three gates

---

## Section 4 — SNN Backbones (Cells 16–18)

| Cell | ID | Class | Input | Output |
|------|----|-------|-------|--------|
| 16 | — | Section header | — | — |
| 17 | `c2959843` | `SNN_Backbone` | `(B, 25, 2, 34, 34)` | `(spk_sum [B, C], features [B, 512])` |
| 18 | `337c6f19` | `SNN_Backbone_SHD` | `(B, 100, 1, 1, 700)` | `(spk_sum [B, C], features [B, 512])` |

### SNN_Backbone (Conv2D)
```
Conv2d(2,32,5) → MaxPool2d(2) → LIF
Conv2d(32,64,5) → MaxPool2d(2) → LIF
Flatten → FC(1600, 1024) → LIF
FC(1024, 512) → LIF  ← features
FC(512, C) → LIF      ← spk_sum
```

### SNN_Backbone_SHD (Linear)
```
FC(700, 1024) → LIF
FC(1024, 1024) → LIF
FC(1024, 512) → LIF  ← features
FC(512, C) → LIF     ← spk_sum
```

---

## Section 5 — M1–M5 Ablation Models (Cells 19–20)

| Cell | ID | Models | Purpose |
|------|----|--------|---------|
| 19 | — | Section header | — |
| 20 | `6758a9b4` | M1–M5 | Baseline ablations for N-MNIST + SHD variants |

### N-MNIST Models
| Model | Class | Components |
|-------|-------|-----------|
| M1 | `Model_1_Baseline_SNN` | Backbone only, no SCL |
| M2 | `Model_2_Baseline_SNN_SCL` | Backbone + SCL |
| M3 | `Model_3_SNN_Hopfield` | Backbone + Hopfield |
| M4 | `Model_4_SNN_HGRN` | Backbone + HGRN |
| M5 | `Model_5_Full_Hybrid` | Backbone + Hopfield + HGRN |

### SHD Models
| Model | Class |
|-------|-------|
| M1 | `Model_1_SHD` |
| M2 | `Model_2_SHD` |
| M3 | `Model_3_SHD` |
| M4 | `Model_4_SHD` |
| M5 | `Model_5_SHD` |

---

## Section 6 — Training Pipeline (Cells 21–22)

| Cell | ID | Class/Function | Purpose |
|------|----|----------------|---------|
| 21 | — | Section header | — |
| 22 | `bdf7b9d2` | `TrainingTracker` | Early stopping + checkpointing |
| | | `train_step` | Single training epoch |
| | | `test_step` | Single validation epoch |
| | | `train_model` | Full training loop |

### TrainingTracker
- Patience-based early stopping
- Saves best model to `CHECKPOINTS_DIR / {model_name}_best.pth`
- Tracks: train/val loss (CE + SCL), accuracy, LR

### train_model
```python
def train_model(model, train_loader, test_loader, model_name, dataset_name,
                use_contrastive, num_epochs, patience, device):
    ...
    return model, history, best_val_acc
```

---

## Section 7 — M7 Unified Model (Cells 23–24)

| Cell | ID | Class | Purpose |
|------|----|-------|---------|
| 23 | — | Section header | — |
| 24 | `0abb3a74` | `M7_ContinualAdaptiveModel` | **Main tri-modal model** |

### Architecture
```
Input → Backbone (modality-specific)
           ↓
    Features [B, 512]
           ↓
    Adaptive Gate (512→128→2 Softmax)
           ↓
    w_hop · Hopfield(f) + w_hgrn · HGRN(f, 0)
           ↓
    Combined [B, 512]
           ↓
    Task Head (LayerNorm → Linear → C classes)
```

### Key Methods
| Method | Signature | Purpose |
|--------|-----------|---------|
| `_get_features` | `(x, modality)` → `[B, 1, 512]` | Extract embeddings for replay buffer |
| `forward` | `(x, modality, task_name)` → `(logits, gate, features)` | Main forward pass |
| `forward_from_embeddings` | `(embs, task_name)` → `logits` | Replay path (no backbone) |

### Per-Task Heads
```python
self.heads = {
    'nmnist': LayerNorm(512) → Linear(512, 10),
    'shd':    LayerNorm(512) → Linear(512, 20),
    'dvs':    LayerNorm(512) → Linear(512, 11),
}
```

---

## Section 8 — Embedding Replay Buffer (Cells 25–26)

| Cell | ID | Class | Purpose |
|------|----|-------|---------|
| 25 | — | Section header | — |
| 26 | `bf43f3cc` | `EmbeddingReplayBuffer` | Hippocampal indexing replay |

### API
```python
buffer = EmbeddingReplayBuffer(feature_dim=512, samples_per_class=50)
buffer.populate(model, loader, modality, task_id, label_offset=0)
buffer.sample(batch_size, task_ids=None)  # → (embs [B,512], labels [B])
buffer.stats()  # → {'total': N, 'num_classes': K, 'by_task': {...}}
```

---

## Section 9 — ContinualLearner (Cells 27–28)

| Cell | ID | Class | Purpose |
|------|----|-------|---------|
| 27 | — | Section header | — |
| 28 | `813c47c4` | `ContinualLearner` | Sequential task trainer with replay |

### Key Methods
| Method | Purpose |
|--------|---------|
| `evaluate(loader, modality, task_name)` | Test accuracy |
| `train_task(..., use_replay, replay_task_ids, ...)` | Train one task |
| `compute_metrics()` | CL forgetting + FWT |

### train_task Signature
```python
def train_task(self, task_name, modality,
               train_loader, val_loader, test_loader,
               num_epochs=50, patience=12, lr=1e-3,
               use_replay=False, replay_task_ids=None,
               use_scl=True, scl_weight=0.1,
               step_id=0)
```

---

## Section 10 — Main CL Experiments (Cells 29–30)

| Cell | ID | Purpose | Runtime |
|------|----|---------|---------|
| 29 | — | Section header | — |
| 30 | `ec05df78` | **Tri-modal CL** (T1→T2→T3) | ~4–5 hrs |

### What It Does
1. Runs `run_one_seed(seed, condition, use_replay)` for each seed
2. Conditions: `no_replay`, `replay`
3. Seeds: `[42, 123, 456]`

### T1→T2→T3 Flow
See [[Tri-Modal CL Sequence]] for full protocol.

### Outputs
- `ICONS_M7/results/cl_trimodal_results.csv`
- `ICONS_M7/results/checkpoints/M7_*_seed{N}.pth`
- `ICONS_M7/results/plots/training_*.png`
- HuggingFace uploads (if `HF_REPO` set)

---

## Sections 11–22 — Analysis & Baselines (Cells 31–54)

| Section | Cells | Purpose | Runtime |
|---------|-------|---------|---------|
| 11 | 31–32 | Load results CSV | <1s |
| 12 | 33–34 | Summary + LaTeX tables | <1s |
| 13 | 35–36 | Publication figures | <1s |
| 14 | 37–38 | Gate specialisation analysis | ~2 min |
| 15 | 39–40 | **EWC baseline** | ~2 hrs |
| 16 | 41–42 | **Joint training upper bound** | ~2 hrs |
| 17 | 43–44 | **Component ablation** | ~3 hrs |
| 18 | 45–46 | Gate collapse fix | ~30 min |
| 19 | 47–48 | Energy efficiency (SynOps) | <1s |
| 20 | 49–50 | Comprehensive results tables | <1s |
| 21 | 51–52 | All publication figures | <1s |
| 22 | 53–54 | Final numbers for paper | <1s |

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `buffer.total_stored` AttributeError | Old code used non-existent attribute | Changed to `buffer.stats()['total']` |
| `buffer.sample` unpacks 3 values | Old code tried `embs, lbls, _` | Changed to `embs, lbls = buffer.sample(...)` |
| `hist_t3` only shows last epoch | Initialized inside loop | Moved before `for epoch in range(...)` |
| `forgetting_dvs` nonsense | Compared SHD acc vs DVS acc | Split into proper `forgetting_nm_t3` + `forgetting_shd_t3` |
| DVS download fails | IBM/tonic server issue | Try/except → `RuntimeError` with SSC-35 fallback |
| HF uploads skipped | `HF_REPO` not set | Export env var before running |

# Tri-Modal CL Sequence

## Protocol

Sequential training across three neuromorphic datasets with embedding replay.

```
T1: N-MNIST (Visual, 10 classes)
    ↓
T2: SHD (Audio, 20 classes) + replay T1 embeddings
    ↓
T3: DVS-Gesture (Visual-Motion, 11 classes) + replay T1+T2 embeddings
```

## Cell 30: Main CL Cell (`ec05df78`)

### Hyperparameters

```python
CL_SEEDS    = [42, 123, 456]
CL_EPOCHS   = 50
CL_PATIENCE = 12
CL_LR       = 1e-3
```

### Two Conditions

| Condition | `use_replay` | Replay Source |
|-----------|--------------|---------------|
| No Replay | `False` | — |
| + Replay  | `True` | Task embeddings from buffer |

### Task Map

| Step | Task Name | Modality | `task_id` | `num_classes` |
|------|-----------|----------|-----------|---------------|
| T1 | `nmnist` | `nmnist` | 1 | 10 |
| T2 | `shd` | `shd` | 2 | 20 |
| T3 | `dvs` | `dvs` | 3 | 11 |

### Training Flow (per seed)

#### T1: N-MNIST
- Train on `train_loader_nmnist`
- Validate on `val_loader_nmnist`
- Evaluate on `test_loader_nmnist` → `acc_nm_s1`
- Evaluate zero-shot SHD → `acc_shd_s1`
- Populate replay buffer with T1 embeddings

#### T2: SHD
- Train on `train_loader_shd_cl`
- Validate on `val_loader_shd`
- Evaluate both test sets:
  - N-MNIST → `acc_nm_after_t2`
  - SHD → `acc_shd_after_t2`
- Replay: sample T1 embeddings via `buffer.sample(32)`
- Populate replay buffer with T2 embeddings

#### T3: DVS-Gesture
- Train on `train_loader_dvs_cl`
- Validate on `val_loader_dvs`
- Evaluate all three test sets:
  - N-MNIST → `acc_nm_after_t3`
  - SHD → `acc_shd_after_t3`
  - DVS → `acc_dvs_after_t3`
- Replay: sample T1+T2 embeddings via `buffer.sample(32, task_ids=[1, 2])`
- Replay loss added against `nmnist` and `shd` heads

### Metrics Computed

```python
forgetting_t2     = acc_nm_s1 - acc_nm_after_t2          # T1 forgetting after T2
fwt_t2            = acc_shd_after_t2 - acc_shd_s1        # T2 forward transfer
forgetting_nm_t3  = acc_nm_after_t2 - acc_nm_after_t3    # NM forgetting after T3
forgetting_shd_t3 = acc_shd_after_t2 - acc_shd_after_t3  # SHD forgetting after T3
```

### Outputs

- `ICONS_M7/results/cl_trimodal_results.csv` — updated after **every seed**
- `ICONS_M7/results/checkpoints/M7_*_seed{N}.pth` — best model per task
- `ICONS_M7/results/plots/training_*_seed{N}_*.png` — training curves
- HuggingFace dataset repo — uploaded after every seed

### Replay Buffer Spec

| Param | Value |
|-------|-------|
| Feature dim | 512 |
| Samples per class | 50 |
| Buffer size (T1) | 10 × 50 = 500 embeddings |
| Buffer size (T2) | 30 × 50 = 1,500 embeddings |
| Buffer size (T3) | 41 × 50 = 2,050 embeddings |
| Total memory | ~4 MB |

### Gate Trajectory

The adaptive gate weights are logged every 5 epochs:
- `w_hop` (Hopfield) vs `w_hgrn` (HGRN)
- Expected: `w_hgrn` increases under replay (specialisation)

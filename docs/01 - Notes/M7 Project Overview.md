# M7 Project Overview

## Title
**NeurIPS 2026 — M7: Modality-Adaptive SNN with Embedding Replay**

## Core Idea
Cross-Modal Continual Learning in Spiking Neural Networks using:
- Modern Hopfield associative memory
- HGRN temporal gating
- Supervised Contrastive Loss (SCL)
- Embedding replay buffer (hippocampal indexing)

## Key Components

### Datasets (Tri-Modal)
| Task | Dataset | Modality | Classes |
|------|---------|----------|---------|
| T1 | N-MNIST | Visual | 10 |
| T2 | SHD | Audio | 20 |
| T3 | DVS-Gesture | Visual Motion | 11 |

### Architecture
- **Backbones**: Conv2D (N-MNIST/DVS) + Linear (SHD)
- **Shared memory**: Modern Hopfield (256 patterns × 512-d)
- **Temporal gating**: Improved HGRN Gate (512→512)
- **Adaptive gate**: 512→128→2 (Softmax) → [w_hop, w_hgrn]
- **Per-task heads**: LayerNorm → Linear per task

### CL Metrics
- `forgetting_t2` = R_T1_after_T1 − R_T1_after_T2
- `fwt_t2` = R_T2_after_T2 − R_T2_after_T1 (zero-shot)
- `forgetting_nm_t3` = R_T1_after_T2 − R_T1_after_T3
- `forgetting_shd_t3` = R_T2_after_T2 − R_T2_after_T3

## Codebase
- Main notebook: `NeurIPS2026_M7_clean (3) (1).ipynb`
- DVS loader: `neurips1_dvs.py`
- Helpers: `neurips3.py`

## Run Order
1. Cells 1→14 (setup, ~20 min)
2. Cell 10 (DVS loader)
3. Cell 11 (DVS smoke test)
4. Cell 30 (main CL, ~4–5 hrs for 3 seeds)

## Key Rules
1. **Run cells in order** — smoke test must complete before main CL
2. **Set `HF_REPO`** for HuggingFace uploads after every seed
3. **Flag immediately** if DVS download fails or accuracy < chance → SSC-35 fallback

# NeurIPS 2026 — M7: Modality-Adaptive SNN with Embedding Replay

## Core Idea

Build a **cross-modal continual learning system** that learns sequentially across three neuromorphic datasets:

1. **T1 (Visual)**: N-MNIST — 10 classes, 34×34 event frames
2. **T2 (Auditory)**: SHD — 20 classes, 700-channel spike trains
3. **T3 (Visual-Motion)**: DVS-Gesture — 11 classes, 128×128 events

## Architecture

```
Input → Modality Backbone → 512-d Features
                              ↓
                    Adaptive Gate [w_hop, w_hgrn]
                              ↓
            Hopfield Memory ←→ HGRN Gate
                              ↓
                       Combined Representation
                              ↓
                    Per-Task Classification Head
```

## Components

- **SNN Backbone**: Conv2D for visual, Linear for audio
- **Modern Hopfield Layer**: Associative memory (256 patterns × 512-d)
- **HGRN Gate**: Temporal gating with LayerNorm
- **Adaptive Gate**: 512→128→2 Softmax routing
- **Embedding Replay Buffer**: ~100 KB vs 80 MB raw replay
- **Supervised Contrastive Loss**: Engram formation

## Continual Learning Protocol

| Step | Task | Modality | Replay Source |
|------|------|----------|---------------|
| T1 | N-MNIST | Visual | — |
| T2 | SHD | Audio | T1 embeddings |
| T3 | DVS-Gesture | Visual-Motion | T1 + T2 embeddings |

## Key Results (Baseline)

- N-MNIST after T2: **97.71 ± 0.08%**
- SHD after T2: **78.00 ± 1.68%**
- Forgetting: **0.01 ± 0.02pp**
- FWT: **72.87 ± 1.56pp**

## Files

- `NeurIPS2026_M7_clean (3) (1).ipynb` — Main experiment notebook
- `neurips1_dvs.py` — DVS loader module
- `neurips3.py` — Supporting utilities

## Run Order

1. Cells 1→14 (setup, ~20 min)
2. Cell 10 (DVS loader)
3. Cell 11 (DVS smoke test)
4. Cell 30 (main CL, ~4–5 hrs)
5. Cells 32→54 (analysis & figures)

## Environment

```bash
export HF_REPO="your-username/your-dataset-repo"
python -m pip install snntorch tonic transformers datasets
```

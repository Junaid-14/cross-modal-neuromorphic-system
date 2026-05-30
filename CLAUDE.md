# CLAUDE.md — M7 Tri-Modal Continual Learning

Guidance for Claude Code working in this repository.

## What This Is

A self-contained research experiment: **M7 — Modality-Adaptive SNN with Embedding
Replay** (NeurIPS 2026). A single spiking neural network learns sequentially across
three neuromorphic datasets of different modalities (visual → audio → visual-motion)
without catastrophic forgetting, using per-task heads + embedding replay
(~4 MB of embeddings vs ~80 MB raw). Includes a pipeline to deploy onto SpiNNaker
neuromorphic hardware.

This repo holds BOTH the code and its documentation:

| Path | What's here |
|------|-------------|
| `code/` | All executable code: training notebook, standalone runner, SpiNNaker tests. |
| `docs/` | Obsidian notes: experimental manifest, architecture deep-dive, SpiNNaker port plan, lab notes. |
| `configs/` | Experiment configs. |
| `results/` | CSVs and figures (gitignored). |
| `code/ICONS_M7/` | Datasets (~23 GB) + checkpoints (~862 MB), gitignored — symlinked, not copied. |

**Boundary rule:** Operate freely on `code/`, `docs/`, `configs/`, and `results/`
WITHIN this repo. Do NOT reach into sibling experiment repos (`../spinnaker-mapping/`,
etc.) or the personal vault (`../research/`) — those are separate repos with their own
CLAUDE.md. Code and docs here are meant to be edited together; keep them in sync (if a
hyperparameter changes in code, update the manifest in `docs/`).

## Stack

PyTorch 2.11.0, snntorch 0.9.4, tonic 1.6.0, sPyNNaker 7.x (hardware deployment).
snnTorch and sPyNNaker have different neuron semantics — parameters do NOT transfer
automatically. Flag portability gaps rather than assuming them.

## Commands

```bash
source code/.venv/bin/activate

# Full tri-modal continual learning (3 seeds, ~4-5 hrs on MPS)
python code/m7_trimodal_run.py

# Publication figures (3-seed mean±std)
python code/generate_figures_3seed.py

# SpiNNaker integration tests (require sPyNNaker installed)
python code/test_spinnaker_hello.py
python code/test_spinnaker_backbone_e2e.py   # requires a checkpoint
```

No formal test suite. Validation = notebook smoke tests + standalone `test_spinnaker_*.py`.
`m7_trimodal_run.py` is the **reproducibility entry point** — keep it self-contained,
no imports beyond standard packages.

## Critical Run Rules

- Notebook cells execute in STRICT order. The DVS smoke test (Cell 11) must pass before
  the main CL run (Cell 30).
- `modify_notebook.py` overwrites the notebook in place — back up first.
- All 3 seeds `[42, 123, 456]` must run when evaluating CL-pipeline changes.
- Device auto-selects MPS → CUDA → CPU. On macOS, `num_workers=0` is required (tonic
  multiprocessing limitation).
- CSV results are saved incrementally after every seed — don't trust cloud instance
  persistence. The results CSV has duplicate rows from reruns: filter `step.isna()`
  then `drop_duplicates()`.

## Architecture (summary — full detail in docs/M7_Architecture_Deep_Dive.md)

Event input (3 modalities, sequential T1→T2→T3) → modality-specific SNN backbones
(Conv2D visual / MLP audio, all 512-d out) → parallel memory (ModernHopfieldLayer +
ImprovedHGRNGate) → adaptive gate (512→128→2 softmax, learns w_hop vs w_hgrn per
modality) → per-task heads (frozen when inactive) → loss `CE + 0.1*SCL + replay_loss`.

Per-task heads isolate gradients; embedding replay (512-d, reservoir-sampled) prevents
forgetting. The gate only learns to use HGRN under replay (without replay, w_hgrn → 0).

| Task | Dataset | Modality | Classes |
|------|---------|----------|---------|
| T1 | N-MNIST | Visual | 10 |
| T2 | SHD | Audio | 20 |
| T3 | DVS-Gesture | Visual-Motion | 11 |

Key design: LIF β=0.9 (≈ tau_m 10ms on SpiNNaker); time_steps=25 fixed across
modalities; DVS downsampled 128→34 to match N-MNIST backbone; SHD via custom
`events_to_dense` (100×700); val split 10%, fixed `manual_seed(42)`.

## SpiNNaker Port (see docs/SpiNNaker_Port_Plan.md)

- Phase 1 (inference): PyTorch weights → PyNN `IF_curr_exp` + `AllToAllConnector`.
- Phase 2 (host-mediated CL): host computes updates, injects into live sim.
- Phase 3 (on-chip STDP): stretch goal.
- Hard constraint: SpiNNaker runs PyNN, not PyTorch. Conv2d must flatten to dense
  projections. Treat fan-in/neuron-per-core/memory limits as binding, not abstract.

## Known Issues

| Issue | Fix |
|-------|-----|
| `buffer.total_stored` AttributeError | `buffer.stats()['total']` |
| `buffer.sample` returns 2 not 3 | `embs, lbls = buffer.sample(...)` |
| `hist_t3` only last epoch | init `hist_t3` before epoch loop |
| DVS download fails | fallback SSC-35 (contact Blessing) |
| DVS smoke test acc ≤ 10.1% | RuntimeError — check preprocessing |

## Research Conventions (non-negotiable)

- The notebook is canonical; standalone `.py` files are extracts kept in sync.
- Seeds are explicit — always set and report torch/numpy seeds.
- **Hypotheses are not results.** Hardware-similarity claims are hypotheses needing
  evidence; never write them up as established.
- **Never fabricate** numbers, citations, or hardware specs. If unknown, say so.
- Never refactor in ways that silently change numerical results.
- When reporting accuracy, also report spike rate/sparsity and timestep count.
- Never overwrite a file without showing a diff first; wait for confirmation.

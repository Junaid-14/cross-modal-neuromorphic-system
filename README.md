# M7 — Tri-Modal Continual Learning on Spiking Neural Networks

> A single spiking neural network that learns three sensory modalities in sequence
> (vision → audio → motion) without forgetting the earlier ones — and a pipeline to
> run it on SpiNNaker neuromorphic hardware. NeurIPS 2026 submission.

## Start here (newcomer's path)

1. **What it does & why it works** → `docs/M7_Architecture_Deep_Dive.md`
2. **How to reproduce results** → `docs/EXPERIMENTAL_MANIFEST.md` (seeds, hyperparameters, expected numbers)
3. **Hardware deployment** → `docs/SpiNNaker_Port_Plan.md`
4. **Run it** → see "Quickstart" below

## Repository map

```
m7-trimodal-cl/
├── code/        all executable code (training, figures, SpiNNaker tests)
├── docs/        documentation & lab notes (Obsidian vault)
├── configs/     experiment configurations
├── results/     output CSVs & figures        (gitignored)
└── code/ICONS_M7/  datasets + checkpoints     (gitignored; symlinked, ~24 GB)
```

## Quickstart

```bash
source code/.venv/bin/activate
pip install -r code/requirements.txt          # snntorch, tonic, transformers, ...
python code/m7_trimodal_run.py                 # 3 seeds, ~4–5 hrs on MPS
python code/generate_figures_3seed.py          # publication figures
```

GPU with ~25 GB VRAM recommended. Datasets auto-download via `tonic` (~23 GB).

## The experiment in one table

| Task | Dataset | Modality | Classes |
|------|---------|----------|---------|
| T1 | N-MNIST | Visual | 10 |
| T2 | SHD | Audio | 20 |
| T3 | DVS-Gesture | Visual-Motion | 11 |

**Headline result:** near-zero catastrophic forgetting via per-task heads + embedding
replay (~4 MB of stored embeddings instead of ~80 MB of raw data).

## Status

Active branch: `feature/dvs-gesture-t3`. See `docs/01-Notes/` for the running log.

## A note on claims

Hardware-similarity claims in this repo are **hypotheses under investigation**, not
established results, unless explicitly backed by the manifest. Reported accuracy always
travels with its cost metrics (spike rate / sparsity / timestep count).

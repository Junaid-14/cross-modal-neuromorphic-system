# Junaid's Notebook Handoff (SHD-20 Repro + Report)

This package is the handoff bundle for the SHD-20 standalone sweep and technical report notebook.

## What To Share (Minimum)

1. `notebook/Junaids_Notebook.ipynb`
2. `results/final_specs.json`
3. `results/research_sweep_results.json`

This is enough to run the **report notebook** top-to-bottom (assuming notebook dependencies are installed).

## What To Share (Full Repro Bundle)

Use the generated archive:

- `deliverables/Junaids_Notebook_RunPod_Handoff.zip`

## Naming Note

- The bundle is still **SHD-20-specific** because the packaged results/plots are for the SHD-20 sweep.
- The notebook/readme were renamed for easier teammate handoff; dataset names in the results remain unchanged by design.

It includes:

- Report notebook (`notebook/Junaids_Notebook.ipynb`)
- Sweep outputs (`results/*.json`)
- Standalone sweep CLI (`run_experiments.py`)
- Structural plasticity + temporal engram + energy methodology code (`training/`, `Models/`, `visualization.py`)
- SHD loader (`Dataloaders/shd_loader.py`)
- Dependency file (`requirements.txt`)
- Experiment log (`experiment_log.txt`, if present)

## Exact Specifications (Current SHD-20 Smoke Run)

These are the exact settings used for the **validated smoke run** whose results are currently in `results/`.

- Dataset: `SHD-20` (Spiking Heidelberg Digits, 20 classes)
- Command:
  - `python3 run_experiments.py --num-workers 0 --datasets shd --num-runs 1 --num-epochs 6`
- Device: `mps` (Apple Silicon GPU via PyTorch MPS)
- Batch size: `16`
- Epochs: `6`
- Seeds used: `[42]`
- Base seed: `42`

### Optimizer / Training Settings

- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- LR scheduler: `CosineAnnealingLR(T_max=num_epochs, eta_min=1e-6)`
- Gradient clipping: `1.0`
- Early stopping patience: `7`
- Loss (this sweep): `CrossEntropyLoss` only (`use_contrastive=False`)

## Full Run Settings (Submission-Grade SHD Sweep)

Recommended command for the full statistical run:

- `python3 run_experiments.py --num-workers 0 --datasets shd --num-runs 5 --num-epochs 30`

This implies (with default base seed `42`) the seed sequence:

- `[42, 43, 44, 45, 46]`

## Structural Plasticity (“Breeding & Killing”) Code

### Where the code lives

- Core algorithm: `Models/structural_plasticity.py`
- Integration into training loop: `training/pipeline.py`

### Hyperparameters (defaults used by sweep CLI)

- Structural plasticity enabled only in config: `structural_plasticity`
- Frequency: every `5` epochs (`structural_plasticity_interval=5`)
- Firing-rate profiling batches: `50` (`structural_plasticity_max_batches=50`)
- Kill start epoch: `10` (`structural_plasticity_kill_start_epoch=10`)
- Pruning threshold (silent neuron criterion): `0.01`
- Alpha donor fraction (top-firing neurons): `0.05`
- Breeding noise sigma: `0.01`

### Birth / Spawn mechanism (implemented)

- Compute mean firing rate per output neuron/channel from LIF spikes.
- Identify “dead” neurons: firing rate `< kill_threshold`.
- Select “alpha” neurons = top `alpha_frac` by firing rate.
- Replace dead neuron weights by cloning alpha neuron weights + Gaussian noise (`breed_sigma`).

## Baseline Comparison (Shadow ANN + Energy Methodology)

### Where the code lives

- ANN shadow model + cloning:
  - `training/pipeline.py` (`ANNShadowModel`, `create_ann_baseline`)
- FLOP counter:
  - `training/pipeline.py` (`calculate_model_flops`)
- Energy ratio computation:
  - `training/pipeline.py` (`compute_energy_improvement_ratio`)
- Sweep integration callback:
  - `training/pipeline.py` (`make_energy_methodology_callback`)

### How energy is measured

- SNN cost proxy: **SynOps** (from spike counts collected via hooks on LIF layers)
- ANN cost proxy: **FLOPs** (manual counting of Conv/Linear + Hopfield matmul approximation)

### Energy formula (implemented)

- `Energy_ANN = FLOPs * 100 pJ`
- `Energy_SNN = Mean SynOps * 0.1 pJ`
- `Efficiency Ratio = Energy_ANN / Energy_SNN`

The ~`406x` label in the report comes directly from the aggregated metric in:

- `results/final_specs.json` → `compute_comparison.energy_improvement_ratio.mean`

## Temporal Engram Analysis (Silhouette-over-Time)

### Where the code lives

- Temporal snapshot hook:
  - `training/pipeline.py` (`TemporalEngramSnapshotHook`)
- Callback factory:
  - `training/pipeline.py` (`make_temporal_silhouette_callback`)
- Seed aggregation:
  - `training/pipeline.py` (`run_seed_iterator`)
- Visualization helper:
  - `visualization.py` (`plot_engram_evolution`)

### What it does

- Captures latent LIF spike embeddings at:
  - `t25`, `t50`, `t75`, `t100`
- Computes silhouette score on a validation subset (`<=512` samples)
- Runs under `torch.no_grad()` for evaluation
- Aggregates mean/std across seeds

### “75% audio duration” evidence

The current smoke-run artifacts already show silhouette values at `t75` and `t100` for both:

- `baseline_snn::SHD-20`
- `structural_plasticity::SHD-20`

Source:

- `results/final_specs.json` → `temporal_silhouette`

## Notebook Requirements (for teammate / RunPod)

The notebook (`Research_Synthesis_V2.ipynb`) is **path-robust** (no hardcoded absolute paths) and loads JSON from `results/`.

### Required Python packages for notebook execution

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `ipython`
- `jupyterlab` (or `jupyter`)

### If reproducing experiments (not just notebook)

Also install:

- `torch`
- `snntorch`
- `tonic`
- `scikit-learn`
- `tqdm`

## Efficiency Expectation Note

- The reported energy improvement ratio (e.g., ~406x or a target ~603x) is computed from **SynOps/FLOPs + fixed pJ constants**, not GPU wall-clock speed.
- Running on RunPod RTX 2000 Ada will improve runtime, but the ratio changes only if model behavior (spike counts/training outcome/config) changes.

## Important Hardware Note

- The **notebook** is hardware-agnostic and runs on RunPod as long as the JSON result files are present.
- `run_experiments.py` now supports `--device {auto,cuda,mps,cpu}`.
- For RunPod (RTX 2000 Ada), use:
  - `python3 run_experiments.py --device cuda --datasets shd --num-workers 4`
- For Apple Silicon, use:
  - `python3 run_experiments.py --device mps --datasets shd --num-workers 0`

[x] SpikeCounterHook: Implemented (counts spikes + SynOps).
[x] Seed Iterator: Implemented (5-run stats + Mean/Std).
[x] Temporal Analysis: Added temporal snapshot silhouette callback (25%, 50%, 75%, 100%) with 512-sample subset + `torch.no_grad()` evaluation and 5-seed aggregation support.
[x] Visualization: Added `plot_engram_evolution()` in `visualization.py` to compare Vision vs Audio temporal silhouette formation speed with mean/std bands.
[x] Efficiency Comparison: Added ANN shadow baseline + FLOP counter + Energy Improvement Factor ($Energy_{ANN} / Energy_{SNN}$) computation with configurable pJ constants and final_specs export.
[ ] SHD 20-Class: Execute the 5-seed run on the full 20-class Heidelberg dataset.

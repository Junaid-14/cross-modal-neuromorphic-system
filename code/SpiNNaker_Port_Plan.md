# M7 → SpiNNaker Port Plan

> Porting the M7 tri-modal continual learning system to SpiNNaker neuromorphic hardware.
> Version: 1.0 | Date: 2026-05-07

---

## 1. Executive Summary

**Objective:** Deploy the M7 tri-modal continual learning SNN on SpiNNaker hardware to demonstrate energy-efficient inference and (optionally) on-chip continual learning.

**SpiNNaker Platform:** massively parallel ARM-based neuromorphic architecture (University of Manchester). Typical deployment uses SpiNN-5 boards (1,036,800 cores) or smaller SpiNN-3 boards (4 chips, 72 cores).

**Approach:** Three-phase port — (1) inference-only deployment of trained PyTorch weights, (2) host-mediated continual learning with on-chip inference, (3) on-chip STDP-based plasticity (stretch goal).

**Key Constraint:** SpiNNaker does not support backpropagation, AdamW, or PyTorch. The model must be reimplemented in PyNN/sPyNNaker with rate-based or spike-based approximations of all M7 components.

---

## 2. SpiNNaker Platform Overview

| Attribute | Spec | Implication for M7 |
|-----------|------|-------------------|
| **Core** | ARM968E-S, ~200 MHz | No GPU acceleration. Serial execution per core. |
| **Memory/core** | 32 KB DTCM + 64 KB ITCM | Entire model must partition across thousands of cores. A single 512-d layer exceeds one core. |
| **Neuron models** | IF_curr_exp, Izhikevich, custom | `snn.Leaky(beta=0.9)` maps to `IF_curr_exp` with tuned tau_m. No native "beta" parameter. |
| **Synapse models** | Static, STDP, neuromodulated | No AdamW. Weight updates via STDP or host-injected changes. |
| **Precision** | Fixed-point (32-bit) or software FP | Slight weight quantization. Test tolerance ±0.5%. |
| **Time step** | Typically 1 ms | 25-frame input becomes 25 ms presentation. DVS events map naturally. |
| **Programming** | PyNN, sPyNNaker, SpiNNTools | Complete rewrite of model definition. PyTorch code cannot run directly. |
| **Conv2D** | Not native in PyNN | Must use `ConvolutionConnector` (sPyNNaker extension) or flatten to dense. |
| **LayerNorm** | Not native | Must approximate with weight scaling or omit. |
| **Softmax** | Not native | Must implement via WTA (winner-take-all) circuit or rate decoding. |

**Critical Realization:** SpiNNaker is designed for *biologically plausible SNN simulation*, not for running PyTorch models. The M7 architecture must be *reconstructed* in PyNN, not *translated* from PyTorch.

---

## 3. Component Mapping: M7 → SpiNNaker

### 3.1 SNN_Backbone (Visual: N-MNIST, DVS-Gesture)

**PyTorch implementation:**
```
Conv2d(2, 32, 5) → MaxPool2d(2) → LIF(beta=0.9)
Conv2d(32, 64, 5) → MaxPool2d(2) → LIF(beta=0.9)
Linear(1600, 1024) → LIF(beta=0.9)
Linear(1024, 512) → LIF(beta=0.9)
Linear(512, num_classes) → LIF(beta=0.9)
```

**SpiNNaker mapping:**

| PyTorch Layer | PyNN Equivalent | Feasibility | Risk |
|---------------|----------------|-------------|------|
| Conv2d | `ConvolutionConnector` (sPyNNaker) | ⚠️ Medium | `ConvolutionConnector` supports 2D conv but kernel sizes are limited. May need to flatten to dense projections. |
| MaxPool2d | `PoolingConnector` or subsampling | ⚠️ Medium | Pooling in spiking networks is non-trivial. Subsampling (taking every 2nd neuron) is easier. |
| LIF(beta=0.9) | `IF_curr_exp(tau_m=10.0, v_reset=0.0)` | ✅ High | `beta=0.9` at 1ms step ≈ `tau_m = 10ms` (since `beta = exp(-dt/tau_m)`). Exact match requires calibration. |
| Linear | `Projection` with `AllToAllConnector` | ✅ High | Standard dense layer. Easy to map. |

**Memory analysis (32 KB DTCM/core ≈ 8,192 float32 weights):**
- Conv1: 2×32×5×5 = 1,600 weights = 6.4 KB ✅ (1 core)
- Conv2: 32×64×5×5 = 51,200 weights = 204.8 KB → **7 cores**
- FC1: 1600×1024 = 1,638,400 weights = 6.55 MB → **205 cores**
- FC2: 1024×512 = 524,288 weights = 2.10 MB → **66 cores**
- FC3 (unused in M7): 512×11 = 5,632 weights = 22.5 KB ✅ (1 core)

**Total visual backbone:** ~280 cores for weights + ~10 cores for neurons = **~290 cores per visual backbone**.

### 3.2 Audio Backbone (SHD)

**PyTorch implementation:**
```
Linear(700, 1024) → LIF
Linear(1024, 1024) → LIF
Linear(1024, 512) → LIF
Linear(512, 20) → LIF
```

**SpiNNaker mapping:**
- All Linear layers → `Projection` with `AllToAllConnector`.
- FC1: 700×1024 = 716,800 weights = 2.87 MB → **90 cores**.
- FC2: 1024×1024 = 1,048,576 weights = 4.19 MB → **131 cores**.
- FC3: 1024×512 = 524,288 weights = 2.10 MB → **66 cores**.

**Total audio backbone:** ~290 cores for weights + ~10 cores for neurons = **~300 cores**.

### 3.3 ModernHopfieldLayer

**PyTorch implementation:**
```python
attn = softmax(x @ M.T / sqrt(512))
out = attn @ M
return LayerNorm(x + out)
```

**SpiNNaker mapping options:**

**Option A: Rate-based approximation (recommended)**
- Population of 100 neurons (one per pattern).
- Each pattern neuron receives weighted input from all 512 feature neurons.
- Softmax → winner-take-all (WTA) inhibition: strongest pattern neuron suppresses others.
- Output = residual + WTA winner's pattern weights.
- **Cores needed:** ~20 (100 patterns × 512 weights = 51,200, plus WTA circuit).

**Option B: Spike-based softmax**
- Implement softmax via recurrent inhibitory network.
- Computationally expensive. Not recommended for first port.

**Risk:** The Hopfield layer is the most complex component to port. The WTA approximation may not preserve the "soft" attention behavior of the PyTorch version.

### 3.4 ImprovedHGRNGate

**PyTorch implementation:** GRU with reset/update gates, LayerNorm, tanh.

**SpiNNaker mapping:**
- GRU recurrence: Supported via recurrent `Projection`.
- Reset gate (`r * h_prev`): **Problematic.** Elementwise multiplication of two spike trains is not natively supported.

**Solution approaches:**

1. **Rate-based GRU (recommended):** Decode spike rates (sliding window count), perform multiplication in rate space, re-encode as spikes. Requires inter-core communication for rate decoding.

2. **Simplified spiking GRU:** Omit multiplicative gating. Use additive gating only:
   ```
   h_t = (1-z) * h_{t-1} + z * tanh(W * [x; h_{t-1}])
   ```
   Where `z` is a binary gate (spike/no-spike). This loses expressivity but is implementable.

3. **Host-mediated:** Compute GRU update on host PC, inject new hidden state as spike pattern. Defeats the purpose of on-chip processing.

**Cores needed:** ~15-25 depending on implementation.

### 3.5 Adaptive Gate

**PyTorch implementation:** `Linear(512→128) → ReLU → Linear(128→2) → Softmax`

**SpiNNaker mapping:**
- Two-layer MLP: Easy to map as two populations with `AllToAllConnector`.
- ReLU: `IF_curr_exp` with zero reset voltage acts as ReLU-ish.
- Softmax: **Not native.** Options:
  - WTA circuit (2 neurons, mutual inhibition). Winner = argmax, not softmax.
  - Rate decoding on host, compute softmax, inject back as input currents.
  - Approximate softmax with normalized spike rates.

**Recommendation:** Use rate decoding for the gate. The gate weights are small (512×128 + 128×2 = 65,792 params) and can be computed quickly on the host between time steps.

**Cores needed:** ~5 (if on-chip) or 0 (if host-mediated).

### 3.6 Task Heads

**PyTorch implementation:** `LayerNorm(512) → Linear(512→C)` where C ∈ {10, 20, 11}.

**SpiNNaker mapping:**
- Linear: `AllToAllConnector`. Easy.
- LayerNorm: **Not native.** Omit or approximate with per-neuron threshold adjustment.

**Cores needed:** ~3 per head (10+20+11 = 41 output neurons, trivial).

### 3.7 SupervisedContrastiveLoss

**PyTorch implementation:** Batch-wise cosine similarity matrix with label masking.

**SpiNNaker mapping:** **Not directly implementable.**

**Alternatives:**
1. **Omit on SpiNNaker:** Train off-chip with SCL, deploy fixed weights on-chip. (Recommended for Phase 1)
2. **STDP-based approximation:** Use spike-timing dependent plasticity to strengthen connections between co-active neurons. This is biologically plausible but not mathematically equivalent to SCL.
3. **Host-mediated SCL:** Compute embeddings on-chip, send to host, compute SCL loss, send gradients back. Extremely slow due to communication overhead.

### 3.8 EmbeddingReplayBuffer

**PyTorch implementation:** Stores 512-d float tensors in host RAM (~4 MB).

**SpiNNaker mapping:**
- **Cannot store on-chip:** 4 MB exceeds per-core memory by 100×.
- **Host-mediated replay (recommended):** Host PC stores embeddings, injects them as spike packets into the on-chip network during replay phases.
- **Synaptic weight replay:** The "memory" is encoded in synaptic weights. No explicit buffer needed. But this is not the same as embedding replay.

---

## 4. Implementation Phases

### Phase 1: Inference-Only Deployment (Weeks 1–3)
**Goal:** Run trained M7 on SpiNNaker for forward pass only. No on-chip learning.

**Steps:**
1. Export PyTorch weights to NumPy arrays.
2. Build PyNN model with `sPyNNaker` backend.
3. Map each layer to appropriate `Population` and `Projection` objects.
4. Load weights into projections.
5. Implement input encoding: convert event frames to spike trains (rate coding or temporal coding).
6. Run inference on N-MNIST, SHD, DVS-Gesture test sets.
7. Compare accuracy with PyTorch baseline. Target: ±1% of baseline.

**Deliverable:** Working inference pipeline with accuracy report.

### Phase 2: Host-Mediated Continual Learning (Weeks 4–6)
**Goal:** Execute the T1→T2→T3 protocol with weight updates computed on host, deployed to SpiNNaker.

**Steps:**
1. Run T1 inference on SpiNNaker. Extract embeddings via on-chip spike monitoring.
2. Send embeddings to host. Store in host-side replay buffer.
3. Compute weight updates on host (PyTorch training loop with replay).
4. Deploy updated weights to SpiNNaker.
5. Run T2 inference. Repeat for T3.
6. Measure forgetting and FWT. Compare with pure-PyTorch results.

**Deliverable:** Full continual learning protocol running with SpiNNaker as inference accelerator.

### Phase 3: On-Chip STDP Plasticity (Weeks 7–10, Stretch Goal)
**Goal:** Perform weight updates on SpiNNaker using STDP instead of host-side backprop.

**Steps:**
1. Replace static projections with STDP-enabled projections for task heads.
2. Implement a neuromodulated STDP rule (e.g., reward-modulated STDP) for the adaptive gate.
3. Design a spike-based replay mechanism (re-activate stored patterns via synaptic potentiation).
4. Train T1 on-chip with STDP.
5. Evaluate T1 accuracy.
6. Attempt T2 training with replay via re-activation of T1 patterns.

**Deliverable:** Proof-of-concept on-chip plasticity. Accuracy will likely be lower than host-trained baseline.

---

## 5. Risk Analysis

| Risk ID | Risk Description | Probability | Impact | Mitigation |
|---------|-----------------|-------------|--------|------------|
| R1 | **Conv2D not supported well in sPyNNaker.** ConvolutionConnector may have bugs or limitations for 5×5 kernels. | High | High | Fall back to flattening conv layers into dense projections with weight sharing. Accuracy may drop slightly. |
| R2 | **Hopfield WTA approximation loses soft attention behavior.** Hard WTA may cause brittle switching between patterns. | Medium | High | Test both WTA and rate-decoded softmax. If WTA fails, implement rate-decoded gate on host. |
| R3 | **GRU multiplicative gating cannot be implemented.** Elementwise multiplication of spike trains is not supported. | High | Medium | Simplify to additive gating only. Document accuracy drop. |
| R4 | **LayerNorm omission causes training instability.** PyTorch model relies on LayerNorm in 6+ places. | Medium | Medium | Calibrate per-neuron thresholds and time constants to approximate LayerNorm's effect. Test without LayerNorm in PyTorch first. |
| R5 | **Memory overflow per core.** FC layers have millions of weights that may not fit in 32KB DTCM. | Medium | High | Partition weights across multiple cores using `sPyNNaker`'s auto-partitioning. Test on smaller layers first. |
| R6 | **Weight quantization reduces accuracy.** SpiNNaker uses fixed-point or software FP. Precision loss. | Medium | Medium | Quantize PyTorch weights to 16-bit, test accuracy drop. If >1%, investigate 32-bit software FP mode. |
| R7 | **Input encoding mismatch.** PyTorch uses frame-based tensors. SpiNNaker expects spike times or rates. | Medium | High | Implement Poisson rate coding for frames. Validate that spike count correlates with pixel intensity. |
| R8 | **Timing synchronization.** 25 time steps in PyTorch ≠ 25 ms on SpiNNaker. Presentation duration matters. | Medium | Medium | Calibrate: find the SpiNNaker presentation time (ms) that yields equivalent spike counts to PyTorch's 25 steps. |
| R9 | **No sPyNNaker installed in environment.** The current venv lacks sPyNNaker. | Low | Low | Install via `pip install sPyNNaker`. Requires Python 3.8–3.11. |
| R10 | **Hardware access unavailable.** No physical SpiNNaker board. | Medium | High | Use `sPyNNaker` simulator backend (software simulation) for development and testing. Deploy to hardware only for final validation. |
| R11 | **Phase 3 (on-chip STDP) fails entirely.** STDP may not converge to useful weights for complex tasks. | High | High | Make Phase 3 explicitly a stretch goal. Success criteria: "any non-random accuracy" rather than "match PyTorch baseline." |
| R12 | **Adaptive gate rate-decoding too slow.** Host-mediated gate adds communication latency. | Medium | Medium | Batch gate computations: decode rates every N ms instead of every 1 ms. |

**Top 3 Risks:**
1. **R1 (Conv2D limitation)** → May force architecture simplification.
2. **R3 (GRU gating)** → May require dropping HGRN entirely, changing the paper's core claim.
3. **R11 (STDP failure)** → Phase 3 may be infeasible. Plan for Phase 2 as the primary deliverable.

---

## 6. Resource Requirements

### Software
```bash
# SpiNNaker toolchain
pip install sPyNNaker==7.0.0
pip install SpiNNaker_PACMAN
pip install spinn_front_end_common

# For weight conversion
pip install numpy scipy h5py

# For visualization
pip install matplotlib
```

**Note:** sPyNNaker requires Python 3.8–3.11. The current venv uses 3.11 ✅.

### Hardware
| Phase | Hardware | Access Method |
|-------|----------|--------------|
| Phase 1 | SpiNNaker simulator (sPyNNaker backend) | Local laptop |
| Phase 1 validation | SpiNN-5 board (1M cores) or SpiNN-3 (72 cores) | Remote access via SpiNNaker portal or physical board |
| Phase 2 | Same as Phase 1 | Same |
| Phase 3 | SpiNN-5 board strongly recommended | Physical or remote |

**Minimum viable hardware:** SpiNN-5 board (48 chips, 864 cores usable). M7 needs ~600 cores total, so SpiNN-3 (72 cores) is **insufficient**. SpiNN-5 is required for the full model.

### Personnel
| Role | Time | Tasks |
|------|------|-------|
| Neuromorphic Engineer | 4 weeks | PyNN model construction, weight mapping, debugging |
| ML Engineer | 2 weeks | Weight export, accuracy validation, baseline comparison |
| Hardware Admin | 1 week | Board access, toolchain setup, remote connectivity |

---

## 7. Timeline

| Week | Phase | Milestone |
|------|-------|-----------|
| 1 | Phase 1 | Install sPyNNaker. Build single-layer test (FC 512→10). Validate spike behavior. |
| 2 | Phase 1 | Build full visual backbone. Map Conv2d→ConvolutionConnector or dense fallback. |
| 3 | Phase 1 | Integrate Hopfield (WTA), adaptive gate (host-mediated), task heads. End-to-end inference. |
| 3 | Phase 1 | **Checkpoint:** Accuracy within ±1% of PyTorch on all 3 test sets. |
| 4 | Phase 2 | Implement host-mediated replay buffer. T1→T2 protocol with host weight updates. |
| 5 | Phase 2 | Add T3. Full continual learning protocol. Measure forgetting and FWT. |
| 6 | Phase 2 | **Checkpoint:** Forgetting and FWT within ±2pp of PyTorch baseline. |
| 7 | Phase 3 | Research STDP rules for task heads. Implement neuromodulated STDP. |
| 8 | Phase 3 | On-chip T1 training. Evaluate accuracy. |
| 9 | Phase 3 | Attempt T2 with replay via pattern reactivation. |
| 10 | Phase 3 | **Checkpoint:** Proof-of-concept on-chip plasticity. Document accuracy gap. |

---

## 8. Proofreading & Self-Review

### Accuracy Checklist
- [x] SpiNNaker core memory: 32 KB DTCM confirmed from SpiNNaker documentation.
- [x] `IF_curr_exp` neuron model: confirmed available in sPyNNaker.
- [x] `ConvolutionConnector`: confirmed in sPyNNaker 7.x (beta quality).
- [x] PyNN does not have native LayerNorm: confirmed.
- [x] sPyNNaker supports Python 3.11: confirmed (requires 3.8–3.11).
- [x] STDP is supported: confirmed (`STDPTimingDependence`, `STDPWeightDependence`).
- [x] No native backprop: confirmed. SpiNNaker is forward-only or STDP-based.

### Feasibility Checklist
- [x] Total core estimate (~600) fits on SpiNN-5 (1M cores) easily.
- [x] Total core estimate exceeds SpiNN-3 (72 cores) by ~8×. Model compression or SpiNN-5 required.
- [x] Weight sizes verified: FC1 (1.6M weights) is the largest. At 4 B/weight, needs ~205 cores at 32 KB/core.
- [x] Memory per core verified: 32 KB ≈ 8,192 floats. Original estimate of ~50 cores was off by 4×. Corrected to ~205 cores.

**⚠️ CRITICAL ISSUE FOUND DURING PROOFREADING:**

FC1 has 1,638,400 weights. At 4 bytes per float32, that's **6.55 MB**. Divided across 50 cores: **131 KB per core**. But each core only has **32 KB DTCM**.

**This means the visual backbone FC1 CANNOT fit on 50 cores. It needs ~205 cores minimum.**

Let me recalculate:
- Conv1: 1,600 weights × 4B = 6.4 KB ✅ (1 core)
- Conv2: 51,200 weights × 4B = 204.8 KB → **7 cores**
- FC1: 1,638,400 weights × 4B = 6,553.6 KB → **205 cores**
- FC2: 524,288 weights × 4B = 2,097.2 KB → **66 cores**
- FC3: 5,632 weights × 4B = 22.5 KB ✅ (1 core)

**Visual backbone total: ~280 cores.**

Audio backbone:
- FC1: 716,800 weights × 4B = 2,867.2 KB → **90 cores**
- FC2: 1,048,576 weights × 4B = 4,194.3 KB → **131 cores**
- FC3: 524,288 weights × 4B = 2,097.2 KB → **66 cores**

**Audio backbone total: ~290 cores.**

Hopfield: 51,200 weights × 4B = 204.8 KB → **7 cores**
HGRN (simplified): ~50K weights → **7 cores**
Gate (host-mediated): 0 cores
Heads: trivial

**TOTAL M7 MODEL: ~600 cores.**

This fits comfortably on SpiNN-5 (1M cores) but **exceeds SpiNN-3 (72 cores)** by 8×. If the available hardware is SpiNN-3, the model must be compressed.

**Note on auto-partitioning:** sPyNNaker's PACMAN toolchain automatically partitions synaptic matrices across cores. The manual estimates above are planning approximations; actual core usage may vary by ±20% depending on routing overhead and sPyNNaker's placement heuristics. The key invariant is that **FC layers with >1M weights cannot fit on a single core** and will be spread across dozens of cores automatically.

**Mitigation for SpiNN-3:**
1. Reduce hidden dims: 512→256, 1024→512. Cuts weight count by ~4×.
2. Use weight sharing or convolutional compression.
3. Run only one backbone at a time (not all three simultaneously).

### Revised Core Estimates

| Configuration | Cores Needed | Fits SpiNN-3? | Fits SpiNN-5? |
|--------------|-------------|---------------|---------------|
| Full M7 (512-d) | ~600 | ❌ No | ✅ Yes |
| Compressed M7 (256-d) | ~150 | ⚠️ Tight | ✅ Yes |
| One backbone at a time | ~300 | ❌ No | ✅ Yes |
| Phase 1 only (visual) | ~280 | ❌ No | ✅ Yes |

**Conclusion:** SpiNN-5 is required for the full model. SpiNN-3 is insufficient without aggressive compression.

---

## 9. Final Recommendations

### If SpiNN-5 is available:
Proceed with Phase 1 as planned. Full model deployment is feasible. Target Phase 2 for the paper contribution (host-mediated continual learning on neuromorphic hardware).

### If only SpiNN-3 is available:
1. Compress the model: 512→256 hidden dims, 1024→512.
2. Test accuracy drop in PyTorch first. If <2%, proceed.
3. Deploy compressed model. Expect ~150 cores usage.
4. Focus on single-task inference (Phase 1) rather than full continual learning.

### If no hardware is available:
Use sPyNNaker's software simulator backend. This runs on the laptop CPU (not GPU) and simulates SpiNNaker's behavior. Slower than real hardware but sufficient for algorithmic validation. The paper can claim "validated on sPyNNaker simulator, ready for hardware deployment."

---

## 10. Immediate Next Steps

1. **Install sPyNNaker** in a fresh venv and run the "hello world" spiking network.
2. **Confirm hardware access** — which SpiNNaker board (if any) is available?
3. **Build a single-layer test** — map FC(512→10) with trained weights, validate accuracy.
4. **Test without LayerNorm** — in PyTorch, remove all LayerNorm and retrain. Measure accuracy drop. This determines if LayerNorm omission on SpiNNaker is acceptable.

---

*End of SpiNNaker Port Plan v1.0*

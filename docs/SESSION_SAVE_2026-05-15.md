# Session Save: M7 → SpiNNaker Porting (2026-05-15)

> Saved before computer restart. Resume from here.

---

## ✅ What Was Accomplished in This Session

### 1. sPyNNaker Installed and Validated
- **Package**: `sPyNNaker==7.4.1` + `PyNN==0.12.4` + all dependencies
- **Config**: `~/.spynnaker.cfg` set to `virtual_board=True`, `width=8`, `height=8`
- **Venv**: `/Users/agent1/Documents/neuromorphic/neurips/.venv` (Python 3.11)

### 2. Hello World Test Passed
- 50 Poisson sources → 100 LIF neurons mapped successfully
- Toolchain, placement, and routing all working

### 3. Real Trained Weights Mapped (FC 512→10)
- Loaded `final_nmnist_nmnist_replay.pt`
- Extracted N-MNIST head: `heads.nmnist.1` (Linear 512→10)
- Mapped 4,049 synapses (after thresholding |w| > 0.001) to virtual SpiNNaker
- ✅ Placement SUCCESS | ✅ Routing SUCCESS

### 4. Stress Test: FC(1600→1024) — CRITICAL FINDING
**Layer**: M7 visual backbone FC1 — 1,600 inputs → 1,024 outputs = **1,638,400 synapses**

| Test | Connector | Neurons/Core | Result |
|------|-----------|--------------|--------|
| #1 | `FromListConnector` (custom weights) | 256 | ❌ FAILED — `SynapseRowTooBigException` |
| #2 | `AllToAllConnector` (uniform weight) | 128 | ✅ PASSED — mapped cleanly |

**Root cause**: `FromListConnector` does NOT allow the splitter to partition the target population, so it tries to put all 1,024 post-neurons on one core → synaptic row of 1,024 entries exceeds the 256 hard limit.

**Implication**: Loading custom trained weights for large layers requires **manual chunking** into multiple projections (e.g., 8 chunks of 128 neurons each).

### 5. Homeostatic Plasticity Question Answered
**Answer: NO — M7 does not implement homeostatic plasticity.**

What's present:
- LayerNorm (deep learning normalization, not biological homeostasis)
- L2 normalization
- Fixed LIF β=0.9 (no adaptive thresholds)

What's absent:
- Synaptic scaling
- Intrinsic plasticity
- Metaplasticity
- Firing rate homeostasis

**Note**: If Phase 3 (on-chip STDP) is attempted, homeostatic mechanisms may be needed to prevent runaway excitation.

---

## 🔧 Current Environment State

```bash
# Venv location
/Users/agent1/Documents/neuromorphic/neurips/.venv

# Key installed packages
sPyNNaker==7.4.1
PyNN==0.12.4
SpiNNFrontEndCommon==7.4.1
SpiNNaker_PACMAN==7.4.1

# sPyNNaker config
~/.spynnaker.cfg
  virtual_board = True
  width = 8
  height = 8

# Test scripts created
test_spinnaker_hello.py              # Basic toolchain validation
test_spinnaker_fc512to10.py          # Real trained weights (small layer)
test_spinnaker_fc1600to1024.py       # FromListConnector stress test (FAILED)
test_spinnaker_fc1600to1024_alltoall.py  # AllToAllConnector stress test (PASSED)
```

---

## 📋 Open Decisions / Next Steps (Ready to Resume)

### Option A: Chunked FromListConnector Test (HIGH PRIORITY)
Rewrite the FC1 test to split the 1,024 output neurons into 8 chunks of 128, each with its own `FromListConnector` projection. Verify trained weights can be loaded for large layers.

**Value**: Unblocks the entire weight-loading strategy for the M7 port.
**Effort**: ~15 minutes.

### Option B: Full Weight Exporter Script
Build a systematic exporter that:
1. Loads a full M7 checkpoint
2. Extracts all layer weights
3. Converts to sPyNNaker-compatible format
4. Handles chunking automatically for layers > 256 outputs

**Value**: Complete pipeline for Phase 1 deployment.
**Effort**: ~30 minutes.

### Option C: Fix Known PyTorch Bugs
- Fix `_get_features` returning `(B, 1, 512)` instead of `(B, 512)`
- Remove unused backbone heads (`fc3`, `fc4`)

**Value**: Cleaner code before porting.
**Effort**: ~10 minutes.

### Option D: LayerNorm Ablation Test
Retrain M7 without LayerNorm and measure accuracy drop.

**Value**: Determines if LayerNorm omission on SpiNNaker is acceptable.
**Effort**: ~4–5 hours.

### Option E: Homeostatic Plasticity Research
Investigate adding synaptic scaling or threshold adaptation to M7 for on-chip STDP stability.

**Value**: Makes Phase 3 more feasible.
**Effort**: Research + implementation, 1–2 days.

---

## 🚨 Key Risk Update (from this session)

**Original Plan Risk R5**: "Memory overflow per core" — FC layers have millions of weights that may not fit in 32KB DTCM.

**Update**: This risk is MITIGATED by sPyNNaker's auto-partitioning. The 1.6M-synapse FC1 layer mapped successfully with `AllToAllConnector` + `set_number_of_neurons_per_core=128`. The toolchain handles large layers across multiple cores automatically.

**NEW RISK**: `FromListConnector` (custom weights) cannot be auto-partitioned. Manual chunking is required. This adds complexity to the weight-loading pipeline but is solvable.

---

## 📁 Files Modified/Created This Session

| File | Action | Purpose |
|------|--------|---------|
| `SpiNNaker_Port_Plan.md` | Created | Full 10-week roadmap with risk analysis |
| `~/.spynnaker.cfg` | Created + edited | Virtual board configuration |
| `test_spinnaker_hello.py` | Created | Toolchain smoke test |
| `test_spinnaker_fc512to10.py` | Created | Real weights on small layer |
| `test_spinnaker_fc1600to1024.py` | Created | FromListConnector stress test |
| `test_spinnaker_fc1600to1024_alltoall.py` | Created | AllToAllConnector stress test |
| `SESSION_SAVE_2026-05-15.md` | Created | This file |

---

## 🔄 How to Resume

After restarting:
1. Re-activate venv: `source /Users/agent1/Documents/neuromorphic/neurips/.venv/bin/activate`
2. Verify sPyNNaker: `python -c "import pyNN.spiNNaker; print('OK')"`
3. Check config: `cat ~/.spynnaker.cfg` (should still have virtual_board=True)
4. Read this file: `cat SESSION_SAVE_2026-05-15.md`
5. Pick next step from "Open Decisions" above

---

*Session saved: 2026-05-15 | Ready to resume after restart*

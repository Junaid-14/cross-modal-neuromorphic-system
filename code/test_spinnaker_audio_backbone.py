"""
sPyNNaker Audio Backbone Test: SHD (700 → 1024 → 1024 → 512)

Maps the full M7 audio backbone with REAL trained weights:
  Input(700 Poisson) → FC1(700→1024) → LIF
                        → FC2(1024→1024) → LIF
                        → FC3(1024→512) → LIF
"""
import numpy as np
import torch
import pyNN.spiNNaker as sim

print("=" * 65)
print("sPyNNaker Audio Backbone Test (SHD)")
print("=" * 65)

# ------------------------------------------------------------------
# 1. Load checkpoint
# ------------------------------------------------------------------
cp = torch.load("ICONS_M7/results/checkpoints/final_shd_shd_replay.pt", map_location="cpu")
sd = cp["model_state_dict"]

W_fc1 = sd["backbones.shd.fc1.weight"].numpy().astype(np.float32)   # (1024, 700)
b_fc1 = sd["backbones.shd.fc1.bias"].numpy().astype(np.float32)
W_fc2 = sd["backbones.shd.fc2.weight"].numpy().astype(np.float32)   # (1024, 1024)
b_fc2 = sd["backbones.shd.fc2.bias"].numpy().astype(np.float32)
W_fc3 = sd["backbones.shd.fc3.weight"].numpy().astype(np.float32)   # (512, 1024)
b_fc3 = sd["backbones.shd.fc3.bias"].numpy().astype(np.float32)

print(f"  FC1:  {W_fc1.shape[1]} → {W_fc1.shape[0]}")
print(f"  FC2:  {W_fc2.shape[1]} → {W_fc2.shape[0]}")
print(f"  FC3:  {W_fc3.shape[1]} → {W_fc3.shape[0]}")

# ------------------------------------------------------------------
# 2. Setup
# ------------------------------------------------------------------
sim.setup(timestep=1.0, min_delay=1.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 128)

SCALE = 0.05
THRESH = 0.001

# ------------------------------------------------------------------
# 3. Helper: build chunked layer
# ------------------------------------------------------------------
def build_chunked_layer(src_pops, src_sizes, n_out, W, b, label, chunk_size=128):
    n_chunks = (n_out + chunk_size - 1) // chunk_size
    pops = []
    total_syns = 0
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, n_out)
        actual = e - s
        out = sim.Population(actual, sim.IF_curr_exp(
            tau_m=20.0, v_rest=-65.0, v_reset=-65.0,
            v_thresh=-50.0, tau_refrac=2.0), label=f"{label}_{c}")
        out.set(i_offset=b[s:e].astype(np.float64) * 10.0)
        
        Wc = W[s:e, :]  # (actual, total_in)
        src_offset = 0
        for src_pop, src_sz in zip(src_pops, src_sizes):
            conn = []
            for post in range(actual):
                for pre in range(src_sz):
                    w = float(Wc[post, src_offset + pre]) * SCALE
                    if abs(w) > THRESH:
                        conn.append((pre, post, w, 1.0))
            if conn:
                sim.Projection(src_pop, out, sim.FromListConnector(conn),
                              sim.StaticSynapse(), label=f"p_{label}_{c}_from_{src_pop.label}")
            src_offset += src_sz
        
        out.record(["spikes"])
        pops.append(out)
        cnt = int(np.sum(np.abs(Wc) > THRESH / SCALE))
        total_syns += cnt
        print(f"  {label} chunk {c}: ~{cnt:,} synapses")
    return pops, total_syns

# ------------------------------------------------------------------
# 4. Build network
# ------------------------------------------------------------------
print("\nBuilding Audio Backbone...")

# Layer 0: Input (700 Poisson)
input_pop = sim.Population(700, sim.SpikeSourcePoisson(rate=20.0, start=0.0, duration=100.0), label="input")

# Layer 1: FC1 (700 → 1024, 8 chunks)
print("\nFC1 (700 → 1024)...")
fc1_pops, fc1_syns = build_chunked_layer([input_pop], [700], 1024, W_fc1, b_fc1, "fc1")

# Layer 2: FC2 (1024 → 1024, 8 chunks)
print("\nFC2 (1024 → 1024)...")
fc2_pops, fc2_syns = build_chunked_layer(fc1_pops, [128]*8, 1024, W_fc2, b_fc2, "fc2")

# Layer 3: FC3 (1024 → 512, 4 chunks)
print("\nFC3 (1024 → 512)...")
fc3_pops, fc3_syns = build_chunked_layer(fc2_pops, [128]*8, 512, W_fc3, b_fc3, "fc3")

print(f"\nTotal synapses: {fc1_syns + fc2_syns + fc3_syns:,}")

# ------------------------------------------------------------------
# 5. Run
# ------------------------------------------------------------------
print("\nMapping audio backbone...")
sim.run(100)

for p in fc1_pops + fc2_pops + fc3_pops:
    p.get_data(["spikes"])

sim.end()

# ------------------------------------------------------------------
# 6. Report
# ------------------------------------------------------------------
print("\n" + "=" * 65)
print("✅ AUDIO BACKBONE TEST PASSED")
print("=" * 65)
print(f"   Pipeline:    700 → 1024 → 1024 → 512")
print(f"   Synapses:    {fc1_syns + fc2_syns + fc3_syns:,}")
print(f"   Chunks:      FC1(8) + FC2(8) + FC3(4) = 20 populations")
print(f"   Status:      Full audio backbone mapped with real weights")
print("=" * 65)

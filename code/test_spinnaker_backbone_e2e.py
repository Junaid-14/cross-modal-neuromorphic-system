"""
sPyNNaker End-to-End Visual Backbone Test (FC Pipeline)

Maps the full M7 visual backbone from flattened conv features
to classification head, with REAL trained weights at every layer:

  Input(1600 Poisson) → FC1(1600→1024) → LIF
                         → FC2(1024→512) → LIF
                         → Head(512→10) → LIF
"""
import numpy as np
import torch
import pyNN.spiNNaker as sim

print("=" * 65)
print("sPyNNaker End-to-End Backbone Test")
print("=" * 65)

# ------------------------------------------------------------------
# 1. Load checkpoint
# ------------------------------------------------------------------
cp = torch.load("ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt", map_location="cpu")
sd = cp["model_state_dict"]

W_fc1 = sd["backbones.nmnist.fc1.weight"].numpy().astype(np.float32)  # (1024, 1600)
b_fc1 = sd["backbones.nmnist.fc1.bias"].numpy().astype(np.float32)
W_fc2 = sd["backbones.nmnist.fc2.weight"].numpy().astype(np.float32)  # (512, 1024)
b_fc2 = sd["backbones.nmnist.fc2.bias"].numpy().astype(np.float32)
W_head = sd["heads.nmnist.1.weight"].numpy().astype(np.float32)       # (10, 512)
b_head = sd["heads.nmnist.1.bias"].numpy().astype(np.float32)

print(f"  FC1:   {W_fc1.shape[1]} → {W_fc1.shape[0]}")
print(f"  FC2:   {W_fc2.shape[1]} → {W_fc2.shape[0]}")
print(f"  Head:  {W_head.shape[1]} → {W_head.shape[0]}")

# ------------------------------------------------------------------
# 2. Setup
# ------------------------------------------------------------------
sim.setup(timestep=1.0, min_delay=1.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 128)

SCALE = 0.05
THRESH = 0.001

# ------------------------------------------------------------------
# 3. Build FC1: 1600 → 1024 (8 chunks of 128)
# ------------------------------------------------------------------
print("\nBuilding FC1 (1600 → 1024)...")
input_pop = sim.Population(1600, sim.SpikeSourcePoisson(rate=20.0, start=0.0, duration=100.0), label="input")

fc1_pops = []
fc1_syns = 0
for c in range(8):
    s, e = c * 128, min((c + 1) * 128, 1024)
    actual = e - s
    out = sim.Population(actual, sim.IF_curr_exp(tau_m=20.0, v_rest=-65.0, v_reset=-65.0,
                                                  v_thresh=-50.0, tau_refrac=2.0), label=f"fc1_{c}")
    out.set(i_offset=b_fc1[s:e].astype(np.float64) * 10.0)
    
    Wc = W_fc1[s:e, :]  # (actual, 1600)
    conn = []
    for post in range(actual):
        for pre in range(1600):
            w = float(Wc[post, pre]) * SCALE
            if abs(w) > THRESH:
                conn.append((pre, post, w, 1.0))
    
    sim.Projection(input_pop, out, sim.FromListConnector(conn), sim.StaticSynapse(), label=f"p_fc1_{c}")
    out.record(["spikes"])
    fc1_pops.append(out)
    fc1_syns += len(conn)
    print(f"  FC1 chunk {c}: {len(conn):,} synapses")

# ------------------------------------------------------------------
# 4. Build FC2: 1024 → 512 (4 chunks of 128)
# ------------------------------------------------------------------
print("\nBuilding FC2 (1024 → 512)...")
fc2_pops = []
fc2_syns = 0
for c in range(4):
    s, e = c * 128, min((c + 1) * 128, 512)
    actual = e - s
    out = sim.Population(actual, sim.IF_curr_exp(tau_m=20.0, v_rest=-65.0, v_reset=-65.0,
                                                  v_thresh=-50.0, tau_refrac=2.0), label=f"fc2_{c}")
    out.set(i_offset=b_fc2[s:e].astype(np.float64) * 10.0)
    
    Wc = W_fc2[s:e, :]  # (actual, 1024)
    
    # Connect from each FC1 chunk
    for fc1_idx, fc1_pop in enumerate(fc1_pops):
        fc1_start = fc1_idx * 128
        fc1_end = min(fc1_start + 128, 1024)
        fc1_sz = fc1_end - fc1_start
        
        conn = []
        for post in range(actual):
            for pre in range(fc1_sz):
                w = float(Wc[post, fc1_start + pre]) * SCALE
                if abs(w) > THRESH:
                    conn.append((pre, post, w, 1.0))
        
        if conn:
            sim.Projection(fc1_pop, out, sim.FromListConnector(conn), sim.StaticSynapse(),
                          label=f"p_fc2_{c}_from_{fc1_idx}")
    
    out.record(["spikes"])
    fc2_pops.append(out)
    # Count from full weight matrix
    cnt = int(np.sum(np.abs(Wc) > THRESH / SCALE))
    fc2_syns += cnt
    print(f"  FC2 chunk {c}: ~{cnt:,} synapses")

# ------------------------------------------------------------------
# 5. Build Head: 512 → 10
# ------------------------------------------------------------------
print("\nBuilding Head (512 → 10)...")
head_pop = sim.Population(10, sim.IF_curr_exp(tau_m=20.0, v_rest=-65.0, v_reset=-65.0,
                                               v_thresh=-50.0, tau_refrac=2.0), label="head")
head_pop.set(i_offset=b_head.astype(np.float64) * 10.0)

head_syns = 0
for fc2_idx, fc2_pop in enumerate(fc2_pops):
    fc2_start = fc2_idx * 128
    fc2_end = min(fc2_start + 128, 512)
    fc2_sz = fc2_end - fc2_start
    
    conn = []
    for post in range(10):
        for pre in range(fc2_sz):
            w = float(W_head[post, fc2_start + pre]) * SCALE
            if abs(w) > THRESH:
                conn.append((pre, post, w, 1.0))
    
    if conn:
        sim.Projection(fc2_pop, head_pop, sim.FromListConnector(conn), sim.StaticSynapse(),
                      label=f"p_head_from_{fc2_idx}")
        head_syns += len(conn)

head_pop.record(["spikes"])
print(f"  Head: {head_syns:,} synapses")

print(f"\nTotal synapses: {fc1_syns + fc2_syns + head_syns:,}")

# ------------------------------------------------------------------
# 6. Run mapping
# ------------------------------------------------------------------
print("\nMapping end-to-end backbone...")
sim.run(100)

for p in fc1_pops + fc2_pops + [head_pop]:
    p.get_data(["spikes"])

sim.end()

# ------------------------------------------------------------------
# 7. Report
# ------------------------------------------------------------------
print("\n" + "=" * 65)
print("✅ END-TO-END BACKBONE TEST PASSED")
print("=" * 65)
print(f"   Pipeline:    1600 → 1024 → 512 → 10")
print(f"   Synapses:    {fc1_syns + fc2_syns + head_syns:,}")
print(f"   Layers:      Input + FC1(8) + FC2(4) + Head")
print(f"   Status:      Full FC backbone mapped with real weights")
print("=" * 65)

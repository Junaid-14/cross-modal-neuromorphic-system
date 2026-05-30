"""
sPyNNaker Chunked FromListConnector Test: FC(1600→1024)

Splits the 1,024 output neurons into 8 chunks of 128.
Each chunk gets its own FromListConnector projection with
a subset of the trained weights. This stays under the 256
synapse-per-row limit that blocked the single-projection approach.
"""
import numpy as np
import torch
import pyNN.spiNNaker as sim

print("=" * 65)
print("sPyNNaker CHUNKED FromListConnector Test: FC(1600→1024)")
print("=" * 65)

# ------------------------------------------------------------------
# 1. Load checkpoint & extract FC1 weights
# ------------------------------------------------------------------
checkpoint_path = "ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt"
print(f"\nLoading checkpoint: {checkpoint_path}")
cp = torch.load(checkpoint_path, map_location="cpu")
sd = cp["model_state_dict"]

W_torch = sd["backbones.nmnist.fc1.weight"]  # shape (1024, 1600)
b_torch = sd["backbones.nmnist.fc1.bias"]    # shape (1024,)

W = W_torch.numpy().astype(np.float32)
b = b_torch.numpy().astype(np.float32)

n_out, n_in = W.shape
chunk_size = 128
n_chunks = n_out // chunk_size  # 1024 / 128 = 8

print(f"  Weight matrix:  {n_out} × {n_in} = {W.size:,} parameters")
print(f"  Chunking:       {n_chunks} chunks × {chunk_size} neurons")

# ------------------------------------------------------------------
# 2. Build sPyNNaker network with chunked projections
# ------------------------------------------------------------------
sim.setup(timestep=1.0, min_delay=1.0)

# Limit neurons per core to ensure rows stay small
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, chunk_size)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, chunk_size)

# Single input population
input_pop = sim.Population(n_in, sim.SpikeSourcePoisson(
    rate=20.0, start=0.0, duration=100.0
), label="input_1600")

# Create 8 output chunks as separate populations
output_pops = []
projections = []
scale = 0.05

for chunk_idx in range(n_chunks):
    start_neuron = chunk_idx * chunk_size
    end_neuron = start_neuron + chunk_size
    
    # Extract weights for this chunk: (128, 1600)
    W_chunk = W[start_neuron:end_neuron, :]  # shape (128, 1600)
    b_chunk = b[start_neuron:end_neuron]      # shape (128,)
    
    # Create connection list for this chunk
    # pre: 0..1599, post: 0..127 (local to this chunk)
    conn_list = []
    for post_local in range(chunk_size):
        for pre in range(n_in):
            w = float(W_chunk[post_local, pre]) * scale
            if abs(w) > 0.001:
                conn_list.append((pre, post_local, w, 1.0))
    
    # Create output population for this chunk
    out_pop = sim.Population(chunk_size, sim.IF_curr_exp(
        tau_m=20.0, v_rest=-65.0, v_reset=-65.0,
        v_thresh=-50.0, tau_refrac=2.0, i_offset=0.0
    ), label=f"output_chunk_{chunk_idx}")
    
    # Apply biases
    biases = b_chunk.astype(np.float64) * 10.0
    out_pop.set(i_offset=biases)
    
    # Create projection
    proj = sim.Projection(
        input_pop, out_pop,
        sim.FromListConnector(conn_list),
        sim.StaticSynapse(),
        label=f"fc_chunk_{chunk_idx}"
    )
    
    out_pop.record(["spikes"])
    output_pops.append(out_pop)
    projections.append((proj, len(conn_list)))
    
    print(f"  Chunk {chunk_idx}: {len(conn_list):,} synapses → {chunk_size} neurons")

total_synapses = sum(count for _, count in projections)
print(f"\nTotal synapses across all chunks: {total_synapses:,}")

# ------------------------------------------------------------------
# 3. Run mapping
# ------------------------------------------------------------------
print("\nMapping chunked FC(1600→1024) to virtual SpiNNaker...")
print("(Each chunk has its own projection — should avoid row limit)")

sim.run(100)

for pop in output_pops:
    pop.get_data(["spikes"])

sim.end()

# ------------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------------
print("\n" + "=" * 65)
print("✅ CHUNKED FromListConnector TEST PASSED")
print("=" * 65)
print(f"   Layer:          {n_in} → {n_out} (M7 visual backbone FC1)")
print(f"   Chunks:         {n_chunks} × {chunk_size} neurons")
print(f"   Synapses:       {total_synapses:,} (with |w| > 0.001 threshold)")
print(f"   Mapping:        SUCCESS — no row-size overflow")
print(f"   Status:         Custom trained weights mapped successfully")
print("=" * 65)
print("\nIMPLICATION: We CAN load real M7 weights into sPyNNaker!")
print("             Large layers just need manual chunking.")
print("=" * 65)

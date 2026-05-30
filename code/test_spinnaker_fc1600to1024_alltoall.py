"""
sPyNNaker Stress Test: FC(1600→1024) using AllToAllConnector

Tests whether the 1.6M synapse layer can be mapped at ALL,
using the standard AllToAllConnector (which may handle 
splitting better than FromListConnector).
"""
import numpy as np
import pyNN.spiNNaker as sim

print("=" * 65)
print("sPyNNaker STRESS TEST: FC(1600→1024) with AllToAllConnector")
print("=" * 65)

n_in = 1600
n_out = 1024

print(f"\nLayer: {n_in} → {n_out}")
print(f"Synapses: {n_in * n_out:,}")

sim.setup(timestep=1.0, min_delay=1.0)

# Aggressively limit neurons per core to keep synaptic rows under 256
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 128)

input_pop = sim.Population(n_in, sim.SpikeSourcePoisson(
    rate=20.0, start=0.0, duration=100.0
), label=f"input_{n_in}")

output_pop = sim.Population(n_out, sim.IF_curr_exp(
    tau_m=20.0, v_rest=-65.0, v_reset=-65.0,
    v_thresh=-50.0, tau_refrac=2.0, i_offset=0.0
), label=f"output_{n_out}")

# AllToAllConnector with uniform weight
proj = sim.Projection(
    input_pop, output_pop,
    sim.AllToAllConnector(),
    sim.StaticSynapse(weight=0.03, delay=1.0),
    label="fc1600to1024"
)

output_pop.record(["spikes"])

print(f"\nMapping {n_in * n_out:,}-synapse layer...")
sim.run(100)
sim.end()

print("\n" + "=" * 65)
print("✅ FC(1600→1024) MAPPED SUCCESSFULLY with AllToAllConnector")
print("=" * 65)
print(f"   Synapses:  {n_in * n_out:,}")
print(f"   Status:    Placement & routing succeeded")
print("=" * 65)

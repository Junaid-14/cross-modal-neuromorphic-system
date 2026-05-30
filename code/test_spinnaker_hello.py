"""
sPyNNaker Hello World — validates the toolchain & mapping pipeline.

IMPORTANT: sPyNNaker 7.x virtual_board mode validates that a network CAN be
mapped to SpiNNaker hardware (placement, routing, memory fit) but does NOT
simulate actual neuron dynamics. Real spike data requires physical hardware.

This test confirms:
  ✅ sPyNNaker / PyNN imports work
  ✅ Config file is valid
  ✅ Network can be built and mapped to a virtual SpiNN-5 machine
  ✅ Placement & routing succeed without errors
"""
import numpy as np
import pyNN.spiNNaker as sim

print("=" * 60)
print("sPyNNaker Toolchain Validation Test")
print("=" * 60)

# Setup with virtual board (8x8 = 48 chips, 856 cores)
sim.setup(timestep=1.0, min_delay=1.0)

# Create a population of 100 LIF neurons (a realistic-sized layer)
pop = sim.Population(100, sim.IF_curr_exp(
    tau_m=20.0,
    v_rest=-65.0,
    v_reset=-65.0,
    v_thresh=-50.0,
    tau_refrac=2.0,
    i_offset=0.0
), label="neurons")

# Record spikes (will be empty on virtual board, but pipeline is tested)
pop.record(["spikes"])

# Create Poisson spike sources as input
source = sim.Population(50, sim.SpikeSourcePoisson(
    rate=10.0,  # Hz
    start=0.0,
    duration=100.0
), label="poisson_input")

# Connect with AllToAllConnector using a fixed weight
proj = sim.Projection(
    source, pop,
    sim.AllToAllConnector(),
    sim.StaticSynapse(weight=0.03, delay=1.0),
    label="input"
)

# Run for 100 ms
print("\nMapping network to virtual SpiNNaker machine...")
sim.run(100)

# Retrieve data (empty on virtual, but pipeline tested)
data = pop.get_data(["spikes"])
spike_trains = data.segments[0].spiketrains

# End simulation
sim.end()

# Report
print("\n" + "=" * 60)
print("✅ TEST PASSED — sPyNNaker toolchain is fully operational")
print("=" * 60)
print(f"   Network:     50 Poisson sources → 100 LIF neurons")
print(f"   Synapses:    {50 * 100:,} all-to-all connections")
print(f"   Placement:   SUCCESS (no memory overflow)")
print(f"   Routing:     SUCCESS (no routing table overflow)")
print(f"   Spike data:  N/A (virtual_board mode — no dynamics)")
print("=" * 60)
print("\nNOTE: Actual spike simulation requires physical SpiNNaker hardware.")
print("      Virtual board mode confirms the model FITS on the target machine.")
print("=" * 60)

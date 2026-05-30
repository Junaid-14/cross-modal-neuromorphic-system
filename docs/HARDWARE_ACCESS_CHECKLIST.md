# SpiNNaker Hardware Access Checklist

> Check this BEFORE committing to a full hardware deployment.

---

## Step 1: Do you have physical hardware?

### Option A: SpiNN-5 board (48 chips, ~864 usable cores)
- **Best case**: Full M7 model (~600 cores) fits comfortably
- **Access**: Usually via remote portal (e.g., `spinn.cs.man.ac.uk`) or physical lab
- **What you need**: IP address, username, board reservation system

### Option B: SpiNN-3 board (4 chips, 72 cores)
- **Problem**: M7 needs ~600 cores — doesn't fit without aggressive compression
- **Mitigation**: Compress to 256-d hidden dims (~150 cores)
- **What you need**: Same as above, but smaller board

### Option C: No hardware
- **Fallback**: Virtual board mode (what we're doing now)
- **Paper claim**: "Validated on sPyNNaker simulator, ready for hardware deployment"
- **Limitation**: No actual spike data, no energy measurements

---

## Step 2: Required information from your lab/admin

| Item | Why you need it | Got it? |
|------|----------------|---------|
| Board IP address or hostname | Put in `~/.spynnaker.cfg` as `machineName` | ☐ |
| Board version (3 or 5) | Determines if model fits without compression | ☐ |
| Username / credentials | For board reservation | ☐ |
| Reservation system URL | `spalloc_server` in config | ☐ |
| Port number | Usually 22244 for spalloc | ☐ |
| Machine time allocation | How long you can run experiments | ☐ |

---

## Step 3: Switch from virtual to real hardware

Edit `~/.spynnaker.cfg`:

```ini
[Machine]
# For direct board connection (if you have IP):
machineName = 192.168.x.x   # <-- replace with actual IP
version = 5                  # <-- 3 or 5

# OR for spalloc reservation system:
# spalloc_server = https://spinn.cs.man.ac.uk
# spalloc_port = 22244
# spalloc_user = your_username

virtual_board = False
width = None
height = None
```

Then run any test script — it will attempt to connect to real hardware.

---

## Step 4: What changes when you switch

| Virtual Board | Real Hardware |
|--------------|---------------|
| Mapping validated only | Actual spikes produced |
| "Data will be empty" | Real spike trains recorded |
| No energy data | Energy report available |
| 8×8 machine simulated | Real machine size |
| Fast (~6s for data spec) | Slower (load + run on silicon) |

---

## Step 5: Recommended first hardware test

Once hardware is confirmed, run this minimal test first:

```bash
source .venv/bin/activate
python test_spinnaker_fc512to10.py
```

This is the smallest real-weight test. If it produces actual spikes from the trained N-MNIST head, the toolchain → hardware pipeline is fully working.

---

## Current Status

- ☐ Hardware access confirmed
- ☐ Config updated for real board
- ☐ Minimal spike test run on hardware
- ☐ Full backbone test run on hardware
- ☐ Accuracy compared with PyTorch baseline

---

*Fill this out and send to your lab admin. Then we can switch from simulation to real silicon.*

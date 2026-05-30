#!/usr/bin/env python3
"""Monitor train.log and kill training when val accuracy plateaus at >= 99%."""

import subprocess
import time
import re
import sys

LOG_FILE = "/Users/agent1/Documents/neuromorphic/neurips/train.log"
TARGET_ACC = 99.0
MIN_DELTA = 0.05  # stop if change between epochs is < 0.05%
PID = 16672

def get_latest_val_accuracies():
    """Parse the log file and return list of val accuracies in order."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    
    val_accs = []
    for line in lines:
        match = re.search(r'Val:\s+(\d+\.\d+)%', line)
        if match:
            val_accs.append(float(match.group(1)))
    return val_accs

def main():
    print(f"Monitoring {LOG_FILE} for Val >= {TARGET_ACC}% with plateau detection...")
    print(f"Will kill PID {PID} when plateau at >= {TARGET_ACC}% (delta < {MIN_DELTA}%).")
    
    while True:
        val_accs = get_latest_val_accuracies()
        
        if len(val_accs) >= 2:
            last = val_accs[-1]
            prev = val_accs[-2]
            delta = abs(last - prev)
            
            if last >= TARGET_ACC and delta < MIN_DELTA:
                print(f"\n✅ Plateau detected! Last 2 val accuracies: {prev:.2f}% → {last:.2f}% (delta={delta:.3f}%)")
                print(f"Killing training process (PID {PID})...")
                subprocess.run(["kill", "-15", str(PID)])
                time.sleep(2)
                result = subprocess.run(["ps", "-p", str(PID)], capture_output=True)
                if result.returncode == 0:
                    subprocess.run(["kill", "-9", str(PID)])
                    print("Force killed.")
                else:
                    print("Gracefully terminated.")
                sys.exit(0)
        
        if val_accs:
            status = f"  Latest val: {val_accs[-3:]}" if len(val_accs) >= 3 else f"  Latest val: {val_accs}"
            print(status, end="\r", flush=True)
        
        time.sleep(3)

if __name__ == "__main__":
    main()

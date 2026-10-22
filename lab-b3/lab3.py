#!/usr/bin/env python3
"""Lab B3: chained job array, periodic checkpoints only (no signal handling).

Counts from 0 to 600, one step per second (sleep = stand-in for real work).
Saves progress to $CKPT_DIR/state.json every 30 steps. Each array task
resumes from that file. When the time limit kills a task, up to 29 steps of
work since the last checkpoint are lost; that loss is the point of the lab.
"""
import json
import os
import time

CKPT_DIR = os.environ.get("CKPT_DIR", ".")
CK = os.path.join(CKPT_DIR, "state.json")
TOTAL = 600   # total "work" in steps (~10 minutes)
EVERY = 30    # checkpoint interval in steps

i = 0
if os.path.exists(CK):
    with open(CK) as f:
        i = json.load(f)["i"]
print(f"Resuming at i={i} (array task {os.environ.get('SLURM_ARRAY_TASK_ID', '-')})",
      flush=True)

while i < TOTAL:
    time.sleep(1)          # one unit of "work"
    i += 1
    if i % EVERY == 0:
        tmp = CK + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"i": i}, f)
        os.replace(tmp, CK)    # atomic: never leaves a half-written state file
        print(f"checkpoint i={i}", flush=True)

print(f"DONE (i={i})", flush=True)

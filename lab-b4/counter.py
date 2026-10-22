
import json, os, signal, time
CK, STOP = "state.json", False

def on_usr1(signum, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGUSR1, on_usr1)

i = 0
if os.path.exists(CK):
    with open(CK) as f:
        i = json.load(f)["i"]
print(f"Resuming at i={i}, restart count={os.environ.get('SLURM_RESTART_COUNT', 0)}", flush=True)

while i < 600:
    time.sleep(1)
    i += 1
    if i % 30 == 0 or STOP:
        with open(CK + ".tmp", "w") as f:
            json.dump({"i": i}, f)
        os.replace(CK + ".tmp", CK)
        print(f"checkpoint i={i}", flush=True)
        if STOP:
            break
print("DONE" if i >= 600 else "stopped for requeue", flush=True)

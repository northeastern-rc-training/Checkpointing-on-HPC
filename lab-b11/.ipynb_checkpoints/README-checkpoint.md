# Lab: threaded holes on a CMM, a four-part study with checkpointing

A Jupyter study of real CMM data: one part with four M3x0.5 threaded holes,
26,098 points, three tips, touch trigger (TT) and scanning, CW and CCW, and
7 to 100 hits. A calibrated **thread locator** in each hole gives the true
centre of the threaded hole. The study answers four questions and, along the
way, teaches checkpointing in OnDemand and in batch.

| Part | Question |
|---|---|
| A. Process | Are the threads consistent in size and form, and is the thread aligned with the surface the probe touches? |
| B. CNC | Are the holes where they should be? |
| C. Defects in parts and tooling | Is any thread lost or damaged, and what defect could each setup catch? |
| D. CMM and measuring procedure | Which tip, mode, direction and number of hits to use? |

## Files

| File | Purpose |
|---|---|
| `thread_study.ipynb` | The study (Parts A–D) |
| `cmm_study.py` | Fits, thread-locator reference, categorisation, sweep, 3D figures |
| `cmm_twin.py` | Geometry, digital twin, defect simulation and detection |
| `data/02_CMM_Threads_Points_Analysis-Data_Science.csv` | Raw CMM export |
| `run_study.sbatch` | The same notebook, headless, as a chained job array |
| `requirements.txt` | Python packages |

## Setup (once)

```bash
module load explorer anaconda3/2024.06
conda create --name twin_env -c conda-forge python=3.12 -y      # with the pkgs_dirs fix
source activate twin_env
pip install --no-cache-dir -r requirements.txt
python -m ipykernel install --user --name twin_env --display-name "Python (twin_env)"
```

## Presets and timing

Set the preset with `TWIN_PRESET` (default `demo`):

| Preset | Twin training | Detection sweep | Run time on 1–2 CPU cores |
|---|---|---|---|
| `quick` | measured points only, 5 folds × 30 epochs | 912 simulated defects | about 20 seconds |
| `demo` | + 24 augmented holes, 5 folds × 100 epochs | 4,788 simulated defects | about 14 minutes |

Parts A, B and D, and the data checks in C1, need no training and appear
within seconds. Checkpoints and figures go to
`/scratch/$USER/thread_study_<preset>`, or to `./study_ckpt_<preset>` if
`/scratch` doesn't exist. Override this with `TWIN_CKPT_DIR`. To start over,
delete that directory.

## Running the lab

**OnDemand (Lab B11)**
1. Start a Jupyter session of about 15 minutes with the `twin_env` kernel and choose **Run All**.
2. During the twin training or the sweep, delete the session or restart the kernel.
3. Start a new session and **Run All** again. You should see:
   - Stage 1 loads from cache.
   - The training prints `resumed at epoch … (validation check OK …)`.
   - The sweep skips the runs that are already done.

**Batch (graduation)**
```bash
mkdir -p logs
sbatch run_study.sbatch      # 5 tasks of 8 minutes, one at a time; stops once STATUS=DONE
```
OnDemand and batch share the same default checkpoint directory, so a study
started interactively can be finished by the batch job.

## Checkpoints

| Stage | Checkpoint | On rerun |
|---|---|---|
| Data, fits, locator reference, categories | `cache/` (`joblib.Memory`) | Loads in about a second |
| Digital twin | `twin_fold{0-4}.joblib`: model + optimizer state, every epoch, atomic | Resumes and verifies the restored model |
| Detection sweep | `sweep/run_NNNN.json`, one per measured run | Skips finished runs |
| Outputs | `points_categorized.csv`, `figures/*.html`, `STATUS` | Rebuilt from saved files |

## Findings from this part (from the test run)

**A. Process**
- **Size.** The crest radius is 25–27 µm over nominal on every hole, so the minor diameter is about 0.05 mm oversize. Ovality and trilobing are 0.5–3 µm.
- **Alignment.** The centre found by probing the thread is 26–45 µm from the thread-locator centre. That gap is the same for every tip, mode and hit count, so it's a property of the part. Two explanations fit:
  - drill-vs-tap misalignment: the ball rides on the crests, which the drill makes, while the locator seats on the flanks, which the tap makes;
  - a tilted thread axis, because the locator was measured about 22 mm above the thread.

**B. CNC**
- **Hole positions.** Holes 1 and 2 are within 4 and 12 µm of nominal. Holes 3 and 4 are 80 and 96 µm off, in almost the same direction (4° apart), which points to a common cause such as the work offset, the fixture or drift.
- **Against tolerance.** Per axis, all holes are within ±0.127 mm. As a diametral position, hole 4 is at 0.19 mm.
- **Reference quality.** The three tips agree on the locator centres within 7 µm.

**C. Defects**
- **Point categories.** No inner-thread point is out of tolerance: 19,692 points are OK (under 50% of tolerance), 6,270 are WATCH (50–80%) and 40 are NEAR_LIMIT. Hole 4 uses the most tolerance because of its position.
- **Local defects.** None: the largest deviation from a run's own smooth form is 3.7σ, and there are no coverage gaps.
- **Detection limits from simulated defects:**
  - Scanning catches a 20 µm lost flank 89–100% of the time at every setting.
  - TT needs 50 hits to catch a 40 µm lost flank reliably.
  - TT with 7–10 hits misses missing-data stretches up to 60°.
  - TT with 100 hits raises false alarms on 12% of healthy runs.

**D. Measuring procedure**
- **Centre.** Every setup misses the locator centre by 30–39 µm, and more hits don't help. For position, measure the thread locator.
- **Helix handedness.** The measuring programs used both handednesses. Right-hand programs (TT CW, SCAN CCW) read smoother threads, with an ovality term of 1.5 vs. 8.9 µm for scanning. That points to a right-hand thread, so the helix should be programmed to match.
- **Mode.** Scanning is quieter (4.3 vs. 5.5 µm) and records about 5 points per "hit" setting. That density is what makes local defects detectable.
- **Tip.** The 1.5 × 20 mm and 2 × 30 mm tips agree best with the locator; the 1.5 × 30 mm tip is about 4 µm worse.

**Limits of this data**
- It covers one part, and the machining order isn't known, so process capability and tool-wear trends need more parts.
- The locator export is 8 points at one height, so its axis (tilt or wobble) can't be computed. Measure it as a cylinder or cone over two or more heights, or export its axis vector.

## Tested

Testing used scikit-learn 1.8 and pandas 3.0.
- **End-to-end runs.** The notebook ran end to end with both presets.
- **Repeated interruption.** It was interrupted 12 times (`quick`) and 3 times (`demo`). The models and every sweep file were bit-for-bit identical to an uninterrupted run.
- **Not tested.** The plotly figures, the OnDemand kernel and the `nbconvert` batch path weren't available in the test environment, so check those on the cluster first.

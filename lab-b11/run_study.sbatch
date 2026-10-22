#!/bin/bash
# Run the four-part thread study notebook headless as a chained job array.
# Each task executes the notebook until its time is nearly up; the notebook
# saves progress and the next task continues. Tasks after completion exit at once.
# Before submitting: mkdir -p logs
#SBATCH --job-name=thread-study
#SBATCH --array=0-4%1
#SBATCH --time=00:08:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%A_%a.out
##SBATCH --partition=<cpu-partition>      # uncomment and adjust if needed

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load explorer anaconda3/2024.06     # adjust
source activate twin_env                   # environment from README.md

export TWIN_PRESET="${TWIN_PRESET:-demo}"
# Same directory the notebook uses by default in an OnDemand session, so
# interactive and batch runs share (and continue) the same checkpoints.
export TWIN_CKPT_DIR="${TWIN_CKPT_DIR:-/scratch/$USER/thread_study_$TWIN_PRESET}"
export TWIN_SAFETY_SECONDS=90              # stop this long before the time limit

echo "=== $(date) task=$SLURM_ARRAY_TASK_ID preset=$TWIN_PRESET ckpt=$TWIN_CKPT_DIR"
if grep -qs DONE "$TWIN_CKPT_DIR/STATUS"; then
    echo "Study already complete; nothing to do."
    exit 0
fi

mkdir -p executed
jupyter nbconvert --to notebook --execute thread_study.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed \
    --output "thread_study_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.ipynb"
RC=$?
echo "=== $(date) nbconvert exit code $RC, STATUS: $(cat "$TWIN_CKPT_DIR/STATUS" 2>/dev/null)"
exit $RC

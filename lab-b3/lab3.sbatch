#!/bin/bash
# Lab B3: 5 array tasks, one at a time (%1), each limited to 3 minutes.
# Expected: tasks 0-3 end in TIMEOUT after advancing the count by ~150 each;
# the count reaches 600 during task 3; task 4 starts, sees i=600, prints DONE at once.
# Before submitting: mkdir -p logs
#SBATCH --job-name=lab3-array
#SBATCH --array=0-4%1
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --partition=short

cd "$SLURM_SUBMIT_DIR"
export CKPT_DIR="${CKPT_DIR:-/scratch/$USER/lab3}"
mkdir -p "$CKPT_DIR"

echo "=== $(date) array_job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
python -u lab3.py

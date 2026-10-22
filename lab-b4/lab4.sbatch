#!/bin/bash
#SBATCH --job-name=restartable
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300        # signal the batch shell 300 s before the limit
#SBATCH --requeue
#SBATCH --open-mode=append          # keep logs across restarts
#SBATCH --output=%x-%j.out

module load miniforge3/26.7.2-0
source activate pytorch_env

CKPT_DIR=/scratch/$USER/$SLURM_JOB_NAME
mkdir -p "$CKPT_DIR"
echo "Start: restart count = ${SLURM_RESTART_COUNT:-0}"

handler() {
    echo "USR1 received at $(date); asking app to checkpoint"
    kill -USR1 "$APP_PID"
    wait "$APP_PID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap handler USR1


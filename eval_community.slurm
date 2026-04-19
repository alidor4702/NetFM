#!/bin/bash
#SBATCH --job-name=netfm_comm
#SBATCH --output=logs/comm_%j.out
#SBATCH --error=logs/comm_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --cpus-per-task=8
#SBATCH --propagate=NONE

mkdir -p logs outputs/testing

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python3 -u -W ignore -m src.eval_community "$@"

echo "End: $(date)"

#!/bin/bash
#SBATCH --job-name=netfm_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --cpus-per-task=8
#SBATCH --propagate=NONE
#SBATCH --mail-user=ali.dor@student-cs.fr
#SBATCH --mail-type=END,FAIL

mkdir -p logs outputs/testing

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Start: $(date)"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python3 -u -W ignore -m src.evaluate "$@"

echo "End: $(date)"

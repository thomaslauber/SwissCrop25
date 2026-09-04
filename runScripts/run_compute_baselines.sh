#!/bin/bash
set -e  # Stop if any command fails

############################
### Print Slurm Commands ###
############################
if [ "$SLURM_NODEID" == "0" ]; then
    echo "===== SLURM ENVIRONMENT VARIABLES ====="
    env | grep ^SLURM_
    echo "======================================="
fi
export PYTHONUNBUFFERED=1
############################

# Activate virtual environment
ln -sfn /srv/.venv $PWD/.venv
source .venv/bin/activate

python scripts/analysis/compute_baselines.py

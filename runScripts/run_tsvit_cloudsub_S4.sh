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

############################
### Set environment ########
############################
GPUS_PER_NODE=4
export OMP_NUM_THREADS=1
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
############################

############################
#### Set network ###########
############################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=13348
############################

# Activate virtual environment
ln -sfn /srv/.venv $PWD/.venv
source .venv/bin/activate

bash -c "\
    python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:13348 \
    --max_restarts=0 \
    train_tsvit.py \
    --epochs 15 \
    --num_workers 12 \
    --bias_initialization \
    --no_normalize_timestamps \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --val_every 1 \
    --ablation_split S4 \
    --batch_size 16 \
    --accumulate_steps 1 \
    --res_dir ./storage/tsvit_cloudsub_S4 \
    --checkpoint_steps 200 \
    --seed 7777"

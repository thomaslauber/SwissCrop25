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
MASTER_PORT=13352
############################

# Activate virtual environment
ln -sfn /srv/.venv $PWD/.venv
source .venv/bin/activate

bash -c "\
    python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=0 \
    train_galileo.py \
    --model nano \
    --freeze_encoder \
    --accumulate_steps 4 \
    --epochs 15 \
    --num_workers 16 \
    --bias_initialization \
    --use_temperature_calendar \
    --use_temperature_subsampling \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --val_every 1 \
    --ablation_split S4 \
    --res_dir ./storage/galileo_nano_frozen_gddsub_S4 \
    --seed 7777"

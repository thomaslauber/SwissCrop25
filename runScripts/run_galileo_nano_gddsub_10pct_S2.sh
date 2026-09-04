#!/bin/bash
set -e

if [ "$SLURM_NODEID" == "0" ]; then
    echo "===== SLURM ENVIRONMENT VARIABLES ====="
    env | grep ^SLURM_
    echo "======================================="
fi
export PYTHONUNBUFFERED=1

GPUS_PER_NODE=4
export OMP_NUM_THREADS=1
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=13349

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
    --epochs 15 \
    --num_workers 12 \
    --bias_initialization \
    --use_temperature_calendar \
    --use_temperature_subsampling \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --val_every 1 \
    --ablation_split S2 \
    --accumulate_steps 4 \
    --train_dataset_portion 0.1 \
    --res_dir ./storage/galileo_nano_gddsub_10pct_S2 \
    --seed 7777"

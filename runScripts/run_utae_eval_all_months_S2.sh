#!/bin/bash
set -e

if [ "$SLURM_NODEID" == "0" ]; then
    echo "===== SLURM ENVIRONMENT VARIABLES =====" && env | grep ^SLURM_ && echo "======================================="
fi
export PYTHONUNBUFFERED=1

GPUS_PER_NODE=4
export OMP_NUM_THREADS=1
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=13360

ln -sfn /srv/.venv $PWD/.venv
source .venv/bin/activate

echo "=== eval_all_months: UTAE S4 (single pass, 12 month cutoffs) ==="
python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=0 \
    train_utae.py \
    --PE v1 \
    --epochs 15 \
    --num_workers 12 \
    --use_temperature_calendar \
    --use_temperature_subsampling \
    --no_normalize_timestamps \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --ablation_split S2 \
    --eval_all_months \
    --res_dir ./storage/utae_bench_S2 \
    --seed 7777

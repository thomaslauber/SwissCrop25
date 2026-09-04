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
# Ensure real-time logging
export PYTHONUNBUFFERED=1
############################

cwd=$SCRATCH/020_crop1990/SwissCrop25
cd "$cwd"
# Activate virtual environment *inside* the container, if needed
source /srv/.venv/bin/activate

cwd=$SCRATCH/020_crop1990/SwissCrop25/scripts/preprocessing
cd "$cwd"

echo "Starting python task"

# ---- GTs: all years (each .tar -> .tar.json next to the .tar) ----
python generate_kerchunk.py \
    "$SCRATCH/020_crop1990/data/CGDD/"{2019,2020,2021,2022,2023,2024,2025}.tar

# # ---- Sentinel-2 webdataset (folder of .tar files per year) ----
# python generate_kerchunk.py \
#     "$STORE/data/satellite/sentinel2/raw/CH/"{2019,2020,2021,2022,2023,2024,2025}

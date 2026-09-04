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

# Run your Python training script

# ---- Sentinel 2 (single source, no band renaming, no Tier-1 filtering) ----
# python generateGT.py \
#     -o "$SCRATCH/020_crop1990/data/GTs_Sentinel" \
#     -y 2025 \
#     -p "$STORE/data/landuse" \
#     -i "$STORE/data/satellite/sentinel2/raw/CH/2025.json" \
#     --num_workers 80

# ---- Landsat 2023-2025: single source (LS 8/9 only) ----
# Bands are auto-renamed (OLI_* -> Blue/Green/Red/NIR/SWIR1/SWIR2).
# Only Tier-1 scenes included in stats.
# python generateGT.py \
#     -o "$SCRATCH/020_crop1990/data/GTs_Landsat" \
#     -y 2025 \
#     -p "$STORE/data/landuse" \
#     -i "$STORE/data/satellite/landsat/raw/CH/89/2025.zarr.zip.json" \
#     --num_workers 80

# ---- Landsat 2019-2022: two sources (LS 8/9 + LS 7) ----
# Tiles from both sources are combined: one GT per spatial location,
# stats include Tier-1 observations from both sensors with unified band names.
python generateGT.py \
    -o "$SCRATCH/020_crop1990/data/GTs_Sentinel" \
    -y 2019 \
    -p "$STORE/data/landuse" \
    -i "$STORE/data/satellite/sentinel2/raw/CH/2019.json" \
    --num_workers 64
    
    #    "$STORE/data/satellite/landsat/raw/CH/7/2019.zarr.zip.json" \
# cd "$SCRATCH/020_crop1990/swiss_crop_thermal/data/GTs_Sentinel/2020"
# tar -cvf "../2020.tar" *

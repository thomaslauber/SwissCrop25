# SwissCrop25

A national benchmark dataset and codebase for operational crop mapping in Switzerland, providing Sentinel-2 time series, daily temperature data, and parcel-level crop type labels across seven growing seasons (2019–2025).

**SwissCrop25: A National Multi-Year Benchmark for Operational Crop Mapping**
(TerraBytes II Workshop, ECCV 2026) — [[Paper]](https://arxiv.org/abs/2608.09497) [[Dataset]](https://huggingface.co/datasets/EOA-team/SwissCrop25) [[Team]](https://www.eoa-team.net/)

![Dataset overview](./assets/SwissCrop25_overview.png)

---

## Repository structure

```
src/                    model architectures and dataset loaders
  backbones/            U-TAE and supporting modules (L-TAE, FiLM, FPN, ConvLSTM, ConvGRU)
  tsvit/                TSViT implementation
  learning/             loss functions and metrics
train_utae.py           training script for U-TAE
train_tsvit.py          training script for TSViT
train_galileo.py        training script for Galileo
scripts/
  preprocessing/        dataset construction (ground truth, temperature, Kerchunk indices)
  analysis/             baselines, per-class metrics, in-season evaluation
  figures/              figure generation scripts
  tables/               LaTeX table generation scripts
runScripts/             SLURM run scripts for all experiments and splits
storage/                per-experiment evaluation metrics (.pkl)
results/
  figures/              generated figures
  tables/               generated LaTeX table bodies
```

---

## Environment

### Training

Built on [NVIDIA PyTorch container 24.12](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-12.html). Additional packages:

```bash
pip install -r requirements.txt
```

### Preprocessing

Built on the [b-data R+QGIS devcontainer](https://github.com/b-data/data-science-devcontainers/blob/main/.devcontainer/r-qgisprocess/devcontainer.json). Additional Python packages:

```bash
pip install -r requirements_preprocess.txt
```

The ground truth generation script (`scripts/preprocessing/generateGT.py`) calls R via subprocess. Required R packages: `terra`, `stars`, `sf`, `exactextractr`, `optparse`.

---

## Data

The dataset is hosted on Hugging Face: [EOA-team/SwissCrop25](https://huggingface.co/datasets/EOA-team/SwissCrop25).

After downloading, pass the dataset root via `--dataset_folder`:

```bash
--dataset_folder /path/to/SwissCrop25
```

The training scripts expect `sentinel2/`, `labels/`, and `temperature/` as subdirectories, matching the Hugging Face repository layout.

---

## Training

All three models are trained with distributed data-parallel using `torch.distributed.run`. Example for a single node with 4 GPUs (LOYO split S1):

**U-TAE** (GDD substitution + GDD positional encoding):
```bash
python -m torch.distributed.run --nproc_per_node=4 train_utae.py \
    --dataset_folder /path/to/SwissCrop25 \
    --epochs 15 \
    --num_workers 12 \
    --bias_initialization \
    --no_normalize_timestamps \
    --use_gdd_pe \
    --use_temperature_calendar \
    --use_temperature_subsampling \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --ablation_split S1 \
    --res_dir ./storage/utae_gddsub_gddpe_S1 \
    --seed 7777
```

**TSViT** (GDD substitution + GDD positional encoding):
```bash
python -m torch.distributed.run --nproc_per_node=4 train_tsvit.py \
    --dataset_folder /path/to/SwissCrop25 \
    --epochs 15 \
    --num_workers 12 \
    --bias_initialization \
    --variable_t \
    --use_gdd_pe \
    --use_temperature_calendar \
    --use_temperature_subsampling \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --batch_size 4 \
    --accumulate_steps 4 \
    --ablation_split S1 \
    --res_dir ./storage/tsvit_gddsub_gddpe_S1 \
    --seed 7777
```

**Galileo-nano** (fine-tuned, GDD substitution):
```bash
python -m torch.distributed.run --nproc_per_node=4 train_galileo.py \
    --dataset_folder /path/to/SwissCrop25 \
    --model nano \
    --epochs 15 \
    --num_workers 12 \
    --bias_initialization \
    --no_normalize_timestamps \
    --variable_t \
    --use_temperature_calendar \
    --use_temperature_subsampling \
    --satellite sentinel \
    --w_ce 1.0 \
    --use_class_balance_loss \
    --batch_size 2 \
    --accumulate_steps 8 \
    --ablation_split S1 \
    --res_dir ./storage/galileo_nano_gddsub_S1 \
    --seed 7777
```

Complete SLURM run scripts for all experiments and splits are provided in `runScripts/`.

### Galileo weights

Galileo pretrained weights are available on Hugging Face. Download them to `galileo_weights/models/{nano,tiny,base}/` before running Galileo experiments.

---

## Reproducing results

Evaluation metrics are stored as `.pkl` files in `storage/` after training. To reproduce tables and figures from the paper, use the scripts in `scripts/tables/` and `scripts/figures/`. For example:

```bash
python scripts/tables/generate_results_table.py
python scripts/figures/confmat_figure.py
```

Baselines (majority vote, previous year) can be recomputed with:

```bash
python scripts/analysis/compute_baselines.py
```

---

## Dataset

### Overview

SwissCrop25 is designed to evaluate crop mapping systems under realistic operational conditions. Unlike datasets that focus on a single aspect of the crop mapping problem, SwissCrop25 jointly supports:

- **Scene completeness**: joint cropland delineation and crop type classification
- **Temporal generalisation**: seven-year LOYO evaluation with genuine interannual weather variability
- **Fine-grained classification**: 65-class crop taxonomy including grassland management types
- **In-season usability**: evaluation under progressively truncated time series

The dataset covers the full Swiss territory (41,285 km²) and is derived from Switzerland's official annual agricultural parcel declarations (LNF), the Swiss equivalent of the EU LPIS.

### Data structure

SwissCrop25 is stored in [WebDataset](https://github.com/webdataset/webdataset) format — sequential tar archives that enable efficient streaming during distributed training without requiring random-access storage. [Kerchunk](https://github.com/fsspec/kerchunk) index files are provided alongside each annual folder for direct random access.

#### Sentinel-2

10 spectral bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (visible, red-edge, NIR, SWIR). All bands resampled to 10 m. Imagery is BRDF-corrected and sourced from Microsoft Planetary Computer. Each observation includes two cloud masks: the native Sentinel-2 SCL flag and the multi-class CloudSEN12+ score.

Shape per cube: `(T, 10, 128, 128)`, where T is the number of available observations.

#### Temperature

Per-cube cumulative growing degree day (GDD) time series derived from MeteoSwiss gridded daily mean temperature (1 km resolution), accumulated from 1 January with base temperature 0 °C.

Shape per cube: `(T,)`, aligned to the Sentinel-2 observation timestamps.

#### Labels

Parcel-level crop type labels rasterised at 10 m using a fractional coverage approach. Label values are integers in [0, 70]: 0 is background (unlabelled), 1–65 are crop classes, 66–70 are non-crop land cover classes (forest, water, built-up, unproductive land, wetland).

Shape per cube: `(128, 128)`.

### Labels

#### Sources

Crop labels are derived from the Swiss Agricultural Cultivated Areas dataset (LNF, *Landwirtschaftliche Nutzfläche*), compiled from annual cantonal land-use declarations and published by the Swiss Federal Office for Agriculture (FOAG). The LNF assigns a single crop or land-use code to each parcel per year based on the primary crop declared by 1 June.

Non-crop land cover classes are sourced from the Swiss national topographic landscape model (swissTLM3D, Federal Office of Topography swisstopo).

#### Class hierarchy

The 140 raw LNF codes are aggregated into 70 classes for training and evaluation (65 crop + 5 non-crop), with 8 codes excluded due to insufficient temporal coverage, low sample counts, or ambiguous mixed-culture definitions.

Agricultural classes are organised in a four-level hierarchy:

| Level | Count | Description |
|---|---|---|
| lv1 (leaf) | 65 | Output classes used for training and evaluation |
| lv2 | 59 | Intermediate crop types |
| lv3 | 22 | Crop-type groups |
| lv4 | 8 | Top-level categories (arable, grassland, permanent, …) |

All 65 crop classes are additionally mapped to HCAT4, enabling compatibility with [EuroCrops](https://github.com/maja601/EuroCrops) and EU LPIS datasets. The full class list, LNF codes, hierarchy, and per-year area statistics are provided in `metadata/crop_classes.csv` in the dataset.

#### Class distribution

The dataset exhibits a pronounced long-tail distribution spanning nearly five orders of magnitude (class imbalance ratio >180,000:1), from dominant classes such as Intensive Meadow and Winter Wheat to minority crops such as Safflower and Tobacco. This imbalance reflects the actual structure of Swiss agriculture and is intentionally preserved.

#### Per-year statistics

| Year | Parcels | Cubes | Coverage |
|---|---:|---:|---|
| 2019 | 740,597 | 10,546 | Partial |
| 2020 | 1,536,321 | 21,439 | Partial |
| 2021 | 2,023,603 | 26,089 | Complete |
| 2022 | 2,004,326 | 26,089 | Complete |
| 2023 | 2,074,936 | 26,089 | Complete |
| 2024 | 2,116,874 | 26,090 | Complete |
| 2025 | 2,146,317 | 26,843 | Complete |
| **Total** | **12,642,978** | **163,185** | |

Partial years exclude cantons with <90% LNF coverage relative to their maximum observed extent.

### Evaluation protocol

SwissCrop25 defines a five-fold **leave-one-year-out (LOYO)** protocol:

| Split | Test year | Val year | Train years |
|---|---|---|---|
| S1 | 2021 | 2020 | 2019, 2022–2025 |
| S2 | 2022 | 2021 | 2019–2020, 2023–2025 |
| S3 | 2023 | 2022 | 2019–2021, 2024–2025 |
| S4 | 2024 | 2023 | 2019–2022, 2025 |
| S5 | 2025 | 2024 | 2019–2023 |

2019–2020 are used as additional training data only due to incomplete national coverage.

Evaluation proceeds in two stages:
1. **Binary cropland mask** (agricultural vs. non-agricultural): IoU_ag, precision, recall, F1
2. **Crop type classification** (65 classes, restricted to agricultural pixels): OA, GIoU, mIoU, mF1

OA and GIoU are micro-averaged; mIoU and mF1 are macro-averaged over crop classes.

### Benchmarks

Results for three architectures under the LOYO protocol (mean ± std across five splits, best temporal encoding configuration per model):

| Model | OA (%) | GIoU (%) | mIoU (%) | mF1 (%) |
|---|---:|---:|---:|---:|
| U-TAE | 77.7 ± 1.5 | 63.5 ± 2.0 | 35.8 ± 2.3 | 45.7 ± 2.4 |
| TSViT | 77.1 ± 1.2 | 62.7 ± 1.6 | 48.1 ± 2.7 | 60.7 ± 2.5 |
| Galileo-nano (FT) | 72.9 ± 1.3 | 57.4 ± 1.6 | 30.4 ± 2.2 | 41.1 ± 2.5 |

Full per-split and per-class results are reported in the [paper](https://arxiv.org/abs/2608.09497).

---

## Citation

```bibtex
@misc{lauber2026swisscrop25nationalmultiyearbenchmark,
  title         = {SwissCrop25: A National Multi-Year Benchmark for Operational Crop Mapping},
  author        = {Thomas Lauber and Mehmet Ozgur Turkoglu and Sélène Ledain and Helge Aasen},
  year          = {2026},
  eprint        = {2608.09497},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.09497}
}
```

---

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

You may share and adapt the data, including for commercial use, provided you give appropriate credit as described below.

### Attribution

When using this dataset, please cite:

- Contains modified Copernicus Sentinel-2 data 2019–2025, processed by Agroscope.
- Contains data from MeteoSwiss (Federal Office of Meteorology and Climatology), Open Government Data.
- Contains cantonal agricultural land-use data (Nutzungsflächen / Surfaces d'utilisation / Superfici d'utilizzazione) from all 26 Swiss cantons (AG, AI, AR, BE, BL, BS, FR, GE, GL, GR, JU, LU, NE, NW, OW, SG, SH, SO, SZ, TG, TI, UR, VD, VS, ZG, ZH), obtained via [geodienste.ch](https://www.geodienste.ch/services/lwb_nutzungsflaechen), snapshot dates 2019–2025. Cantons with specific attribution requirements:
  - `Quelle: Nutzungsflächen, Kanton Graubünden`
  - `Fonte: Amministrazione cantonale - Canton Ticino`
  - `Source: Géodonnées Etat de Vaud`
  - `Kanton St.Gallen`
  - `Kanton Nidwalden, Amt für Landwirtschaft`
  - `Kanton Obwalden, Amt für Landwirtschaft und Umwelt`

### Disclaimer

Data provided as is, without warranty. Not for navigation or legally-binding use. Contains no personal or farm-identifying attributes.

---

## Contact

- Thomas Lauber — thomas.lauber@agroscope.admin.ch
- Helge Aasen — helge.aasen@agroscope.admin.ch
- [Earth Observation of Agroecosystems Team, Agroscope](https://www.eoa-team.net/)

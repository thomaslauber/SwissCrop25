import os
import random
from collections import defaultdict
from contextlib import contextmanager
import json
import io
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import zarr
import fsspec
import webdataset as wds

# Global zarr cache shared across all dataset instances
zarr_cache = {}


class WebDataLoaderWithLen:
    """Wrapper around WebDataset DataPipeline that provides __len__ method."""
    def __init__(self, dataloader, length, dataset=None):
        self.dataloader = dataloader
        self._length = length
        self.dataset = dataset  # Store reference to original dataset for accessing attributes

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter(self.dataloader)


class SatelliteDataset(Dataset):

    # ── Sensor augmentation constants ─────────────────────────────────────────
    # Landsat Collection 2 SR ↔ DN scaling
    _LC2_SCALE  = 2.75e-05
    _LC2_OFFSET = -0.2

    # Roy harmonization coefficients (keyed by generic band name)
    # OLI→TM direction: TM_SR = slope * OLI_SR + intercept
    _TM_AUG_HARMONIZATION = {
        'Blue':  {'slope': 0.8431, 'intercept': 0.0167},
        'Green': {'slope': 0.8703, 'intercept': 0.0150},
        'Red':   {'slope': 0.8886, 'intercept': 0.0146},
        'NIR':   {'slope': 0.8896, 'intercept': 0.0377},
        'SWIR1': {'slope': 0.9945, 'intercept': 0.0160},
        'SWIR2': {'slope': 1.0123, 'intercept': 0.0051},
    }
    # TM→OLI direction: OLI_SR = slope * TM_SR + intercept  (separate Roy regression, not a mathematical inverse)
    _TM_TO_OLI_HARMONIZATION = {
        'Blue':  {'slope': 1.0556, 'intercept': -0.0123},
        'Green': {'slope': 1.015,  'intercept': -0.0067},
        'Red':   {'slope': 0.9861, 'intercept': -0.0063},
        'NIR':   {'slope': 0.966,  'intercept':  0.0031},
        'SWIR1': {'slope': 0.874,  'intercept':  0.0067},
        'SWIR2': {'slope': 0.8534, 'intercept':  0.0083},
    }

    # Per-band Gaussian noise sigma (SR units) from LEOHS RMSE
    _OLI_DEGRADE_SIGMA = {      # per-band RMSE from ~800K co-located ETM+/OLI pairs (LEOHS, Switzerland)
        'Blue':  0.026472,
        'Green': 0.026092,
        'Red':   0.028344,
        'NIR':   0.047183,
        'SWIR1': 0.033383,
        'SWIR2': 0.024078,
    }
    _ETM_DEGRADE_SIGMA = {      # ~0.5× OLI values, ETM+ has ~2× better SNR than TM (Chander et al. 2009)
        'Blue':  0.013,
        'Green': 0.013,
        'Red':   0.014,
        'NIR':   0.024,
        'SWIR1': 0.017,
        'SWIR2': 0.012,
    }

    # Generic band names for combining OLI (LS8/9), TM (LS4/5), and ETM+ (LS7)
    BAND_RENAME_MAP = {
        'OLI_B2': 'Blue',  'OLI_B3': 'Green', 'OLI_B4': 'Red',
        'OLI_B5': 'NIR',   'OLI_B6': 'SWIR1', 'OLI_B7': 'SWIR2',
        'TM_B1':  'Blue',  'TM_B2':  'Green', 'TM_B3':  'Red',
        'TM_B4':  'NIR',   'TM_B5':  'SWIR1', 'TM_B7':  'SWIR2',
        'ETM_B1': 'Blue',  'ETM_B2': 'Green', 'ETM_B3': 'Red',
        'ETM_B4': 'NIR',   'ETM_B5': 'SWIR1', 'ETM_B7': 'SWIR2',
    }

    SENSOR_BAND_MAPS = {
        "OLI": {   # Landsat 8/9
            "OLI_B2": "Blue",
            "OLI_B3": "Green",
            "OLI_B4": "Red",
            "OLI_B5": "NIR",
            "OLI_B6": "SWIR1",
            "OLI_B7": "SWIR2",
        },
        "TM": {    # Landsat 5
            "TM_B1": "Blue",
            "TM_B2": "Green",
            "TM_B3": "Red",
            "TM_B4": "NIR",
            "TM_B5": "SWIR1",
            "TM_B7": "SWIR2",
        },
        "ETM": {   # Landsat 7
            "ETM_B1": "Blue",
            "ETM_B2": "Green",
            "ETM_B3": "Red",
            "ETM_B4": "NIR",
            "ETM_B5": "SWIR1",
            "ETM_B7": "SWIR2",
        }
    }

    def __init__(
        self,
        satellite_paths,
        gt_paths,
        temp_paths,
        label_sheet_file="./SwissCrop25.xlsx",
        label_columns="4th_tier_ENG",
        ignore_index=None,
        bands=[
            's2_B02', 's2_B03', 's2_B04', 's2_B08',
            's2_B05', 's2_B06', 's2_B07', 's2_B8A', 's2_B12'
        ],
        cloud_band="s2_mask",
        band_stats=None,
        condition='open_sky',
        sample_percentage=1.0,
        truncate_portion=1.0,  # Portion of time dimension to keep (1.0 = no truncation)
        temporal_length=24,
        seed=42,
        use_temperature_calendar=False,
        use_temperature_subsampling=False,
        no_sliding_subsample=False,
        use_temperature_calendar_no_sliding_subsample=False,
        augmentation=True,
        simulate_landsat=False,
        revisit_time=8,
        temp_mean_var='CGDD_mean',      # Variable name for scalar GDD (subsampling/PE)
        temp_spatial_var=None,    # Variable name for spatial GDD (attention) - None to disable
        temp_stats=None,
        cgdd_bounds_gpkg=None,  # Path(s) to GPKG file(s) with climatological CGDD bounds
        use_fixed_temperature_subsampling=False,  # Enable climatological GDD subsampling
        mode='training',  # 'training' or 'prediction'
        filter_years=None,  # Optional: list of years to filter (e.g., [2019, 2020, 2021])
        # Multi-head configuration (set these to enable 3-head decoder mode)
        multi_head_fine_column=None,    # e.g. "Crop_Label_lv1"
        multi_head_medium_column=None,  # e.g. "Crop_Label_lv2" (for per-branch hierarchical loss)
        multi_head_coarse_column=None,  # e.g. "Crop_Label_lv3"
        grassland_coarse_values=None,   # e.g. ["Grassland"]
        arable_coarse_values=None,      # e.g. ["Arable Land", "Permanent"]
        # Sensor-aware degradation augmentation (source-named: OLI→TM-like, ETM+→ETM+-noisy)
        oli_degrade_prob=0.0,           # P(OLI tile degraded to TM-like via quant+noise); replaces tm_aug_prob
        etm_degrade_prob=0.0,           # P(ETM+ tile degraded with 8-bit quant+noise)
        spectral_sparsify=False,        # Shorthand: sets both degrade probs to 0.5
        sensor_isolation_prob=0.0,      # P(per direction) of loading only OLI or only ETM+/TM paths
        harmonize_oli_to_tm=False,      # Load-time inverse Roy: shift all OLI timesteps to TM space (spectral only)
        harmonize_tm_to_oli=False,      # Load-time forward Roy: shift all TM/ETM+ timesteps to OLI space
        use_sensor_flag=False,          # Append sensor-identity channel to images
        sensor_flag_map=None,           # Dict mapping sensor type to flag value, e.g. {"OLI": 0, "TM": 1, "ETM": 2}
                                        # Defaults to {"OLI": 0, "TM": 1, "ETM": 1} (TM/ETM indistinguishable)
        temporal_sparsify=False,        # 50% chance per sample of reducing obs to TM-era density (8-14 obs/season)
        is_tm_data=False,               # Mark this dataset as real TM data (sensor flag=1 always)
        tier_lookup=None,               # Path to Tier1 CSV (scene_id, collection_category)
        alpine_mask_path=None,          # Path to binary COG for Sömmerungsgebiet remapping
        normalize_timestamps=True,      # Z-score GDD and DOY; set False to return raw values for sinusoidal PE
        verbose=True,                   # Print dataset summary on construction
    ):
        # Validate mode
        if mode not in ['training', 'prediction']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'training' or 'prediction'")
        self.mode = mode

        # Setup
        self.satellite_paths = satellite_paths
        self.gt_paths = gt_paths
        self.temp_paths = temp_paths
        self.filter_years = filter_years  # Store year filter
        self.label_sheet_file = label_sheet_file
        self.label_columns = label_columns
        self.ignore_index = ignore_index
        self.bands = bands
        self.cloud_band = cloud_band
        self.band_stats = band_stats
        self.condition = condition
        self.sample_percentage = sample_percentage
        self.truncate_portion = truncate_portion
        self.temporal_length = temporal_length  # always 24; truncation handled via time_stamps slicing
        self.use_temperature_calendar = use_temperature_calendar
        self.use_temperature_subsampling = use_temperature_subsampling
        self.use_fixed_temperature_subsampling = use_fixed_temperature_subsampling
        self.no_sliding_subsample = no_sliding_subsample
        self.use_temperature_calendar_no_sliding_subsample = use_temperature_calendar_no_sliding_subsample
        self.augmentation = augmentation
        self.temporal_sparsify = bool(temporal_sparsify)
        self.temporal_sparsify_range = (6, 24)
        self.min_temporal_keep = 4
        self.simulate_landsat = simulate_landsat
        self.revisit_time = revisit_time
        self.verbose = verbose
        self.normalize_timestamps = normalize_timestamps
        if spectral_sparsify:
            oli_degrade_prob = 0.5
            etm_degrade_prob = 0.5
        self.oli_degrade_prob = oli_degrade_prob
        self.etm_degrade_prob = etm_degrade_prob
        self.spectral_sparsify = bool(spectral_sparsify)
        self.sensor_isolation_prob = sensor_isolation_prob
        self.harmonize_oli_to_tm = harmonize_oli_to_tm
        self.harmonize_tm_to_oli = harmonize_tm_to_oli
        self.use_sensor_flag = use_sensor_flag
        self.sensor_flag_map = sensor_flag_map if sensor_flag_map is not None else {"OLI": 0, "TM": 1, "ETM": 1}
        self.is_tm_data = is_tm_data
        if self.harmonize_oli_to_tm and self.oli_degrade_prob > 0.0:
            raise ValueError(
                "harmonize_oli_to_tm and oli_degrade_prob > 0 must not be used together: "
                "OLI timesteps would be quantised+noised then Roy-shifted (double correction)."
            )
        if self.harmonize_tm_to_oli and self.etm_degrade_prob > 0.0:
            raise ValueError(
                "harmonize_tm_to_oli and etm_degrade_prob > 0 must not be used together: "
                "ETM+ timesteps would be degraded then Roy-shifted to OLI space (double correction)."
            )

        # Load Tier1 scene IDs for Landsat filtering
        if tier_lookup is not None:
            _t1_df = pd.read_csv(tier_lookup, usecols=["scene_id", "collection_category"])
            self.tier1_scene_ids = set(_t1_df.loc[_t1_df["collection_category"] == "T1", "scene_id"])
            if self.verbose:
                print(f"Loaded {len(self.tier1_scene_ids)} Tier1 scene IDs from {tier_lookup}")
        else:
            self.tier1_scene_ids = None

        # Ensure list inputs
        self.satellite_paths = self._ensure_list(satellite_paths)
        self.gt_paths = self._ensure_list(gt_paths)
        self.temp_paths = self._ensure_list(temp_paths)
        self.label_columns = self._ensure_list(label_columns)

        # Normalize satellite_paths to list-of-lists for uniform multi-sensor handling.
        # Sentinel (flat list of strings) becomes [[path], [path], ...].
        # Landsat (list of lists) is already in the right shape.
        if self.satellite_paths and not isinstance(self.satellite_paths[0], list):
            self.satellite_paths = [[f] for f in self.satellite_paths]

        # Fix the seed for reproducibility
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Detect data mode
        self.is_webdataset = self._detect_sentinel2_mode()

        # Load and cache all data sources
        self.data_sources = self._load_and_cache_data_sources()

        # Get available tiles from all sources
        if self.filter_years is not None and self.verbose:
            print(f"Filtering tiles to only include years: {self.filter_years}")
        self.data_files = self._get_available_tiles()

        # Subsample dataset if requested
        if self.sample_percentage < 1.0:
            n_samples = int(len(self.data_files) * self.sample_percentage)
            self.data_files = random.sample(self.data_files, n_samples)

        # Filter out corrupted tiles if exclusion list exists
        self.data_files = self._filter_corrupted_tiles(self.data_files)

        # For WebDataset mode, preload GT and Calendar data into cache (after sampling)
        if self.is_webdataset:
            self.gt_cache, self.cal_cache = self._load_gt_and_calendar_cache()
            # Precompute available tile names for efficient filtering
            self.available_tile_names = {
                self._extract_tile_name(s2[0] if isinstance(s2, list) else s2)
                for s2, _, _ in self.data_files
            }

        # Configure GDD variable names
        self.gdd_mean_var = temp_mean_var
        # Handle string 'None' or None for spatial variable
        if temp_spatial_var == 'None' or temp_spatial_var is None:
            self.gdd_spatial_var = None
        else:
            self.gdd_spatial_var = temp_spatial_var
        self.use_spatial_gdd = self.gdd_spatial_var is not None
        self.temp_stats = temp_stats

        # Validate fixed temperature subsampling parameters
        if self.use_fixed_temperature_subsampling and self.use_temperature_subsampling:
            raise ValueError(
                "Cannot use both use_fixed_temperature_subsampling and use_temperature_subsampling. "
                "Choose one subsampling method."
            )

        if self.use_fixed_temperature_subsampling and not self.use_temperature_calendar:
            raise ValueError(
                "use_fixed_temperature_subsampling requires use_temperature_calendar=True"
            )

        if self.use_fixed_temperature_subsampling and cgdd_bounds_gpkg is None:
            raise ValueError(
                "cgdd_bounds_gpkg must be provided when use_fixed_temperature_subsampling=True"
            )

        # Load CGDD bounds if fixed temperature subsampling is enabled
        self.cgdd_bounds = None
        if self.use_fixed_temperature_subsampling:
            self.cgdd_bounds = self._load_cgdd_bounds(cgdd_bounds_gpkg)

        # Channel statistics - dynamically load from stats files if available
        if self.band_stats:
            self.channel_stats = self.band_stats
            if self.verbose:
                print("Loaded channel statistics from provided dictionary.")
        else:
            loaded_channel_stats = self._load_channel_stats()
            if loaded_channel_stats is not None:
                self.channel_stats = loaded_channel_stats
                if self.verbose:
                    print("Loaded channel statistics from parquet files.")
            else:
                # Fallback to hardcoded statistics
                self.channel_stats = {
                    # Landsat generic bands (whole-dataset estimates)
                    'Blue':  {'mean': 22466.66, 'std': 16016.28},
                    'Green': {'mean': 22584.65, 'std': 14830.52},
                    'Red':   {'mean': 22475.58, 'std': 14905.39},
                    'NIR':   {'mean': 25530.33, 'std': 12065.82},
                    'SWIR1': {'mean': 15433.29, 'std': 5831.74},
                    'SWIR2': {'mean': 13848.24, 'std': 5059.05},
                    # Sentinel-2 bands
                    's2_B02': {'mean': 1962.10, 'std': 5731.00},
                    's2_B03': {'mean': 2106.59, 'std': 5656.44},
                    's2_B04': {'mean': 2028.38, 'std': 5662.33},
                    's2_B08': {'mean': 3797.36, 'std': 5362.84},
                    's2_B05': {'mean': 2430.80, 'std': 5590.77},
                    's2_B06': {'mean': 3380.80, 'std': 5408.80},
                    's2_B07': {'mean': 3681.89, 'std': 5354.31},
                    's2_B8A': {'mean': 3851.88, 'std': 5307.00},
                    's2_B11': {'mean': 2163.20, 'std': 5302.00},
                    's2_B12': {'mean': 1582.97, 'std': 5161.84},
                }
                if self.verbose:
                    print("Using hardcoded channel statistics.")
        self.channel_means = np.array([self.channel_stats[band]['mean'] for band in self.bands], dtype=np.float32)
        self.channel_stds = np.array([self.channel_stats[band]['std'] for band in self.bands], dtype=np.float32)
        if self.verbose:
            print(f"Channel Means: {[f'{mean:.2f}' for mean in self.channel_means]}")
            print(f"Channel Stds: {[f'{std:.2f}' for std in self.channel_stds]}")

        # GDD statistics - load from stats files
        if self.temp_stats:
            self.gdd_stats = self.temp_stats
            if self.verbose:
                print("Using provided GDD statistics.")
        else:
            gdd_stats = self._load_gdd_stats()
            if gdd_stats is not None:
                self.gdd_stats = gdd_stats
                if self.verbose:
                    print("Loaded GDD statistics from parquet files.")
            else:
                # Fallback to hardcoded statistics
                self.gdd_stats = {'mean': 1300.03, 'std': 1253.74}
                if self.verbose:
                    print("Using hardcoded GDD statistics.")
            if self.verbose:
                print(f"GDD statistics: mean={self.gdd_stats['mean']:.2f}, std={self.gdd_stats['std']:.2f}")

        # Multi-head configuration attributes
        self.multi_head_fine_column = multi_head_fine_column
        self.multi_head_medium_column = multi_head_medium_column
        self.multi_head_coarse_column = multi_head_coarse_column
        self.grassland_coarse_values = grassland_coarse_values or ["Grassland"]
        self.arable_coarse_values = arable_coarse_values or ["Arable Land", "Permanent"]
        # Populated by _map_lnf_code_to_ground_truth when multi-head is active
        self.grassland_coarse_ids = []
        self.arable_coarse_ids = []
        self.num_grassland_classes = 0
        self.num_arable_classes = 0
        self.num_coarse_classes = 0

        # Map LNF codes to ground truth classes
        self.target_mapping = None
        self.num_classes = 0
        self.mapping_dict = None
        self.child_maps = None
        self._map_lnf_code_to_ground_truth(self.label_columns)

        # Alpine Pasture remapping via Sömmerungsgebiet mask
        self.alpine_mask = None
        self.alpine_mask_transform = None
        self.alpine_pasture_class_id = None
        self.extensive_pasture_class_id = None
        self.pasture_class_ids = frozenset()
        if alpine_mask_path is not None:
            import rasterio as _rio
            with _rio.open(alpine_mask_path) as _src:
                self.alpine_mask = _src.read(1)          # (H, W) uint8
                self.alpine_mask_transform = _src.transform
            # mapping_dict[0] is the fine-class name→ID dict (grassland head or lv1)
            fine_map = self.mapping_dict[0]
            self.alpine_pasture_class_id = fine_map.get("Alpine Pasture")
            self.extensive_pasture_class_id = fine_map.get("Extensive Pasture")
            self.pasture_class_ids = frozenset(
                v for k, v in fine_map.items() if "Pasture" in k and k != "Alpine Pasture"
            )
            if self._is_rank_zero() and self.verbose:
                print(f"Alpine mask: {alpine_mask_path}")
                print(f"  Alpine Pasture class ID: {self.alpine_pasture_class_id}")
                print(f"  Extensive Pasture class ID (outside-zone fallback): {self.extensive_pasture_class_id}")
                print(f"  Pasture class IDs to remap: {sorted(self.pasture_class_ids)}")

        if self._is_rank_zero() and self.verbose:
            print(f"Number of classes: {self.num_classes}")
            print(f"Temporal length: {self.temporal_length}")
            print(f"Number of S2 bands: {len(self.bands)}")
            print(f"Bands: {self.bands}")
            print(f"Dataset size: {len(self.data_files)}")
            print(f"Dataset mode: {self.mode}")
            print(f"Class mapping: {self.mapping_dict}")
            print(f"LNF code mapping: {self.target_mapping}")
            print(f"use_temperature_calendar: {self.use_temperature_calendar}")
            print(f"use_temperature_subsampling: {self.use_temperature_subsampling}")
            print(f"use_fixed_temperature_subsampling: {self.use_fixed_temperature_subsampling}")
            print(f"no_sliding_subsample: {self.no_sliding_subsample}")
            print(f"use_temperature_calendar_no_sliding_subsample: {self.use_temperature_calendar_no_sliding_subsample}")
            print(f"truncate_portion: {self.truncate_portion}")
            print(f"temp_mean_var: {self.gdd_mean_var}")
            print(f"temp_spatial_var: {self.gdd_spatial_var}")
            print(f"use_spatial_gdd: {self.use_spatial_gdd}")
            if self.use_fixed_temperature_subsampling:
                print(f"CGDD bounds loaded for {len(self.cgdd_bounds)} tiles")
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """Get item - works for both JSON reference and WebDataset modes."""
        if self.is_webdataset:
            raise NotImplementedError(
                "In WebDataset mode, use _create_webdataset_loader() to get an iterable dataset. "
                "__getitem__ is not supported for WebDataset mode."
            )

        s2_files, gt_file, temp_calendar_file = self.data_files[idx]

        # Sensor isolation: randomly restrict to one sensor type BEFORE zarr loading so that
        # CGDD temporal subsampling draws from a realistic single-sensor temporal distribution.
        # Path convention: '/89/' = OLI (LS8/9); '/7/' = ETM+ (LS7); '/45/' = TM (LS4/5).
        # 0.2 per direction ≈ proportional to ~26% TM-only + ~29% ETM+-only inference years.
        if self.augmentation and self.sensor_isolation_prob > 0.0 and isinstance(s2_files, list):
            _r = random.random()
            if _r < self.sensor_isolation_prob:
                _filtered = [p for p in s2_files if p is None or '/89/' in str(p)]
                if any(p is not None for p in _filtered):
                    s2_files = _filtered
            elif _r < 2 * self.sensor_isolation_prob:
                _filtered = [p for p in s2_files if p is None or '/89/' not in str(p)]
                if any(p is not None for p in _filtered):
                    s2_files = _filtered

        s2_ref = s2_files[0] if isinstance(s2_files, list) else s2_files
        try:
            # Load and merge all sensor zarrs into a single dict-like structure.
            # If sensor isolation left an empty set (e.g. L7 2022 post-retirement has no
            # Tier-1 obs), fall back to the full path list so no tile is ever skipped.
            try:
                s2_data = self._load_and_merge_sensor_data(s2_files)
            except RuntimeError:
                s2_files_full = self.data_files[idx][0]
                s2_data = self._load_and_merge_sensor_data(s2_files_full)
            temp_data = self._load_zarr(temp_calendar_file)

            # Load GT only if available (training mode or prediction with GT)
            if gt_file is not None:
                gt_data = self._load_zarr(gt_file)
            else:
                gt_data = None  # Prediction mode without GT

            return self._process_tile_data(s2_data, gt_data, temp_data, s2_file_path=s2_ref)
        except Exception as e:
            import traceback
            tile_name = self._extract_tile_name(s2_ref)
            print(f"ERROR processing tile {tile_name} (idx={idx})")
            print(f"  S2 files: {self.data_files[idx][0]}")
            print(f"  GT file: {gt_file}")
            print(f"  Temp file: {temp_calendar_file}")
            print(f"  Error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Public Functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_dataloader(self, batch_size, num_workers=4, shuffle=True, shuffle_buffer=1000,
                         resampled=False, rank=None, world_size=None, pin_memory=True,
                         drop_last=False, prefetch_factor=4, persistent_workers=True, **kwargs):
        """
        Create appropriate DataLoader for the dataset mode.
        Handles both WebDataset (with DDP support) and standard PyTorch DataLoader (with DistributedSampler).

        Args:
            batch_size: Batch size per GPU/process
            num_workers: Number of worker processes for data loading
            shuffle: Whether to shuffle data (ignored if using DDP, as sampler handles it)
            shuffle_buffer: Buffer size for shuffling in WebDataset mode
            resampled: Whether to use resampled WebDataset (for DDP)
            rank: Process rank for DDP (None = auto-detect from torch.distributed)
            world_size: Total number of processes for DDP (None = auto-detect from torch.distributed)
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            prefetch_factor: Number of batches to prefetch per worker (default: 4)
            persistent_workers: Whether to keep workers alive between epochs (default: True)
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            DataLoader compatible object
        """
        # Auto-detect DDP settings if not provided
        if rank is None or world_size is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
            except Exception:
                pass
        is_distributed = rank is not None and world_size is not None

        if self.is_webdataset:
            # WebDataset mode with DDP support
            # Use resampled=True to handle uneven distribution of filtered samples across tar files
            # This ensures each GPU can generate exactly the required number of batches
            # even if its assigned tar files don't contain enough valid samples
            dataset = self._create_webdataset_loader(resampled=resampled, shuffle=shuffle, shuffle_buffer=shuffle_buffer)

            # Use WebLoader for efficient parallel loading
            dataloader = wds.WebLoader(
                dataset,
                batch_size=None,  # Batching done below
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                **kwargs
            )

            # Add batching and ensure full batches for DDP
            # Note: Shuffling already done in dataset pipeline, no need to shuffle again here
            # Samples are already fully processed by workers via _process_tile_data
            def collate_processed_samples(batch):
                """Collate already-processed samples into batched tensors."""
                # batch is list of ((images, timestamps), ground_truth, alpine_slice) tuples
                images_list = [sample[0][0] for sample in batch]
                timestamps_list = [sample[0][1] for sample in batch]
                gt_list = [sample[1] for sample in batch]
                alpine_list = [sample[2] for sample in batch]

                # Stack into batches
                images = torch.stack(images_list)
                # timestamps is always a tuple: (gdd, doy) or (gdd, doy, spatial_gdd)
                timestamps = tuple(
                    torch.stack([t[i] for t in timestamps_list])
                    for i in range(len(timestamps_list[0]))
                )

                # Handle hierarchical ground truth
                if isinstance(gt_list[0], list):
                    ground_truth = [torch.stack([gt[i] for gt in gt_list])
                                  for i in range(len(gt_list[0]))]
                else:
                    ground_truth = torch.stack(gt_list)

                alpine_batch = torch.stack(alpine_list)

                return (images, timestamps), ground_truth, alpine_batch

            dataloader = dataloader.batched(
                batch_size,
                partial=not drop_last,
                collation_fn=collate_processed_samples
            )

            # CRITICAL: Apply .with_epoch() AFTER batching to limit number of batches
            # Calculate batches per GPU (split_by_node already distributed the data)
            if is_distributed and world_size is not None:
                # Match PyTorch DistributedSampler logic:
                # Total batches = total_samples // batch_size (floor division if drop_last)
                # Then divide by world_size to get batches per GPU
                if drop_last:
                    total_batches = len(self.data_files) // batch_size
                    batches_per_gpu = total_batches // world_size
                else:
                    # If not drop_last, use ceiling for total batches, then floor for per-GPU
                    total_batches = (len(self.data_files) + batch_size - 1) // batch_size
                    batches_per_gpu = total_batches // world_size

            else:
                if drop_last:
                    batches_per_gpu = len(self.data_files) // batch_size
                else:
                    batches_per_gpu = (len(self.data_files) + batch_size - 1) // batch_size

            # Apply with_epoch at the BATCH level to stop after correct number of batches
            dataloader = dataloader.with_epoch(batches_per_gpu)

            # Wrap the dataloader to provide __len__ method and dataset reference
            dataloader = WebDataLoaderWithLen(dataloader, batches_per_gpu, dataset=self)

            return dataloader

        else:
            # Standard PyTorch DataLoader with DistributedSampler for DDP
            from torch.utils.data import DataLoader

            if is_distributed:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(
                    self,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=drop_last
                )
                dataloader = DataLoader(
                    self,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    **kwargs
                )
            else:
                dataloader = DataLoader(
                    self,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    **kwargs
                )

            return dataloader

    def subsample_to_most_diverse_tiles(
        self,
        pixel_count_files=None,
        diversity_percentage=0.1,
        diversity_metric='inverse_class_freq'
    ):
        """
        Select the most diverse tiles based on class distribution.

        Args:
            pixel_count_files: Paths to parquet files with pixel counts
            diversity_percentage: Fraction of tiles to select (0.1 = 10%)
            diversity_metric: 'num_classes', 'shannon_entropy', 'inverse_class_freq', 'log_inverse_class_freq'

        Returns:
            List of (s2_path, gt_path, cal_path) tuples with diverse tiles
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Load pixel counts using helper
        all_tile_counts, class_pixel_counts, total_pixels = self._load_pixel_counts(pixel_count_files)

        if not all_tile_counts:
            print("No pixel count data found. Returning original data_files.")
            return self.data_files

        print(f"Calculating diversity scores using metric: {diversity_metric}")

        # Calculate diversity score for each tile
        # Map spatial tile name -> all data_files indices (multiple per tile in multi-year)
        from collections import defaultdict as _dd
        tile_name_to_idxs = _dd(list)
        for idx, (s2, _, _) in enumerate(self.data_files):
            name = self._extract_tile_name(s2[0] if isinstance(s2, list) else s2, include_dates=False)
            tile_name_to_idxs[name].append(idx)
        tile_diversity_scores = []

        _seen_tiles: set = set()
        for tile_name, counts in all_tile_counts:
            if tile_name in _seen_tiles or tile_name not in tile_name_to_idxs:
                continue
            _seen_tiles.add(tile_name)

            # Use first index as representative for diversity scoring
            tile_idx = tile_name_to_idxs[tile_name][0]

            # Map LNF codes to target classes for this tile
            tile_class_counts = defaultdict(float)
            for lnf_code, count in counts.items():
                lnf_code_int = int(float(lnf_code))
                if lnf_code_int in self.target_mapping[0]:
                    target_class = self.target_mapping[0][lnf_code_int]
                    # Skip background
                    if target_class == 0 or (self.ignore_index is not None and target_class == self.ignore_index):
                        continue
                    tile_class_counts[target_class] += count

            if not tile_class_counts:
                continue

            # Calculate diversity score
            if diversity_metric == 'num_classes':
                diversity_score = len(tile_class_counts)

            elif diversity_metric == 'shannon_entropy':
                total_tile_pixels = sum(tile_class_counts.values())
                entropy = 0
                for count in tile_class_counts.values():
                    p = count / total_tile_pixels
                    if p > 0:
                        entropy -= p * np.log(p)
                diversity_score = entropy

            elif "inverse_class_freq" in diversity_metric:
                # Inverse frequency weighting
                diversity_score = 0
                for target_class, count in tile_class_counts.items():
                    global_class_pixels = class_pixel_counts[target_class]
                    class_frequency = global_class_pixels / total_pixels

                    # Inverse frequency weight with smoothing
                    inv_freq_weight = 1.0 / (class_frequency + 1e-6)

                    # log smoothing
                    if diversity_metric == "log_inverse_class_freq":
                        inv_freq_weight = np.log1p(inv_freq_weight)

                    # Binary presence weighted by inverse frequency
                    diversity_score += inv_freq_weight

            else:
                raise ValueError(f"Unknown diversity_metric: {diversity_metric}")

            tile_diversity_scores.append((tile_idx, diversity_score, tile_name))

        # Sort and select top percentage
        tile_diversity_scores.sort(key=lambda x: x[1], reverse=True)
        num_diverse_tiles = max(1, int(len(self.data_files) * diversity_percentage))
        selected_tiles = tile_diversity_scores[:num_diverse_tiles]

        # Select tiles (same for both modes now)
        diverse_data_files = [self.data_files[tile_idx] for tile_idx, _, _ in selected_tiles]

        print(f"\nDiversity selection summary:")
        print(f"  Original dataset size: {len(self.data_files)}")
        print(f"  Selected diverse tiles: {len(diverse_data_files)} ({diversity_percentage*100:.1f}%)")
        print(f"  Average selected diversity score: {np.mean([score for _, score, _ in selected_tiles]):.2f}")
        print(f"  Min selected diversity score: {selected_tiles[-1][1]:.2f}")
        print(f"  Max selected diversity score: {selected_tiles[0][1]:.2f}")
        print(f"  Average diversity score across whole dataset: {np.mean([score for _, score, _ in tile_diversity_scores]):.2f}")
        print(f"  Median diversity score across whole dataset: {np.median([score for _, score, _ in tile_diversity_scores]):.2f}")

        return diverse_data_files

    def upsample_tiles_with_minority_classes(
        self,
        pixel_count_files=None,
        minority_threshold=0.005,
        target_ratio=0.02,
        max_replicas=10,
        min_class_pixels=100,
        top_k_tiles_per_class=None,
        target_mapping_override=None,
        mapping_dict_override=None,
    ):
        """
        Upsample tiles containing minority classes by duplicating them.

        Args:
            pixel_count_files: Paths to parquet files with pixel counts
            minority_threshold: Classes below this fraction are considered rare
            target_ratio: Target representation as fraction of dataset size
            max_replicas: Maximum times a single tile can be duplicated
            min_class_pixels: Minimum pixels of rare class required in tile
            top_k_tiles_per_class: Only use top K tiles per class

        Returns:
            List of (s2_path, gt_path, cal_path) tuples with upsampled tiles
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        verbose = self._is_rank_zero()

        # Load pixel counts using helper
        all_tile_counts, class_pixel_counts, total_pixels = self._load_pixel_counts(
            pixel_count_files, target_mapping_override=target_mapping_override)

        # Build name lookup immediately so minority print can use it
        _md = mapping_dict_override if mapping_dict_override is not None else self.mapping_dict[0]
        class_id_to_name = {v: k for k, v in _md.items()}

        if not all_tile_counts:
            if verbose:
                print("No pixel count data found. Returning original data_files.")
            return self.data_files

        # Identify minority classes
        minority_classes = set()
        for target_class, pixel_count in class_pixel_counts.items():
            # Skip background/ignore_index
            if self.ignore_index is not None and target_class == self.ignore_index:
                continue
            if target_class == 0:
                continue

            ratio = pixel_count / total_pixels
            if ratio < minority_threshold:
                minority_classes.add(target_class)
                if verbose:
                    name = class_id_to_name.get(target_class, f"class_{target_class}")
                    print(f"Minority class {name}: {ratio*100:.3f}% of pixels")

        if not minority_classes:
            if verbose:
                print(f"No minority classes found below {minority_threshold*100}% threshold.")
            return self.data_files

        if verbose:
            print(f"\nFound {len(minority_classes)} minority classes to upsample")

        # Find tiles containing minority classes
        minority_class_tiles = defaultdict(list)
        # Map spatial tile name -> all data_files indices (multiple per tile in multi-year)
        from collections import defaultdict as _dd
        tile_name_to_idxs = _dd(list)
        for idx, (s2, _, _) in enumerate(self.data_files):
            name = self._extract_tile_name(s2[0] if isinstance(s2, list) else s2, include_dates=False)
            tile_name_to_idxs[name].append(idx)

        _seen_tiles_upsample: set = set()
        for tile_name, counts in all_tile_counts:
            if tile_name in _seen_tiles_upsample or tile_name not in tile_name_to_idxs:
                continue
            _seen_tiles_upsample.add(tile_name)

            for lnf_code, pixel_count in counts.items():
                if pixel_count < min_class_pixels:
                    continue

                lnf_code_int = int(float(lnf_code))
                _tm2 = target_mapping_override if target_mapping_override is not None else self.target_mapping[0]
                if lnf_code_int in _tm2:
                    target_class = _tm2[lnf_code_int]
                    if target_class in minority_classes:
                        # Add ALL year-indices so every year-instance can be upsampled
                        for tile_idx in tile_name_to_idxs[tile_name]:
                            minority_class_tiles[target_class].append((tile_idx, pixel_count))

        # Sort and select top K tiles per class
        for target_class in minority_class_tiles:
            minority_class_tiles[target_class].sort(key=lambda x: x[1], reverse=True)
            if top_k_tiles_per_class is not None:
                minority_class_tiles[target_class] = minority_class_tiles[target_class][:top_k_tiles_per_class]

        # Calculate replication factors
        tile_replicas = defaultdict(int)
        original_dataset_size = len(self.data_files)
        capped_tiles = 0
        capped_class_counts = defaultdict(int)  # class_name -> number of capped tiles

        for target_class, tiles_info in minority_class_tiles.items():
            if not tiles_info:
                continue

            current_class_tiles = len(tiles_info)
            target_class_tiles = max(int(original_dataset_size * target_ratio), current_class_tiles)
            total_class_pixels = sum(pc for _, pc in tiles_info)
            class_name = class_id_to_name.get(target_class, f"class_{target_class}")

            for tile_idx, pixel_count in tiles_info:
                weight = pixel_count / total_class_pixels
                replicas_raw = int(np.ceil(weight * (target_class_tiles - current_class_tiles)))
                replicas = min(replicas_raw, max_replicas)
                if replicas_raw > max_replicas:
                    capped_tiles += 1
                    capped_class_counts[class_name] += 1
                tile_replicas[tile_idx] = max(tile_replicas[tile_idx], replicas)

        # Build upsampled data_files list
        upsampled_data_files = list(self.data_files)
        for tile_idx, num_replicas in tile_replicas.items():
            tile_data = self.data_files[tile_idx]
            for _ in range(num_replicas):
                upsampled_data_files.append(tile_data)

        np.random.shuffle(upsampled_data_files)

        # Compute upsampled class pixel percentages
        tile_pixel_counts = defaultdict(lambda: defaultdict(float))
        _tm_for_counts = target_mapping_override if target_mapping_override is not None else self.target_mapping[0]
        for tile_name, lnf_counts in all_tile_counts:
            for lnf_code, count in lnf_counts.items():
                lnf_code_int = int(float(lnf_code))
                if lnf_code_int in _tm_for_counts:
                    target_class = _tm_for_counts[lnf_code_int]
                    tile_pixel_counts[tile_name][target_class] += count

        upsampled_class_pixel_counts = dict(class_pixel_counts)
        for tile_idx, num_replicas in tile_replicas.items():
            _s2_ref = self.data_files[tile_idx][0]
            tile_name = self._extract_tile_name(_s2_ref[0] if isinstance(_s2_ref, list) else _s2_ref, include_dates=False)
            for target_class, count in tile_pixel_counts[tile_name].items():
                upsampled_class_pixel_counts[target_class] = (
                    upsampled_class_pixel_counts.get(target_class, 0) + num_replicas * count
                )

        upsampled_total_pixels = sum(
            v for k, v in upsampled_class_pixel_counts.items()
            if k != 0 and (self.ignore_index is None or k != self.ignore_index)
        )

        if verbose:
            print(f"\nUpsampling summary:")
            print(f"  Original dataset size: {original_dataset_size}")
            print(f"  Upsampled dataset size: {len(upsampled_data_files)}")
            print(f"  Tiles duplicated: {len(tile_replicas)}")
            print(f"  Total new tiles added: {len(upsampled_data_files) - original_dataset_size}")
            if tile_replicas:
                print(f"  Average replicas per duplicated tile: {np.mean(list(tile_replicas.values())):.2f}")
            print(f"  Tiles capped at max_replicas ({max_replicas}): {capped_tiles}")
            if capped_class_counts:
                sorted_capped = sorted(capped_class_counts.items(), key=lambda x: x[1], reverse=True)
                class_str = ", ".join(f"{name}: {count}" for name, count in sorted_capped)
                print(f"  Classes with capped tiles: {class_str}")
                print(f"  WARNING: target_ratio may not be fully achieved for these classes.")

            print(f"\n  Class pixel percentages (before -> after upsampling):")
            sorted_classes = sorted(
                [(k, v) for k, v in class_pixel_counts.items()
                 if k != 0 and (self.ignore_index is None or k != self.ignore_index)],
                key=lambda x: x[1], reverse=True
            )
            for target_class, orig_count in sorted_classes:
                name = class_id_to_name.get(target_class, f"class_{target_class}")
                before_pct = orig_count / total_pixels * 100
                after_count = upsampled_class_pixel_counts.get(target_class, 0)
                after_pct = after_count / upsampled_total_pixels * 100
                print(f"    {name}: {before_pct:.3f}% -> {after_pct:.3f}%")

        return upsampled_data_files

    def get_class_weights(self, pixel_count_files=None, beta=0.9):
        """
        Calculate class weights using effective number of samples.
        Refer to:
        Cui et al. (2019): Class-Balanced Loss Based on Effective Number of Samples.
            https://doi.org/10.48550/arXiv.1901.05555


        Args:
            pixel_count_files: Paths to pixel count files
            beta: Hyperparameter for effective number calculation

        Returns:
            np.ndarray: Class weights
        """
        _, class_pixel_counts, _ = self._load_pixel_counts(pixel_count_files)

        if not class_pixel_counts:
            print("No class weight files found.")
            return np.ones(self.num_classes[0] + 1, dtype=np.float32)

        # Get the class counts for the mapped classes
        class_counts = np.zeros(self.num_classes[0] + 1, dtype=np.float32)
        for target_class, count in class_pixel_counts.items():
            class_counts[target_class] = count

        # Remove ignore_index if set
        if self.ignore_index is not None:
            class_counts[self.ignore_index] = 0

        # Identify missing classes
        missing = class_counts == 0

        # Calculate effective number and weights
        if beta == 1.0:
            # Classic normalized inverse-frequency weights
            weights = 1.0 / (class_counts + 1e-6)
            weights[missing] = 1e-6
            weights = weights / np.sum(weights) * (self.num_classes[0] + 1)
        else:
            effective_num = 1.0 - np.power(beta, class_counts)
            effective_num[effective_num <= 0] = 1e-6
            weights = (1.0 - beta) / effective_num
            weights[missing] = 1e-6
            weights = weights / np.sum(weights) * (self.num_classes[0] + 1)
        return weights


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Private Functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~ Basic Functions ~~~ #
    @staticmethod
    def _ensure_list(x):
            return x if isinstance(x, (list, tuple)) else [x]

    def _load_cgdd_bounds(self, gpkg_paths):
        """
        Load CGDD bounds from GPKG file(s) into memory.

        Args:
            gpkg_paths: str or list of str, paths to GPKG files

        Returns:
            dict: {tile_name: {'cgdd_min': float, 'cgdd_max': float}}
        """
        if gpkg_paths is None:
            raise ValueError(
                "cgdd_bounds_gpkg must be provided when use_fixed_temperature_subsampling=True"
            )

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for loading CGDD bounds. "
                "Install with: pip install geopandas"
            )

        gpkg_paths = self._ensure_list(gpkg_paths)
        cgdd_bounds = {}

        for gpkg_path in gpkg_paths:
            if not os.path.exists(gpkg_path):
                raise ValueError(f"CGDD bounds file not found: {gpkg_path}")

            print(f"Loading CGDD bounds from: {gpkg_path}")
            gdf = gpd.read_file(gpkg_path)

            # Validate required columns
            required_cols = ['tile_name', 'cgdd_start_p10', 'cgdd_end_p90']
            missing_cols = [col for col in required_cols if col not in gdf.columns]
            if missing_cols:
                raise ValueError(
                    f"GPKG file missing required columns: {missing_cols}. "
                    f"Available columns: {list(gdf.columns)}"
                )

            # Build lookup dictionary
            for _, row in gdf.iterrows():
                tile_name = row['tile_name']
                cgdd_bounds[tile_name] = {
                    'cgdd_min': float(row['cgdd_start_p10']),
                    'cgdd_max': float(row['cgdd_end_p90'])
                }

            print(f"  Loaded bounds for {len(gdf)} tiles")

        print(f"Total CGDD bounds loaded: {len(cgdd_bounds)} tiles")
        return cgdd_bounds

    @staticmethod
    def _extract_tile_name(path, include_dates=True):
        """
        Extract tile name from path.

        Args:
            path: Full path to zarr file
            include_dates: If False, returns only tile identifier (e.g., 'S2_262230_5113980')
                          If True, returns full name with dates (default behavior)

        Returns:
            str: Tile name
        """
        name = os.path.basename(path)

        # Remove extensions
        if name.endswith('.zarr.zip'):
            name = name[:-9]
        elif name.endswith('.zarr'):
            name = name[:-5]
        else:
            name = os.path.splitext(name)[0]

        # If include_dates=False, remove date suffix
        if not include_dates:
            # Pattern: PREFIX_XCOORD_YCOORD_STARTDATE_ENDDATE
            # Split by underscore and keep first 3 parts
            parts = name.split('_')
            if len(parts) >= 3:
                # Check if parts[3] looks like a date (8 digits)
                if len(parts) >= 4 and parts[3].isdigit() and len(parts[3]) == 8:
                    # Has date suffix, return only prefix + coords
                    return '_'.join(parts[:3])
            return name

        return name

    def _get_tile_identifier(self, file_path):
        """Get tile identifier from file path for CGDD bounds lookup."""
        return self._extract_tile_name(file_path, include_dates=False)

    def _detect_sentinel2_mode(self):
        """Detect if we should use WebDataset based on satellite_paths type."""
        # Check first satellite_paths (satellite_paths is list-of-lists after normalization)
        first_dir = self.satellite_paths[0]
        if isinstance(first_dir, list):
            first_dir = first_dir[0]

        # If it's a directory (not ending in .json, .zarr, etc.), check its contents
        if os.path.isdir(first_dir):
            # Check if directory contains .tar files (WebDataset format)
            tar_files = glob.glob(os.path.join(first_dir, "*.tar"))
            if tar_files:
                return True

            # Check if directory contains .zarr files (regular zarr format)
            zarr_files = glob.glob(os.path.join(first_dir, "*.zarr"))
            if zarr_files:
                return False  # Use regular zarr mode, not WebDataset

            # If neither found, raise error
            raise ValueError(
                f"Directory {first_dir} does not contain .tar files (WebDataset) or .zarr files (regular zarr). "
                f"Please check the path."
            )

        # Otherwise use JSON reference mode (path is likely a .json file)
        return False
    
    @staticmethod
    def _downsample(arr, block_size=3, method='mean'):
        from scipy import stats
        from numpy.lib.stride_tricks import as_strided

        def _block_view(A, block_size=block_size):
            shape = (A.shape[0]//block_size, A.shape[1]//block_size, block_size, block_size)
            strides = (A.strides[0]*block_size, A.strides[1]*block_size, A.strides[0], A.strides[1])
            return as_strided(A, shape=shape, strides=strides)

        pad_y = (block_size - arr.shape[0] % block_size) % block_size
        pad_x = (block_size - arr.shape[1] % block_size) % block_size
        arr = np.pad(arr, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
        blocks = _block_view(arr)

        if method == 'mean':
            return np.nanmean(blocks, axis=(2,3))
        elif method == 'median':
            return np.nanmedian(blocks, axis=(2,3))
        elif method == 'mode':
            flattened = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
            mode_result, _ = stats.mode(flattened, axis=2, nan_policy='omit')
            return mode_result
        else:
            raise ValueError("Method must be one of 'mean', 'median', or 'mode'.")

    @staticmethod
    def _upsample(arr, block_size=3, target_shape=(128, 128)):
        arr_big = np.repeat(np.repeat(arr, block_size, axis=0), block_size, axis=1)
        arr_big = arr_big[:target_shape[0], :target_shape[1]]
        return arr_big


    # ~~~ Webdataset Functions ~~~ #
    @staticmethod
    def _load_zarr_from_bytes(zarr_bytes):
        """Load zarr group from bytes (for WebDataset)."""
        mapper = fsspec.filesystem("zip", fo=io.BytesIO(zarr_bytes), block_size=None).get_mapper("")

        # Strategy 1: Try consolidated metadata with zarr v2
        try:
            zarr_group = zarr.open_consolidated(mapper, mode="r", zarr_format=2)
            return zarr_group
        except (KeyError, FileNotFoundError, zarr.errors.GroupNotFoundError):
            pass

        # Strategy 2: Try regular zarr group with v2
        try:
            zarr_group = zarr.open_group(mapper, mode="r", zarr_format=2)
            return zarr_group
        except zarr.errors.GroupNotFoundError:
            pass

        # Strategy 3: Try without specifying zarr_format (auto-detect)
        try:
            zarr_group = zarr.open_group(mapper, mode="r")
            return zarr_group
        except zarr.errors.GroupNotFoundError:
            pass

        # Strategy 4: Try opening as zarr v3
        try:
            zarr_group = zarr.open_group(mapper, mode="r", zarr_format=3)
            return zarr_group
        except zarr.errors.GroupNotFoundError:
            pass

        # If all strategies fail, raise an error
        raise ValueError("Could not open zarr group from bytes - tried all strategies")
    
    def _process_webdataset_sample(self, sample):
        """
        Process a WebDataset sample: decode Sentinel2 data and attach GT/Calendar.
        Returns a tuple ready for __getitem__ style processing.
        In prediction mode, gt_data may be None.
        """
        # Extract tile name using consistent method
        tile_name = sample.get('__key__', '')
        tile_name = self._extract_tile_name(tile_name)

        # Load Sentinel2 zarr from bytes
        zarr_bytes = sample.get('zarr.zip', None)
        if zarr_bytes is None:
            raise ValueError(f"No zarr.zip found in sample {tile_name}")

        s2_data = self._load_zarr_from_bytes(zarr_bytes)

        # Get GT and Calendar from cache
        gt_data = self.gt_cache.get(tile_name)
        cal_data = self.cal_cache.get(tile_name)

        # Validate calendar (always required)
        if cal_data is None:
            raise ValueError(f"Calendar data not found for tile {tile_name}")

        # Validate GT only in training mode
        if self.mode == 'training' and gt_data is None:
            raise ValueError(f"GT data not found for tile {tile_name} in training mode")

        # Return in format expected by processing logic
        # gt_data may be None in prediction mode (dummy created in _process_tile_data)
        # Include tile_name for fixed temperature subsampling
        return (s2_data, gt_data, cal_data, tile_name)
    
    def _create_webdataset_loader(self, resampled=False, shuffle=True, shuffle_buffer=1000):
        """
        Create a WebDataset loader for streaming Sentinel2 data.
        This should be called from the training script to get an iterable dataset.
        """
        if not self.is_webdataset:
            raise ValueError("_create_webdataset_loader() can only be called in WebDataset mode")

        # Collect all tar files from all sentinel2 directories
        tar_urls = []
        for s2_dir in self.satellite_paths:
            tar_files = sorted(glob.glob(os.path.join(s2_dir, "*.tar")))
            tar_urls.extend(tar_files)

        print(f"Found {len(tar_urls)} tar files for WebDataset streaming")

        # Create WebDataset pipeline with DDP support
        # Note: shardshuffle should happen BEFORE split_by_node/split_by_worker
        shardshuffle_size = len(tar_urls) if shuffle else False
        dataset = wds.WebDataset(
            tar_urls,
            resampled=resampled,
            shardshuffle=shardshuffle_size,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker
        )

        # Add shuffling if requested (shuffles samples within shards)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer, initial=shuffle_buffer)

        # Filter by available tiles BEFORE decoding (more efficient - only decode needed samples)
        # Process samples fully in worker threads (including _process_tile_data)
        dataset = (
            dataset
            .select(self._filter_by_available_tiles)  # Filter before decode - __key__ available from tar metadata
            .decode()
            .map(self._process_webdataset_sample)
            .map(lambda sample: self._process_tile_data(sample[0], sample[1], sample[2], s2_file_path=sample[3]))
        )

        # Note: .with_epoch() will be applied in create_dataloader() at the batch level
        # to ensure correct stopping behavior per GPU
        return dataset


    # ~~~ Functions to get tiles and load data ~~~ #
    def _get_available_tiles(self):
        """
        Get available tiles from all sources (S2, GT, Calendar) using data_sources.
        - Training mode: Requires S2 ∩ GT ∩ Calendar (all three)
        - Prediction mode: Requires S2 ∩ Calendar (GT optional)
        - Filters by year if filter_years is specified
        """
        data_files = []

        # Extract tile prefix helper function
        def get_tile_prefix(filename):
            # Always return the spatial-only prefix LS_X_Y (first 3 parts).
            # GT tiles have no date suffix (LS_X_Y) while satellite/CGDD tiles do
            # (LS_X_Y_STARTDATE_ENDDATE). Using only 3 parts ensures consistent
            # matching across all three sources regardless of date presence.
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            if len(parts) >= 3:
                return f"{parts[0]}_{parts[1]}_{parts[2]}"
            return name

        # Helper to extract year from filename
        def get_year_from_filename(filename):
            """Extract year from tile filename (e.g., S2_262620_5110700_20200103 -> 2020)"""
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            if len(parts) >= 4:
                date_str = parts[3]
                return int(date_str[:4])
            return None

        # Loop over years
        for i in range(len(self.satellite_paths)):
            sensor_zarr_paths = self.satellite_paths[i]  # list of per-sensor zarr paths for this year

            # Build merged s2_index: tile_prefix -> {sensor_zarr_path: group_name}
            # Tiles are collected from all sensor zarrs; missing sensors are silently skipped.
            s2_index = {}
            for sensor_path in sensor_zarr_paths:
                sensor_files = self.data_sources.get(sensor_path, (None, []))[1]
                for f in sensor_files:
                    prefix = get_tile_prefix(f)
                    if prefix not in s2_index:
                        s2_index[prefix] = {}
                    s2_index[prefix][sensor_path] = f

            cal_files = self.data_sources.get(self.temp_paths[i], (None, []))[1]
            cal_index = {get_tile_prefix(f): f for f in cal_files}

            if self.mode == 'training':
                # Training mode: Require all three sources
                gt_dir = self.gt_paths[i]
                gt_files = self.data_sources.get(gt_dir, (None, []))[1]
                gt_index = {get_tile_prefix(f): f for f in gt_files}

                # Find tiles available in all three sources (by prefix)
                all_tiles = set(gt_index) & set(s2_index) & set(cal_index)

                for tile in sorted(all_tiles):
                    gt_file = gt_index[tile]
                    cal_file = cal_index[tile]
                    sensor_group_map = s2_index[tile]  # {sensor_zarr_path: group_name}

                    # Filter by year if specified (use first available sensor's group name)
                    if self.filter_years is not None:
                        any_group = next(iter(sensor_group_map.values()))
                        tile_year = get_year_from_filename(any_group)
                        if tile_year not in self.filter_years:
                            continue

                    # Build list of full s2 paths (one per sensor that has this tile)
                    s2_paths = [
                        os.path.join(sp, gn) for sp, gn in sensor_group_map.items()
                    ]
                    gt_path = os.path.join(gt_dir, gt_file)
                    cal_path = os.path.join(self.temp_paths[i], cal_file)

                    data_files.append((s2_paths, gt_path, cal_path))

            else:  # prediction mode
                # Prediction mode: S2 ∩ Calendar (GT optional)
                all_tiles = set(s2_index) & set(cal_index)

                # Get GT index if GT dirs provided
                gt_index = {}
                if i < len(self.gt_paths) and self.gt_paths[i]:
                    gt_dir = self.gt_paths[i]
                    gt_files = self.data_sources.get(gt_dir, (None, []))[1]
                    gt_index = {get_tile_prefix(f): f for f in gt_files}

                for tile in sorted(all_tiles):
                    cal_file = cal_index[tile]
                    sensor_group_map = s2_index[tile]

                    # Filter by year if specified
                    if self.filter_years is not None:
                        any_group = next(iter(sensor_group_map.values()))
                        tile_year = get_year_from_filename(any_group)
                        if tile_year not in self.filter_years:
                            continue

                    # Build list of full s2 paths
                    s2_paths = [
                        os.path.join(sp, gn) for sp, gn in sensor_group_map.items()
                    ]
                    cal_path = os.path.join(self.temp_paths[i], cal_file)

                    # Check if GT exists for this tile
                    if tile in gt_index:
                        gt_file = gt_index[tile]
                        gt_path = os.path.join(gt_dir, gt_file)
                    else:
                        gt_path = None  # No GT for this tile

                    data_files.append((s2_paths, gt_path, cal_path))

        return data_files

    def _filter_corrupted_tiles(self, data_files):
        """
        Filter out corrupted tiles based on exclusion list.
        Looks for 'corrupted_temperature_tiles.txt' in the current directory.
        """
        exclusion_file = os.path.join(os.getcwd(), 'corrupted_temperature_tiles.txt')

        if not os.path.exists(exclusion_file):
            # No exclusion file, return all tiles
            return data_files

        # Load corrupted tile names
        with open(exclusion_file, 'r') as f:
            corrupted_tiles = set(line.strip() for line in f if line.strip())

        # Filter out corrupted tiles
        original_count = len(data_files)
        filtered_files = [
            (s2, gt, cal) for s2, gt, cal in data_files
            if self._extract_tile_name(s2[0] if isinstance(s2, list) else s2) not in corrupted_tiles
        ]

        excluded_count = original_count - len(filtered_files)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} corrupted tiles from dataset (based on {exclusion_file})")

        return filtered_files

    def _filter_by_available_tiles(self, sample):
        """Filter function for WebDataset - only keep tiles with GT/Calendar data."""
        # Extract tile name from key
        tile_name = sample.get('__key__', '')
        tile_name = self._extract_tile_name(tile_name)

        # Check against precomputed available tile names
        return tile_name in self.available_tile_names

    def _load_and_merge_sensor_data(self, s2_paths):
        """Load one or more sensor zarrs, rename bands to generic names, filter Tier1, merge along time.

        For Sentinel (or any case where bands don't need renaming), acts as a thin
        wrapper around _load_zarr and returns the zarr group directly.
        For Landsat multi-sensor, loads each zarr, applies BAND_RENAME_MAP, filters to
        T1 scenes, concatenates along the time axis, and returns a plain dict of
        numpy arrays keyed by generic band name.
        """
        if isinstance(s2_paths, str):
            s2_paths = [s2_paths]

        # Check whether band renaming is needed (Landsat) or not (Sentinel).
        # self.bands contains generic names (Blue/Green/...) which are the VALUES of
        # BAND_RENAME_MAP, not the keys (OLI_B2/ETM_B1/...).
        needs_rename = any(b in self.BAND_RENAME_MAP.values() for b in self.bands)

        # Single path with no renaming needed → return zarr group directly (Sentinel fast path).
        # Sentinel zarr keys already match self.bands (s2_B02 etc.), so no merge loop needed.
        if not needs_rename and len(s2_paths) == 1:
            return self._load_zarr(s2_paths[0])

        all_times = []
        all_qa = []
        all_band_data = {b: [] for b in self.bands}
        all_sensor_flags = []

        for path in s2_paths:
            try:
                data = self._load_zarr(path)
            except Exception as e:
                print(f"Warning: skipping sensor zarr {path}: {e}")
                continue

            # Tier1 time mask
            n_times = len(data['time'][:])
            time_mask = np.ones(n_times, dtype=bool)
            if self.tier1_scene_ids is not None:
                try:
                    scene_ids = np.array(data['scene_id'][:])
                    time_mask = np.array([sid in self.tier1_scene_ids for sid in scene_ids])
                except KeyError:
                    pass  # no scene_id in this zarr, keep all timesteps

            # Convert relative timestamps (days since tile start) to absolute DOY
            # so that cross-sensor dedup compares calendar dates, not sensor-local offsets.
            doy_offset = 0
            for sep in ('.zarr.zip/', '.json/', '.zarr/'):
                if sep in path:
                    group_key = path.split(sep, 1)[1].lstrip('/')
                    parts = group_key.split('_')
                    if len(parts) >= 4 and len(parts[3]) == 8 and parts[3].isdigit():
                        from datetime import date as _date
                        start_str = parts[3]
                        start = _date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:]))
                        doy_offset = (start - _date(start.year, 1, 1)).days
                    break

            times = (data['time'][:] + doy_offset)[time_mask]
            if len(times) == 0:
                continue

            # Determine sensor type to select the correct band rename map.
            sensor_type = None
            for st, bmap in self.SENSOR_BAND_MAPS.items():
                for raw_name in bmap:
                    try:
                        _ = data[raw_name]
                        sensor_type = st
                        break
                    except (KeyError, FileNotFoundError):
                        continue
                if sensor_type is not None:
                    break
            if sensor_type is None:
                print(f"Warning: skipping sensor {path} — could not determine sensor type")
                continue

            band_map = self.SENSOR_BAND_MAPS[sensor_type]
            generic_to_raw = {v: k for k, v in band_map.items()}

            # Load bands into a temp dict — catches both KeyError and FileNotFoundError
            # (reference stores raise FileNotFoundError when a chunk file is absent).
            temp_band_data = {}
            missing_bands = []
            for generic_band in self.bands:
                raw_band = generic_to_raw.get(generic_band)
                if raw_band is None:
                    missing_bands.append(generic_band)
                    continue
                try:
                    temp_band_data[generic_band] = data[raw_band][:][time_mask]
                except (KeyError, FileNotFoundError):
                    missing_bands.append(raw_band)

            if missing_bands:
                print(f"Warning: skipping sensor {path} | sensor={sensor_type} | missing={missing_bands}")
                continue

            # Atomic append — all_times, all_qa, all_sensor_flags, all_band_data stay in sync.
            all_times.append(times)
            all_qa.append(data['QA_PIXEL'][:][time_mask])
            flag_val = np.int8(self.sensor_flag_map.get(sensor_type, 1))
            all_sensor_flags.append(np.full(int(time_mask.sum()), flag_val, dtype=np.int8))
            for generic_band in self.bands:
                all_band_data[generic_band].append(temp_band_data[generic_band])

        if not all_times:
            raise RuntimeError(f"No valid Tier1 observations found for paths: {s2_paths}")

        merged_time = np.concatenate(all_times)
        merged_qa = np.concatenate(all_qa)
        sort_idx = np.argsort(merged_time, kind='stable')

        result = {
            'time': merged_time[sort_idx],
            'QA_PIXEL': merged_qa[sort_idx],
            'sensor_flag': np.concatenate(all_sensor_flags)[sort_idx],
        }
        for b in self.bands:
            if all_band_data[b]:
                result[b] = np.concatenate(all_band_data[b])[sort_idx]

        return result

    @staticmethod
    def _load_zarr(path):
        # Case 1: zipped zarr file
        if '.zarr.zip/' in path or path.endswith('.zarr.zip'):
            store_path, sep, group_key = path.partition('.zarr.zip')
            store_path += '.zarr.zip'
            group_key = group_key.lstrip('/')

            # Open and cache the zarr group
            if store_path not in zarr_cache:
                store = zarr.storage.ZipStore(store_path, mode='r')
                try:
                    zarr_group = zarr.open_consolidated(store, mode='r')
                except Exception:
                    # Consolidated metadata missing or corrupted, open without it
                    # Explicitly disable consolidated metadata (zarr v3 compatibility)
                    try:
                        zarr_group = zarr.open(store, mode='r', use_consolidated=False)
                    except TypeError:
                        # zarr v2 doesn't have use_consolidated parameter
                        zarr_group = zarr.open(store, mode='r')
                # Always cache the opened group
                zarr_cache[store_path] = zarr_group

            # Return the requested group or array
            if group_key == "":
                return zarr_cache[store_path]
            else:
                # Try to access the key from the cached group
                try:
                    return zarr_cache[store_path][group_key]
                except KeyError:
                    # Key not found in consolidated metadata (corrupted)
                    # Fallback: reopen without consolidated and try again
                    store = zarr.storage.ZipStore(store_path, mode='r')
                    try:
                        zarr_group = zarr.open(store, mode='r', use_consolidated=False)
                    except TypeError:
                        zarr_group = zarr.open(store, mode='r')
                    # Update cache with non-consolidated version
                    zarr_cache[store_path] = zarr_group
                    return zarr_group[group_key]

        # Case 2: JSON reference
        elif path.endswith(".json") or ".json/" in path:
            store_path, sep, group_key = path.partition(".json")
            store_path += ".json"
            group_key = group_key.lstrip("/")
            if store_path not in zarr_cache:
                import json as _json
                with open(store_path) as _f:
                    _ref = _json.load(_f)
                # Resolve relative template paths against the JSON's own directory
                # so the JSON is portable (not tied to an absolute path on one machine)
                if "templates" in _ref:
                    _base = os.path.dirname(os.path.abspath(store_path))
                    _ref["templates"] = {
                        k: os.path.join(_base, v) if not os.path.isabs(v) and "://" not in v else v
                        for k, v in _ref["templates"].items()
                    }
                mapper = fsspec.filesystem("reference", fo=_ref).get_mapper("")
                zarr_cache[store_path] = zarr.open_group(mapper, mode="r")
            return zarr_cache[store_path] if group_key == "" else zarr_cache[store_path][group_key]

        # Case 3: local zarr file
        elif '.zarr/' in path or path.endswith('.zarr'):
            store_path, sep, group_key = path.partition('.zarr')
            store_path += '.zarr'
            group_key = group_key.lstrip('/')

            # Open and cache the zarr group
            if store_path not in zarr_cache:
                store = zarr.storage.LocalStore(store_path)
                try:
                    zarr_group = zarr.open_consolidated(store, mode='r')
                except Exception:
                    # Consolidated metadata missing or corrupted, open without it
                    # Explicitly disable consolidated metadata (zarr v3 compatibility)
                    try:
                        zarr_group = zarr.open(store, mode='r', use_consolidated=False)
                    except TypeError:
                        # zarr v2 doesn't have use_consolidated parameter
                        zarr_group = zarr.open(store, mode='r')
                # Always cache the opened group
                zarr_cache[store_path] = zarr_group

            # Return the requested group or array
            if group_key == "":
                return zarr_cache[store_path]
            else:
                # Try to access the key from the cached group
                try:
                    return zarr_cache[store_path][group_key]
                except KeyError:
                    # Key not found in consolidated metadata (corrupted)
                    # Fallback: reopen without consolidated and try again
                    store = zarr.storage.LocalStore(store_path)
                    try:
                        zarr_group = zarr.open(store, mode='r', use_consolidated=False)
                    except TypeError:
                        zarr_group = zarr.open(store, mode='r')
                    # Update cache with non-consolidated version
                    zarr_cache[store_path] = zarr_group
                    return zarr_group[group_key]

        else:
            raise ValueError(f"Unsupported zarr path format: {path}")

    def _load_and_cache_data_sources(self):
        """
        Unified function to load and cache data sources.
        Handles all data source types:
        1) Plain directory with .zarr files -> list files, no caching
        2) Directory with .zgroup (zarr group) -> open and cache group
        3) Directory with .tar files (webdataset) -> get names from .json, no caching
        4) .zarr.zip file -> open via ZipStore and cache
        5) .json file (kerchunk) -> load via fsspec and cache

        Returns:
            dict: Mapping of path -> (zarr_group or None, list of file/key names)
        """
        data_sources = {}
        # satellite_paths is list-of-lists; flatten to individual zarr paths
        flat_satellite_paths = [p for year_paths in self.satellite_paths for p in year_paths]
        paths = flat_satellite_paths + self.gt_paths + self.temp_paths

        for path in paths:
            if not isinstance(path, str):
                continue

            # Case 5: JSON kerchunk reference
            if path.endswith('.json') and os.path.isfile(path):
                # Get file names from JSON directly
                with open(path) as f:
                    ref_data = json.load(f)
                file_names = sorted({
                    key.split("/")[0]
                    for key in ref_data["refs"].keys()
                    if "/" in key and not key.split("/")[0].startswith('.')
                })

                # Cache the group
                if path not in zarr_cache:
                    # Resolve relative template paths to absolute, matching _load_zarr Case 2
                    _base = os.path.dirname(os.path.abspath(path))
                    if "templates" in ref_data:
                        ref_data["templates"] = {
                            k: os.path.join(_base, v) if not os.path.isabs(v) and "://" not in v else v
                            for k, v in ref_data["templates"].items()
                        }
                    mapper = fsspec.filesystem("reference", fo=ref_data).get_mapper("")
                    try:
                        zarr_group = zarr.open_group(mapper, mode="r", use_consolidated=False)
                    except TypeError:
                        zarr_group = zarr.open_group(mapper, mode="r")
                    zarr_cache[path] = zarr_group

                data_sources[path] = (zarr_cache[path], file_names)

            # Case 4: .zarr.zip file
            elif path.endswith('.zarr.zip') and os.path.isfile(path):
                if path not in zarr_cache:
                    store = zarr.storage.ZipStore(path, mode='r')
                    try:
                        zarr_group = zarr.open_consolidated(store, mode='r')
                    except Exception:
                        zarr_group = zarr.open(store, mode='r')
                    zarr_cache[path] = zarr_group
                    # print(f"Cached .zarr.zip: {path}")

                file_names = list(zarr_cache[path].keys())
                data_sources[path] = (zarr_cache[path], file_names)

            # Cases 1-3: Directory
            elif os.path.isdir(path):
                # Case 3: WebDataset with .tar files
                tar_files = glob.glob(os.path.join(path, "*.tar"))
                if tar_files:
                    # Get file names from accompanying .json file
                    json_file = path.rstrip('/') + ".json"
                    if os.path.isfile(json_file):
                        with open(json_file) as f:
                            ref_data = json.load(f)
                        file_names = sorted({
                            key.split("/")[0]
                            for key in ref_data["refs"].keys()
                            if "/" in key and not key.split("/")[0].startswith('.')
                        })
                    else:
                        print(f"Warning: WebDataset directory {path} has no .json file list")
                        file_names = []

                    # No caching for webdataset
                    data_sources[path] = (None, file_names)

                # Case 2: Zarr group (directory with .zgroup)
                elif os.path.isfile(os.path.join(path, '.zgroup')):
                    if path not in zarr_cache:
                        store = zarr.storage.LocalStore(path)
                        try:
                            zarr_group = zarr.open_consolidated(store, mode='r')
                        except Exception:
                            zarr_group = zarr.open(store, mode='r')
                        zarr_cache[path] = zarr_group
                        print(f"Cached zarr group directory: {path}")

                    file_names = list(zarr_cache[path].keys())
                    data_sources[path] = (zarr_cache[path], file_names)

                # Case 1: Plain directory with .zarr files
                else:
                    files = [f for f in os.listdir(path) if f.endswith('.zarr')]
                    # No caching for plain directories
                    data_sources[path] = (None, files)

        print(f"Data sources loaded. Cached {len(zarr_cache)} zarr groups.")
        return data_sources

    def _load_gt_and_calendar_cache(self):
        """
        Load GT and Calendar data into memory for WebDataset mode.
        Uses self.data_files which contains (s2_path, gt_path, cal_path) tuples.
        In prediction mode, GT may be None for some tiles.
        """
        gt_cache = {}
        cal_cache = {}

        for s2_path, gt_path, cal_path in self.data_files:
            # Extract tile name from s2_path using consistent method
            tile_name = self._extract_tile_name(s2_path)

            try:
                # Always load calendar
                cal_data = self._load_zarr(cal_path)
                cal_cache[tile_name] = cal_data

                # Load GT only if path exists (training mode or prediction with GT)
                if gt_path is not None:
                    gt_data = self._load_zarr(gt_path)
                    gt_cache[tile_name] = gt_data
                else:
                    # Prediction mode: Store None (dummy created in processing)
                    gt_cache[tile_name] = None

            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load tile {tile_name}: {e}")
                continue

        return gt_cache, cal_cache

    # ~~~ Ground Truth Functions ~~~ #
    def _map_lnf_code_to_ground_truth(self, label_columns=["4th_tier_ENG"]):
        """
        Create hierarchical mappings (any number of levels) from an Excel or CSV label sheet.

        For multi-head classification (when multi_head_fine_column and
        multi_head_coarse_column are set), creates separate fine-grained mappings
        for Grassland and Arable heads.  Output y_levels layout in that case:
            y_levels[0] = grassland head labels (fine classes for grassland pixels, 0 elsewhere)
            y_levels[1] = arable head labels    (fine classes for arable pixels,    0 elsewhere)
            y_levels[2] = coarse labels         (all pixels)

        Args:
            label_columns (list[str]): Ordered list of column names representing hierarchy levels
        """
        # Load the LNF file
        ext = os.path.splitext(self.label_sheet_file)[1].lower()
        if ext in ['.xls', '.xlsx']:
            label_sheet = pd.read_excel(self.label_sheet_file)
        elif ext == '.csv':
            label_sheet = pd.read_csv(self.label_sheet_file)

        # Remove rows that should not be used
        if "Exclude" in label_sheet.columns:
            label_sheet = label_sheet[label_sheet["Exclude"] != True]

        # ---- Multi-head branch ----
        is_multihead = (
            self.multi_head_fine_column is not None
            and self.multi_head_coarse_column is not None
            and self.multi_head_fine_column in label_columns
            and self.multi_head_coarse_column in label_columns
        )

        if is_multihead:
            fine_col   = self.multi_head_fine_column
            coarse_col = self.multi_head_coarse_column
            grassland_values = set(self.grassland_coarse_values)
            arable_values    = set(self.arable_coarse_values)

            # Coarse mapping: all unique values in coarse_col
            coarse_unique  = sorted(label_sheet[coarse_col].dropna().unique())
            coarse_mapping = {name: i + 1 for i, name in enumerate(coarse_unique)}
            coarse_target  = {0: 0, -1: 0}

            # Grassland fine mapping: fine classes for grassland pixels only
            g_df             = label_sheet[label_sheet[coarse_col].isin(grassland_values)]
            g_fine_classes   = sorted(g_df[fine_col].dropna().unique())
            grassland_mapping = {name: i + 1 for i, name in enumerate(g_fine_classes)}
            grassland_target  = {0: 0, -1: 0}

            # Arable fine mapping: fine classes for arable+permanent pixels only
            a_df            = label_sheet[label_sheet[coarse_col].isin(arable_values)]
            a_fine_classes  = sorted(a_df[fine_col].dropna().unique())
            arable_mapping  = {name: i + 1 for i, name in enumerate(a_fine_classes)}
            arable_target   = {0: 0, -1: 0}

            # Derive domain IDs from the coarse mapping
            grassland_coarse_ids = [coarse_mapping[v] for v in self.grassland_coarse_values if v in coarse_mapping]
            arable_coarse_ids    = [coarse_mapping[v] for v in self.arable_coarse_values    if v in coarse_mapping]

            # Store on self so train script can read them after dataset creation
            self.grassland_coarse_ids = grassland_coarse_ids
            self.arable_coarse_ids    = arable_coarse_ids
            self.num_grassland_classes = len(grassland_mapping)
            self.num_arable_classes    = len(arable_mapping)
            self.num_coarse_classes    = len(coarse_mapping)

            # --- Per-branch lv2/lv3 mappings for hierarchical loss within each branch ---
            lv2_col = self.multi_head_medium_column
            grassland_lv2_mapping = None
            arable_lv2_mapping    = None
            grassland_lv3_mapping = None
            arable_lv3_mapping    = None
            grassland_child_maps  = None
            arable_child_maps     = None
            grassland_lv2_target  = None
            arable_lv2_target     = None
            grassland_lv3_target  = None
            arable_lv3_target     = None

            if lv2_col is not None and lv2_col in label_sheet.columns:
                # lv2: medium classes within each domain
                g_lv2_classes = sorted(g_df[lv2_col].dropna().unique())
                a_lv2_classes = sorted(a_df[lv2_col].dropna().unique())
                grassland_lv2_mapping = {name: i + 1 for i, name in enumerate(g_lv2_classes)}
                arable_lv2_mapping    = {name: i + 1 for i, name in enumerate(a_lv2_classes)}
                grassland_lv2_target  = {0: 0, -1: 0}
                arable_lv2_target     = {0: 0, -1: 0}
                # child_maps level 0: {lv1_local_id: [lv2_local_ids]} (fine → medium)
                g_child_map = {i: [] for i in range(1, len(grassland_mapping) + 1)}
                a_child_map = {i: [] for i in range(1, len(arable_mapping) + 1)}

                # lv3: coarse-within-domain (re-indexed coarse classes per domain)
                g_lv3_classes = sorted(g_df[coarse_col].dropna().unique())
                a_lv3_classes = sorted(a_df[coarse_col].dropna().unique())
                grassland_lv3_mapping = {name: i + 1 for i, name in enumerate(g_lv3_classes)}
                arable_lv3_mapping    = {name: i + 1 for i, name in enumerate(a_lv3_classes)}
                grassland_lv3_target  = {0: 0, -1: 0}
                arable_lv3_target     = {0: 0, -1: 0}
                # child_maps level 1: {lv2_local_id: [lv3_local_ids]} (medium → coarse-within-domain)
                g_child_map2 = {i: [] for i in range(1, len(grassland_lv2_mapping) + 1)}
                a_child_map2 = {i: [] for i in range(1, len(arable_lv2_mapping) + 1)}

            if self._is_rank_zero():
                print(f"\n=== Multi-Head Configuration ===")
                print(f"Fine column:    {fine_col}")
                print(f"Medium column:  {lv2_col}")
                print(f"Coarse column:  {coarse_col}")
                print(f"Grassland head: {len(grassland_mapping)} classes (coarse IDs {grassland_coarse_ids})")
                print(f"  Classes: {g_fine_classes}")
                print(f"Arable head:    {len(arable_mapping)} classes (coarse IDs {arable_coarse_ids})")
                print(f"  Classes: {a_fine_classes}")
                print(f"Coarse head:    {len(coarse_mapping)} classes")
                print(f"  Classes: {coarse_unique}")
                if grassland_lv2_mapping is not None:
                    print(f"Grassland lv2:  {len(grassland_lv2_mapping)} medium classes (for hierarchical loss)")
                    print(f"Arable lv2:     {len(arable_lv2_mapping)} medium classes (for hierarchical loss)")
                    print(f"Grassland lv3:  {len(grassland_lv3_mapping)} coarse-within-domain classes")
                    print(f"Arable lv3:     {len(arable_lv3_mapping)} coarse-within-domain classes")
                print(f"================================\n")

            # Assign LNF codes to all mappings
            for _, row in label_sheet.iterrows():
                lnf        = int(row["LNF_code"])
                coarse_val = row[coarse_col]
                fine_val   = row[fine_col]
                coarse_target[lnf] = coarse_mapping.get(coarse_val, 0)
                if coarse_val in grassland_values:
                    grassland_target[lnf] = grassland_mapping.get(fine_val, 0)
                    arable_target[lnf]    = 0
                    if grassland_lv2_mapping is not None:
                        lv2_val = row[lv2_col]
                        lv1_id  = grassland_mapping.get(fine_val)
                        lv2_id  = grassland_lv2_mapping.get(lv2_val) if pd.notna(lv2_val) else None
                        grassland_lv2_target[lnf] = lv2_id if lv2_id is not None else 0
                        if lv1_id and lv2_id:
                            g_child_map[lv1_id].append(lv2_id)
                        lv3_id = grassland_lv3_mapping.get(coarse_val)
                        grassland_lv3_target[lnf] = lv3_id if lv3_id is not None else 0
                        if lv2_id and lv3_id:
                            g_child_map2[lv2_id].append(lv3_id)
                elif coarse_val in arable_values:
                    grassland_target[lnf] = 0
                    arable_target[lnf]    = arable_mapping.get(fine_val, 0)
                    if arable_lv2_mapping is not None:
                        lv2_val = row[lv2_col]
                        lv1_id  = arable_mapping.get(fine_val)
                        lv2_id  = arable_lv2_mapping.get(lv2_val) if pd.notna(lv2_val) else None
                        arable_lv2_target[lnf] = lv2_id if lv2_id is not None else 0
                        if lv1_id and lv2_id:
                            a_child_map[lv1_id].append(lv2_id)
                        lv3_id = arable_lv3_mapping.get(coarse_val)
                        arable_lv3_target[lnf] = lv3_id if lv3_id is not None else 0
                        if lv2_id and lv3_id:
                            a_child_map2[lv2_id].append(lv3_id)
                else:
                    grassland_target[lnf] = 0
                    arable_target[lnf]    = 0

            if grassland_lv2_mapping is not None:
                # 2-transition child_maps: [lv1→lv2, lv2→lv3]
                grassland_child_maps = [
                    {k: sorted(set(v)) for k, v in g_child_map.items()},
                    {k: sorted(set(v)) for k, v in g_child_map2.items()},
                ]
                arable_child_maps = [
                    {k: sorted(set(v)) for k, v in a_child_map.items()},
                    {k: sorted(set(v)) for k, v in a_child_map2.items()},
                ]

            # y_levels order: [grass_lv1, arable_lv1, coarse, grass_lv2, arable_lv2, grass_lv3, arable_lv3]
            mapping_dict_list   = [grassland_mapping, arable_mapping, coarse_mapping]
            target_mapping_list = [grassland_target,  arable_target,  coarse_target]
            num_classes_list    = [len(grassland_mapping), len(arable_mapping), len(coarse_mapping)]
            if grassland_lv2_mapping is not None:
                mapping_dict_list.extend([grassland_lv2_mapping, arable_lv2_mapping,
                                          grassland_lv3_mapping, arable_lv3_mapping])
                target_mapping_list.extend([grassland_lv2_target, arable_lv2_target,
                                            grassland_lv3_target, arable_lv3_target])
                num_classes_list.extend([len(grassland_lv2_mapping), len(arable_lv2_mapping),
                                         len(grassland_lv3_mapping), len(arable_lv3_mapping)])

            self.mapping_dict        = mapping_dict_list
            self.target_mapping      = target_mapping_list
            self.num_classes         = num_classes_list
            self.child_maps          = []
            self.grassland_child_maps = grassland_child_maps
            self.arable_child_maps    = arable_child_maps
            return

        n_levels = len(label_columns)

        # Remove rows that should not be used
        if "Exclude" in label_sheet.columns:
            label_sheet = label_sheet[label_sheet["Exclude"] != True]

        # Storage for per-level mappings and target dictionaries
        level_mappings = []
        target_mappings = []
        num_classes = []

        # Map unique values for all levels to integers (name → integer ID)
        for level_idx, col in enumerate(label_columns):
            mapping = {name: i + 1 for i, name in enumerate(label_sheet[col].unique())}
            level_mappings.append(mapping)
            target_mapping = {0: 0, -1: 0}
            target_mappings.append(target_mapping)
            num_classes.append(len(mapping))

        # Assign all integers to LNF codes
        for _, row in label_sheet.iterrows():
            lnf = int(row["LNF_code"])
            for level_idx, col in enumerate(label_columns):
                target_mappings[level_idx][lnf] = level_mappings[level_idx][row[col]]

        # Build child maps between consecutive levels
        child_maps = []
        for parent_idx in range(n_levels - 1):
            parent_map = level_mappings[parent_idx]
            child_map = level_mappings[parent_idx + 1]
            mapping = {i: [] for i in range(1, len(parent_map) + 1)}
            for _, row in label_sheet.iterrows():
                parent_id = parent_map[row[label_columns[parent_idx]]]
                child_id = child_map[row[label_columns[parent_idx + 1]]]
                mapping[parent_id].append(child_id)
            # remove duplicates
            mapping = {k: sorted(set(v)) for k, v in mapping.items()}
            child_maps.append(mapping)

        self.mapping_dict = level_mappings
        self.target_mapping = target_mappings
        self.child_maps = child_maps
        self.num_classes = num_classes

    def _is_rank_zero(self):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        return True

    def _load_pixel_counts(self, pixel_count_files=None, target_mapping_override=None):
        """
        Load pixel count data from parquet/CSV files.

        Correctly handles duplicate tiles in data_files (from upsampling):
        if a tile appears N times, its pixel counts are counted N times.

        Args:
            pixel_count_files: Paths to pixel count files (uses GT dirs if None)

        Returns:
            tuple: (all_tile_counts, class_pixel_counts, total_pixels)
                - all_tile_counts: List of (tile_name, {lnf_code: count}) tuples
                - class_pixel_counts: Dict of {target_class: total_pixel_count}
                - total_pixels: Total pixels across all non-background classes
        """
        # Get paths to pixel count files
        if pixel_count_files is None:
            import re as _re
            path_list = []
            for i, path in enumerate(self.gt_paths):
                year = None
                candidate_dirs = []

                if path and path != '':
                    dirname = os.path.dirname(path) if not os.path.isdir(path) else path
                    basename = os.path.basename(path)
                    m = _re.search(r'(\d{4})', basename)
                    if m:
                        year = m.group(1)
                    candidate_dirs.append(dirname)

                # Fallback for prediction mode (empty gt_path): use satellite_paths[i]
                if year is None and i < len(self.satellite_paths):
                    for sp in self.satellite_paths[i]:
                        m = _re.search(r'(\d{4})', os.path.basename(sp))
                        if m:
                            year = m.group(1)
                            candidate_dirs.append(os.path.dirname(sp))
                            break

                if year is None:
                    continue

                for cdir in candidate_dirs:
                    candidate = os.path.join(cdir, f"{year}_pixelCounts")
                    if os.path.exists(candidate + ".parquet") or os.path.exists(candidate + ".csv"):
                        path_list.append(candidate)
                        break
        else:
            path_list = self._ensure_list(pixel_count_files)

        # Load all tile counts
        all_tile_counts = []
        # Extract tile names from data_files using consistent method
        # Count how many times each tile appears (for handling upsampled/duplicated tiles)
        from collections import Counter
        gt_files_list = [self._extract_tile_name(s2[0] if isinstance(s2, list) else s2, include_dates=False) for s2, _, _ in self.data_files]
        tile_frequency = Counter(gt_files_list)
        unique_gt_files = list(tile_frequency.keys())

        for path in path_list:
            try:
                path_pq = path + ".parquet"
                df = pd.read_parquet(path_pq)
                # Normalize tile names to prefix format (strip date suffix) so that
                # both Sentinel (dated: S2_X_Y_YYYYMMDD_YYYYMMDD) and Landsat (prefix: LS_X_Y)
                # parquets match the unique_gt_files set which always uses include_dates=False.
                df['tile'] = df['tile'].apply(lambda t: self._extract_tile_name(t, include_dates=False))
                df = df[df['tile'].isin(unique_gt_files)]

                # Check format: long format (tile, lnf_code, count) or old wide format (tile, counts)
                if 'counts' in df.columns:
                    # Old wide format with nested dicts
                    for _, row in df.iterrows():
                        tile_name = row['tile']
                        counts = eval(row['counts']) if isinstance(row['counts'], str) else row['counts']
                        # Convert None values to 0 and ensure numeric
                        counts = {k: (0 if v is None else v) for k, v in counts.items()}
                        # Add this tile once for each occurrence in data_files
                        for _ in range(tile_frequency[tile_name]):
                            all_tile_counts.append((tile_name, counts))
                else:
                    # New long format: group by tile and reconstruct dict
                    # Detect the property column (e.g., 'lnf_code')
                    property_col = [col for col in df.columns if col not in ['tile', 'count']][0]
                    grouped = df.groupby('tile')
                    for tile_name, group in grouped:
                        counts = dict(zip(group[property_col].astype(str), group['count']))
                        # Convert None values to 0 and ensure numeric
                        counts = {k: (0 if v is None else v) for k, v in counts.items()}
                        # Add this tile once for each occurrence in data_files
                        for _ in range(tile_frequency[tile_name]):
                            all_tile_counts.append((tile_name, counts))
            except Exception:
                try:
                    path_csv = path + ".csv"
                    df = pd.read_csv(path_csv)
                    df['tile'] = df['tile'].apply(lambda t: self._extract_tile_name(t, include_dates=False))
                    df = df[df['tile'].isin(unique_gt_files)]

                    for _, row in df.iterrows():
                        tile_name = row['tile']
                        counts = eval(row['counts']) if isinstance(row['counts'], str) else row['counts']
                        counts = {k: (0 if v is None else v) for k, v in counts.items()}
                        # Add this tile once for each occurrence in data_files
                        for _ in range(tile_frequency[tile_name]):
                            all_tile_counts.append((tile_name, counts))
                except Exception as e:
                    print(f"Warning: Could not load pixel counts from {path}: {e}")

        if not all_tile_counts:
            return [], {}, 0

        # Map LNF codes to target classes and calculate pixel counts
        class_pixel_counts = defaultdict(float)

        _tm = target_mapping_override if target_mapping_override is not None else self.target_mapping[0]
        for tile_name, counts in all_tile_counts:
            for lnf_code, count in counts.items():
                lnf_code_int = int(float(lnf_code))
                if lnf_code_int in _tm:
                    target_class = _tm[lnf_code_int]
                    class_pixel_counts[target_class] += count

        # Calculate total pixels (exclude background/ignore_index)
        total_pixels = 0
        for target_class, pixel_count in class_pixel_counts.items():
            # Skip background/ignore_index
            if self.ignore_index is not None and target_class == self.ignore_index:
                continue
            if target_class == 0:
                continue
            total_pixels += pixel_count

        return all_tile_counts, dict(class_pixel_counts), total_pixels

    def _load_channel_stats(self, stats_files=None):
        """
        Load channel statistics (mean, std) from parquet files generated by generateGT.py.

        Args:
            stats_files: Paths to stats parquet files (uses satellite_paths if None)

        Returns:
            dict: {band_name: {'mean': float, 'std': float}}
        """
        # Get paths to stats files
        if stats_files is None:
            import re as _re
            path_list = []
            for i, path in enumerate(self.gt_paths):
                year = None
                candidate_dirs = []

                if path and path != '':
                    dirname = os.path.dirname(path) if not os.path.isdir(path) else path
                    basename = os.path.basename(path)
                    m = _re.search(r'(\d{4})', basename)
                    if m:
                        year = m.group(1)
                    candidate_dirs.append(dirname)

                # Fallback for prediction mode (empty gt_path): use satellite_paths[i]
                if year is None and i < len(self.satellite_paths):
                    for sp in self.satellite_paths[i]:
                        m = _re.search(r'(\d{4})', os.path.basename(sp))
                        if m:
                            year = m.group(1)
                            candidate_dirs.append(os.path.dirname(sp))
                            break

                if year is None:
                    continue

                for cdir in candidate_dirs:
                    stats_path = os.path.join(cdir, f"{year}_stats.parquet")
                    if os.path.exists(stats_path):
                        path_list.append(stats_path)
                        break
        else:
            path_list = self._ensure_list(stats_files)

        if not path_list:
            print("Warning: No stats files found. Using hardcoded channel statistics.")
            return None

        # Load stats from all files
        all_band_stats = []
        tile_names = {self._extract_tile_name(s2[0] if isinstance(s2, list) else s2, include_dates=False) for s2, _, _ in self.data_files}

        for stats_path in path_list:
            try:
                df = pd.read_parquet(stats_path)

                # Filter to only tiles in our dataset
                # Normalize parquet tile names to prefix format (strip date suffix) so that
                # both Sentinel (dated: S2_X_Y_YYYYMMDD_YYYYMMDD) and Landsat (prefix: LS_X_Y)
                # parquets match the tile_names set which always uses include_dates=False.
                df_tile_match = df['tile'].apply(lambda t: self._extract_tile_name(t, include_dates=False))
                df_filtered = df[df_tile_match.isin(tile_names)]

                if df_filtered.empty:
                    continue

                # Extract stats for each tile (long format: one row per tile-band)
                for tile_name, tile_df in df_filtered.groupby('tile'):
                    tile_stats = {}
                    for _, row in tile_df.iterrows():
                        if row['band'] not in self.bands:
                            continue
                        tile_stats[row['band']] = {
                            'count': int(row['count']),
                            'mean': float(row['mean']),
                            'std': float(row['std'])
                        }
                    if tile_stats:
                        all_band_stats.append(tile_stats)

            except Exception as e:
                print(f"Warning: Could not load channel stats from {stats_path}: {e}")

        if not all_band_stats:
            print("Warning: No valid stats loaded. Using hardcoded channel statistics.")
            return None

        # Combine statistics across all tiles (weighted by pixel count)
        # This follows the same logic as combine_image_stats in generateGT.py
        bands = all_band_stats[0].keys()
        combined_stats = {}

        for band in bands:
            counts = np.array([s[band]['count'] for s in all_band_stats if s[band]['count'] > 0])
            means = np.array([s[band]['mean'] for s in all_band_stats if s[band]['count'] > 0])
            stds = np.array([s[band]['std'] for s in all_band_stats if s[band]['count'] > 0])

            if len(counts) == 0:
                continue

            # Compute weighted global mean
            global_mean = np.sum(means * counts) / np.sum(counts)

            # Compute weighted variance
            global_var = (
                np.sum((counts - 1) * stds**2 + counts * (means - global_mean)**2)
                / (np.sum(counts) - 1)
            )
            global_std = np.sqrt(global_var)

            combined_stats[band] = {
                'mean': float(global_mean),
                'std': float(global_std)
            }

        return combined_stats

    def _load_gdd_stats(self, stats_files=None):
        """
        Load GDD/temperature calendar statistics from parquet files.

        Args:
            stats_files: Paths to stats parquet files (uses temp_paths if None)

        Returns:
            dict: {'mean': float, 'std': float}
        """
        # Get paths to stats files
        if stats_files is None:
            path_list = []
            for path in self.temp_paths:
                # Extract year from the path
                dirname = os.path.dirname(path) if not os.path.isdir(path) else path
                basename = os.path.basename(path)

                # Try to extract year from filename
                year_match = None
                if basename:
                    import re
                    year_match = re.search(r'(\d{4})', basename)

                if year_match:
                    year = year_match.group(1)
                    # Look for stats file in parent directory
                    parent_dir = os.path.dirname(dirname) if not os.path.isdir(path) else dirname
                    stats_path = os.path.join(parent_dir, f"{year}_stats.parquet")
                    if os.path.exists(stats_path):
                        path_list.append(stats_path)
        else:
            path_list = self._ensure_list(stats_files)

        if not path_list:
            print("Warning: No stats files found for GDD. Using default GDD statistics.")
            return None

        # Load GDD stats from all files
        gdd_values = []
        tile_names = {self._extract_tile_name(s2[0] if isinstance(s2, list) else s2, include_dates=False) for s2, _, _ in self.data_files}

        for stats_path in path_list:
            try:
                df = pd.read_parquet(stats_path)

                # Filter to only tiles in our dataset
                # Normalize parquet tile names to prefix format (same logic as _load_channel_stats)
                df_tile_match = df['tile'].apply(lambda t: self._extract_tile_name(t, include_dates=False))
                df_filtered = df[df_tile_match.isin(tile_names)]

                if df_filtered.empty:
                    continue

                # Look for the configured GDD mean variable
                for _, row in df_filtered.iterrows():
                    tile_stats = row['stats']
                    if isinstance(tile_stats, str):
                        import ast
                        tile_stats = ast.literal_eval(tile_stats)

                    # Use the configured GDD variable name
                    if self.gdd_mean_var in tile_stats:
                        # Extract individual GDD values (not aggregated mean/std)
                        # We need the raw distribution, so we'll use the tile-level stats
                        count = tile_stats[self.gdd_mean_var]['count']
                        mean = tile_stats[self.gdd_mean_var]['mean']
                        std = tile_stats[self.gdd_mean_var]['std']

                        if count > 0 and mean is not None:
                            # Approximate the distribution (not perfect, but reasonable)
                            # We collect weighted samples
                            gdd_values.extend([mean] * min(int(count / 1000), 100))  # Sample for efficiency

            except Exception as e:
                print(f"Warning: Could not load GDD stats from {stats_path}: {e}")

        if not gdd_values:
            print("Warning: No valid GDD stats loaded. Using default GDD statistics.")
            return None

        gdd_array = np.array(gdd_values)
        gdd_stats = {
            'mean': float(np.mean(gdd_array)),
            'std': float(np.std(gdd_array))
        }

        return gdd_stats


    # ~~~ Functions for individual samples ~~~ #   
    def _remove_duplicate_timestamps(self, time_stamps, s2_data):
        keep_indices = np.arange(len(time_stamps))
        unique_timestamps, idx, counts = np.unique(time_stamps, return_index=True, return_counts=True)
        duplicated_timestamps = unique_timestamps[counts > 1]

        if len(duplicated_timestamps) == 0:
            return keep_indices, time_stamps
        else:
            # Get indices of all duplicated timesteps
            indices_duplicates = []
            for ts in duplicated_timestamps:
                indices_duplicates.append(np.where(time_stamps == ts)[0])
            all_duplicated_indices = np.concat(indices_duplicates)

            # Load cloud mask for duplicated timesteps.
            # Both sensors return: 0=clear, 1=cloud/NULL, 2=shadow, 3=snow.
            cloud_mask = self._get_cloud_mask(s2_data, all_duplicated_indices)

            # For each duplicate group, keep the scene with fewest hard-cloud pixels.
            # Shadow (2) and snow (3) are not counted here — a shadow scene is
            # still preferable to a cloudy one.  GDD subsampling later maximises
            # clear pixels (==0), which naturally penalises shadow and snow.
            remove_indices = []
            start = 0
            for dup_group in indices_duplicates:
                group_size = len(dup_group)
                group_cloud_mask = cloud_mask[start:start + group_size]

                # Count hard-cloud/NULL pixels only (value 1)
                cloud_null_counts = np.sum(group_cloud_mask == 1, axis=(1, 2))

                # Keep the duplicate with FEWEST cloud/NULL pixels
                best_idx_in_group = np.argmin(cloud_null_counts)
                remove_indices.append(np.delete(dup_group, best_idx_in_group))
                start += group_size

            keep_indices = np.delete(keep_indices, np.concat(remove_indices))
            time_stamps = time_stamps[keep_indices]
            return keep_indices, time_stamps

    def _pick_one_of_duplicate_timestamps(self, time_stamps, s2_data):
        keep_indices = np.arange(len(time_stamps))
        # Get the duplicated time steps
        unique_timestamps, idx, counts = np.unique(time_stamps, return_index=True, return_counts=True)
        duplicated_timestamps = unique_timestamps[counts > 1]
        if len(duplicated_timestamps) == 0:
            # If no duplicates, return
            return keep_indices, time_stamps
        else:
            # If duplicates, pick one of the duplicates randomly
            indices_duplicates = [np.where(time_stamps == ts)[0] for ts in duplicated_timestamps]
            selected_indices = [np.random.choice(dup_group) for dup_group in indices_duplicates]
            remove_indices = [np.delete(dup_group, np.where(dup_group == selected_indices[_])[0][0]) for _, dup_group in enumerate(indices_duplicates)]
            # Keep only the time steps you identified
            keep_indices = np.delete(keep_indices, np.concat(remove_indices))
            return keep_indices, unique_timestamps

    def _degrade_oli(self, images_dn, bands):
        """Degrade OLI (12-bit) to TM-like quality: 8-bit quantisation + per-band Gaussian noise.

        Roy spectral shift (OLI→TM) is handled separately via harmonize_oli_to_tm.
        Must NOT be combined with harmonize_oli_to_tm (double correction).

        Args:
            images_dn: np.ndarray (C, T, H, W) in Landsat Collection 2 DN units
            bands:     list of generic band names in channel order

        Returns:
            np.ndarray (C, T, H, W) in DN units
        """
        sr = images_dn * self._LC2_SCALE + self._LC2_OFFSET  # DN → SR
        sr = np.clip(sr, 0.0, 1.0)
        sr = np.round(sr * 255.0) / 255.0  # 8-bit quantisation (TM is 8-bit)

        sigmas = np.array(
            [self._OLI_DEGRADE_SIGMA[b] for b in bands],
            dtype=np.float32,
        ).reshape(-1, 1, 1, 1)
        sr = sr + np.random.randn(*sr.shape).astype(np.float32) * sigmas

        return (sr - self._LC2_OFFSET) / self._LC2_SCALE  # SR → DN

    def _degrade_etm(self, images_dn, bands):
        """Degrade ETM+ (8-bit) with realistic ETM+ sensor noise.

        ETM+ acquisition is 8-bit (same as TM), so 8-bit quantisation is applied.
        Noise sigma is ~0.5× OLI-degrade sigma, reflecting ETM+'s ~2× better SNR
        relative to TM (Chander et al. 2009).

        Args:
            images_dn: np.ndarray (C, T, H, W) in Landsat Collection 2 DN units
            bands:     list of generic band names in channel order

        Returns:
            np.ndarray (C, T, H, W) in DN units
        """
        sr = images_dn * self._LC2_SCALE + self._LC2_OFFSET  # DN → SR
        sr = np.clip(sr, 0.0, 1.0)
        sr = np.round(sr * 255.0) / 255.0  # 8-bit quantisation (ETM+ is 8-bit like TM)

        sigmas = np.array(
            [self._ETM_DEGRADE_SIGMA[b] for b in bands],
            dtype=np.float32,
        ).reshape(-1, 1, 1, 1)
        sr = sr + np.random.randn(*sr.shape).astype(np.float32) * sigmas

        return (sr - self._LC2_OFFSET) / self._LC2_SCALE  # SR → DN

    def _sliding_window_subsample(self, time_stamps, cloud_mask):
        T = len(time_stamps)
        num_seg = self.temporal_length
        seg_size = max(1, T // num_seg)
        if self.condition == 'cloud':
            cov = np.sum(cloud_mask == 1, axis=(1, 2))
        else:
            cov = np.sum(cloud_mask == 0, axis=(1, 2))
        sel = []
        for start in range(0, T, seg_size):
            end = min(start + seg_size, T)
            idxs = range(start, end)
            if self.condition == 'cloud':
                best = min(idxs, key=lambda x: cov[x])
            else:
                best = max(idxs, key=lambda x: cov[x])
            sel.append(best)
        sel = sorted(sel[:num_seg])
        return np.array(time_stamps)[sel]

    def _adaptive_temperature_subsampling(self, temp_calendar, time_stamps, cloud_mask, condition="open_sky"):
        tmin, tmax = temp_calendar.min(), temp_calendar.max()
        target = (tmax - tmin) / self.temporal_length

        # Initialize with -1 to mark empty bins
        sel = [-1] * self.temporal_length
        cur = tmin
        bin_idx = 0

        # Fill bins with best samples from each temperature range
        while bin_idx < self.temporal_length and cur < tmax:
            nxt = cur + target
            wnd = np.where((temp_calendar >= cur) & (temp_calendar < nxt))[0]
            wnd = wnd[wnd < len(time_stamps)]
            if len(wnd) > 0:
                if condition == 'cloud':
                    cc = np.sum(cloud_mask[wnd] == 1, axis=(1, 2))
                    best = wnd[np.argmin(cc)]
                else:
                    cs = np.sum(cloud_mask[wnd] == 0, axis=(1, 2))
                    best = wnd[np.argmax(cs)]                  
                sel[bin_idx] = best
            # else: leave sel[bin_idx] = -1 to mark empty bin
            cur = nxt
            bin_idx += 1

        # Handle case where we have too many samples (remove worst ones)
        # This preserves the -1 markers for empty bins
        valid_indices = [i for i, idx in enumerate(sel) if idx >= 0]
        if len(valid_indices) > self.temporal_length:
            # Get actual indices and their cloud scores
            actual_indices = [sel[i] for i in valid_indices]
            if condition == 'cloud':
                cc = np.sum(cloud_mask[actual_indices] == 1, axis=(1, 2))
                # Remove the worst (highest cloud cover)
                worst_positions = np.argsort(cc)[-len(valid_indices) + self.temporal_length:]
            else:
                cs = np.sum(cloud_mask[actual_indices] == 0, axis=(1, 2))
                # Remove the worst (lowest clear sky)
                worst_positions = np.argsort(cs)[:len(valid_indices) - self.temporal_length]

            for pos in worst_positions:
                sel[valid_indices[pos]] = -1

        # Build output arrays with placeholders for empty bins
        time_stamps_out = []
        temp_cal_out = []
        for idx in sel:
            if idx >= 0:
                # Valid bin: use actual data
                time_stamps_out.append(time_stamps[idx])
                temp_cal_out.append(temp_calendar[idx])
            else:
                # Empty bin: use -1 as placeholder (will be masked by UTAE)
                time_stamps_out.append(-1)
                temp_cal_out.append(-1)

        return (
            np.array(time_stamps_out),
            np.array(temp_cal_out)
        )

    def _fixed_temperature_subsampling(self, temp_calendar, time_stamps, cloud_mask, cgdd_min, cgdd_max, condition="open_sky"):
        """
        Subsample time series using fixed climatological CGDD bounds.

        The output sequence has temporal_length slots structured as:
          [0]         : best image with GDD < cgdd_min  ("before season"), or -1 if none
          [1 .. T-2]  : T-2 GDD-uniform bins spanning [cgdd_min, cgdd_max]
          [T-1]       : best image with GDD > cgdd_max  ("after season"),  or -1 if none

        Args:
            temp_calendar: CGDD values for each timestamp (T,)
            time_stamps: Day-of-year timestamps (T,)
            cloud_mask: Cloud mask array (T, H, W)
            cgdd_min: Climatological minimum CGDD (from cgdd_start_p10)
            cgdd_max: Climatological maximum CGDD (from cgdd_end_p90)
            condition: 'open_sky' or 'cloud' - determines which images to prefer

        Returns:
            tuple: (selected_timestamps, selected_temp_calendar)
                   Arrays of length temporal_length with -1 for empty slots
        """
        def _best_idx(wnd):
            """Return the index of the best image in wnd according to condition."""
            if len(wnd) == 0:
                return -1
            if condition == 'cloud':
                cc = np.sum(cloud_mask[wnd] == 1, axis=(1, 2))
                return int(wnd[np.argmin(cc)])
            else:
                cs = np.sum(cloud_mask[wnd] == 0, axis=(1, 2))
                return int(wnd[np.argmax(cs)])

        core_length = self.temporal_length - 2  # slots reserved for GDD-uniform bins

        # Handle edge case: no temperature variation (e.g. glacial tiles with [0,0] range)
        if cgdd_max <= cgdd_min:
            total = len(time_stamps)
            core_indices = np.linspace(0, total - 1, core_length, dtype=int).tolist()
            all_indices = [0] + core_indices + [total - 1]
            return time_stamps[all_indices], temp_calendar[all_indices]

        # --- Core bins: T-2 uniform GDD bins in [cgdd_min, cgdd_max] ---
        target = (cgdd_max - cgdd_min) / core_length
        sel = [-1] * core_length
        cur = cgdd_min
        bin_idx = 0

        while bin_idx < core_length and cur < cgdd_max:
            nxt = cur + target
            wnd = np.where((temp_calendar >= cur) & (temp_calendar < nxt))[0]
            wnd = wnd[wnd < len(time_stamps)]
            sel[bin_idx] = _best_idx(wnd)
            cur = nxt
            bin_idx += 1

        # Handle too many samples (remove worst ones)
        valid_indices = [i for i, idx in enumerate(sel) if idx >= 0]
        if len(valid_indices) > core_length:
            actual_indices = [sel[i] for i in valid_indices]
            if condition == 'cloud':
                cc = np.sum(cloud_mask[actual_indices] == 1, axis=(1, 2))
                worst_positions = np.argsort(cc)[-len(valid_indices) + core_length:]
            else:
                cs = np.sum(cloud_mask[actual_indices] == 0, axis=(1, 2))
                worst_positions = np.argsort(cs)[:len(valid_indices) - core_length]
            for pos in worst_positions:
                sel[valid_indices[pos]] = -1

        # --- Boundary images ---
        wnd_before = np.where(temp_calendar < cgdd_min)[0]
        wnd_after  = np.where(temp_calendar > cgdd_max)[0]
        before_idx = _best_idx(wnd_before)
        after_idx  = _best_idx(wnd_after)

        # --- Build output arrays ---
        def _slot(idx):
            if idx >= 0:
                return int(time_stamps[idx]), int(temp_calendar[idx])
            return -1, -1

        time_stamps_out = []
        temp_cal_out = []

        ts_b, gdd_b = _slot(before_idx)
        time_stamps_out.append(ts_b)
        temp_cal_out.append(gdd_b)

        for idx in sel:
            ts, gdd = _slot(idx)
            time_stamps_out.append(ts)
            temp_cal_out.append(gdd)

        ts_a, gdd_a = _slot(after_idx)
        time_stamps_out.append(ts_a)
        temp_cal_out.append(gdd_a)

        return (
            np.array(time_stamps_out, dtype=time_stamps.dtype),
            np.array(temp_cal_out, dtype=temp_calendar.dtype)
        )

    def _get_cloud_mask(self, s2_data, indices=None):
        """
        Get cloud/quality mask from s2_data.

        Args:
            s2_data: Zarr group containing cloud band data
            indices: Optional indices to load (for memory efficiency)

        Returns:
            4-category mask matching the Sentinel-2 convention (both sensors):
                0 = clear         (fully usable)
                1 = cloud / NULL  (hard invalid — no surface signal)
                2 = shadow        (cloud shadow — altered reflectance)
                3 = snow          (snow/ice — no crop signal)

            This encoding lets downstream functions apply graduated logic:
              - duplicate removal  (==1): picks scene with fewest hard-cloud pixels;
                shadow and snow don't count against a scene for dedup.
              - GDD subsampling    (==0): maximises truly clear pixels;
                shadow and snow naturally reduce the score.
        """
        if indices is not None:
            cloud_mask = s2_data[self.cloud_band][indices]
        else:
            cloud_mask = s2_data[self.cloud_band][:]

        if self.cloud_band != "QA_PIXEL":
            # Sentinel-2: s2_mask band
            # Original: 0=clear, 1=cloud, 2=shadow, 3=snow, 4=NULL
            # Map NULL (4) → 1 (cloud/hard-invalid); keep 0/1/2/3 otherwise.
            cloud_mask = np.where(cloud_mask == 4, 1, cloud_mask)
            return cloud_mask.astype(int)
        else:
            # Landsat Collection 2 QA_PIXEL bit flags → 4-category mask
            # matching the Sentinel-2 convention above.
            #
            # Correct QA_PIXEL bit positions:
            #   Bit 0 = Fill, Bit 2 = Cirrus, Bit 3 = Cloud,
            #   Bit 4 = Cloud Shadow, Bit 5 = Snow
            #   Bit 6 = Clear flag (1=clear) — NOT used as an invalid indicator.
            #
            # Priority: cloud/fill/cirrus (1) > shadow (2) > snow (3) > clear (0)
            fill_bit         = 1 << 0  # 1
            cirrus_bit       = 1 << 2  # 4
            cloud_bit        = 1 << 3  # 8
            cloud_shadow_bit = 1 << 4  # 16
            snow_bit         = 1 << 5  # 32

            is_cloud  = (((cloud_mask & fill_bit)   != 0) |
                         ((cloud_mask & cirrus_bit)  != 0) |
                         ((cloud_mask & cloud_bit)   != 0))
            is_shadow = ((cloud_mask & cloud_shadow_bit) != 0) & ~is_cloud
            is_snow   = ((cloud_mask & snow_bit)    != 0) & ~is_cloud & ~is_shadow

            result = np.zeros(cloud_mask.shape, dtype=np.int32)
            result[is_cloud]  = 1
            result[is_shadow] = 2
            result[is_snow]   = 3
            return result

    def _process_tile_data(self, s2_data, gt_data, temp_data, s2_file_path=None):
        """Common processing logic for both modes.

        Args:
            s2_data: Zarr group with S2 data
            gt_data: Zarr group with GT data (or None for prediction mode)
            temp_data: Zarr group with temperature calendar data
            s2_file_path: Optional path to S2 file (needed for fixed temperature subsampling)
        """
        # Load time steps
        time_stamps = s2_data['time'][:]

        # If simulate landsat, remove time stamps
        if self.simulate_landsat:
            time_stamps_unique = np.unique(time_stamps)
            filtered = [time_stamps_unique[0]]
            for t in time_stamps_unique[1:]:
                if t - filtered[-1] >= self.revisit_time:
                    filtered.append(t)
            time_stamps = np.array(filtered)

        # Remove duplicate time steps
        keep_indices, time_stamps = self._remove_duplicate_timestamps(time_stamps, s2_data)
        # keep_indices, time_stamps = self._pick_one_of_duplicate_timestamps(time_stamps, s2_data)

        # Truncate time dim
        if self.truncate_portion < 1.0:
            # total = images.shape[1]
            new_t = max(1, int(365 * self.truncate_portion))
            idx_new_t = np.searchsorted(time_stamps, new_t, side='right')
            time_stamps = time_stamps[:idx_new_t]
        else:
            idx_new_t = len(time_stamps)

        # Keep track of original day-of-year timestamps (needed for spatial GDD indexing)
        time_stamps_original = time_stamps.copy()

        # Subsampling
        if self.use_fixed_temperature_subsampling:
            # Fixed/Climatological temperature subsampling using CGDD bounds from GPKG
            cloud_mask = self._get_cloud_mask(s2_data, keep_indices)[:idx_new_t]
            fill_value = 0
            temp_cal = temp_data[self.gdd_mean_var][:].astype(np.float32)
            temp_cal = temp_cal[time_stamps.tolist()]
            temp_cal = np.nan_to_num(temp_cal, nan=fill_value).astype(int)

            # Get tile identifier and look up CGDD bounds
            if s2_file_path is None:
                raise ValueError(
                    "s2_file_path must be provided when use_fixed_temperature_subsampling=True"
                )
            tile_id = self._get_tile_identifier(s2_file_path)
            if tile_id not in self.cgdd_bounds:
                raise ValueError(
                    f"Tile '{tile_id}' not found in CGDD bounds. "
                    f"Available tiles: {len(self.cgdd_bounds)}. "
                    f"File: {s2_file_path}"
                )

            bounds = self.cgdd_bounds[tile_id]
            cgdd_min = bounds['cgdd_min']
            cgdd_max = bounds['cgdd_max']

            # Apply fixed subsampling
            time_stamps_subset, temp_cal_subset = self._fixed_temperature_subsampling(
                temp_cal, time_stamps, cloud_mask, cgdd_min, cgdd_max, condition=self.condition
            )

            # Build idx_images with explicit handling of missing bins (-1 markers)
            idx_images = []
            for ts in time_stamps_subset:
                if ts >= 0:  # Valid timestamp
                    matching_indices = np.where(time_stamps == ts)[0]
                    if len(matching_indices) > 0:
                        idx_images.append(matching_indices[0])
                    else:
                        idx_images.append(0)  # Fallback dummy
                else:  # Missing bin marker (-1)
                    idx_images.append(0)  # Use any valid index, will be replaced

            idx_images = np.array(idx_images)
            time_stamps = temp_cal_subset

        elif self.use_temperature_calendar and self.use_temperature_subsampling:
            cloud_mask = self._get_cloud_mask(s2_data, keep_indices)[:idx_new_t]
            fill_value = 0
            temp_cal = temp_data[self.gdd_mean_var][:].astype(np.float32)
            temp_cal = temp_cal[time_stamps.tolist()]
            # Replace NaN with fill value before converting to int
            temp_cal = np.nan_to_num(temp_cal, nan=fill_value).astype(int)
            time_stamps_subset, temp_cal_subset = self._adaptive_temperature_subsampling(temp_cal, time_stamps, cloud_mask)

            # Build idx_images with explicit handling of missing bins (-1 markers)
            idx_images = []
            for ts in time_stamps_subset:
                if ts >= 0:  # Valid timestamp
                    matching_indices = np.where(time_stamps == ts)[0]
                    if len(matching_indices) > 0:
                        idx_images.append(matching_indices[0])
                    else:
                        # Fallback: use index 0 as dummy (will be replaced later)
                        idx_images.append(0)
                else:  # Missing bin marker (-1)
                    idx_images.append(0)  # Use any valid index, will be replaced with dummy

            idx_images = np.array(idx_images)
            time_stamps = temp_cal_subset

        elif self.use_temperature_calendar:
            cloud_mask = self._get_cloud_mask(s2_data, keep_indices)[:idx_new_t]
            fill_value = 0
            temp_cal = temp_data[self.gdd_mean_var][:].astype(np.float32)
            time_stamps_subset = self._sliding_window_subsample(time_stamps, cloud_mask)
            idx_images = np.nonzero(np.isin(time_stamps, time_stamps_subset))[0]
            time_stamps_cal = temp_cal[time_stamps_subset.tolist()]
            # Replace NaN with fill value before converting to int
            time_stamps = np.nan_to_num(time_stamps_cal, nan=fill_value).astype(int)

        elif self.use_temperature_calendar_no_sliding_subsample:
            cloud_mask = self._get_cloud_mask(s2_data, keep_indices)[:idx_new_t]
            temp_cal = temp_data[self.gdd_mean_var][:].astype(np.float32)
            total = len(time_stamps)
            idx_images = indices = np.linspace(0, total - 1, self.temporal_length, dtype=int)
            time_stamps = temp_cal[indices]

        elif self.no_sliding_subsample:
            total = len(time_stamps)
            idx_images = indices = np.linspace(0, total - 1, self.temporal_length, dtype=int)
            time_stamps = time_stamps[indices]

        else:
            cloud_mask = self._get_cloud_mask(s2_data, keep_indices)[:idx_new_t]
            time_stamps_subset = self._sliding_window_subsample(time_stamps, cloud_mask)
            idx_images = np.nonzero(np.isin(time_stamps, time_stamps_subset))[0]
            time_stamps = time_stamps_subset

        # Load the data
        bands_data = [s2_data[band][keep_indices][idx_images].astype(np.float32) for band in self.bands]
        images = np.stack(bands_data, axis=0)  # C x T x H x W

        # Compute original sensor masks once before any degradation/harmonization.
        # Both degrade and harmonize blocks use these original masks so they compose naturally.
        if 'sensor_flag' in s2_data:
            sf_raw = s2_data['sensor_flag'][keep_indices][idx_images]  # (T,) copy
            original_oli_mask   = (sf_raw == 0).copy()
            original_flag1_mask = (sf_raw == 1).copy()  # flag=1 = ETM+ in training, TM at inference
        else:
            original_oli_mask   = None
            original_flag1_mask = None

        # Step 1: OLI degradation (training only, before normalisation).
        # Degrades OLI timesteps to TM-like quality (8-bit quant + noise).
        # is_tm_data=True means this is a real-TM dataset — no OLI to degrade.
        is_tm = self.is_tm_data
        if not is_tm and self.augmentation and self.oli_degrade_prob > 0.0:
            if random.random() < self.oli_degrade_prob:
                is_tm = True
                if original_oli_mask is not None:
                    if original_oli_mask.any():
                        images[:, original_oli_mask, :, :] = self._degrade_oli(
                            images[:, original_oli_mask, :, :], self.bands
                        )
                    s2_data['sensor_flag'][keep_indices[idx_images[original_oli_mask]]] = 1
                else:
                    images = self._degrade_oli(images, self.bands)

        # Step 1b: ETM+ degradation (training only, before normalisation).
        # original_flag1_mask = flag==1 = ETM+ timesteps in training (no real TM in training).
        # Both TM and ETM+ are 8-bit acquisition → same quantisation step as _degrade_oli().
        if self.augmentation and self.etm_degrade_prob > 0.0:
            if random.random() < self.etm_degrade_prob:
                if original_flag1_mask is not None and original_flag1_mask.any():
                    images[:, original_flag1_mask, :, :] = self._degrade_etm(
                        images[:, original_flag1_mask, :, :], self.bands
                    )
                elif original_flag1_mask is None and self.is_tm_data:
                    images = self._degrade_etm(images, self.bands)

        # Step 2: Deterministic load-time Roy harmonization (applied in DN space before normalisation).
        # Uses original_oli_mask / original_flag1_mask so Roy applies to the same timesteps as degrade,
        # even if degrade already updated sensor_flag. Dummy timesteps are safe: zeroed AFTER normalisation.
        if self.harmonize_oli_to_tm or self.harmonize_tm_to_oli:
            coeff_table  = self._TM_AUG_HARMONIZATION if self.harmonize_oli_to_tm else self._TM_TO_OLI_HARMONIZATION
            target_mask  = original_oli_mask if self.harmonize_oli_to_tm else original_flag1_mask
            flag_after   = 1 if self.harmonize_oli_to_tm else 0
            if target_mask is not None and target_mask.any():
                for i, band in enumerate(self.bands):
                    coeff = coeff_table.get(band)
                    if coeff:
                        inter_sc = (coeff['intercept'] - self._LC2_OFFSET) / self._LC2_SCALE
                        images[i, target_mask, :, :] = coeff['slope'] * images[i, target_mask, :, :] + inter_sc
                s2_data['sensor_flag'][keep_indices[idx_images[target_mask]]] = flag_after
            elif target_mask is None and self.is_tm_data and self.harmonize_tm_to_oli:
                # Single-sensor TM dataset with no sensor_flag: all timesteps are TM
                for i, band in enumerate(self.bands):
                    coeff = self._TM_TO_OLI_HARMONIZATION.get(band)
                    if coeff:
                        inter_sc = (coeff['intercept'] - self._LC2_OFFSET) / self._LC2_SCALE
                        images[i] = coeff['slope'] * images[i] + inter_sc

        # Normalize
        images = (images - self.channel_means[:, None, None, None]) / self.channel_stds[:, None, None, None]

        # Replace dummy positions (where time_stamps < 0) with pad_value (0) AFTER normalization
        # This ensures UTAE will correctly mask them since pad_mask checks (input == self.pad_value)
        if self.use_fixed_temperature_subsampling or (self.use_temperature_calendar and self.use_temperature_subsampling):
            for i, ts in enumerate(time_stamps):
                if ts < 0:  # Dummy position marker
                    images[:, i, :, :] = 0.0  # Exactly pad_value

        images = np.transpose(images, (1, 0, 2, 3))  # T x C x H x W
        H, W = images.shape[2], images.shape[3]

        # Load GT (or create dummy if None for prediction mode)
        if gt_data is not None:
            # Detect format by checking dimensionality
            lnf_code_data = gt_data['lnf_code']

            if lnf_code_data.ndim == 2:
                # Old single-band format: (H, W)
                lnf = lnf_code_data[:]
                lnf = np.where(lnf == None, 0, lnf).astype(np.int32)
                # Map lnf to target labels
                gt_list = [
                    np.vectorize(lambda x: mapping.get(x, 0))(lnf).astype(np.int32)
                    for mapping in self.target_mapping
                ]
            elif lnf_code_data.ndim == 3:
                # New multi-band coverage_fractions format: (num_bands, H, W)
                # Aggregate coverage fractions by target class BEFORE taking argmax
                # This correctly handles cases where multiple lnf_codes map to same target

                # Load band data
                data_arr = gt_data['lnf_code'][:]  # (num_bands, H, W)
                band_coords = gt_data['band'][:]  # Array of lnf_codes
                lnf_codes = band_coords.astype(int)

                H, W = data_arr.shape[1], data_arr.shape[2]

                # Validate spatial dimensions
                if H != 128 or W != 128:
                    raise ValueError(
                        f"Unexpected spatial dimensions for tile: {data_arr.shape}. "
                        f"Expected (num_bands, 128, 128)"
                    )

                # For each hierarchy level, aggregate coverage by target class
                gt_list = []
                for level_idx, mapping in enumerate(self.target_mapping):
                    num_classes = self.num_classes[level_idx] + 1  # +1 for background
                    aggregated = np.zeros((num_classes, H, W), dtype=np.float32)

                    # Aggregate coverage fractions by target class
                    for band_idx, lnf_code in enumerate(lnf_codes):
                        target_class = mapping.get(int(lnf_code), 0)  # Default to 0 (background)
                        aggregated[target_class] += data_arr[band_idx]

                    # Take argmax to get final label
                    gt = np.argmax(aggregated, axis=0).astype(np.int32)
                    gt_list.append(gt)
            else:
                raise ValueError(
                    f"Unexpected lnf_code dimensions: {lnf_code_data.ndim}. "
                    f"Expected 2 (old format) or 3 (new format)"
                )
        else:
            # Prediction mode without GT: create dummy ground truth (all zeros)
            H, W = images.shape[2], images.shape[3]
            gt_list = [np.zeros((H, W), dtype=np.int32) for _ in self.target_mapping]

        # Alpine mask slice — extract H×W crop aligned with this tile
        if self.alpine_mask is not None and s2_file_path is not None:
            stem = os.path.splitext(os.path.basename(s2_file_path))[0]
            parts = stem.split('_')
            tile_left = float(parts[1])
            tile_top  = float(parts[2])
            tr = self.alpine_mask_transform
            col_off = int(round((tile_left - tr.c) / tr.a))
            row_off = int(round((tr.f - tile_top) / abs(tr.e)))
            mH, mW = self.alpine_mask.shape
            r0 = max(row_off, 0);       r1 = min(row_off + H, mH)
            c0 = max(col_off, 0);       c1 = min(col_off + W, mW)
            alpine_slice = np.zeros((H, W), dtype=np.uint8)
            if r1 > r0 and c1 > c0:
                alpine_slice[r0 - row_off:r1 - row_off,
                             c0 - col_off:c1 - col_off] = self.alpine_mask[r0:r1, c0:c1]
        else:
            alpine_slice = np.zeros((H, W), dtype=np.uint8)

        # Augmentation
        if self.augmentation:
            # Random turn (0, 90, 180, 270 degrees)
            if random.random() < 0.5:
                k = random.randint(1, 3)
                images = np.rot90(images, k=k, axes=(2, 3))             # T x C x H x W
                if len(gt_list) == 1:
                    gt_list[0] = np.rot90(gt_list[0], k=k, axes=(0, 1)) # H x W
                else:
                    gt_list = [np.rot90(gt, k=k, axes=(0, 1)) for gt in gt_list]
                alpine_slice = np.rot90(alpine_slice, k=k)

            # Random horizontal flip (flip W)
            if random.random() < 0.5:
                images = np.flip(images, axis=3)                        # T x C x H x W
                if len(gt_list) == 1:
                    gt_list[0] = np.flip(gt_list[0], axis=1)            # H x W
                else:
                    gt_list = [np.flip(gt, axis=1) for gt in gt_list]
                alpine_slice = np.flip(alpine_slice, axis=1)

            # Random vertical flip (flip H)
            if random.random() < 0.5:
                images = np.flip(images, axis=2)                        # T x C x H x W
                if len(gt_list) == 1:
                    gt_list[0] = np.flip(gt_list[0], axis=0)            # H x W
                else:
                    gt_list = [np.flip(gt, axis=0) for gt in gt_list]
                alpine_slice = np.flip(alpine_slice, axis=0)

        # Finalize GTs - ensure contiguous arrays to avoid storage resize errors
        if len(gt_list) == 1:
            ground_truth = torch.from_numpy(gt_list[0].copy())
        else:
            ground_truth = [torch.from_numpy(gt.copy()) for gt in gt_list]

        alpine_slice_tensor = torch.from_numpy(alpine_slice.copy())

        # GDD timestamps: z-score or keep raw depending on normalize_timestamps
        valid_mask = time_stamps >= 0
        time_stamps_mean_norm = time_stamps.astype(np.float32)
        if self.normalize_timestamps and valid_mask.any():
            time_stamps_mean_norm[valid_mask] = (
                (time_stamps[valid_mask] - self.gdd_stats['mean']) / self.gdd_stats['std']
            )

        # DOY timestamps: z-score or keep raw (1–365) depending on normalize_timestamps
        DOY_MEAN, DOY_STD = 183.0, 105.1
        doy_values = time_stamps_original[idx_images].astype(np.float32)
        if self.normalize_timestamps:
            doy_norm = np.zeros_like(doy_values)
            if valid_mask.any():
                doy_norm[valid_mask] = (doy_values[valid_mask] - DOY_MEAN) / DOY_STD
            doy_tensor = torch.from_numpy(doy_norm.copy())
        else:
            doy_tensor = torch.from_numpy(doy_values.copy())

        # Load and normalize spatial GDD if enabled
        if self.use_spatial_gdd:
            # Load spatial CGDD data
            try:
                cgdd_spatial = temp_data[self.gdd_spatial_var][:]  # Shape: (365, H, W)
            except Exception as e:
                print(f"ERROR loading spatial GDD variable '{self.gdd_spatial_var}'")
                print(f"  Available variables in temp_data: {list(temp_data.keys())}")
                if hasattr(temp_data, 'attrs'):
                    print(f"  Temp data attributes: {dict(temp_data.attrs)}")
                print(f"  Error: {type(e).__name__}: {e}")
                raise

            # Index by day-of-year to get the selected timesteps
            # We need to use the same indexing logic as for mean GDD, which is applied via idx_images to time_stamps_original
            # So we select from the 365-day calendar using time_stamps_original[idx_images]
            selected_days = time_stamps_original[idx_images]
            cgdd_spatial = cgdd_spatial[selected_days.tolist()]  # Shape: (T, H, W)

            # Apply same augmentations as images
            if self.augmentation:
                # Random rotation
                if random.random() < 0.5:
                    k = random.randint(1, 3)
                    cgdd_spatial = np.rot90(cgdd_spatial, k=k, axes=(1, 2))
                # Random horizontal flip
                if random.random() < 0.5:
                    cgdd_spatial = np.flip(cgdd_spatial, axis=1)
                # Random vertical flip
                if random.random() < 0.5:
                    cgdd_spatial = np.flip(cgdd_spatial, axis=2)

            # Normalize using same stats as mean GDD
            valid_mask_spatial = cgdd_spatial >= 0
            cgdd_spatial_norm = cgdd_spatial.astype(np.float32)
            if valid_mask_spatial.any():
                cgdd_spatial_norm[valid_mask_spatial] = (
                    (cgdd_spatial[valid_mask_spatial] - self.gdd_stats['mean']) /
                    self.gdd_stats['std']
                )

            # Handle -1 markers (missing bins) - set to 0 (pad value)
            cgdd_spatial_norm[~valid_mask_spatial] = 0.0

            # Create tuple structure: (mean_gdd, doy, spatial_gdd)
            time_stamps = (
                torch.from_numpy(time_stamps_mean_norm.copy()),  # (T,)
                doy_tensor,                                       # (T,)
                torch.from_numpy(cgdd_spatial_norm.copy())       # (T, H, W)
            )
        else:
            # Scalar mode: (mean_gdd, doy)
            time_stamps = (
                torch.from_numpy(time_stamps_mean_norm.copy()),  # (T,)
                doy_tensor,                                       # (T,)
            )

        # To tensors - ensure contiguous arrays (important after np.flip/np.rot90)
        images = torch.from_numpy(images.copy())

        # Temporal sparsification: simulate TM-era observation density (applied 50% of the time)
        if self.augmentation and self.temporal_sparsify and torch.rand(1).item() < 0.5:
            occupied_idx = (images.abs().sum(dim=(1, 2, 3)) > 0).nonzero(as_tuple=False).squeeze(1)
            n_occupied = len(occupied_idx)
            if n_occupied > self.min_temporal_keep:
                lo, hi = self.temporal_sparsify_range
                n_keep = int(torch.randint(lo, hi + 1, (1,)).item())
                n_keep = max(self.min_temporal_keep, min(n_keep, n_occupied))
                keep_idx = occupied_idx[torch.randperm(n_occupied)[:n_keep]]
                drop_mask = torch.zeros(images.shape[0], dtype=torch.bool)
                drop_mask[occupied_idx] = True
                drop_mask[keep_idx] = False
                images[drop_mask] = 0.0
                ts_list = list(time_stamps)
                ts_list[0][drop_mask] = 0.0  # gdd_normalized
                ts_list[1][drop_mask] = 0.0  # doy_normalized
                if len(ts_list) > 2:
                    ts_list[2][drop_mask] = 0.0  # spatial_gdd (T, H, W)
                time_stamps = tuple(ts_list)

        # Append sensor-identity channel: 0.0 = OLI, 1.0 = TM
        if self.use_sensor_flag:
            T, C, H, W = images.shape
            if 'sensor_flag' in s2_data:
                # Per-timestep flag from multi-sensor merge: OLI=0, TM/ETM=1
                sf = s2_data['sensor_flag'][keep_indices][idx_images].astype(np.float32)
                sensor_ch = torch.from_numpy(
                    np.broadcast_to(sf[:, None, None, None], (T, 1, H, W)).copy()
                ).to(images.dtype)
            else:
                # Fallback: tile-level flag (Sentinel or is_tm_data)
                sensor_val = 1.0 if is_tm else 0.0
                sensor_ch  = torch.full((T, 1, H, W), sensor_val, dtype=images.dtype)
            # Keep padded timesteps (all channels == 0) consistent: set sensor flag to 0 there too
            if self.use_fixed_temperature_subsampling or \
               (self.use_temperature_calendar and self.use_temperature_subsampling):
                pad_mask = (images == 0.0).all(dim=1, keepdim=True)  # (T, 1, H, W)
                sensor_ch[pad_mask.expand_as(sensor_ch)] = 0.0
            images = torch.cat([images, sensor_ch], dim=1)  # (T, C+1, H, W)

        return (images, time_stamps), ground_truth, alpine_slice_tensor


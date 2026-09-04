#!/usr/bin/env python3
# generate_compute_table.py
# Model overview + computational cost table for SwissCrop25 supplementary.
#
# Total params: instantiated from S1 conf.json for each model variant.
# Trainable params: read from conf.json (mean over LOYO splits).
# Training time: sum of per-epoch train_epoch_time from trainlog.json; epochs missing
# train_epoch_time (due to job restarts) are extrapolated using the mean of recorded epochs.
# Inference time: mean val_epoch_time from trainlog.json.
# Both time metrics are averaged across LOYO splits.
#
# Requires: galileo_weights/models/{nano,base}/ present at project root.
#
# Usage:
#   python scripts/tables/generate_compute_table.py

import json
import types
import numpy as np
import torch
from pathlib import Path

ROOT    = Path(__file__).parents[2]
STORAGE = ROOT / "storage"
SPLITS  = ["S5", "S4", "S3", "S2", "S1"]

# (display_name, storage_prefix, display_type, pretrained, arch_key)
MODELS = [
    ("U-TAE",                 "utae_gddsub_gddpe",             "Conv.+attn.",    False, "utae"),
    ("TSViT",                 "tsvit_gddsub_gddpe", "Transformer",    False, "tsvit"),
    ("Galileo-nano",          "galileo_nano_gddsub",           "FM",             True,  "galileo"),
    ("Galileo-nano (frozen)", "galileo_nano_frozen_gddsub",     "FM",             True,  "galileo"),
    ("Galileo-base (frozen)", "galileo_base_frozen_gddsub",     "FM",             True,  "galileo"),
]

OUTPUTS = [
    ROOT / "results/tables/tab_compute_body.tex",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/tables/tab_compute_body.tex",
]


# ── Model instantiation ───────────────────────────────────────────────────────

def _load_conf(prefix):
    p = STORAGE / f"{prefix}_S1" / "conf.json"
    return json.load(open(p))


def _instantiate_utae(conf):
    import sys
    sys.path.insert(0, str(ROOT))
    from src import model_utils
    config = types.SimpleNamespace(**conf)
    return model_utils.get_model(config, mode="semantic")


def _instantiate_tsvit(conf):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.tsvit.TSViTdense import TSViT
    model_config = {
        "img_res":        conf["img_res"],
        "patch_size":     conf["patch_size"],
        "max_seq_len":    conf["max_seq_len"],
        "num_channels":   conf["num_bands"] + 1,
        "num_classes":    conf["num_classes"],
        "dim":            conf["dim"],
        "temporal_depth": conf["temporal_depth"],
        "spatial_depth":  conf["spatial_depth"],
        "heads":          conf["heads"],
        "dim_head":       conf["dim_head"],
        "dropout":        conf["dropout"],
        "emb_dropout":    conf["emb_dropout"],
        "pool":           conf["pool"],
        "scale_dim":      conf["scale_dim"],
    }
    return TSViT(model_config)


def _instantiate_galileo(conf):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.galileo_segmentation import GalileoForSegmentation
    encoder_path = str(ROOT / conf["galileo_encoder_path"])
    return GalileoForSegmentation(
        num_classes=conf["num_classes"],
        encoder_path=encoder_path,
        patch_size=conf["galileo_patch_size"],
        tile_size=conf["tile_size"],
        tile_chunk=conf["tile_chunk"],
        freeze_encoder=conf["freeze_encoder"],
        head_mode=conf["head_mode"],
        head_hidden_dim=conf["head_hidden_dim"],
        bands=conf["bands"][:conf["num_bands"]],
        channel_stats=conf["channel_stats"],
        use_lora=conf["lora"],
        lora_rank=conf["lora_rank"],
        lora_alpha=conf["lora_alpha"],
    )


_INSTANTIATORS = {
    "utae":    _instantiate_utae,
    "tsvit":   _instantiate_tsvit,
    "galileo": _instantiate_galileo,
}


def count_total_params(prefix, arch_key):
    conf = _load_conf(prefix)
    with torch.no_grad():
        model = _INSTANTIATORS[arch_key](conf)
    return sum(p.numel() for p in model.parameters()) / 1e6


# ── Existing readers ──────────────────────────────────────────────────────────

def read_n_params(prefix):
    vals = []
    for split in SPLITS:
        p = STORAGE / f"{prefix}_{split}" / "conf.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        if "N_params" in d:
            vals.append(d["N_params"])
    return np.mean(vals) / 1e6 if vals else None


def read_times(prefix):
    train_h_vals = []
    infer_min_vals = []
    for split in SPLITS:
        p = STORAGE / f"{prefix}_{split}" / "trainlog.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        epochs = list(d.values())
        n_total = len(epochs)

        avail_train = [e["train_epoch_time"] for e in epochs if "train_epoch_time" in e]
        if avail_train:
            mean_epoch_s = np.mean(avail_train)
            train_h_vals.append(mean_epoch_s * n_total / 3600)

        val_times = [e["val_epoch_time"] for e in epochs if "val_epoch_time" in e]
        if val_times:
            infer_min_vals.append(np.mean(val_times) / 60)

    train_h   = np.mean(train_h_vals)   if train_h_vals   else None
    infer_min = np.mean(infer_min_vals) if infer_min_vals else None
    return train_h, infer_min


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    lines = ["% auto-generated by generate_compute_table.py — do not edit manually"]

    for i, (name, prefix, disp_type, pretrained, arch_key) in enumerate(MODELS):
        print(f"Processing {name}...", flush=True)

        total_params           = count_total_params(prefix, arch_key)
        trainable_params       = read_n_params(prefix)
        train_h, infer_min     = read_times(prefix)

        pretrained_str   = r"\checkmark" if pretrained else "---"
        total_str        = f"{total_params:.1f}"  if total_params    is not None else "---"
        trainable_str    = f"{trainable_params:.1f}" if trainable_params is not None else "---"
        train_str        = f"{train_h:.1f}"       if train_h         is not None else "---"
        infer_str        = f"{infer_min:.1f}"     if infer_min       is not None else "---"

        sep = r" \\" if i < len(MODELS) - 1 else ""
        lines.append(
            f"{name:<25} & {disp_type:<15} & {pretrained_str} & "
            f"{total_str} & {trainable_str} & {train_str} & {infer_str}{sep}"
        )
        print(
            f"  type={disp_type}  pretrained={pretrained}  "
            f"total={total_str}M  trainable={trainable_str}M  "
            f"train={train_str}h  infer={infer_str}min"
        )

    body = "\n".join(lines)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()

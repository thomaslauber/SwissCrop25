"""
Galileo Foundation Model — Segmentation Wrapper for T3S Crop Mapping
====================================================================

Thin wrapper around the *official* Galileo encoder from:
    third_party/galileo/single_file_galileo.py

This file does NOT reimplement Galileo. It only:
  1. Maps our 9‑band S2 input to Galileo's expected MaskedOutput format.
  2. Handles de‑normalisation (our dataset stats) → re‑normalisation (Galileo
     pretraining stats) so the pretrained weights see the right distribution.
  3. Tiles the input image into spatial windows before encoding.
     Galileo was pretrained on 96×96 pixel instances; feeding 128×128
     directly causes OOM on 24 GB GPUs due to O(N²) self‑attention.
     Default tile_size=16 splits 128×128 into 64 tiles of 16×16.
  4. Adds a segmentation head on top of the encoder patch features.

Reference:
    Tseng et al., "Galileo: Learning Global and Local Features in Pretrained
    Remote Sensing Models", arXiv:2502.09356
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from einops import rearrange
from peft import LoraConfig, inject_adapter_in_model

from .single_file_galileo import (
    Encoder,
    SPACE_TIME_BANDS,
    SPACE_BANDS,
    TIME_BANDS,
    STATIC_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    SPACE_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    S2_BANDS as GALILEO_S2_BANDS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent  # project root

# ============================================================================
# Band mapping: our s2_* band names → Galileo SPACE_TIME_BANDS names
# ============================================================================
# SPACE_TIME_BANDS: VV VH B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 NDVI
#                    0  1  2  3  4  5  6  7  8   9  10  11   12

_S2_NAME_TO_GALILEO = {
    "s2_B02": "B2",  "s2_B03": "B3",  "s2_B04": "B4",  "s2_B05": "B5",
    "s2_B06": "B6",  "s2_B07": "B7",  "s2_B08": "B8",  "s2_B8A": "B8A",
    "s2_B11": "B11", "s2_B12": "B12",
}

# S2 mask groups that we *unmask* (set to 0 = "seen by encoder")
S2_MASK_GROUPS = [
    idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
]

# ============================================================================
# Galileo pretraining normalisation stats for SPACE_TIME_BANDS (13 channels)
# ============================================================================
GALILEO_ST_MEAN = [
    -11.728724, -18.855582,            # VV, VH
    1395.340873, 1338.402692,           # B2, B3
    1343.098838, 1543.860798,           # B4, B5
    2186.202207, 2525.093285,           # B6, B7
    2410.337719, 2750.285465,           # B8, B8A
    2234.911100, 1474.531127,           # B11, B12
    0.289212,                           # NDVI
]
GALILEO_ST_STD = [
    4.887146, 5.730270,
    917.704144, 913.298842,
    1092.678724, 1047.220608,
    1048.010161, 1143.690303,
    1098.979178, 1204.472755,
    1145.977406, 980.242984,
    0.272094,
]


def _build_renorm_params(bands, channel_stats):
    """
    Compute per‑band renorm params from dataset channel_stats:
        galileo_norm = our_norm * scale + offset
    """
    scales, offsets, dst_indices = [], [], []
    for band_name in bands:
        gal_name = _S2_NAME_TO_GALILEO.get(band_name)
        if gal_name is None:
            raise ValueError(f"Band '{band_name}' has no Galileo mapping in _S2_NAME_TO_GALILEO")
        dst_idx = SPACE_TIME_BANDS.index(gal_name)
        our_mean = channel_stats[band_name]["mean"]
        our_std  = channel_stats[band_name]["std"]
        gal_mean = GALILEO_ST_MEAN[dst_idx]
        gal_std  = GALILEO_ST_STD[dst_idx]
        scales.append(our_std / gal_std)
        offsets.append((our_mean - gal_mean) / gal_std)
        dst_indices.append(dst_idx)
    return scales, offsets, dst_indices


# ============================================================================
# Segmentation Head
# ============================================================================
class SegmentationHead(nn.Module):
    """
    Per‑patch features → per‑pixel class logits.

    mode="single": 1×1 conv (default — minimal added params, clean baseline)
    mode="multi":  3‑layer conv with BN + GELU
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256,
                 upsample_factor: int = 4, mode: str = "single"):
        super().__init__()
        self.upsample_factor = upsample_factor
        if mode == "single":
            self.head = nn.Conv2d(in_dim, num_classes, 1)
        elif mode == "multi":
            self.head = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, num_classes, 1),
            )
        else:
            raise ValueError(f"Unknown head mode: {mode!r}. Use 'single' or 'multi'.")

    def forward(self, x):
        out = self.head(x)
        if self.upsample_factor > 1:
            out = F.interpolate(out, scale_factor=self.upsample_factor,
                                mode="bilinear", align_corners=False)
        return out


# ============================================================================
# Main Model: GalileoForSegmentation
# ============================================================================
class GalileoForSegmentation(nn.Module):
    """
    Wraps the official Galileo Encoder + a segmentation head.

    Key design — **tiling**:
        Galileo's self‑attention is O(N²) where N = all unmasked tokens in
        a spatial window.  For 128×128 with patch_size=4 that's 32×32×T×bands
        ≈ 120k tokens → 900 GiB attention matrix → instant OOM.

        We split the image into tiles (default 32×32 pixels), encode each
        tile independently, then reassemble the per‑patch feature map.  With
        tile_size=16 and patch_size=4, each tile produces a 4×4 patch grid.

    Forward signature matches UTAE:
        model(x, batch_positions=dates)
          x     : (B, T, C, H, W)  — our normalised S2 data
          dates : (B, T)            — DOY timestamps → months for Galileo PE
        Returns : (B, num_classes, H, W)
    """

    def __init__(
        self,
        num_classes: int = 53,
        encoder_path: Optional[str] = None,
        patch_size: int = 4,
        tile_size: int = 16,
        tile_chunk: int = 16,
        month: int = 6,
        head_mode: str = "single",
        head_hidden_dim: int = 256,
        freeze_encoder: bool = False,
        bands: Optional[list] = None,
        channel_stats: Optional[dict] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        """
        Args:
            num_classes:     Number of output classes.
            encoder_path:    Path to Galileo weights folder. None → tiny.
            patch_size:      Galileo ViT patch size (must divide tile_size).
            tile_size:       Spatial window fed to encoder (default 16).
                             Galileo pretrained on 96×96; but each 16×16 tile
                             has 4×4 patches × 24T × 5 groups = 1,920 tokens.
                             128/16 = 8×8 = 64 tiles.
            tile_chunk:      Max tiles per encoder call (controls peak GPU mem).
            month:           Fallback month if batch_positions is None.
            head_mode:       "single" (1×1 conv) or "multi" (3‑layer).
            head_hidden_dim: Hidden dim for multi‑layer head.
            freeze_encoder:  Freeze encoder weights (linear probe).
        """
        super().__init__()
        assert tile_size % patch_size == 0, \
            f"tile_size ({tile_size}) must be divisible by patch_size ({patch_size})"

        self.patch_size = patch_size
        self.tile_size = tile_size
        self.tile_chunk = tile_chunk
        self.month = month
        self.num_classes = num_classes
        self.ph = tile_size // patch_size   # patches per tile (height)
        self.pw = tile_size // patch_size   # patches per tile (width)

        # ----- Load official Galileo encoder -----
        if encoder_path is not None:
            folder = Path(encoder_path)
        else:
            folder = _REPO_ROOT / "galileo_weights" / "models" / "nano"

        print(f"[Galileo] Loading encoder from: {folder}")
        assert (folder / "config.json").exists(), \
            f"config.json not found in {folder}"
        assert (folder / "encoder.pt").exists(), \
            f"encoder.pt not found in {folder}"

        self.encoder = Encoder.load_from_folder(folder, device=torch.device("cpu"))
        self.embedding_size = self.encoder.embedding_size

        # Verify weights are loaded (not random init)
        with open(folder / "config.json") as f:
            enc_cfg = json.load(f)["model"]["encoder"]
        print(f"[Galileo] Config: embed={enc_cfg.get('embedding_size')}, "
              f"depth={enc_cfg.get('depth')}, heads={enc_cfg.get('num_heads')}")

        # Spot-check: compare a param against the checkpoint
        ckpt = torch.load(folder / "encoder.pt", map_location="cpu")
        sample_key = next(iter(ckpt.keys()))
        model_val = dict(self.encoder.named_parameters())[
            sample_key.replace(".backbone", "")]
        if torch.equal(model_val.data, ckpt[sample_key]):
            print(f"[Galileo] ✓ Pretrained weights verified (checked: {sample_key})")
        else:
            print(f"[Galileo] ✗ WARNING: weight mismatch for {sample_key}!")
        del ckpt

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.use_lora = use_lora

        if use_lora:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q", "k", "v", "proj"],
                lora_dropout=0.0,
                bias="none",
            )
            inject_adapter_in_model(lora_cfg, self.encoder, adapter_name="default")
            for name, param in self.encoder.named_parameters():
                param.requires_grad_("lora_" in name)

        # ----- Segmentation head -----
        self.head = SegmentationHead(
            in_dim=self.embedding_size,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            upsample_factor=patch_size,
            mode=head_mode,
        )

        # ----- Re‑normalisation buffers (not learned) -----
        if bands is None or channel_stats is None:
            raise ValueError("bands and channel_stats must be provided")
        scales, offsets, dst_indices = _build_renorm_params(bands, channel_stats)
        self._dst_indices = dst_indices  # Python list of Galileo SPACE_TIME_BANDS indices
        self.register_buffer("_renorm_scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("_renorm_offsets", torch.tensor(offsets, dtype=torch.float32))

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_lora = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_head = sum(p.numel() for p in self.head.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_str = f" | lora: {n_lora:,}" if use_lora else ""
        print(f"[Galileo] encoder: {n_enc:,}{lora_str} | head: {n_head:,} | "
              f"trainable: {n_train:,}")
        print(f"  tile_size={tile_size}, patch_size={patch_size}, "
              f"head_mode={head_mode!r}, tile_chunk={tile_chunk}")

    # ------------------------------------------------------------------
    def _prepare_tiles(self, x_tiles: torch.Tensor, is_real=None):
        """
        Convert tile tensor from our normalisation to Galileo's input format.

        Args:
            x_tiles: (N, T, C, ts, ts) our‑normalised tiles
            is_real: (N, T) bool — True for real observations, False for zero-padded slots
        Returns:
            Tuple of (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        """
        N, T, C, ts_h, ts_w = x_tiles.shape
        device = x_tiles.device
        dtype = x_tiles.dtype

        # 1. space_time_x: (N, ts_h, ts_w, T, 13)
        s_t_x = torch.zeros(N, ts_h, ts_w, T, len(SPACE_TIME_BANDS),
                            device=device, dtype=dtype)
        x_hwt = rearrange(x_tiles, "n t c h w -> n h w t c")
        for src_idx in range(C):
            dst_idx = self._dst_indices[src_idx]
            s_t_x[:, :, :, :, dst_idx] = (
                x_hwt[:, :, :, :, src_idx] * self._renorm_scales[src_idx]
                + self._renorm_offsets[src_idx]
            )

        # 2. space_time mask: unmask S2 groups only
        n_st_groups = len(SPACE_TIME_BANDS_GROUPS_IDX)
        s_t_m = torch.ones(N, ts_h, ts_w, T, n_st_groups,
                           device=device, dtype=dtype)
        for g in S2_MASK_GROUPS:
            s_t_m[:, :, :, :, g] = 0

        # Re-mask empty timesteps: s_t_m=1 physically removes tokens via remove_masked_tokens
        if is_real is not None:
            # is_real: (N, T) → broadcast to (N, 1, 1, T, 1) to match s_t_m
            empty = ~is_real  # (N, T) True where timestep is zero-padded
            s_t_m = s_t_m.masked_fill(empty[:, None, None, :, None], 1)

        # 3. Other modalities: zeros + fully masked
        sp_x = torch.zeros(N, ts_h, ts_w, len(SPACE_BANDS),
                           device=device, dtype=dtype)
        sp_m = torch.ones(N, ts_h, ts_w, len(SPACE_BAND_GROUPS_IDX),
                          device=device, dtype=dtype)
        t_x = torch.zeros(N, T, len(TIME_BANDS), device=device, dtype=dtype)
        t_m = torch.ones(N, T, len(TIME_BAND_GROUPS_IDX),
                         device=device, dtype=dtype)
        st_x = torch.zeros(N, len(STATIC_BANDS), device=device, dtype=dtype)
        st_m = torch.ones(N, len(STATIC_BAND_GROUPS_IDX),
                          device=device, dtype=dtype)

        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m

    # ------------------------------------------------------------------
    def _encode_tiles(self, x_tiles, months_tiles, is_real=None):
        """
        Encode tiles in chunks to limit GPU memory.

        Args:
            x_tiles:      (N, T, C, ts, ts) our‑normalised tiles
            months_tiles: (N, T) month indices
            is_real:      (N, T) bool — True for real observations
        Returns:
            features: (N, ph*pw, D) per‑patch pooled features
        """
        N = x_tiles.shape[0]
        feat_list = []

        for start in range(0, N, self.tile_chunk):
            end = min(start + self.tile_chunk, N)
            chunk_x = x_tiles[start:end]
            chunk_months = months_tiles[start:end]
            chunk_is_real = is_real[start:end] if is_real is not None else None

            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = \
                self._prepare_tiles(chunk_x, is_real=chunk_is_real)

            if self.use_lora:
                # Gradient checkpointing: free intermediate activations after each
                # encoder call (attention scores, Q/K/V, MLP) and recompute during
                # backward. Without this, 64 calls × 340 MB/call = ~90 GB OOM.
                _patch_size = self.patch_size
                def _enc_fn(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months):
                    return self.encoder(
                        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
                        patch_size=_patch_size,
                    )
                enc_out = torch.utils.checkpoint.checkpoint(
                    _enc_fn,
                    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, chunk_months,
                    use_reentrant=False,
                )
            else:
                enc_out = self.encoder(
                    s_t_x, sp_x, t_x, st_x,
                    s_t_m, sp_m, t_m, st_m,
                    chunk_months,
                    patch_size=self.patch_size,
                )
            s_t_enc, sp_enc, t_enc, st_enc, \
                s_t_m_enc, sp_m_enc, t_m_enc, st_m_enc, _ = enc_out

            # Pool tokens per spatial patch → (chunk, ph*pw, D)
            feat = Encoder.apply_mask_and_average_tokens_per_patch(
                s_t_enc, sp_enc, t_enc, st_enc,
                s_t_m_enc, sp_m_enc, t_m_enc, st_m_enc,
            )
            feat_list.append(feat)

        return torch.cat(feat_list, dim=0)  # (N, ph*pw, D)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, batch_positions=None, strip_empty_timesteps=False):
        """
        Args:
            x:                      (B, T, C, H, W) our‑normalised S2 data
            batch_positions:         (B, T) DOY timestamps → months for Galileo PE
            strip_empty_timesteps:   if True, mask zero-padded timesteps out of
                                     Galileo attention (controlled by --variable_t).
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, T, C, H, W = x.shape
        TS = self.tile_size
        device = x.device

        # ── 0. Detect real vs zero-padded timesteps (only when requested) ──
        # Caller sets strip_empty_timesteps=True via --variable_t; when False,
        # zero-padded slots pass through to Galileo attention unchanged.
        if strip_empty_timesteps:
            is_real = x.abs().sum(dim=(2, 3, 4)) > 0
        else:
            is_real = None

        # ── 0b. Pad to be divisible by tile_size ──
        pad_h = (TS - H % TS) % TS
        pad_w = (TS - W % TS) % TS
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad = H + pad_h
        W_pad = W + pad_w
        n_tiles_h = H_pad // TS
        n_tiles_w = W_pad // TS
        n_tiles = n_tiles_h * n_tiles_w

        # ── 1. Split into tiles ──
        # (B, T, C, H_pad, W_pad) → (B*n_h*n_w, T, C, TS, TS)
        x_tiles = rearrange(
            x, "b t c (nh th) (nw tw) -> (b nh nw) t c th tw",
            th=TS, tw=TS,
        )

        # ── 2. Months from batch_positions ──
        if batch_positions is not None:
            doy = batch_positions.float().clamp(1, 365)
            months = ((doy - 1) / 30.44).long().clamp(0, 11)  # (B, T)
        else:
            months = torch.ones(B, T, device=device, dtype=torch.long) * self.month

        # Expand months per tile: (B, T) → (B*n_tiles, T)
        months_tiles = months.unsqueeze(1).expand(B, n_tiles, T).reshape(
            B * n_tiles, T)
        if is_real is not None:
            is_real_tiles = is_real.unsqueeze(1).expand(B, n_tiles, T).reshape(
                B * n_tiles, T)
        else:
            is_real_tiles = None

        # ── 3. Encode all tiles (chunked) ──
        features = self._encode_tiles(x_tiles, months_tiles, is_real=is_real_tiles)
        # features: (B*n_h*n_w, ph*pw, D)

        # ── 4. Reassemble spatial feature map ──
        # → (B, D, n_h*ph, n_w*pw) = (B, D, H_pad/P, W_pad/P)
        features = rearrange(
            features, "(b nh nw) (ph pw) d -> b d (nh ph) (nw pw)",
            b=B, nh=n_tiles_h, nw=n_tiles_w, ph=self.ph, pw=self.pw,
        )

        # ── 5. Segmentation head (upsamples by patch_size) ──
        logits = self.head(features)  # (B, num_classes, H_pad, W_pad)

        # ── 6. Crop back to original size ──
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :H, :W]

        return logits


# ============================================================================
# Factory
# ============================================================================
def get_galileo_model(
    num_classes: int = 53,
    encoder_path: Optional[str] = None,
    patch_size: int = 4,
    tile_size: int = 16,
    tile_chunk: int = 16,
    freeze_encoder: bool = False,
    head_mode: str = "single",
    head_hidden_dim: int = 256,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    **kwargs,
) -> GalileoForSegmentation:
    return GalileoForSegmentation(
        num_classes=num_classes,
        encoder_path=encoder_path,
        patch_size=patch_size,
        tile_size=tile_size,
        tile_chunk=tile_chunk,
        freeze_encoder=freeze_encoder,
        head_mode=head_mode,
        head_hidden_dim=head_hidden_dim,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
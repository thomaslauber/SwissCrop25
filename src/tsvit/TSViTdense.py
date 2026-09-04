import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .module import Attention, PreNorm, FeedForward
import numpy as np


def get_params_values(args, key, default=None):
    if (key in args) and (args[key] is not None):
        return args[key]
    return default


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, key_padding_mask=None):
        for attn, ff in self.layers:
            x = attn(x, key_padding_mask=key_padding_mask) * 0.1 + x
            x = ff(x) * 0.1 + x
        return self.norm(x)


class TSViT(nn.Module):
    """
    Temporal-Spatial ViT for dense segmentation.

    Expects input x of shape [B, T, H, W, C+1] where the last channel is the
    day-of-year timestamp (raw float, e.g. 1..365).  Inside forward the model
    permutes to [B, T, C+1, H, W], strips the timestamp channel and uses it as
    a continuous temporal position embedding.

    Output: [B, num_classes, H, W]
    """

    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.temporal_depth = model_config.get('temporal_depth', model_config.get('depth', 4))
        self.spatial_depth = model_config.get('spatial_depth', model_config.get('depth', 4))
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}
        num_patches = self.num_patches_1d ** 2
        # num_channels counts spectral bands only (timestamp is stripped)
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),
        )
        self.to_temporal_embedding_input = nn.Linear(1, self.dim)
        nn.init.xavier_uniform_(self.to_temporal_embedding_input.weight)
        nn.init.zeros_(self.to_temporal_embedding_input.bias)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        nn.init.normal_(self.temporal_token, std=0.02)
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        nn.init.normal_(self.space_pos_embedding, std=0.02)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                             self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )

    def forward(self, x):
        # x: [B, T, H, W, C+1]  (last channel = day-of-year timestamp)
        x = x.permute(0, 1, 4, 2, 3)   # [B, T, C+1, H, W]
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]          # [B, T] timestamps
        x = x[:, :, :-1]                 # [B, T, C, H, W] spectral only

        # Detect zero-padded timesteps before patch embedding
        is_real = x.abs().sum(dim=(2, 3, 4)) > 0  # (B, T)

        xt = xt.unsqueeze(-1)            # [B, T, 1]
        temporal_pos_embedding = self.to_temporal_embedding_input(xt)  # [B, T, dim]

        x = self.to_patch_embedding(x)  # [(B*num_patches), T, dim]
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)

        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d',
                                     b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # Build key_padding_mask: False for class tokens, True where timestep is empty
        # shape: (B*num_patches, 1, 1, num_classes+T) for broadcasting over heads and queries
        B_flat = B * self.num_patches_1d ** 2
        is_real_flat = is_real.unsqueeze(1).expand(B, self.num_patches_1d ** 2, T).reshape(B_flat, T)
        cls_visible = torch.zeros(B_flat, self.num_classes, dtype=torch.bool, device=x.device)
        kpm = torch.cat([cls_visible, ~is_real_flat], dim=1).view(B_flat, 1, 1, -1)

        x = self.temporal_transformer(x, key_padding_mask=kpm)
        x = x[:, :self.num_classes]

        x = (x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim)
              .permute(0, 2, 1, 3)
              .reshape(B * self.num_classes, self.num_patches_1d ** 2, self.dim))
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = (x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2)
              .permute(0, 2, 3, 1)
              .reshape(B, H, W, self.num_classes)
              .permute(0, 3, 1, 2))  # [B, num_classes, H, W]
        return x

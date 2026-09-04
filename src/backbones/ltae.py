import copy

import numpy as np
import torch
import torch.nn as nn

from src.backbones.positional_encoding import PositionalEncoder


class LTAE2d_TimeChannel(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
        add_time_channel=True,
        time_encoding_method="replicate",  # "replicate" or "key_only"
        add_doy_channel=False,
    ):
        """
        Lightweight Temporal Attention Encoder with Time Channel Support.
        
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the MLP layers.
            dropout (float): Dropout rate.
            d_model (int): Dimension of the feature space.
            T (int): Period for positional encoding.
            return_att (bool): Return attention masks.
            positional_encoding (bool): Use positional encoding.
            add_time_channel (bool): Add time as additional channel(s).
            time_encoding_method (str): How to add time information:
                - "replicate": Add n_head time channels (one per head) - RECOMMENDED
                - "key_only": Add time only to keys, not values (more efficient)
            add_doy_channel (bool): Also add DOY as n_head additional channels alongside GDD.
                Only active when add_time_channel=True. Doubles the time channel count.
        """
        super(LTAE2d_TimeChannel, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head
        self.add_time_channel = add_time_channel
        self.time_encoding_method = time_encoding_method
        self.add_doy_channel = add_doy_channel

        # Number of time signals: 1 (GDD only) or 2 (GDD + DOY)
        n_time_signals = 2 if (add_time_channel and add_doy_channel) else 1

        # Determine effective input channels after time channel addition
        if add_time_channel and time_encoding_method == "replicate":
            self.effective_in_channels = in_channels + n_head * n_time_signals
        else:
            self.effective_in_channels = in_channels

        if d_model is not None:
            self.d_model = d_model
            # If adding time channels, adjust input dimension
            if add_time_channel and time_encoding_method == "replicate":
                self.inconv = nn.Conv1d(in_channels + n_head * n_time_signals, d_model, 1)
            else:
                self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # Attention mechanism
        if add_time_channel and time_encoding_method == "key_only":
            # Modified attention that adds time to keys
            self.attention_heads = MultiHeadAttention_TimeInKeys(
                n_head=n_head, d_k=d_k, d_in=self.d_model,
                num_time_signals=n_time_signals
            )
        else:
            # Standard attention (time already in values if using "replicate")
            self.attention_heads = MultiHeadAttention(
                n_head=n_head, d_k=d_k, d_in=self.d_model
            )

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.effective_in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend([
                nn.Linear(self.mlp[i], self.mlp[i + 1]),
                nn.BatchNorm1d(self.mlp[i + 1]),
                nn.ReLU(),
            ])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, time_values=None):
        """
        Args:
            x: (B, T, C, H, W) input tensor
            batch_positions: (B, T) position indices for positional encoding
            pad_mask: (B, T) padding mask
            time_values: (B, T) or (B, T, H, W) actual time values (e.g., GDD, day of year)
                - (B, T): scalar time values, broadcast to all pixels
                - (B, T, H, W): spatial time values, pixel-specific
                NO NORMALIZATION - raw values are used!
        """
        sz_b, seq_len, d, h, w = x.shape

        # Add time channel(s) BEFORE any processing
        if self.add_time_channel:
            # Unpack dual signals if present: time_values can be a 2-tuple (gdd, doy)
            if self.add_doy_channel and isinstance(time_values, tuple):
                gdd_values, doy_values = time_values[0], time_values[1]
            else:
                gdd_values = time_values[0] if isinstance(time_values, tuple) else time_values
                doy_values = None

            if gdd_values is None:
                # Fallback to batch_positions or sequential
                if batch_positions is not None:
                    gdd_values = batch_positions.float()
                else:
                    gdd_values = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(sz_b, -1).float()

            if self.time_encoding_method == "replicate":
                # Create n_head copies of GDD channel (one per head)
                if gdd_values.dim() == 2:  # (B, T) - scalar mode
                    gdd_channel = gdd_values.view(sz_b, seq_len, 1, 1, 1).expand(
                        sz_b, seq_len, self.n_head, h, w
                    )
                elif gdd_values.dim() == 4:  # (B, T, H, W) - spatial mode
                    gdd_channel = gdd_values.unsqueeze(2).expand(
                        sz_b, seq_len, self.n_head, h, w
                    )
                else:
                    raise ValueError(f"gdd_values must be 2D (scalar) or 4D (spatial), got shape {gdd_values.shape}")

                if doy_values is not None:
                    # Also add n_head DOY channels alongside GDD channels
                    doy_channel = doy_values.view(sz_b, seq_len, 1, 1, 1).expand(
                        sz_b, seq_len, self.n_head, h, w
                    )
                    x = torch.cat([x, gdd_channel, doy_channel], dim=2)
                    d = d + 2 * self.n_head
                else:
                    x = torch.cat([x, gdd_channel], dim=2)
                    d = d + self.n_head
            # For "key_only" method, time is added later in attention
        
        # Standard LTAE processing
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # Pass time_values to attention if using "key_only" method
        if self.add_time_channel and self.time_encoding_method == "key_only" and gdd_values is not None:
            if gdd_values.dim() == 2:  # (B, T) - scalar mode
                gdd_flat = gdd_values.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
                gdd_flat = gdd_flat.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            elif gdd_values.dim() == 4:  # (B, T, H, W) - spatial mode
                gdd_flat = gdd_values.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            else:
                raise ValueError(f"gdd_values must be 2D or 4D for key_only method, got {gdd_values.shape}")

            if doy_values is not None:
                doy_flat = doy_values.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
                doy_flat = doy_flat.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
                time_flat = torch.stack([gdd_flat, doy_flat], dim=-1)  # (B*H*W, T, 2)
            else:
                time_flat = gdd_flat.unsqueeze(-1)  # (B*H*W, T, 1)

            out, attn = self.attention_heads(out, pad_mask=pad_mask, time_values=time_flat)
        else:
            out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention_TimeInKeys(nn.Module):
    """
    Modified Multi-Head Attention that adds time information to keys only.
    More parameter-efficient than replicating time channels.
    """
    def __init__(self, n_head, d_k, d_in, num_time_signals=1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        # Key projection includes time information
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        # Learnable projection for time values: supports 1 (GDD) or 2 (GDD+DOY) signals
        self.time_k = nn.Linear(num_time_signals, n_head * d_k)
        nn.init.normal_(self.time_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, time_values=None, return_comp=False):
        """
        Args:
            v: (sz_b, seq_len, d_in) values
            pad_mask: (sz_b, seq_len) padding mask
            time_values: (sz_b, seq_len) time values - NO NORMALIZATION!
        """
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
        
        # Keys from features
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        
        # Add time information to keys
        # time_values: (sz_b, T, num_signals) — already shaped correctly by LTAE2d forward
        if time_values is not None:
            time_k = self.time_k(time_values).view(sz_b, seq_len, n_head, d_k)
            k = k + time_k  # Add time encoding to keys
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        
        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
            
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn



##### Original LTAE for reference #####

class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

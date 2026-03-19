


# PURE ViT VERSION (Flow magnitude / optional Δm / adaptive multi-scale Δm):
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torchvision
from torch import nn
from torch.nn.init import normal_, constant_

from ops.transforms import *


# --------------------------- Temporal head (TCN-style) ---------------------------
class TemporalConvHead(nn.Module):
    """TCN-style temporal head: [B,T,F] -> [B,F] (GN by default)."""
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        kernel_size: int = 3,
        norm: str = "gn",  # "gn" | "bn"
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim
        pad = kernel_size // 2

        def _norm_layer(c: int):
            if norm.lower() == "bn":
                return nn.BatchNorm1d(c)
            # GroupNorm(1, C) is batch-size-agnostic and works well for Conv1d features
            return nn.GroupNorm(1, c)

        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=kernel_size, padding=pad, bias=False),
            _norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Conv1d(hidden_dim, feat_dim, kernel_size=kernel_size, padding=pad, bias=False),
            _norm_layer(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F] -> [B,F,T]
        x = x.transpose(1, 2).contiguous()
        x = self.net(x)        # [B,F,T]
        x = x.mean(dim=-1)     # [B,F]
        return x


# -------------------- Adaptive Multi-Scale Temporal Difference (Scheme B) --------------------
class AdaptiveMultiScaleTemporalDiff(nn.Module):
    """
    Adaptive Multi-Scale Temporal Difference (AMTD), Scheme B: scale-attention weighting.

    Input : m_seq [B,N,H,W]
    Output: out   [B,N,H,W]  (same shape)

    """
    def __init__(
        self,
        scales: Sequence[int] = (1, 2, 3, 4),
        attn_hidden: int = 32,
        temperature: float = 1.0,
        residual: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.scales = [int(s) for s in scales]
        if any(s <= 0 for s in self.scales):
            raise ValueError(f"AMTD scales must be positive, got {self.scales}")
        self.S = len(self.scales)
        self.temperature = float(temperature)
        self.residual = bool(residual)
        self.eps = float(eps)

        # per-time-step MLP over scale dimension S
        self.mlp = nn.Sequential(
            nn.Linear(self.S, int(attn_hidden), bias=True),
            nn.GELU(),
            nn.Linear(int(attn_hidden), self.S, bias=True),
        )

    @staticmethod
    def _delta_k(m_seq: torch.Tensor, k: int) -> torch.Tensor:
        # m_seq: [B,N,H,W]
        out = torch.zeros_like(m_seq)
        out[:, :k] = m_seq[:, :k]
        out[:, k:] = m_seq[:, k:] - m_seq[:, :-k]
        return out

    def forward(self, m_seq: torch.Tensor) -> torch.Tensor:
        B, N, H, W = m_seq.shape

        # 1) stack deltas: [B,S,N,H,W]
        d = torch.stack([self._delta_k(m_seq, k) for k in self.scales], dim=1)

        # 2) spatial pooling: [B,S,N]
        g = d.mean(dim=(3, 4))

        # 3) per-time attention weights over scales
        g_t = g.permute(0, 2, 1).contiguous()          # [B,N,S]
        logits = self.mlp(g_t)                         # [B,N,S]
        logits = logits / max(self.temperature, self.eps)
        w = torch.softmax(logits, dim=-1)              # [B,N,S]

        # 4) weighted fuse
        d_t = d.permute(0, 2, 1, 3, 4).contiguous()     # [B,N,S,H,W]
        fused = (d_t * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)  # [B,N,H,W]

        if self.residual:
            fused = m_seq + fused

        return fused


# -------------------- Conv tokenization (patch embed + center-crop + attention pool) --------------------
class MagnitudeTokenConv(nn.Module):
    """
    Convert magnitude frames [B,N,H,W] -> frame tokens [B,N,C].
    """
    def __init__(self, out_dim=256, patch_size=16, drop=0.1, crop_border_patches=1):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(1, out_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        self.crop_border_patches = int(crop_border_patches)

        # Spatial attention pooling
        self.attn = nn.Linear(out_dim, 1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, m_seq: torch.Tensor) -> torch.Tensor:
        """
        m_seq: [B,N,H,W]
        return: [B,N,C]
        """
        B, N, H, W = m_seq.shape
        x = m_seq.reshape(B * N, 1, H, W)      # [B*N,1,H,W]
        x = self.proj(x)                       # [B*N,C,Hp,Wp]
        x = self.act(x)
        x = self.drop(x)

        # Center crop patch grid
        b = self.crop_border_patches
        if b > 0 and x.size(-2) > 2 * b and x.size(-1) > 2 * b:
            x = x[:, :, b:-b, b:-b]

        # Flatten patches: [B*N,C,Hp,Wp] -> [B*N,P,C]
        x = x.flatten(2).transpose(1, 2).contiguous()

        # Attention weights over patches -> token
        if x.size(1) == 1:
            token = x[:, 0]
        else:
            w = self.attn(x).squeeze(-1)       # [B*N,P]
            w = torch.softmax(w, dim=1)
            token = (x * w.unsqueeze(-1)).sum(dim=1)

        token = self.norm(token)
        token = token.view(B, N, -1)           # [B,N,C]
        return token


# -------------------- Temporal ViT (TransformerEncoder over N frame tokens) --------------------
class TemporalViT(nn.Module):
    """Transformer encoder over [B,N,C] with learnable absolute pos embedding."""
    def __init__(self, dim=256, depth=4, num_heads=8, mlp_ratio=4.0, drop=0.1, max_len=512):
        super().__init__()
        self.max_len = int(max_len)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        ff_dim = int(dim * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if N > self.max_len:
            raise ValueError(f"TemporalViT got N={N} > max_len={self.max_len}. Increase max_len.")
        x = x + self.pos[:, :N, :]
        x = self.drop(x)
        x = self.encoder(x)  # [B,N,C]
        return x



class TSN(nn.Module):
    """
    Flow magnitude + optional Δm (global) or adaptive multi-scale Δm version.
    Expect Flow input: [B, 2*N, H, W], N=num_segments*new_length
      channel order: u1,v1,u2,v2,...,uN,vN
    """
    def __init__(
        self,
        num_class,
        num_segments,
        modality,
        base_model='vit',
        new_length=None,
        consensus_type='avg',
        before_softmax=True,
        dropout=0.5,
        img_feature_dim=256,
        crop_num=1,
        partial_bn=True,
        print_spec=True,
        pretrain='none',
        is_shift=False,
        shift_div=8,
        shift_place='blockres',
        fc_lr5=False,
        temporal_pool=False,
        non_local=False,
        temporal_head='tcn',
        temporal_dropout=0.2,
        temporal_kernel=3,
        use_magnitude_only=True,
        magnitude_eps=1e-6,
        magnitude_global_delta=True,
        # ----- preprocessing (keep train/test consistent) -----
        magnitude_log1p=True,
        magnitude_frame_norm="none",  # "none" | "per_frame"
        # ----- adaptive multi-scale temporal diff -----
        magnitude_multiscale_delta=True,     # <-- NEW: prefer AMTD
        ms_scales=(1, 2, 3, 4),              # <-- NEW
        ms_attn_hidden=32,                   # <-- NEW
        ms_temperature=1.0,                  # <-- NEW
        ms_residual=False,                   # <-- NEW
        # ----- PURE ViT config -----
        patch_size=16,
        vit_dim=256,
        vit_depth=4,
        vit_heads=8,
        vit_mlp_ratio=4.0,
        vit_drop=0.1,
        max_len=512,
        crop_border_patches=0,#1,
        head_norm="gn",  # temporal head norm: "gn" or "bn"
    ):
        super().__init__()

        self.modality = modality
        self.num_segments = int(num_segments)
        self.before_softmax = before_softmax
        self.dropout = float(dropout)
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = int(new_length)

        self.use_magnitude_only = bool(use_magnitude_only)
        self.magnitude_eps = float(magnitude_eps)
        self.magnitude_global_delta = bool(magnitude_global_delta)

        self.magnitude_log1p = bool(magnitude_log1p)
        self.magnitude_frame_norm = str(magnitude_frame_norm).lower()

        # ---- NEW: adaptive multi-scale temporal diff ----
        self.magnitude_multiscale_delta = bool(magnitude_multiscale_delta)
        self.ms_diff = AdaptiveMultiScaleTemporalDiff(
            scales=ms_scales,
            attn_hidden=ms_attn_hidden,
            temperature=ms_temperature,
            residual=ms_residual,
        )

        self.temporal_head_type = temporal_head
        self.temporal_dropout = float(temporal_dropout)
        self.temporal_kernel = int(temporal_kernel)

        # defaults used by transforms/main.py
        self.input_size = 224
        if self.modality == 'Flow':
            # Flow magnitude: do NOT use ImageNet normalization.
            self.input_mean = [0.0]
            self.input_std = [1.0]
        else:
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        # ---- token conv + temporal vit ----
        self.tokenizer = MagnitudeTokenConv(
            out_dim=vit_dim,
            patch_size=patch_size,
            drop=vit_drop,
            crop_border_patches=crop_border_patches,
        )
        self.token_drop = nn.Dropout(vit_drop)

        self.temporal_vit = TemporalViT(
            dim=vit_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            drop=vit_drop,
            max_len=max_len,
        )

        # temporal head
        if self.temporal_head_type.lower() == 'tcn':
            self.temporal_module = TemporalConvHead(
                feat_dim=vit_dim,
                dropout=self.temporal_dropout,
                kernel_size=self.temporal_kernel,
                norm=head_norm,
            )
        else:
            raise ValueError(f"Unknown temporal_head: {self.temporal_head_type}")

        # classifier head (use TSN --dropout here to fight overfitting)
        self.head_drop = nn.Dropout(p=self.dropout) if self.dropout and self.dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(vit_dim, num_class)
        std = 0.001
        normal_(self.classifier.weight, 0, std)
        constant_(self.classifier.bias, 0)

        if not self.before_softmax:
            self.softmax = nn.Softmax(dim=1)

        if print_spec:
            print(f"""
Initializing PURE-ViT TSN (Magnitude/Δm/AMTD).
Configurations:
    modality:                    {self.modality}
    num_segments:                {self.num_segments}
    new_length:                  {self.new_length}
    magnitude_log1p:             {self.magnitude_log1p}
    magnitude_frame_norm:        {self.magnitude_frame_norm}
    magnitude_multiscale_delta:  {self.magnitude_multiscale_delta}
        ms_scales:               {tuple(ms_scales)}
        ms_attn_hidden:          {ms_attn_hidden}
        ms_temperature:          {ms_temperature}
        ms_residual:             {ms_residual}
    magnitude_global_delta:      {self.magnitude_global_delta} (fallback if AMTD off)
    temporal_head:               {self.temporal_head_type} (norm={head_norm})
    mag->tokens->ViT dim:        {vit_dim}
    patch_size:                  {patch_size}
    center crop patches:         remove {crop_border_patches} border patch(es)
""")

    # ---------- properties required by pipeline ----------
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    # ---------- magnitude & Δm ----------
    def _flow_uv_to_magnitude(self, x_seq_uv: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_seq_uv.shape
        if C % 2 != 0:
            raise ValueError(f"[Flow] Expect even channels (u,v pairs), got C={C}")
        N = C // 2
        uv = x_seq_uv.view(B, N, 2, H, W)
        u = uv[:, :, 0, :, :]
        v = uv[:, :, 1, :, :]
        m = torch.sqrt(u * u + v * v + self.magnitude_eps)
        return m  # [B,N,H,W]

    def _magnitude_global_delta(self, m_seq: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(m_seq)
        out[:, 0] = m_seq[:, 0]
        out[:, 1:] = m_seq[:, 1:] - m_seq[:, :-1]
        return out

    def _frame_norm(self, m_seq: torch.Tensor) -> torch.Tensor:
        if self.magnitude_frame_norm == "none":
            return m_seq
        if self.magnitude_frame_norm == "per_frame":
            mean = m_seq.mean(dim=(2, 3), keepdim=True)
            std = m_seq.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            return (m_seq - mean) / std
        raise ValueError(f"Unknown magnitude_frame_norm={self.magnitude_frame_norm}")

    # ---------- forward ----------
    def forward(self, input: torch.Tensor, no_reshape: bool = False) -> torch.Tensor:
        if no_reshape:
            raise RuntimeError("PURE ViT TSN does not support no_reshape=True path.")

        if self.modality != 'Flow':
            raise ValueError("This version expects modality='Flow' (magnitude / Δm / AMTD).")

        N = self.num_segments * self.new_length
        c_need = 2 * N

        if input.dim() != 4:
            raise ValueError(f"Expect input as [B,C,H,W], got {tuple(input.shape)}")
        if input.size(1) < c_need:
            raise ValueError(f"[Flow] Need >= {c_need} channels (2*N), got {input.size(1)}")

        x_uv = input[:, :c_need, :, :]                 # [B, 2N, H, W]
        m_seq = self._flow_uv_to_magnitude(x_uv)       # [B, N, H, W]

        # preprocessing
        if self.magnitude_log1p:
            m_seq = torch.log1p(m_seq)
        m_seq = self._frame_norm(m_seq)

        # temporal difference (AMTD preferred, else fallback global Δm)
        if self.magnitude_multiscale_delta:
            m_seq = self.ms_diff(m_seq)                # [B,N,H,W]
        elif self.magnitude_global_delta:
            m_seq = self._magnitude_global_delta(m_seq)

        tokens = self.tokenizer(m_seq)                 # [B,N,C]
        tokens = self.token_drop(tokens)

        tokens = self.temporal_vit(tokens)             # [B,N,C]
        video_feat = self.temporal_module(tokens)      # [B,C]

        video_feat = self.head_drop(video_feat)
        logits = self.classifier(video_feat)           # [B,num_class]
        if not self.before_softmax:
            logits = self.softmax(logits)
        return logits

    # ---------- optim policies ----------
    def get_optim_policies(self):

        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn_like = []
        ln = []
        custom_ops = []

        conv_cnt = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                bn_like.extend(list(m.parameters()))

            elif isinstance(m, nn.LayerNorm):
                ln.extend(list(m.parameters()))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f"New atomic module type: {type(m)}. Need to give it a learning policy")

        is_flow_like = (self.modality == 'Flow')
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if is_flow_like else 1, 'decay_mult': 1, 'name': "first_conv_weight"},
            {'params': first_conv_bias,   'lr_mult': 10 if is_flow_like else 2, 'decay_mult': 0, 'name': "first_conv_bias"},
            {'params': normal_weight,     'lr_mult': 1, 'decay_mult': 1, 'name': "normal_weight"},
            {'params': normal_bias,       'lr_mult': 2, 'decay_mult': 0, 'name': "normal_bias"},
            {'params': bn_like,           'lr_mult': 1, 'decay_mult': 0, 'name': "BN/GN scale/shift"},
            {'params': ln,                'lr_mult': 1, 'decay_mult': 0, 'name': "LN scale/shift"},
            {'params': custom_ops,        'lr_mult': 1, 'decay_mult': 1, 'name': "custom_ops"},
            {'params': lr5_weight,        'lr_mult': 5, 'decay_mult': 1, 'name': "lr5_weight"},
            {'params': lr10_bias,         'lr_mult': 10, 'decay_mult': 0, 'name': "lr10_bias"},
        ]

    # ---------- augmentation ----------
    def get_augmentation(self, flip=True):

        if self.modality == 'Flow':
            return torchvision.transforms.Compose([
                GroupMultiScaleCrop(self.input_size, [1.0, 0.95, 0.9]),
                GroupRandomHorizontalFlip(is_flow=False) if flip else IdentityTransform(),
            ])
        else:
            return torchvision.transforms.Compose([
                GroupMultiScaleCrop(self.input_size, [1.0, 0.95, 0.9, 0.85]),
                GroupRandomHorizontalFlip(is_flow=False) if flip else IdentityTransform(),
            ])

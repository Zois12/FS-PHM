"""Physical-time tokenizer and Physics-MoE PHM foundation model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .order_tracking import order_spectrum
from .envelope import EnvelopeSpectrumResidual
from .power_spectrum import PowerFusionHead, PowerSpectrumNormalizer
from .confidence_gate import ConfidenceGatedPowerHead
from .physical_layers import DeformableConv1d, LocalSparseSelfAttention, MorletCWTFeature

BACKBONE_ABLATIONS = ("none", "no-time", "no-frequency", "no-fft", "no-cwt", "no-order",
                      "no-deformable-offsets", "no-local-attention", "uniform-router",
                      "no-fast", "no-slow")


def normalize_backbone_ablation(mode: str) -> str:
    """Validate one or several knockouts, preserving legacy string configs."""
    parts = mode.split("+")
    if any(p not in BACKBONE_ABLATIONS for p in parts) or len(set(parts)) != len(parts):
        raise ValueError(f"Use distinct backbone ablations from {BACKBONE_ABLATIONS}, joined with +")
    if "none" in parts and len(parts) > 1:
        raise ValueError("none cannot be combined with a knockout")
    if len(set(parts) & {"no-fast", "no-slow", "uniform-router"}) > 1:
        raise ValueError("no-fast, no-slow and uniform-router are mutually exclusive")
    return "+".join(p for p in BACKBONE_ABLATIONS if p in parts)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denominator


def _positional_encoding(position: Tensor, dimension: int) -> Tensor:
    """Continuous sinusoidal positions, avoiding a fixed maximum sequence size."""

    half = (dimension + 1) // 2
    frequency = torch.exp(
        torch.arange(half, device=position.device, dtype=position.dtype)
        * (-math.log(10000.0) / max(half, 1))
    )
    angles = position.unsqueeze(-1) * frequency
    result = torch.zeros(*position.shape, dimension, device=position.device, dtype=position.dtype)
    result[..., 0::2] = torch.sin(angles[..., : result[..., 0::2].shape[-1]])
    if result[..., 1::2].shape[-1]:
        result[..., 1::2] = torch.cos(angles[..., : result[..., 1::2].shape[-1]])
    return result


def _physical_statistics(
    signals: Tensor, lengths: Tensor, channel_mask: Optional[Tensor] = None
) -> Tensor:
    """Compute RMS, excess kurtosis, and spectral entropy per sample."""

    batch, channels, length = signals.shape
    positions = torch.arange(length, device=signals.device).view(1, 1, length)
    valid = positions < lengths.view(batch, 1, 1)
    if channel_mask is None:
        channel_mask = torch.ones((batch, channels), device=signals.device, dtype=torch.bool)
    valid_channels = channel_mask.to(signals.dtype).sum(dim=1).clamp_min(1.0)
    point_count = lengths.to(signals.dtype).clamp_min(1.0)
    mask = valid.to(signals.dtype) * channel_mask.to(signals.dtype).unsqueeze(-1)
    count = point_count * valid_channels
    mean = (signals * mask).sum(dim=(1, 2), keepdim=True) / count.view(batch, 1, 1)
    centered = (signals - mean) * mask
    variance = (centered.square().sum(dim=(1, 2)) / count).clamp_min(1e-6)
    rms = torch.sqrt((signals.square() * mask).sum(dim=(1, 2)) / count + 1e-6)
    kurtosis = (centered.pow(4).sum(dim=(1, 2)) / count) / variance.square() - 3.0

    aggregate = (signals * mask).sum(dim=1) / valid_channels.view(batch, 1)
    entropy = aggregate.new_zeros((batch,), dtype=torch.float32)
    for valid_length in torch.unique(lengths).tolist():
        rows = torch.nonzero(lengths == valid_length, as_tuple=True)[0]
        spectrum = torch.fft.rfft(aggregate[rows, :valid_length].float(), dim=-1).abs().square()
        spectrum = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy_values = -(
            spectrum.clamp_min(1e-8) * spectrum.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(spectrum.shape[-1], 2))
        entropy.index_copy_(0, rows, entropy_values)
    return torch.stack((rms, kurtosis, entropy), dim=-1)


@dataclass
class FoundationConfig:
    embed_dim: int = 256
    patch_size: int = 256
    patch_stride: int = 128
    layers: int = 6
    attention_heads: int = 8
    dropout: float = 0.1
    mask_ratio: float = 0.4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    feature_version: int = 2
    physical_patch_seconds: float = 0.004
    local_attention_window: int = 9
    channel_fusion: str = "mean"
    zero_channel_policy: str = "keep"
    backbone_ablation: str = "none"


class ChannelAttentionFusion(nn.Module):
    """Shared feature-based scoring of channels at each physical patch.

    Zero initialization starts at uniform averaging over valid channels. No
    channel identity, raw amplitude, or batch-wide energy statistics are used.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, channel_tokens: Tensor, channel_mask: Tensor) -> Tensor:
        with torch.autocast(device_type=channel_tokens.device.type, enabled=False):
            normalized = F.layer_norm(channel_tokens.float(), (channel_tokens.shape[-1],))
            scores = (normalized * self.score.float()).sum(-1) / math.sqrt(channel_tokens.shape[-1])
            scores = scores.masked_fill(~channel_mask[:, :, None], float("-inf"))
            weights = scores.softmax(dim=1).to(channel_tokens.dtype)
        return (channel_tokens * weights.unsqueeze(-1)).sum(dim=1)


class MeanMaxPooling(nn.Module):
    """A trainable residual from masked mean towards masked maximum.

    Per-feature tanh gates start at zero, preserving the pretrained mean exactly.
    Signed bounded gates can learn to amplify or suppress a maximum response.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, hidden: Tensor, token_mask: Tensor) -> Tensor:
        valid = token_mask.bool()
        mean = _masked_mean(hidden, valid, dim=1)
        maximum = hidden.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(dim=1)
        maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return mean + self.gate.tanh().to(hidden.dtype) * (maximum - mean)


class AdaptiveTokenizer(nn.Module):
    """Channel-independent tokenizer with fixed physical patch duration."""

    def __init__(
        self,
        embed_dim: int = 256,
        patch_size: int = 256,
        patch_stride: int = 128,
        dropout: float = 0.1,
        feature_version: int = 2,
        physical_patch_seconds: float = 0.004,
        channel_fusion: str = "mean",
        zero_channel_policy: str = "keep",
    ) -> None:
        super().__init__()
        if patch_size < 8 or patch_stride < 1:
            raise ValueError("patch_size must be >= 8 and patch_stride must be positive")
        if feature_version not in {1, 2} or physical_patch_seconds <= 0:
            raise ValueError("feature_version must be 1 or 2 and physical_patch_seconds must be positive")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.feature_version = feature_version
        self.physical_patch_seconds = physical_patch_seconds
        mid = max(embed_dim // 2, 32)
        self.time_encoder = nn.Sequential(
            DeformableConv1d(1, mid, kernel_size=7, stride=4, padding=3, max_offset=2.0)
            if feature_version == 2 else nn.Conv1d(1, mid, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(mid, embed_dim, kernel_size=5, stride=4 if feature_version == 2 else 1, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.frequency_encoder = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if feature_version == 2:
            self.cwt = MorletCWTFeature()
            self.cwt_projection = nn.Linear(5, embed_dim)
            self.order_projection = nn.Linear(32, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.period_projection = nn.Linear(1, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.channel_fusion: Optional[ChannelAttentionFusion] = None
        self.configure_channel_fusion(channel_fusion, zero_channel_policy)

    def configure_channel_fusion(self, mode: str, zero_channel_policy: str) -> None:
        if mode not in {"mean", "attention"}:
            raise ValueError("channel fusion must be mean or attention")
        if zero_channel_policy not in {"keep", "drop-zero"}:
            raise ValueError("zero channel policy must be keep or drop-zero")
        if mode == "attention" and self.channel_fusion is None:
            self.channel_fusion = ChannelAttentionFusion(self.embed_dim)
        elif mode == "mean":
            self.channel_fusion = None
        self.zero_channel_policy = zero_channel_policy

    def _pad_for_patches(self, signals: Tensor) -> Tensor:
        length = signals.shape[-1]
        if length <= self.patch_size:
            right = self.patch_size - length
        else:
            remainder = (length - self.patch_size) % self.patch_stride
            right = (self.patch_stride - remainder) % self.patch_stride
        return F.pad(signals, (0, right))

    def _resample_to_physical_grid(
        self, signals: Tensor, lengths: Tensor, window_seconds: Tensor
    ) -> tuple[Tensor, Tensor]:
        target_lengths = torch.round(
            window_seconds.float() * (self.patch_size / self.physical_patch_seconds)
        ).long().clamp_min(1)
        if bool((target_lengths > 32768).any()):
            raise ValueError("physical resampling exceeds 32768 points; increase physical_patch_seconds")
        maximum = int(target_lengths.max().item())
        result = signals.new_zeros((signals.shape[0], signals.shape[1], maximum))
        pairs = sorted(set(zip(lengths.tolist(), target_lengths.tolist())))
        for source_length, target_length in pairs:
            rows = torch.nonzero((lengths == source_length) & (target_lengths == target_length), as_tuple=True)[0]
            source = signals.index_select(0, rows)[..., :source_length].float()
            if target_length == source_length:
                resized = source
            elif target_length < source_length:
                resized = F.interpolate(source, size=target_length, mode="area")
            else:
                resized = F.interpolate(source, size=target_length, mode="linear", align_corners=False)
            result.index_copy_(0, rows, F.pad(resized.to(signals.dtype), (0, maximum - target_length)))
        return result, target_lengths

    def forward(
        self,
        signals: Tensor,
        lengths: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        sample_periods: Optional[Tensor] = None,
        mask_ratio: float = 0.0,
        rpm_traces: Optional[Tensor] = None,
        rpm_trace_valid: Optional[Tensor] = None,
        shaft_angles: Optional[Tensor] = None,
        shaft_angle_valid: Optional[Tensor] = None,
        nominal_rpm: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if signals.ndim == 2:
            signals = signals.unsqueeze(1)
        if signals.ndim != 3:
            raise ValueError("signals must have shape (batch, channels, time)")
        batch, channels, original_length = signals.shape
        device = signals.device
        if lengths is None:
            lengths = torch.full((batch,), original_length, device=device, dtype=torch.long)
        lengths = lengths.to(device=device, dtype=torch.long).clamp(1, original_length)
        if channel_mask is None:
            channel_mask = torch.ones((batch, channels), device=device, dtype=torch.bool)
        channel_mask = channel_mask.to(device=device, dtype=torch.bool)
        if self.zero_channel_policy == "drop-zero":
            # Only explicitly opted-in, exactly zero normalized channels are
            # removed. Padding beyond each sample's valid length is ignored.
            time_valid = torch.arange(original_length, device=device)[None, None, :] < lengths[:, None, None]
            nonzero = signals.masked_fill(~time_valid, 0).ne(0).any(dim=-1)
            channel_mask = channel_mask & nonzero
        if not bool(channel_mask.any(dim=1).all()):
            raise ValueError("Every sample needs an available channel; check zero-channel policy and data")
        periods_provided = sample_periods is not None
        if sample_periods is None:
            sample_periods = signals.new_ones((batch,))
        sample_periods = sample_periods.to(device=device, dtype=signals.dtype).clamp_min(1e-6)

        raw_signals, raw_lengths = signals, lengths
        if self.feature_version == 2 and periods_provided:
            signals, lengths = self._resample_to_physical_grid(signals, lengths, sample_periods)
        padded = self._pad_for_patches(signals)
        patches = padded.unfold(-1, self.patch_size, self.patch_stride)
        # unfold gives (B, C, T, P), and T is dynamic for every padded batch.
        _, _, token_count, _ = patches.shape
        starts = torch.arange(token_count, device=device) * self.patch_stride
        patch_counts = (
            (lengths - self.patch_size).clamp_min(0) + self.patch_stride - 1
        ) // self.patch_stride + 1
        token_mask = torch.arange(token_count, device=device).unsqueeze(0) < patch_counts.unsqueeze(1)

        flattened = patches.reshape(batch * channels * token_count, 1, self.patch_size)
        time_features = self.time_encoder(flattened).flatten(1)
        if "no-time" in getattr(self, "ablation", "none").split("+"):
            time_features = time_features * 0
        spectrum = torch.fft.rfft(flattened.squeeze(1).float(), dim=-1).abs().log1p()
        frequency_features = self.frequency_encoder(spectrum)
        if {"no-fft", "no-frequency"} & set(getattr(self, "ablation", "none").split("+")):
            frequency_features = frequency_features * 0
        if self.feature_version == 2:
            cwt = self.cwt(flattened).mean(dim=-1).flatten(1)
            cwt_features = self.cwt_projection(cwt)
            frequency_features = frequency_features + cwt_features * (0 if {"no-cwt", "no-frequency"} & set(getattr(self, "ablation", "none").split("+")) else 1)
        time_features = time_features.view(batch, channels, token_count, self.embed_dim)
        frequency_features = frequency_features.view(batch, channels, token_count, self.embed_dim)
        gate = torch.sigmoid(self.gate(torch.cat((time_features, frequency_features), dim=-1)))
        channel_tokens = gate * time_features + (1.0 - gate) * frequency_features
        channel_tokens = channel_tokens * channel_mask[:, :, None, None].to(channel_tokens.dtype)
        denominator = channel_mask.sum(dim=1).clamp_min(1).view(batch, 1, 1)
        tokens = (
            channel_tokens.sum(dim=1) / denominator if self.channel_fusion is None
            else self.channel_fusion(channel_tokens, channel_mask)
        )

        positions = starts.to(signals.dtype).unsqueeze(0) / lengths.to(signals.dtype).view(batch, 1).clamp_min(1.0)
        tokens = tokens + _positional_encoding(positions, self.embed_dim)
        tokens = tokens + self.period_projection(sample_periods.log().view(batch, 1, 1))
        if self.feature_version == 2:
            orders, order_valid = order_spectrum(
                raw_signals, raw_lengths, channel_mask, sample_periods,
                rpm_traces=rpm_traces, rpm_trace_valid=rpm_trace_valid,
                shaft_angles=shaft_angles, shaft_angle_valid=shaft_angle_valid,
                nominal_rpm=nominal_rpm,
            )
            order_embedding = self.order_projection(orders.to(tokens.dtype))
            if {"no-order", "no-frequency"} & set(getattr(self, "ablation", "none").split("+")):
                order_embedding = order_embedding * 0
            tokens = tokens + order_embedding.unsqueeze(1) * order_valid[:, None, None]
        tokens = self.output_norm(self.dropout(tokens))

        mask_ratio = float(max(0.0, min(1.0, mask_ratio)))
        if mask_ratio > 0.0:
            random_mask = torch.rand((batch, token_count), device=device) < mask_ratio
            reconstruction_mask = random_mask & token_mask
            tokens = torch.where(
                reconstruction_mask.unsqueeze(-1),
                self.mask_token.view(1, 1, -1).to(tokens.dtype),
                tokens,
            )
        else:
            reconstruction_mask = torch.zeros_like(token_mask)

        target_patches = patches * channel_mask[:, :, None, None].to(patches.dtype)
        target_patches = target_patches.sum(dim=1) / denominator
        physical_stats = _physical_statistics(raw_signals, raw_lengths, channel_mask)
        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "reconstruction_mask": reconstruction_mask,
            "target_patches": target_patches,
            "physical_stats": physical_stats,
        }


class FastTransientBranch(nn.Module):
    def __init__(
        self, embed_dim: int, attention_heads: int, dropout: float,
        feature_version: int = 2, local_attention_window: int = 9,
    ) -> None:
        super().__init__()
        self.feature_version = feature_version
        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=2)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=1)
        self.attention = (
            LocalSparseSelfAttention(
                embed_dim, attention_heads, window_size=local_attention_window, dropout=dropout
            )
            if feature_version == 2 else
            nn.MultiheadAttention(embed_dim, attention_heads, dropout=dropout, batch_first=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        normalized = self.norm(tokens)
        sequence = normalized.transpose(1, 2)
        # Left padding preserves the causal direction while keeping the length.
        sequence = F.pad(sequence, (4, 0))
        sequence = F.gelu(self.conv1(sequence))
        sequence = F.pad(sequence, (2, 0))
        sequence = F.gelu(self.conv2(sequence)).transpose(1, 2)
        if self.feature_version == 2:
            attended = self.attention(sequence, token_mask)
        else:
            attended, _ = self.attention(
                sequence, sequence, sequence, key_padding_mask=~token_mask, need_weights=False
            )
        if "no-local-attention" in getattr(self, "ablation", "none").split("+"):
            attended = attended * 0
        return self.dropout(sequence + attended) * token_mask.unsqueeze(-1).to(tokens.dtype)


class SlowDegradationBranch(nn.Module):
    def __init__(
        self, embed_dim: int, attention_heads: int, dropout: float, feature_version: int = 2
    ) -> None:
        super().__init__()
        self.feature_version = feature_version
        if feature_version == 2:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim),
            )
            self.dropout = nn.Dropout(dropout)
            self.attention_heads = attention_heads
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=attention_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.gate = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        if self.feature_version == 2:
            batch, count, dimension = tokens.shape
            normalized = self.norm1(tokens)

            def heads(projection: nn.Module) -> Tensor:
                return projection(normalized).reshape(
                    batch, count, self.attention_heads, dimension // self.attention_heads
                ).transpose(1, 2)

            attended = F.scaled_dot_product_attention(
                heads(self.q_proj), heads(self.k_proj), heads(self.v_proj),
                attn_mask=token_mask[:, None, None, :],
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            attended = attended.transpose(1, 2).reshape(batch, count, dimension)
            encoded = tokens + self.dropout(self.out_proj(attended))
            encoded = encoded + self.dropout(self.feed_forward(self.norm2(encoded)))
        else:
            encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        gated = encoded * self.gate(tokens)
        return gated * token_mask.unsqueeze(-1).to(tokens.dtype)


class PhysicsMoEBackbone(nn.Module):
    """Fast/slow experts with a physics-statistics routing network."""

    def __init__(
        self,
        embed_dim: int = 256,
        layers: int = 6,
        attention_heads: int = 8,
        dropout: float = 0.1,
        feature_version: int = 2,
        local_attention_window: int = 9,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(embed_dim),
                        "fast": FastTransientBranch(
                            embed_dim, attention_heads, dropout, feature_version, local_attention_window
                        ),
                        "slow": SlowDegradationBranch(embed_dim, attention_heads, dropout, feature_version),
                        "router": nn.Sequential(
                            nn.LayerNorm(embed_dim + 3),
                            nn.Linear(embed_dim + 3, max(embed_dim // 2, 32)),
                            nn.GELU(),
                            nn.Linear(max(embed_dim // 2, 32), 2),
                        ),
                        "dropout": nn.Dropout(dropout),
                    }
                )
                for _ in range(max(1, layers))
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        physical_stats: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch, token_count, _ = tokens.shape
        if physical_stats is None:
            physical_stats = tokens.new_zeros((batch, 3))
        physical_stats = physical_stats.to(tokens.dtype)
        route_history = []
        hidden = tokens
        for layer in self.layers:
            normalized = layer["norm"](hidden)
            fast = layer["fast"](normalized, token_mask)
            slow = layer["slow"](normalized, token_mask)
            stats = physical_stats.unsqueeze(1).expand(-1, token_count, -1)
            probabilities = torch.softmax(layer["router"](torch.cat((normalized, stats), dim=-1)), dim=-1)
            ablation = getattr(self, "ablation", "none").split("+")
            if "uniform-router" in ablation:
                probabilities = probabilities * 0 + 0.5
            elif "no-fast" in ablation:
                fast = fast * 0
                probabilities = probabilities * 0 + probabilities.new_tensor([0., 1.])
            elif "no-slow" in ablation:
                slow = slow * 0
                probabilities = probabilities * 0 + probabilities.new_tensor([1., 0.])
            mixed = probabilities[..., 0:1] * fast + probabilities[..., 1:2] * slow
            hidden = hidden + layer["dropout"](mixed)
            hidden = hidden * token_mask.unsqueeze(-1).to(hidden.dtype)
            route_history.append(probabilities)
        hidden = self.final_norm(hidden)
        pooled = _masked_mean(hidden, token_mask, dim=1)
        return {
            "hidden": hidden,
            "pooled": pooled,
            "route_probabilities": torch.stack(route_history, dim=1),
        }


class PHMFoundationModel(nn.Module):
    """Self-supervised PHM base model used by both training stages."""

    def __init__(self, config: Optional[FoundationConfig] = None) -> None:
        super().__init__()
        self.config = config or FoundationConfig()
        if self.config.embed_dim % self.config.attention_heads != 0:
            raise ValueError("embed_dim must be divisible by attention_heads")
        self.tokenizer = AdaptiveTokenizer(
            embed_dim=self.config.embed_dim,
            patch_size=self.config.patch_size,
            patch_stride=self.config.patch_stride,
            dropout=self.config.dropout,
            feature_version=self.config.feature_version,
            physical_patch_seconds=self.config.physical_patch_seconds,
            channel_fusion=self.config.channel_fusion,
            zero_channel_policy=self.config.zero_channel_policy,
        )
        self.backbone = PhysicsMoEBackbone(
            embed_dim=self.config.embed_dim,
            layers=self.config.layers,
            attention_heads=self.config.attention_heads,
            dropout=self.config.dropout,
            feature_version=self.config.feature_version,
            local_attention_window=self.config.local_attention_window,
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.patch_size),
        )
        self.health_head = nn.Sequential(nn.LayerNorm(self.config.embed_dim), nn.Linear(self.config.embed_dim, 1))
        self.configure_ablation(self.config.backbone_ablation)

    def configure_ablation(self, mode: str) -> None:
        mode = normalize_backbone_ablation(mode)
        parts = set(mode.split("+"))
        if self.config.feature_version != 2 and parts & {"no-cwt", "no-order", "no-deformable-offsets", "no-local-attention"}:
            raise ValueError(f"{mode} requires feature_version=2")
        self.config.backbone_ablation = mode
        self.tokenizer.ablation = mode
        self.backbone.ablation = mode
        for layer in self.backbone.layers:
            layer["fast"].ablation = mode
        for module in self.tokenizer.modules():
            if isinstance(module, DeformableConv1d):
                module.disable_offsets = "no-deformable-offsets" in parts

    def encode(
        self,
        signals: Tensor,
        lengths: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        sample_periods: Optional[Tensor] = None,
        mask_ratio: float = 0.0,
        rpm_traces: Optional[Tensor] = None,
        rpm_trace_valid: Optional[Tensor] = None,
        shaft_angles: Optional[Tensor] = None,
        shaft_angle_valid: Optional[Tensor] = None,
        nominal_rpm: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        tokenized = self.tokenizer(
            signals, lengths, channel_mask, sample_periods=sample_periods, mask_ratio=mask_ratio,
            rpm_traces=rpm_traces, rpm_trace_valid=rpm_trace_valid,
            shaft_angles=shaft_angles, shaft_angle_valid=shaft_angle_valid,
            nominal_rpm=nominal_rpm,
        )
        encoded = self.backbone(
            tokenized["tokens"], tokenized["token_mask"], tokenized["physical_stats"]
        )
        return {**tokenized, **encoded}

    def forward(
        self,
        signals: Tensor,
        lengths: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        sample_periods: Optional[Tensor] = None,
        mask_ratio: Optional[float] = None,
        rpm_traces: Optional[Tensor] = None,
        rpm_trace_valid: Optional[Tensor] = None,
        shaft_angles: Optional[Tensor] = None,
        shaft_angle_valid: Optional[Tensor] = None,
        nominal_rpm: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio
        encoded = self.encode(
            signals, lengths, channel_mask, sample_periods, mask_ratio=ratio,
            rpm_traces=rpm_traces, rpm_trace_valid=rpm_trace_valid,
            shaft_angles=shaft_angles, shaft_angle_valid=shaft_angle_valid,
            nominal_rpm=nominal_rpm,
        )
        encoded["reconstruction"] = self.reconstruction_head(encoded["hidden"])
        encoded["health"] = self.health_head(encoded["hidden"]).squeeze(-1)
        return encoded


class LoRALinear(nn.Module):
    """Small rank-r update used for parameter-efficient downstream tuning."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + self.scaling * self.lora_up(self.dropout(self.lora_down(inputs)))


def inject_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> int:
    """Replace Linear layers below ``module`` and return the number replaced."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(module, nn.MultiheadAttention) and name == "out_proj":
            # MultiheadAttention accesses out_proj.weight/bias directly inside
            # its functional forward; replacing the module breaks that path.
            continue
        if isinstance(module, nn.TransformerEncoderLayer) and name in {"linear1", "linear2"}:
            # The optimized inference path also reads these weights directly.
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += inject_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return replaced


class FDHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, pooled: Tensor) -> Tensor:
        return self.classifier(pooled)


class CosineFDHead(nn.Module):
    """Trainable class directions with a fixed positive cosine logit scale."""

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("cosine scale must be finite and positive")
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, pooled: Tensor) -> Tensor:
        return self.scale * F.linear(F.normalize(pooled, dim=-1), F.normalize(self.weight, dim=-1))


class RULHead(nn.Module):
    """Bidirectional self-attention regression head with Weibull parameters."""

    def __init__(self, embed_dim: int, attention_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.weibull = nn.Linear(embed_dim, 2)

    def forward(self, hidden: Tensor, token_mask: Tensor) -> Dict[str, Tensor]:
        attended, _ = self.attention(hidden, hidden, hidden, key_padding_mask=~token_mask, need_weights=False)
        pooled = _masked_mean(self.norm(hidden + attended), token_mask, dim=1)
        raw = self.regressor(pooled).squeeze(-1)
        weibull_raw = self.weibull(pooled)
        return {
            "prediction": raw,
            "weibull_shape": F.softplus(weibull_raw[..., 0]) + 1e-4,
            "weibull_scale": F.softplus(weibull_raw[..., 1]) + 1e-4,
            "pooled": pooled,
        }


class PHMTaskModel(nn.Module):
    """Task-specific wrapper around a pretrained foundation model."""

    def __init__(
        self,
        foundation: PHMFoundationModel,
        task: str,
        num_classes: Optional[int] = None,
        adaptation: str = "lora",
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        fd_head: str = "linear",
        cosine_scale: float = 10.0,
        fd_pooling: str = "mean",
        fd_head_dropout: float = 0.1,
        fd_envelope: bool = False,
        envelope_bins: int = 128,
        envelope_max_hz: float = 1000.0,
        envelope_mode: str = "broadband",
        fd_power_spectrum: bool = False,
        power_bins: int = 256,
        power_channels: int = 2,
        power_logit_scale: float = 10.0,
        confidence_gate: Optional[bool] = None,
        gate_threshold: float = 0.02,
        gate_alpha_low: float = 0.01,
    ) -> None:
        super().__init__()
        task = task.lower()
        if task not in {"fd", "rul"}:
            raise ValueError("task must be fd or rul")
        if task == "fd" and (num_classes is None or num_classes < 2):
            raise ValueError("FD requires num_classes >= 2")
        self.foundation = foundation
        self.task = task
        if fd_pooling not in {"mean", "mean-max"}:
            raise ValueError("fd_pooling must be mean or mean-max")
        if not 0 <= fd_head_dropout < 1:
            raise ValueError("fd_head_dropout must be in [0, 1)")
        if task != "fd" and fd_pooling != "mean":
            raise ValueError("mean-max pooling is an FD-only option")
        self.fd_pool = MeanMaxPooling(foundation.config.embed_dim) if fd_pooling == "mean-max" else None
        if task == "fd":
            if fd_head not in {"linear", "cosine"}:
                raise ValueError("fd_head must be linear or cosine")
            self.head: nn.Module = (
                CosineFDHead(foundation.config.embed_dim, int(num_classes), cosine_scale)
                if fd_head == "cosine" else FDHead(foundation.config.embed_dim, int(num_classes), fd_head_dropout)
            )
        else:
            self.head = RULHead(
                foundation.config.embed_dim,
                attention_heads=foundation.config.attention_heads,
                dropout=foundation.config.dropout,
            )
        self.adaptation = adaptation.lower()
        self._configure_adaptation(
            self.adaptation,
            lora_rank if lora_rank is not None else foundation.config.lora_rank,
            lora_alpha if lora_alpha is not None else foundation.config.lora_alpha,
        )
        if fd_envelope and task != "fd":
            raise ValueError("Envelope residual is currently an FD-only adapter")
        if envelope_mode != "broadband" and not fd_envelope:
            raise ValueError("Multiband envelope requires fd_envelope=True")
        self.fd_envelope = (
            EnvelopeSpectrumResidual(foundation.config.embed_dim, envelope_bins, envelope_max_hz, envelope_mode)
            if fd_envelope else None
        )
        if fd_power_spectrum and (task != "fd" or power_bins < 2 or power_channels < 1):
            raise ValueError("Power-spectrum branch requires FD, bins >=2 and channels >=1")
        self.fd_power_spectrum = PowerSpectrumNormalizer(power_bins * power_channels) if fd_power_spectrum else None
        if fd_power_spectrum:
            self.head = PowerFusionHead(self.head, foundation.config.embed_dim,
                                        power_bins * power_channels, int(num_classes), power_logit_scale)
        if confidence_gate is not None:
            if not fd_power_spectrum or task != "fd":
                raise ValueError("Confidence-gated fusion requires FD with the power-spectrum branch")
            self.head = ConfidenceGatedPowerHead(self.head, gate_threshold, gate_alpha_low, confidence_gate)

    def _configure_adaptation(self, adaptation: str, rank: int, alpha: float) -> None:
        if adaptation not in {"full", "linear", "lora"}:
            raise ValueError("adaptation must be full, linear, or lora")
        if adaptation == "full":
            # The downstream path calls encode(), so these pretraining-only
            # parameters would otherwise be unused trainable DDP parameters.
            for parameter in self.foundation.reconstruction_head.parameters():
                parameter.requires_grad = False
            for parameter in self.foundation.health_head.parameters():
                parameter.requires_grad = False
            self.foundation.tokenizer.mask_token.requires_grad = False
            return
        for parameter in self.foundation.parameters():
            parameter.requires_grad = False
        if adaptation == "lora":
            inject_lora(self.foundation.backbone, rank=rank, alpha=alpha)
            for name, parameter in self.foundation.backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True
        # New fusion parameters are downstream adapters, not frozen pretrained weights.
        if self.foundation.tokenizer.channel_fusion is not None:
            for parameter in self.foundation.tokenizer.channel_fusion.parameters():
                parameter.requires_grad = True

    def pool_fd_features(self, encoded: Dict[str, Tensor]) -> Tensor:
        return encoded["pooled"] if self.fd_pool is None else self.fd_pool(encoded["hidden"], encoded["token_mask"])

    def fd_features(
        self, encoded: Dict[str, Tensor], signals: Tensor, lengths: Optional[Tensor],
        channel_mask: Optional[Tensor], sample_periods: Optional[Tensor],
        power_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Original FD pooling/envelope plus support-normalized spectral input."""
        pooled = self.pool_fd_features(encoded)
        ratio = None
        if self.fd_envelope is not None:
            residual = self.fd_envelope(signals, lengths, channel_mask, sample_periods,
                                        self.foundation.config.zero_channel_policy)
            ratio = residual.detach().float().norm(dim=-1) / pooled.detach().float().norm(dim=-1).clamp_min(1e-6)
            pooled = pooled + residual.to(pooled.dtype)
        if self.fd_power_spectrum is not None:
            if power_features is None:
                raise ValueError("FD power-spectrum branch needs raw per-channel power_features from the dataset")
            spectra = self.fd_power_spectrum(power_features)
            pooled = torch.cat((pooled, spectra.to(pooled.dtype)), dim=-1)
        return pooled, ratio

    def forward(
        self,
        signals: Tensor,
        lengths: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        sample_periods: Optional[Tensor] = None,
        rpm_traces: Optional[Tensor] = None,
        rpm_trace_valid: Optional[Tensor] = None,
        shaft_angles: Optional[Tensor] = None,
        shaft_angle_valid: Optional[Tensor] = None,
        nominal_rpm: Optional[Tensor] = None,
        power_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        encoded = self.foundation.encode(
            signals, lengths, channel_mask, sample_periods, mask_ratio=0.0,
            rpm_traces=rpm_traces, rpm_trace_valid=rpm_trace_valid,
            shaft_angles=shaft_angles, shaft_angle_valid=shaft_angle_valid,
            nominal_rpm=nominal_rpm,
        )
        if self.task == "fd":
            pooled, envelope_ratio = self.fd_features(encoded, signals, lengths, channel_mask, sample_periods,
                                                       power_features)
            result = {**encoded, "pooled": pooled, "logits": self.head(pooled)}
            if isinstance(self.head, ConfidenceGatedPowerHead):
                result["gate_confidence"] = self.head.last_confidence
                result["gate_alpha"] = self.head.last_alpha
            if envelope_ratio is not None:
                result["envelope_residual_ratio"] = envelope_ratio
            return result
        return {**encoded, **self.head(encoded["hidden"], encoded["token_mask"])}


def load_foundation_checkpoint(
    path: str,
    config: Optional[FoundationConfig] = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
    load_weights: bool = True,
) -> PHMFoundationModel:
    """Load a foundation checkpoint, or only its architecture for a random control."""

    payload = torch.load(path, map_location=device, weights_only=True)
    saved_config = payload.get("config") if isinstance(payload, dict) else None
    if config is None and isinstance(saved_config, dict):
        allowed = {key: value for key, value in saved_config.items() if key in FoundationConfig.__annotations__}
        if "feature_version" not in allowed:
            allowed["feature_version"] = 1
        config = FoundationConfig(**allowed)
    model = PHMFoundationModel(config)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if load_weights:
        model.load_state_dict(state, strict=strict)
    return model


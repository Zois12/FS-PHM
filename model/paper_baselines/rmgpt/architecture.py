"""RmGPT components from paper III A-C; unresolved choices are documented.

This is a paper-derived implementation, not verified author source. No T5 or
language weights are used. Input/output tensors retain their channel axis.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RmGPTConfig:
    points: int = 2048
    patch: int = 256
    width: int = 512
    layers: int = 4
    heads: int = 8
    ff_multiplier: int = 4
    prompt_length: int = 10
    max_channels: int = 32
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.points, self.patch, self.width, self.layers, self.heads,
               self.ff_multiplier, self.prompt_length, self.max_channels) < 1:
            raise ValueError("All architecture dimensions must be positive")
        if self.width % self.heads or self.points % self.patch or self.points < 2 * self.patch:
            raise ValueError("Width must divide heads; points must contain >=2 complete patches")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0,1)")


class ChannelTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.channel_norm = nn.LayerNorm(d)
        self.time_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.channel_attention = nn.MultiheadAttention(d, config.heads, config.dropout, batch_first=True)
        self.time_attention = nn.MultiheadAttention(d, config.heads, config.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d * config.ff_multiplier), nn.GELU(),
                                nn.Dropout(config.dropout), nn.Linear(d * config.ff_multiplier, d))
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, channels):
        # B,C,N,D -> B*N,C,D: attention between real sensors at the same token.
        b, c, n, d = x.shape
        value = self.channel_norm(x).permute(0, 2, 1, 3).reshape(b * n, c, d)
        padding = (~channels)[:, None].expand(b, n, c).reshape(b * n, c)
        value = self.channel_attention(value, value, value, key_padding_mask=padding, need_weights=False)[0]
        x = x + self.drop(value.reshape(b, n, c, d).permute(0, 2, 1, 3))
        # B*C,N,D: attention over temporal/token positions within each sensor.
        value = self.time_norm(x).reshape(b * c, n, d)
        value = self.time_attention(value, value, value, need_weights=False)[0]
        x = x + self.drop(value.reshape(b, c, n, d))
        x = x + self.drop(self.ff(self.ff_norm(x)))
        return x.masked_fill(~channels[:, :, None, None], 0)


class TokenBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, c = config.width, config.max_channels
        self.patch_projection = nn.Linear(config.patch, d)
        self.prompt_tokens = nn.Parameter(torch.randn(1, c, config.prompt_length, d) * 0.02)
        self.statistics_projection = nn.Linear(2, d)
        # One learned task anchor followed by magnitude and phase tokens.
        self.task_tokens = nn.Parameter(torch.randn(1, c, 1, d) * 0.02)
        self.magnitude_projection = nn.Linear(config.points // 2 + 1, d)
        self.phase_projection = nn.Linear(config.points // 2 + 1, d)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, d) * 0.02)
        self.blocks = nn.ModuleList(ChannelTimeBlock(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(d)

    def normalize(self, x, channels, visible_points=None):
        if x.ndim != 3 or x.shape[-1] != self.config.points or x.shape[1] > self.config.max_channels:
            raise ValueError("Expected B,C,points with C <= max_channels")
        if channels.shape != x.shape[:2] or not channels.any(dim=1).all():
            raise ValueError("Each sample must have at least one real channel")
        visible = x.float()[..., :visible_points] if visible_points is not None else x.float()
        mean = visible.mean(-1, keepdim=True)
        std = visible.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = ((x.float() - mean) / std).masked_fill(~channels[..., None], 0)
        stats = torch.cat((mean, std), -1)
        return normalized, stats

    def position(self, length, x):
        d = self.config.width
        pos = torch.arange(length, device=x.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, d, 2, device=x.device) * (-math.log(10000.0) / d))
        encoding = torch.zeros(length, d, device=x.device)
        encoding[:, 0::2] = torch.sin(pos * frequency)
        encoding[:, 1::2] = torch.cos(pos * frequency[:encoding[:, 1::2].shape[-1]])
        return encoding.to(x.dtype)[None, None]

    def encode(self, normalized, stats, channels, *, pretrain=False):
        c, p = normalized.shape[1], self.config.patch
        patches = normalized.unfold(-1, p, p)
        if pretrain:
            # The unknown final patch never enters a projection or any statistic.
            signal = self.patch_projection(patches[..., :-1, :])
            signal = torch.cat((signal, self.mask_token.expand(len(signal), c, 1, -1)), 2)
        else:
            signal = self.patch_projection(patches)
        signal = signal + self.position(signal.shape[2], signal)
        prompt = self.prompt_tokens[:, :c] + self.statistics_projection(stats).unsqueeze(2)
        sequence = [prompt.expand(len(signal), -1, -1, -1), signal]
        if not pretrain:
            # Frequency tokens are used downstream only, so future targets cannot
            # leak through FFT during next-patch pretraining.
            with torch.autocast(device_type=normalized.device.type, enabled=False):
                spectrum = torch.fft.rfft(normalized.float())
                magnitude, phase = spectrum.abs().log1p(), torch.angle(spectrum) / math.pi
            sequence.extend((self.task_tokens[:, :c].expand(len(signal), -1, -1, -1),
                             self.magnitude_projection(magnitude).unsqueeze(2),
                             self.phase_projection(phase).unsqueeze(2)))
        hidden = torch.cat(sequence, 2)
        for block in self.blocks:
            hidden = block(hidden, channels)
        return self.output_norm(hidden).masked_fill(~channels[:, :, None, None], 0)


class RmGPT(nn.Module):
    def __init__(self, config=None, task="fd", classes=2):
        super().__init__()
        self.config = config or RmGPTConfig()
        if task not in {"fd", "rul"} or classes < 1:
            raise ValueError("Invalid task or prototype count")
        self.task = task
        self.backbone = TokenBackbone(self.config)
        self.decoder = nn.Linear(self.config.width, self.config.patch)
        self.fault_tokens = nn.Parameter(torch.randn(classes, self.config.max_channels, self.config.width) * 0.02)
        self.regression = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1)) if task == "rul" else None
        self.adaptation = "full"

    def pretrain_loss(self, x, channels):
        normalized, stats = self.backbone.normalize(x, channels, self.config.points - self.config.patch)
        hidden = self.backbone.encode(normalized, stats, channels, pretrain=True)
        prediction = self.decoder(hidden[:, :, -1])
        target = normalized[:, :, -self.config.patch:]
        error = (prediction.float() - target).square().mean(-1)
        loss = error.masked_select(channels).mean()
        return loss, prediction

    def health(self, x, channels):
        normalized, stats = self.backbone.normalize(x, channels)
        hidden = self.backbone.encode(normalized, stats, channels)
        return hidden[:, :, -3:].mean(2)

    def forward(self, x, channels=None):
        if channels is None:
            channels = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        health = self.health(x, channels).float()
        prototypes = self.fault_tokens[:, :x.shape[1]].float()
        mask = channels[:, None, :, None]
        if self.task == "fd":
            # Cosine over the full real-channel health/prototype tensor.
            query = (health[:, None] * mask).flatten(2)
            reference = (prototypes[None] * mask).flatten(2)
            return F.cosine_similarity(query, reference, dim=-1) * 10.0
        difference = ((health[:, None] - prototypes[None]).square() * mask).sum((2, 3))
        distance = (difference / (channels.sum(-1, keepdim=True) * self.config.width)).clamp_min(1e-12).sqrt()
        return self.regression(distance.min(1).values[:, None]).squeeze(-1)

    def set_adaptation(self, mode):
        if mode not in {"full", "prompt"}:
            raise ValueError("Adaptation must be full or prompt")
        self.adaptation = mode
        for parameter in self.parameters():
            parameter.requires_grad_(mode == "full")
        # Token generators and task-specific prototypes comprise adaptation.
        if mode == "prompt":
            for name in ("prompt_tokens", "statistics_projection", "task_tokens",
                         "magnitude_projection", "phase_projection"):
                item = getattr(self.backbone, name)
                if isinstance(item, nn.Parameter):
                    item.requires_grad_(True)
                else:
                    item.requires_grad_(True)
            self.fault_tokens.requires_grad_(True)
            if self.regression is not None:
                self.regression.requires_grad_(True)
        # The reconstruction head never participates in downstream prediction.
        self.decoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.adaptation == "prompt":
            # Frozen backbone dropout must not inject new stochastic features.
            self.backbone.blocks.eval()
        return self

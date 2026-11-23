"""Compact PHM comparison networks with a common signal interface.

Transformer, CNN-LSTM and ResNet18 are standard architecture families. Only the three generic baselines used by the comparison adapters are retained.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BASELINES = ("transformer", "cnn_lstm", "resnet18")

# Names shown as supervised comparisons in RmGPT Tables IV and V. Their
# implementations here remain engineering proxies, not paper reproductions.
PAPER_BASELINES = BASELINES


def prepare_signals(signals: Tensor, lengths: Tensor, channel_mask: Tensor, points: int) -> Tensor:
    """Resample each valid window independently; masked channel average.

    The fixed grid is normalized window time, not a claim of physical sample
    rate invariance. All comparison networks receive exactly this transform.
    """
    if signals.ndim != 3 or points < 64:
        raise ValueError("Expected (batch, channels, time) and at least 64 points")
    if not bool(channel_mask.bool().any(dim=1).all()):
        raise ValueError("Every sample requires a valid channel")
    result = []
    for signal, length, mask in zip(signals, lengths, channel_mask):
        valid = signal[mask.bool(), : int(length)]
        source = valid.unsqueeze(0).float()
        resized = (F.interpolate(source, size=points, mode="area") if int(length) > points else
                   F.interpolate(source, size=points, mode="linear", align_corners=False))
        result.append(resized.mean(1))
    return torch.cat(result, dim=0).unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, before: int, after: int, kernel: int = 7, stride: int = 2):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(before, after, kernel, stride, kernel // 2, bias=False),
                                    nn.BatchNorm1d(after), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, before: int, after: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(before, after, 3, stride, 1, bias=False), nn.BatchNorm1d(after), nn.ReLU(),
                                  nn.Conv1d(after, after, 3, 1, 1, bias=False), nn.BatchNorm1d(after))
        self.skip = nn.Identity() if before == after and stride == 1 else nn.Sequential(
            nn.Conv1d(before, after, 1, stride, bias=False), nn.BatchNorm1d(after))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.main(x) + self.skip(x))


class ResNet18_1D(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(1, width, 7, 2), nn.MaxPool1d(3, 2, 1))
        stages = []
        before = width
        for index, after in enumerate((width, width * 2, width * 4, width * 8)):
            for block in range(2):
                stride = 2 if index > 0 and block == 0 else 1
                stages.append(ResidualBlock(before, after, stride))
                before = after
        self.stages = nn.Sequential(*stages)
        self.output_dim = before

    def forward(self, x: Tensor) -> Tensor:
        return self.stages(self.stem(x)).mean(-1)


class Transformer1D(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.patch = nn.Conv1d(1, width, 32, 16, 8)
        block = nn.TransformerEncoderLayer(width, 4, width * 4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.output_dim = width

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch(x).transpose(1, 2)
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device, dtype=x.dtype) *
                              (-math.log(10000.0) / x.shape[2]))
        positional = torch.zeros_like(x)
        positional[..., 0::2] = torch.sin(position * frequency)
        positional[..., 1::2] = torch.cos(position * frequency[:x[..., 1::2].shape[-1]])
        return self.encoder(x + positional).mean(1)


class CNNLSTM(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(1, width, 9, 4), ConvBlock(width, width, 5, 4),
                                  ConvBlock(width, width, 3, 2))
        self.lstm = nn.LSTM(width, width, batch_first=True, bidirectional=True)
        self.output_dim = width * 2

    def forward(self, x: Tensor) -> Tensor:
        sequence = self.conv(x).transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        return torch.cat((hidden[-2], hidden[-1]), dim=-1)
















def build_encoder(name: str, width: int = 64, layers: int = 2) -> nn.Module:
    if width < 8 or width % 4:
        raise ValueError("--width must be at least 8 and divisible by 4")
    models = {
        "transformer": lambda: Transformer1D(width, layers),
        "cnn_lstm": lambda: CNNLSTM(width),
        "resnet18": lambda: ResNet18_1D(width),
    }
    return models[name.lower()]()


class BaselineModel(nn.Module):
    def __init__(self, architecture: str, task: str, num_classes: int = 0, points: int = 2048,
                 width: int = 64, layers: int = 2):
        super().__init__()
        if architecture.lower() not in BASELINES or task not in {"fd", "rul"}:
            raise ValueError("Unknown architecture or task")
        if task == "fd" and num_classes < 2:
            raise ValueError("FD requires at least two classes")
        if points < 64 or points % 64:
            raise ValueError("--points must be divisible by 64 and at least 64")
        self.architecture, self.task, self.points = architecture.lower(), task, points
        self.encoder = build_encoder(self.architecture, width, layers)
        self.head = nn.Linear(self.encoder.output_dim, num_classes if task == "fd" else 1)
        self.decoder = nn.Linear(self.encoder.output_dim, 64)

    def encode(self, signals: Tensor, lengths: Tensor, channel_mask: Tensor) -> Tensor:
        return self.encoder(prepare_signals(signals, lengths, channel_mask, self.points))

    def forward(self, signals: Tensor, lengths: Tensor, channel_mask: Tensor,
                sample_periods: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        features = self.encode(signals, lengths, channel_mask)
        prediction = self.head(features)
        return {"logits": prediction, "prediction": prediction.squeeze(-1), "features": features}

    def pretraining_loss(self, signals: Tensor, lengths: Tensor, channel_mask: Tensor,
                         mask_ratio: float = 0.4) -> Tensor:
        x = prepare_signals(signals, lengths, channel_mask, self.points)
        target = F.adaptive_avg_pool1d(x, 64).squeeze(1)
        mask = torch.rand(target.shape, device=x.device) < mask_ratio
        mask[:, 0] = True
        expanded = F.interpolate(mask[:, None].float(), size=self.points, mode="nearest").bool()
        masked = x.masked_fill(expanded, 0)
        reconstruction = self.decoder(self.encoder(masked))
        return F.mse_loss(reconstruction[mask], target[mask])

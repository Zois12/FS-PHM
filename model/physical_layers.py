"""Optional physical signal layers implemented with native PyTorch operations."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DeformableConv1d(nn.Module):
    """Conv1D with learned, linearly interpolated sampling offsets.

    Input and output have shape ``(batch, channels, time)``. Zero initialized
    offsets make the initial operation identical to a regular Conv1D. Samples
    outside the input are zero, including at the left and right boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        max_offset: Optional[float] = None,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, stride, dilation, groups) < 1:
            raise ValueError("channel counts, kernel_size, stride, dilation, and groups must be positive")
        if padding < 0 or in_channels % groups or out_channels % groups:
            raise ValueError("padding must be nonnegative and groups must divide channel counts")
        if max_offset is not None and max_offset <= 0:
            raise ValueError("max_offset must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.max_offset = max_offset
        self.offset_conv = nn.Conv1d(
            in_channels, kernel_size, kernel_size, stride=stride,
            padding=padding, dilation=dilation,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.kernel_size
            nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, signals: Tensor) -> Tensor:
        if signals.ndim != 3 or signals.shape[1] != self.in_channels:
            raise ValueError("signals must have shape (batch, in_channels, time)")
        batch, channels, length = signals.shape
        output_length = (
            length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1
        if output_length < 1:
            raise ValueError("kernel is larger than the padded input")

        offsets = self.offset_conv(signals).transpose(1, 2)
        if self.max_offset is not None:
            offsets = offsets.tanh() * self.max_offset
        if getattr(self, "disable_offsets", False):
            offsets = offsets * 0
        base = (
            torch.arange(output_length, device=signals.device, dtype=torch.float32)[:, None]
            * self.stride
            + torch.arange(self.kernel_size, device=signals.device, dtype=torch.float32)[None, :]
            * self.dilation
            - self.padding
        )
        positions = base.unsqueeze(0) + offsets.float()
        lower = positions.floor().long()
        upper = lower + 1
        fraction = (positions - lower).to(signals.dtype)

        def take(indices: Tensor) -> Tensor:
            valid = (indices >= 0) & (indices < length)
            gathered = signals.gather(
                2,
                indices.clamp(0, length - 1).reshape(batch, 1, -1).expand(-1, channels, -1),
            ).reshape(batch, channels, output_length, self.kernel_size)
            return gathered * valid[:, None].to(signals.dtype)

        samples = take(lower) * (1 - fraction[:, None]) + take(upper) * fraction[:, None]
        samples = samples.reshape(
            batch, self.groups, channels // self.groups, output_length, self.kernel_size
        )
        weights = self.weight.reshape(
            self.groups, self.out_channels // self.groups, channels // self.groups, self.kernel_size
        )
        output = torch.einsum(
            "bgitl,gqil->bgqt", samples.to(offsets.dtype), weights.to(offsets.dtype)
        )
        output = output.reshape(batch, self.out_channels, output_length)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).view(1, -1, 1)
        return output


class MorletCWTFeature(nn.Module):
    """Finite-band complex Morlet transform with FFT convolution.

    ``center_frequencies`` are cycles per sample by default. Set
    ``frequencies_hz=True`` and pass each window's duration in seconds to use
    physical frequencies. The result is log-magnitude with shape
    ``(batch, channels, scales, time)``; padded positions are zero. FFTs run in
    float32 for half precision inputs and the result returns to input dtype.
    """

    def __init__(
        self,
        center_frequencies: Sequence[float] = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4),
        *,
        frequencies_hz: bool = False,
        omega0: float = 6.0,
        chunk_size: int = 128,
    ) -> None:
        super().__init__()
        frequencies = tuple(float(value) for value in center_frequencies)
        if not frequencies or any(not math.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError("center_frequencies must contain positive finite values")
        if not frequencies_hz and any(value >= 0.5 for value in frequencies):
            raise ValueError("normalized center_frequencies must be below Nyquist (0.5)")
        if not math.isfinite(omega0) or omega0 <= 0 or chunk_size < 1:
            raise ValueError("omega0 and chunk_size must be positive")
        self.register_buffer("center_frequencies", torch.tensor(frequencies), persistent=True)
        self.frequencies_hz = frequencies_hz
        self.omega0 = float(omega0)
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        signals: Tensor,
        lengths: Optional[Tensor] = None,
        window_seconds: Optional[Tensor] = None,
    ) -> Tensor:
        if signals.ndim != 3:
            raise ValueError("signals must have shape (batch, channels, time)")
        batch, channels, max_length = signals.shape
        if max_length < 1:
            raise ValueError("time dimension must be nonempty")
        if lengths is None:
            lengths = torch.full((batch,), max_length, device=signals.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, device=signals.device, dtype=torch.long)
        if lengths.shape != (batch,) or (lengths < 1).any() or (lengths > max_length).any():
            raise ValueError("lengths must be within [1, time] for every sample")
        if self.frequencies_hz:
            if window_seconds is None:
                raise ValueError("window_seconds is required for frequencies in Hz")
            window_seconds = torch.as_tensor(window_seconds, device=signals.device, dtype=torch.float32)
            if window_seconds.shape != (batch,) or not torch.isfinite(window_seconds).all() or (window_seconds <= 0).any():
                raise ValueError("window_seconds must contain one positive finite duration per sample")

        result = signals.new_zeros((batch, channels, self.center_frequencies.numel(), max_length))
        fft_dtype = torch.float64 if signals.dtype == torch.float64 else torch.float32
        for valid_length in torch.unique(lengths).tolist():
            rows = torch.nonzero(lengths == valid_length, as_tuple=True)[0]
            for row_chunk in rows.split(self.chunk_size):
                if self.frequencies_hz:
                    normalized = (
                        self.center_frequencies.to(device=signals.device, dtype=fft_dtype)[None, :]
                        * window_seconds[row_chunk, None].to(fft_dtype)
                        / valid_length
                    )
                    if (normalized >= 0.5).any():
                        raise ValueError("a center frequency reaches or exceeds the sample Nyquist frequency")
                else:
                    normalized = self.center_frequencies.to(
                        device=signals.device, dtype=fft_dtype
                    )[None, :].expand(row_chunk.numel(), -1)
                scales = self.omega0 / (2 * math.pi * normalized)
                padding = max(1, math.ceil(4 * scales.max().item()))
                fft_length = 1 << (valid_length + 2 * padding - 1).bit_length()
                segment = F.pad(signals[row_chunk, :, :valid_length].to(fft_dtype), (padding, padding))
                spectrum = torch.fft.fft(segment, n=fft_length, dim=-1)
                angular_frequency = (
                    2 * math.pi * torch.fft.fftfreq(fft_length, device=signals.device, dtype=fft_dtype)
                )
                scaled_frequency = scales[:, :, None] * angular_frequency[None, None, :]
                response = (
                    torch.exp(-0.5 * (scaled_frequency - self.omega0).square())
                    - math.exp(-0.5 * self.omega0**2)
                    * torch.exp(-0.5 * scaled_frequency.square())
                )
                response = response * (angular_frequency > 0)[None, None, :]
                coefficients = torch.fft.ifft(
                    spectrum[:, :, None, :] * response[:, None, :, :], dim=-1
                )[..., padding : padding + valid_length]
                magnitudes = coefficients.abs().log1p().to(signals.dtype)
                result.index_copy_(
                    0, row_chunk,
                    F.pad(magnitudes, (0, max_length - valid_length)),
                )
        return result


class LocalSparseSelfAttention(nn.Module):
    """Sliding-window multi-head attention with O(time * window_size) storage.

    ``token_mask`` has shape ``(batch, time)`` and may contain arbitrary holes.
    Invalid queries produce zero output; invalid keys receive zero probability.
    Explicit query/key/value/output Linear layers support the repo's LoRA path.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 9,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if window_size < 1 or not 0 <= dropout < 1:
            raise ValueError("window_size must be positive and dropout must be in [0, 1)")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.causal = causal
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError("tokens must have shape (batch, time, embed_dim)")
        batch, length, _ = tokens.shape
        if length < 1:
            raise ValueError("time dimension must be nonempty")
        if token_mask is None:
            token_mask = torch.ones((batch, length), device=tokens.device, dtype=torch.bool)
        elif token_mask.shape != (batch, length):
            raise ValueError("token_mask must have shape (batch, time)")
        else:
            token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        left = self.window_size - 1 if self.causal else self.window_size // 2
        right = 0 if self.causal else self.window_size - 1 - left

        def heads(projection: nn.Linear) -> Tensor:
            return projection(tokens).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
        key_windows = F.pad(key, (0, 0, left, right)).unfold(2, self.window_size, 1)
        value_windows = F.pad(value, (0, 0, left, right)).unfold(2, self.window_size, 1)
        key_windows = key_windows.permute(0, 1, 2, 4, 3)
        value_windows = value_windows.permute(0, 1, 2, 4, 3)
        valid_keys = F.pad(token_mask, (left, right)).unfold(1, self.window_size, 1)
        scores = torch.einsum("bhtd,bhtwd->bhtw", query, key_windows).float()
        scores = scores * (self.head_dim**-0.5)
        probabilities = torch.softmax(scores.masked_fill(~valid_keys[:, None], -1e9), dim=-1)
        probabilities = probabilities * valid_keys[:, None]
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        context = torch.einsum(
            "bhtw,bhtwd->bhtd", self.dropout(probabilities).to(value.dtype), value_windows
        )
        context = context.transpose(1, 2).reshape(batch, length, self.embed_dim)
        output = self.out_proj(context)
        return output * token_mask.unsqueeze(-1).to(output.dtype)

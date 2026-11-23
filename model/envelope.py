"""Whole-window, per-channel envelope spectra for optional FD adaptation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EnvelopeSpectrumResidual(nn.Module):
    """Broadband or multiband Hilbert envelopes on a physical modulation grid.

    No validation-selected carrier band or bearing-specific frequency is used.
    Projection and optional global band fusion weights are learned. Initialization
    does not consume random numbers, preserving the existing model's RNG sequence.
    """

    def __init__(self, embed_dim: int, bins: int = 128, max_hz: float = 1000.0,
                 mode: str = "broadband") -> None:
        super().__init__()
        if bins < 2 or not math.isfinite(max_hz) or max_hz <= 0:
            raise ValueError("Envelope bins must be >= 2 and max_hz finite and positive")
        if mode not in {"broadband", "multiband"}:
            raise ValueError("Envelope mode must be broadband or multiband")
        self.mode = mode
        self.register_buffer("frequencies", torch.linspace(max_hz / bins, max_hz, bins))
        self.weight = nn.Parameter(torch.zeros(embed_dim, bins))
        if mode == "multiband":
            # Fixed fractions of each sample's Nyquist, NOT validation-tuned Hz.
            self.register_buffer("carrier_bands", torch.tensor([[0.05, 0.40], [0.25, 0.70], [0.55, 0.95]]))
            self.band_logits = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("band_logits", None)

    def band_weights(self) -> Tensor:
        return self.weight.new_ones(1) if self.band_logits is None else self.band_logits.softmax(0)

    def carrier_response(self, length: int, band: Tensor) -> Tensor:
        """Cosine-tapered FFT bandpass with 10% transition at both edges."""
        normalized_frequency = torch.fft.fftfreq(length, device=band.device).abs() * 2
        low, high = band.unbind()
        transition = (high - low) * 0.1
        rise = ((normalized_frequency - low) / transition).clamp(0, 1)
        fall = ((high - normalized_frequency) / transition).clamp(0, 1)
        return (0.5 - 0.5 * torch.cos(torch.pi * rise)) * (0.5 - 0.5 * torch.cos(torch.pi * fall))

    @torch.no_grad()
    def extract_bands(
        self, signals: Tensor, lengths: Tensor | None, channel_mask: Tensor | None,
        window_seconds: Tensor | None, zero_channel_policy: str = "keep",
    ) -> Tensor:
        if signals.ndim == 2:
            signals = signals.unsqueeze(1)
        if signals.ndim != 3:
            raise ValueError("Envelope signals must be (batch, channels, time)")
        batch, channels, padded_length = signals.shape
        if window_seconds is None:
            raise ValueError("Envelope features require physical window duration (sample_periods)")
        device = signals.device
        seconds = window_seconds.to(device=device, dtype=torch.float32)
        if seconds.shape != (batch,) or not bool((torch.isfinite(seconds) & (seconds > 0)).all()):
            raise ValueError("Envelope window durations must be finite, positive and one per sample")
        if lengths is None:
            lengths = torch.full((batch,), padded_length, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)
        if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > padded_length)).any()):
            raise ValueError("Envelope lengths must lie within the padded signal")
        mask = torch.ones((batch, channels), device=device, dtype=torch.bool) if channel_mask is None else channel_mask.to(device=device, dtype=torch.bool)
        if mask.shape != (batch, channels) or zero_channel_policy not in {"keep", "drop-zero"}:
            raise ValueError("Invalid envelope channel mask or zero-channel policy")
        band_count = 4 if self.mode == "multiband" else 1
        features = torch.zeros((batch, band_count, len(self.frequencies)), device=device, dtype=torch.float32)
        # Group equal lengths, cropping BEFORE Hilbert/FFT/Hann. Extra padding or
        # a longer batch peer must not alter a sample's physical frequency axis.
        with torch.autocast(device_type=device.type, enabled=False):
            for length in lengths.unique().tolist():
                rows = (lengths == length).nonzero(as_tuple=True)[0]
                source = signals.index_select(0, rows)[..., :length].float()
                active = mask.index_select(0, rows)
                if zero_channel_policy == "drop-zero":
                    active = active & source.ne(0).any(-1)
                if not bool(active.any(-1).all()):
                    raise ValueError("Envelope spectrum requires at least one available channel per sample")
                source = source.masked_fill(~active.unsqueeze(-1), 0)
                if not bool(torch.isfinite(source).all()):
                    raise ValueError("Nonfinite valid input in envelope spectrum")
                if length < 4:
                    continue  # Too short for a meaningful envelope spectrum.
                source = source - source.mean(-1, keepdim=True)
                spectrum = torch.fft.fft(source, dim=-1)
                hilbert = source.new_zeros(length)
                hilbert[0] = 1
                hilbert[1:(length + 1) // 2] = 2
                if length % 2 == 0:
                    hilbert[length // 2] = 1
                window = torch.hann_window(length, periodic=False, device=device)
                positions = seconds.index_select(0, rows)[:, None] * self.frequencies.float()[None, :]
                last_bin = length // 2
                in_range = positions <= last_bin
                lower = positions.floor().long().clamp(0, last_bin)
                upper = (lower + 1).clamp_max(last_bin)
                fraction = (positions - lower).clamp(0, 1)[:, None, :]
                lower = lower[:, None, :].expand(-1, channels, -1)
                upper = upper[:, None, :].expand(-1, channels, -1)
                descriptors = []
                responses = [None]
                if self.mode == "multiband":
                    responses.extend(self.carrier_response(length, band) for band in self.carrier_bands)
                for band_index, response in enumerate(responses):
                    analytic_spectrum = spectrum * hilbert
                    if response is not None:
                        analytic_spectrum = analytic_spectrum * response
                    envelope = torch.fft.ifft(analytic_spectrum, dim=-1).abs()
                    envelope_mean = envelope.mean(-1, keepdim=True)
                    # Relative modulation, computed independently per carrier band.
                    modulation = (envelope - envelope_mean) / envelope_mean.clamp_min(1e-6)
                    if response is not None:
                        # Avoid amplifying tiny input/FFT roundoff in an empty
                        # carrier band (amplitude floor 1e-4 of input RMS).
                        floor = source.square().mean(-1, keepdim=True).sqrt() * 1e-4
                        modulation = modulation.masked_fill(envelope_mean <= floor.clamp_min(1e-8), 0)
                    amplitude = torch.fft.rfft(modulation * window, dim=-1).abs() * (2 / window.sum())
                    amplitude[..., 0] = 0
                    sampled = amplitude.gather(-1, lower) * (1 - fraction) + amplitude.gather(-1, upper) * fraction
                    band_range = in_range
                    if response is not None:
                        band = self.carrier_bands[band_index - 1]
                        bandwidth = (band[1] - band[0]) * length / (2 * seconds.index_select(0, rows))
                        band_range = band_range & (self.frequencies[None, :] <= bandwidth[:, None])
                    sampled = sampled * band_range[:, None, :] * active[:, :, None]
                    # Combine channel spectra, never raw waveforms.
                    descriptors.append(sampled.log1p().sum(1) / active.sum(1)[:, None])
                features.index_copy_(0, rows, torch.stack(descriptors, dim=1))
        return features

    def fuse(self, descriptors: Tensor) -> Tensor:
        if self.band_logits is None:
            return descriptors[:, 0]
        return (descriptors * self.band_weights()[None, :, None]).sum(dim=1)

    @torch.no_grad()
    def extract(
        self, signals: Tensor, lengths: Tensor | None, channel_mask: Tensor | None,
        window_seconds: Tensor | None, zero_channel_policy: str = "keep",
    ) -> Tensor:
        """Diagnostic descriptor; training uses differentiable fusion in forward."""
        return self.fuse(self.extract_bands(signals, lengths, channel_mask, window_seconds, zero_channel_policy))

    def forward(
        self, signals: Tensor, lengths: Tensor | None, channel_mask: Tensor | None,
        window_seconds: Tensor | None, zero_channel_policy: str = "keep",
    ) -> Tensor:
        features = self.fuse(self.extract_bands(signals, lengths, channel_mask, window_seconds, zero_channel_policy))
        return F.linear(features, self.weight)
